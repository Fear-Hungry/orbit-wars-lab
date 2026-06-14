"""AlphaZero iteration on GridNet: MCTS-improved play → train policy+value → repeat.

The bare MCTS reached −0.77 vs the producer (best of the day) but does NOT scale
with simulations — the value net (BC-trained to mimic the holdwave) misjudges deep
states, so more search trusts a bad evaluator. AlphaZero's fix: train the value net
on REAL outcomes from MCTS-guided play, and train the policy to imitate the MCTS
visit distribution (which is stronger than the raw policy). A better value → a
better MCTS → better data → ... the virtuous cycle. Each iteration: generate
self-improved games, fit policy (to MCTS action) + value (to game outcome), evaluate
the BARE policy (submission-cheap) and the MCTS (with the improved value).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from python.agents.policy import PLANET_N, GridNetActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.action_decoder import decode_gridnet_action, gridnet_planet_mask
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim
from python.train.train_ppo import _scores_margin, evaluate_gridnet_margin
from scripts.gridnet_mcts import mcts_action


def generate_games(model, *, opponent, seeds, steps, n_sim, k, c_puct, max_depth, device):
    """MCTS-guided games vs the opponent; record (obs, mcts_action, mask, outcome)."""
    obs_l, act_l, mask_l, out_l = [], [], [], []
    dev = torch.device(device)
    for seed in seeds:
        seat = seed % 2
        op = 1 - seat
        b = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(episode_steps=steps, enable_comets=True))
        s = b.reset(seed)[0]
        real_opp = make_isolated_opponent(opponent)
        sim_opp = make_isolated_opponent(opponent)
        sim = RustBatchBackend(num_envs=1, num_players=2, seed=0, config=RustConfig(enable_comets=True))
        game_obs, game_act, game_mask = [], [], []
        last = [0.0, 0.0]
        for _ in range(steps):
            a = mcts_action(model, s, seat, opp_model=sim_opp, n_sim=n_sim, k=k, c_puct=c_puct, max_depth=max_depth, sim=sim, device=dev)
            game_obs.append(encode_state(s, seat, DEFAULT_ENCODER_CONFIG).astype(np.float32))
            game_act.append(a)
            game_mask.append(gridnet_planet_mask(s, seat))
            rows = [[0.0, float(seat), float(m[0]), float(m[1]), float(m[2])] for m in decode_gridnet_action(s, seat, a)]
            rows += [[0.0, float(op), float(m[0]), float(m[1]), float(m[2])] for m in real_opp(s, op)]
            arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            out, st = b.step_flat_with_states(arr)
            s = st[0]
            sc = out[0].get("scores") or out[0].get("rewards")
            if sc:
                last = list(sc)
            if out[0].get("done"):
                break
        outcome = _scores_margin(last, seat, 2)
        obs_l += game_obs
        act_l += game_act
        mask_l += game_mask
        out_l += [outcome] * len(game_obs)
    return {
        "obs": np.stack(obs_l), "action": np.stack(act_l),
        "mask": np.stack(mask_l), "outcome": np.asarray(out_l, dtype=np.float32),
    }


def fit(model, data, *, epochs, lr, device, value_coef=1.0):
    dev = torch.device(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=dev)
    act = torch.as_tensor(data["action"], dtype=torch.long, device=dev)
    mask = torch.as_tensor(data["mask"], dtype=torch.bool, device=dev)
    val = torch.as_tensor(data["outcome"], dtype=torch.float32, device=dev)
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            o, a, m, vv = obs[idx], act[idx], mask[idx], val[idx]
            out = model.forward(o)
            mf = m.float()
            launched = (a[..., 0] == 1).float() * mf
            l_ce = F.cross_entropy(out["launch"].reshape(-1, 2), a[..., 0].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            t_ce = F.cross_entropy(out["target"].reshape(-1, out["target"].shape[-1]), a[..., 1].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            f_ce = F.cross_entropy(out["frac"].reshape(-1, out["frac"].shape[-1]), a[..., 2].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            o_ce = F.cross_entropy(out["offset"].reshape(-1, out["offset"].shape[-1]), a[..., 3].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            policy_loss = (l_ce * mf).sum() / mf.sum().clamp_min(1.0) + ((t_ce + f_ce + o_ce) * launched).sum() / launched.sum().clamp_min(1.0)
            value_loss = F.mse_loss(out["value"], vv)
            loss = policy_loss + value_coef * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="artifacts/bc/gridnet_bc_big512.pt")
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-sim", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/bc/gridnet_az.pt")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    sm = ck["summary"]
    H, EH = sm["hidden"], sm["entity_hidden"]
    model = GridNetActorCritic(observation_dim(), entity_hidden=EH, hidden=H).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    base = evaluate_gridnet_margin(model, opponent_name=args.opponent, seeds=6, episode_steps=200, device=args.device)["mean_score_margin"]
    print(json.dumps({"iter": -1, "bare_policy_vs_" + args.opponent: round(base, 4)}), flush=True)

    for it in range(args.iterations):
        data = generate_games(model, opponent=args.opponent,
                              seeds=list(range(it * 50, it * 50 + args.games)),
                              steps=args.steps, n_sim=args.n_sim, k=args.k, c_puct=1.5,
                              max_depth=args.max_depth, device=args.device)
        model.train()
        model = fit(model, data, epochs=args.epochs, lr=3e-4, device=args.device)
        model.eval()
        bare = evaluate_gridnet_margin(model, opponent_name=args.opponent, seeds=6, episode_steps=200, device=args.device)["mean_score_margin"]
        grd = evaluate_gridnet_margin(model, opponent_name="greedy", seeds=4, episode_steps=200, device=args.device)["mean_score_margin"]
        print(json.dumps({"iter": it, "examples": int(data["obs"].shape[0]),
                          "mean_mcts_outcome": round(float(data["outcome"].mean()), 4),
                          "bare_policy_vs_" + args.opponent: round(bare, 4),
                          "vs_greedy": round(grd, 4)}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save({"model_state_dict": model.state_dict(), "summary": {"arch": "gridnet", "hidden": H, "entity_hidden": EH}}, out)
    print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
