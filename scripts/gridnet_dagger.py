"""DAgger (Ross et al. 2011) of the holdwave into the GridNet policy.

BC imitates the holdwave only on the holdwave's OWN self-play states. The trained
policy diverges and visits states the holdwave never saw (covariate shift) — and
there it errs and gets annihilated by planners. DAgger fixes this: roll out the
CURRENT policy, and on the states IT visits, query the holdwave oracle for the
right action, aggregate into the dataset, retrain. Iterating drives the imitation
on the policy's own distribution, which BC cannot reach.

If this pushes margin vs the producer above −1.0, covariate shift (not capacity)
was the wall and there is a path; if it stalls, the planner gap is fundamental.
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
from python.orbit_wars_gym.action_decoder import decode_gridnet_action, gridnet_planet_mask, invert_gridnet_moves
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim
from python.train.train_ppo import evaluate_gridnet_margin
from scripts.train_gridnet_bc import collect as bc_collect
from scripts.train_gridnet_bc import train as bc_train


def _greedy(model, state, player, device):
    mask = torch.as_tensor(gridnet_planet_mask(state, player), dtype=torch.bool, device=device).unsqueeze(0)
    obs = torch.as_tensor(encode_state(state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.forward(obs)
        launch = torch.where(mask, out["launch"].argmax(-1), torch.zeros_like(out["launch"].argmax(-1)))
        a = torch.stack([launch, out["target"].argmax(-1), out["frac"].argmax(-1), out["offset"].argmax(-1)], dim=-1)
    return a[0].cpu().numpy()


def collect_policy_states(model, *, opponent_name, seeds, episode_steps, device):
    """Roll out the CURRENT policy vs an opponent; on each visited state, label seat 0
    with the holdwave oracle's action (invert_gridnet_moves of its moves)."""
    obs_l, act_l, mask_l = [], [], []
    oracle = make_isolated_opponent("pgs")  # holdwave expert
    for seed in seeds:
        b = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(episode_steps=episode_steps, enable_comets=True))
        s = b.reset(seed)[0]
        opp = make_isolated_opponent(opponent_name)
        for _ in range(episode_steps):
            # label THIS visited state with the oracle's action
            oracle_moves = [list(m) for m in oracle(s, 0)]
            a_label, _ = invert_gridnet_moves(s, 0, oracle_moves)
            obs_l.append(encode_state(s, 0, DEFAULT_ENCODER_CONFIG).astype(np.float32))
            act_l.append(a_label)
            mask_l.append(gridnet_planet_mask(s, 0))
            # advance: policy plays seat 0 (so we visit ITS distribution), opp seat 1
            rows = [[0.0, 0.0, float(m[0]), float(m[1]), float(m[2])] for m in decode_gridnet_action(s, 0, _greedy(model, s, 0, device))]
            rows += [[0.0, 1.0, float(m[0]), float(m[1]), float(m[2])] for m in opp(s, 1)]
            arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            out, st = b.step_flat_with_states(arr)
            s = st[0]
            if out[0].get("done"):
                break
    return {"obs": np.stack(obs_l), "action": np.stack(act_l), "mask": np.stack(mask_l)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="artifacts/bc/gridnet_bc_big512.pt")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--dagger-seeds", type=int, default=12, help="policy-rollout games per iter")
    ap.add_argument("--episode-steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--entity-hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mix-opponent", default="producer", help="opponent the policy rolls out against (its hard distribution)")
    ap.add_argument("--out", default="artifacts/bc/gridnet_dagger.pt")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    sm = ck["summary"]
    model = GridNetActorCritic(observation_dim(), entity_hidden=sm["entity_hidden"], hidden=sm["hidden"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # seed dataset: the original BC self-play states (holdwave distribution)
    agg = bc_collect(list(range(16)), 2, args.episode_steps)
    base_margin = evaluate_gridnet_margin(model, opponent_name=args.mix_opponent, seeds=6, episode_steps=256, device=args.device)["mean_score_margin"]
    print(json.dumps({"iter": -1, "margin_vs_" + args.mix_opponent: round(base_margin, 4),
                      "vs_greedy": round(evaluate_gridnet_margin(model, opponent_name="greedy", seeds=6, episode_steps=256, device=args.device)["mean_score_margin"], 4)}), flush=True)

    for it in range(args.iterations):
        # roll out the CURRENT policy on its HARD distribution (vs producer) + label with oracle
        new = collect_policy_states(model, opponent_name=args.mix_opponent,
                                    seeds=list(range(it * 100, it * 100 + args.dagger_seeds)),
                                    episode_steps=args.episode_steps, device=dev)
        for kk in ("obs", "action", "mask"):
            agg[kk] = np.concatenate([agg[kk], new[kk]], axis=0)
        model, metrics = bc_train(agg, epochs=args.epochs, batch_size=256, lr=5e-4, seed=it,
                                  hidden=args.hidden, entity_hidden=args.entity_hidden, device=args.device)
        model = model.to(dev)
        model.eval()
        m_prod = evaluate_gridnet_margin(model, opponent_name=args.mix_opponent, seeds=6, episode_steps=256, device=args.device)["mean_score_margin"]
        m_grd = evaluate_gridnet_margin(model, opponent_name="greedy", seeds=6, episode_steps=256, device=args.device)["mean_score_margin"]
        print(json.dumps({"iter": it, "examples": int(agg["obs"].shape[0]),
                          "target_acc": round(metrics["target_acc"], 4),
                          "margin_vs_" + args.mix_opponent: round(m_prod, 4),
                          "vs_greedy": round(m_grd, 4)}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save({"model_state_dict": model.state_dict(), "summary": {"arch": "gridnet", "hidden": args.hidden, "entity_hidden": args.entity_hidden}}, out)
    print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
