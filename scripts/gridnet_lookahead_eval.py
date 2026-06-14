"""GridNet + policy-rollout lookahead (Tesauro 1995 / AlphaZero policy improvement).

The GridNet policy is reactive (loses −1.0 to planners) but its VALUE net is
excellent (EV 0.9+). So give the policy planning at DECISION time, no new training:
the policy proposes k candidate actions; each is simulated a few steps ahead in the
real engine (reset_from_states), the opponent modelled by the policy itself; the
candidate whose lookahead leads to the best value/score is played. If this crosses
the reactive→planner cliff (beats the producer where the bare policy is annihilated),
search — not more training — is the missing ingredient.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from python.agents.policy import GridNetActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.action_decoder import decode_gridnet_action, gridnet_planet_mask
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim


def _sample_actions(model, obs_t, mask_t, k):
    """k sampled per-planet actions (B=k via repeat)."""
    obs_k = obs_t.repeat(k, 1)
    mask_k = mask_t.repeat(k, 1)
    with torch.no_grad():
        a, _, _, _ = model.get_action_and_value(obs_k, masks={"planet": mask_k})
    return a.cpu().numpy()  # (k, PLANET_N, 4)


def _greedy_action(model, state, player, device):
    mask = torch.as_tensor(gridnet_planet_mask(state, player), dtype=torch.bool, device=device).unsqueeze(0)
    obs = torch.as_tensor(encode_state(state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.forward(obs)
        launch = torch.where(mask, out["launch"].argmax(-1), torch.zeros_like(out["launch"].argmax(-1)))
        a = torch.stack([launch, out["target"].argmax(-1), out["frac"].argmax(-1), out["offset"].argmax(-1)], dim=-1)
    return a[0].cpu().numpy()


def decide_lookahead(model, state, player, *, k, depth, device, opp_model=None):
    """Pick the candidate whose k-env rollout yields the best value. ``opp_model``
    (callable (state,player)->moves) models the opponent in the simulation; if None,
    the GridNet policy itself plays the opponent (optimistic when the real foe is a
    stronger planner)."""
    obs_t = torch.as_tensor(encode_state(state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(gridnet_planet_mask(state, player), dtype=torch.bool, device=device).unsqueeze(0)
    cands = _sample_actions(model, obs_t, mask_t, k)
    opp = 1 - player
    sim = RustBatchBackend(num_envs=k, num_players=2, seed=0, config=RustConfig(enable_comets=True))
    sim.reset_from_states([state] * k)
    states_k = [state] * k
    for step in range(depth):
        rows = []
        for i in range(k):
            my = cands[i] if step == 0 else _greedy_action(model, states_k[i], player, device)
            for mv in decode_gridnet_action(states_k[i], player, my):
                rows.append([float(i), float(player), float(mv[0]), float(mv[1]), float(mv[2])])
            opp_moves = opp_model(states_k[i], opp) if opp_model is not None else \
                decode_gridnet_action(states_k[i], opp, _greedy_action(model, states_k[i], opp, device))
            for mv in opp_moves:
                rows.append([float(i), float(opp), float(mv[0]), float(mv[1]), float(mv[2])])
        arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        _, states_k = sim.step_flat_with_states(arr)
    # evaluate the k resulting states with the value net (seat=player)
    obs_final = torch.as_tensor(
        np.stack([encode_state(states_k[i], player, DEFAULT_ENCODER_CONFIG) for i in range(k)]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        values = model.forward(obs_final)["value"].cpu().numpy()
    return cands[int(np.argmax(values))]


def play(model, opponent_name, seat, seed, *, k, depth, device, steps=256, handicap=1.0, model_opp=False):
    b = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(episode_steps=steps, enable_comets=True))
    s = b.reset(seed)[0]
    opp = make_isolated_opponent(opponent_name)
    last = [0.0, 0.0]
    op = 1 - seat
    # model the real opponent inside the lookahead (fresh instance, no state leak)
    sim_opp = make_isolated_opponent(opponent_name) if model_opp else None
    for _ in range(steps):
        a = decide_lookahead(model, s, seat, k=k, depth=depth, device=device, opp_model=sim_opp)
        rows = [[0.0, float(seat), float(m[0]), float(m[1]), float(m[2])] for m in decode_gridnet_action(s, seat, a)]
        om = opp(s, op)
        if handicap < 1.0:
            om = [[m[0], m[1], max(1.0, float(m[2]) * handicap)] for m in om]
        rows += [[0.0, float(op), float(m[0]), float(m[1]), float(m[2])] for m in om]
        arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        out, st = b.step_flat_with_states(arr)
        s = st[0]
        sc = out[0].get("scores") or out[0].get("rewards")
        if sc:
            last = list(sc)
        if out[0].get("done"):
            break
    s0, s1 = float(last[0]), float(last[1])
    denom = max(abs(s0) + abs(s1), 1.0)
    return (s0 - s1) / denom if seat == 0 else (s1 - s0) / denom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="artifacts/bc/gridnet_bc_2p_big.pt")
    ap.add_argument("--opponents", default="producer,pgs")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-opp", action="store_true", help="model the real opponent inside the lookahead")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = GridNetActorCritic(observation_dim()).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for opp in args.opponents.split(","):
        ms = []
        for sd in range(args.seeds):
            ms.append(0.5 * (play(model, opp, 0, sd, k=args.k, depth=args.depth, device=dev, steps=args.steps, model_opp=args.model_opp)
                             + play(model, opp, 1, sd, k=args.k, depth=args.depth, device=dev, steps=args.steps, model_opp=args.model_opp)))
        print(f"GridNet+lookahead(k={args.k},d={args.depth}) vs {opp}: margem={np.mean(ms):+.3f}")


if __name__ == "__main__":
    main()
