"""Single-env torch BReP eval over N DISTINCT seeds (each reset(seed) = one game),
both seats — an apples-to-apples sample matching how the submission test runs, but
fast (torch). Confirms the policy's margin on a uniform per-seed game sample."""
from __future__ import annotations
import argparse
import numpy as np
import torch

from python.train.train_ppo import _build_policy, _apply_residual_edits, _moves_to_flat_rows
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.encoding import observation_dim, EncoderConfig
from python.orbit_wars_gym.entities import fleet_owner, fleet_ships, planet_owner, planet_ships


def _ships(state, player):
    own = enemy = 0.0
    for p in state.get("planets", []):
        o = planet_owner(p)
        if o == player:
            own += planet_ships(p)
        elif o >= 0:
            enemy += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if o == player:
            own += fleet_ships(f)
        elif o >= 0:
            enemy += fleet_ships(f)
    return own, enemy


def _play(model, k_max, seat, seeds, steps, comets, device):
    opp_seat = 1 - seat
    margins = []
    for seed in seeds:
        backend = RustBatchBackend(num_envs=1, num_players=2, seed=int(seed),
                                   config=RustConfig(enable_comets=comets))
        states = backend.reset(int(seed))
        base_prod = make_isolated_opponent("producer")
        opp_prod = make_isolated_opponent("producer")
        for _ in range(steps):
            state = states[0]
            obs = np.asarray(backend.encoded_states(seat)[0], dtype=np.float32)
            base = [list(m) for m in base_prod(state, seat)]
            mask = torch.zeros(1, k_max, dtype=torch.bool)
            mask[0, : min(len(base), k_max)] = True
            with torch.no_grad():
                logits = model.forward(torch.tensor(obs, device=device).unsqueeze(0))["edit"]
                greedy = torch.where(mask, logits.argmax(-1), torch.zeros_like(logits.argmax(-1)))[0].tolist()
            amoves = _apply_residual_edits(state, base, greedy, k_max)
            omoves = opp_prod(state, opp_seat)
            rows = _moves_to_flat_rows(0, seat, amoves) + _moves_to_flat_rows(0, opp_seat, omoves)
            flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            backend.step_flat_with_encoded_states(flat, seat)
            states = backend.states()
        own, enemy = _ships(states[0], seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/ppo/brep_gpu/c05.pt")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = _build_policy("producer_residual", observation_dim(EncoderConfig()))
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    seeds = list(range(1000, 1000 + args.seeds))
    m0 = _play(model, model.k_max, 0, seeds, args.episode_steps, True, args.device)
    m1 = _play(model, model.k_max, 1, seeds, args.episode_steps, True, args.device)
    print(f"single-env torch BReP vs Producer ({args.seeds} distinct seeds, both seats):")
    print(f"  mean={float(np.mean(m0+m1)):+.4f}  seat0={float(np.mean(m0)):+.4f}  seat1={float(np.mean(m1)):+.4f}")


if __name__ == "__main__":
    main()
