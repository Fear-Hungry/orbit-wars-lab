"""Direct eval for a BReP (producer_residual) checkpoint — no submission export.

Plays the policy vs Producer at 500 steps, BOTH seats, over N seeds (batched), and
reports the seat-averaged normalized ship-score margin (>0 = beats Producer). This
sidesteps the arch-aware submission exporter; it replicates the BReP per-step move
(Producer base plan + learned per-slot edit) directly against the Rust backend."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from python.train.train_ppo import _apply_residual_edits, _moves_to_flat_rows, _build_policy
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import (
    fleet_owner, fleet_ships, planet_owner, planet_ships,
)


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


def _play_seat(model, k_max, agent_seat, seeds, episode_steps, enable_comets, device, opponent="producer"):
    """Agent plays as `agent_seat`; Producer plays the other seat. Batched over seeds.

    Besides the end-state margin, tracks the TIME-TO-ANNIHILATION per env (H-P4,
    DB id=129): the PGS line lost on the real leaderboard via annihilation at steps
    115-238, which the 500-step margin alone cannot see — a candidate must also
    SURVIVE the aggressive regimes."""
    opp_seat = 1 - agent_seat
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=2, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=enable_comets))
    states = backend.reset(int(seeds[0]))
    death_steps: list[int | None] = [None] * n
    # FRESH, distinct instances — the cached pool would hand the SAME Producer to both
    # roles, cross-contaminating its per-game memory across the two seats.
    agent_base = [make_isolated_opponent("producer") for _ in range(n)]   # agent's own base plan
    opp_policy = [make_isolated_opponent(opponent) for _ in range(n)]   # opponent (producer/oep/...)
    for step in range(episode_steps):
        obs = np.asarray(backend.encoded_states(agent_seat))
        base_moves = [[list(m) for m in agent_base[i](states[i], agent_seat)] for i in range(n)]
        mask = np.zeros((n, k_max), dtype=bool)
        for i in range(n):
            mask[i, : min(len(base_moves[i]), k_max)] = True
        with torch.no_grad():
            # GREEDY (argmax) for eval — sampling injects ~P(non-KEEP) random edits
            # every step, which a KEEP-init policy must NOT do (it must reproduce
            # Producer exactly = parity floor). Inactive slots forced to KEEP(0).
            logits = model.forward(torch.as_tensor(obs, dtype=torch.float32, device=device))["edit"]
            greedy = logits.argmax(-1)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            greedy = torch.where(mask_t, greedy, torch.zeros_like(greedy))
        edits = greedy.cpu().numpy()
        rows = []
        for i in range(n):
            agent_moves = _apply_residual_edits(states[i], base_moves[i], edits[i], k_max)
            opp_moves = opp_policy[i](states[i], opp_seat)
            rows.extend(_moves_to_flat_rows(i, agent_seat, agent_moves))
            rows.extend(_moves_to_flat_rows(i, opp_seat, opp_moves))
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, agent_seat)
        states = backend.states()
        for i in range(n):
            if death_steps[i] is None:
                own, _ = _ships(states[i], agent_seat)
                if own <= 0:
                    death_steps[i] = step + 1
    margins = []
    for i in range(n):
        own, enemy = _ships(states[i], agent_seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins, death_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=96)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opponent", default="producer", help="opponent to eval against (producer/oep/...)")
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch = ckpt["config"]["policy_arch"]
    if arch != "producer_residual":
        raise SystemExit(f"checkpoint arch is {arch!r}, expected producer_residual")
    obs_dim = ckpt["model_state_dict"]["edit.weight"].shape  # sanity
    from python.orbit_wars_gym.encoding import observation_dim, EncoderConfig
    model = _build_policy("producer_residual", observation_dim(EncoderConfig()))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    k_max = model.k_max
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    enable_comets = not args.no_comets

    m0, d0 = _play_seat(model, k_max, 0, seeds, args.episode_steps, enable_comets, args.device, args.opponent)
    m1, d1 = _play_seat(model, k_max, 1, seeds, args.episode_steps, enable_comets, args.device, args.opponent)
    seat0, seat1 = float(np.mean(m0)), float(np.mean(m1))
    overall = float(np.mean(m0 + m1))
    deaths = [d for d in d0 + d1 if d is not None]
    last_entropy = float(ckpt.get("summary", {}).get("last_entropy", float("nan")))
    result = {
        "checkpoint": args.checkpoint,
        "mean_score_margin": overall,
        "seat0_margin": seat0,
        "seat1_margin": seat1,
        # H-P4 survival metrics: annihilations are the LB failure mode the
        # 500-step margin misses (PGS died at steps 115-238 vs rushers).
        "annihilated": len(deaths),
        "survival_rate": 1.0 - len(deaths) / max(len(d0) + len(d1), 1),
        "first_death_step": min(deaths) if deaths else None,
        "mean_death_step": float(np.mean(deaths)) if deaths else None,
        "seat0_annihilated": sum(1 for d in d0 if d is not None),
        "seat1_annihilated": sum(1 for d in d1 if d is not None),
        "seeds": args.seeds,
        "episode_steps": args.episode_steps,
        "last_entropy": last_entropy,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
