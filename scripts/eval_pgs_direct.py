"""Direct eval for the PGS bot (heuristic front) — H-P1 floor check + H-P2 triage.

Plays PGS vs a registry opponent at both seats over N seeds (batched envs),
reporting seat-averaged normalized ship-score margin (>0 = beats it), per-seat
margins, and decision-time stats (p95 must stay under ~700ms for Kaggle).

--max-deviations 0 turns the search off: the all-PRODUCER assignment, i.e. the
parity FLOOR — its margin vs Producer measures generator fidelity (~0 expected).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import (
    fleet_owner,
    fleet_ships,
    planet_owner,
    planet_ships,
)
from python.orbit_wars_gym.observation import to_official_observation


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


def _play_seat(agent_seat, seeds, episode_steps, enable_comets, opponent, pgs_config, decision_ms,
               num_players=2, launch_sizes=None):
    from bots.pgs.planner import PGSConfig, make_runtime

    n = len(seeds)
    backend = RustBatchBackend(
        num_envs=n, num_players=num_players, seed=int(seeds[0]),
        config=RustConfig(enable_comets=enable_comets),
    )
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [make_runtime(PGSConfig(**pgs_config)) for _ in range(n)]
    others = [p for p in range(num_players) if p != agent_seat]
    # one isolated opponent instance per (env, other-seat)
    opp_pool = get_isolated_opponents(opponent, n * len(others))
    for _ in range(episode_steps):
        rows = []
        for i in range(n):
            obs = to_official_observation(states[i], agent_seat)
            t0 = time.perf_counter()
            moves = agents[i].act(obs)
            decision_ms.append((time.perf_counter() - t0) * 1000.0)
            for m in moves:
                if len(m) >= 3:
                    rows.append([float(i), float(agent_seat), float(m[0]), float(m[1]), float(m[2])])
                    if launch_sizes is not None:
                        launch_sizes.append(float(m[2]))
            for j, seat in enumerate(others):
                for m in opp_pool[i * len(others) + j](states[i], seat):
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, agent_seat)
        states = backend.states()
    margins = []
    for i in range(n):
        own, enemy = _ships(states[i], agent_seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--agent-seats", default=None,
                    help="comma list of seats the agent plays (default: 0,1 for 2p; 0,2 for 4p)")
    ap.add_argument("--max-deviations", type=int, default=None, help="0 = parity floor (no search)")
    ap.add_argument("--max-passes", type=int, default=None)
    ap.add_argument("--value-horizon", type=int, default=None)
    ap.add_argument("--prod-weight", type=float, default=None)
    ap.add_argument("--deviation-max-step", type=int, default=None)
    ap.add_argument("--arbiter-margin", type=float, default=None)
    ap.add_argument("--scripts", type=str, default=None)
    ap.add_argument("--wave-min-ships", type=float, default=None)
    ap.add_argument("--wave-start-step", type=int, default=None)
    ap.add_argument("--wave-max-delay", type=int, default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.set_num_threads(1)
    pgs_config = {}
    for key in ("max_deviations", "max_passes", "value_horizon", "prod_weight", "deviation_max_step", "arbiter_margin", "scripts", "wave_min_ships", "wave_start_step", "wave_max_delay"):
        v = getattr(args, key)
        if v is not None:
            pgs_config[key] = v

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    enable_comets = not args.no_comets
    if args.agent_seats is not None:
        agent_seats = [int(s) for s in args.agent_seats.split(",")]
    else:
        agent_seats = [0, 1] if args.num_players == 2 else [0, 2]
    decision_ms: list[float] = []
    launch_sizes: list[float] = []
    by_seat = {
        seat: _play_seat(seat, seeds, args.episode_steps, enable_comets, args.opponent,
                         pgs_config, decision_ms, num_players=args.num_players,
                         launch_sizes=launch_sizes)
        for seat in agent_seats
    }
    all_margins = [m for ms in by_seat.values() for m in ms]
    arr = np.asarray(decision_ms)
    result = {
        "agent": "pgs",
        "pgs_config": pgs_config,
        "opponent": args.opponent,
        "num_players": args.num_players,
        "mean_score_margin": float(np.mean(all_margins)),
        **{f"seat{seat}_margin": float(np.mean(ms)) for seat, ms in by_seat.items()},
        "win_rate": float(np.mean([m > 0 for m in all_margins])),
        "seeds": args.seeds,
        "episode_steps": args.episode_steps,
        **{f"per_seed_seat{seat}": {str(s): round(m, 4) for s, m in zip(seeds, ms)}
           for seat, ms in by_seat.items()},
        "decision_ms": {
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        },
        # wave-discipline profile (H-P5 intermediate metric): elite ~0.4-0.5
        # launches/step with 60-80% >=50 ships; our spray baseline ~1.5 / 5%.
        "launch_profile": {
            "launches_per_step": len(launch_sizes) / max(1, len(decision_ms)),
            "median_ships": float(np.median(launch_sizes)) if launch_sizes else 0.0,
            "share_ge50": float(np.mean([s >= 50 for s in launch_sizes])) if launch_sizes else 0.0,
        },
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
