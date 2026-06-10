"""Play one league matchup (2p pair or 4p composition) over batched seeds.

2p: plays BOTH seat orderings per seed. 4p: rotates the composition across
seats per seed (seed i uses rotation i % 4). Winner = argmax final total ships
(official reward semantics: one +1, rest -1). Also records annihilation
(first step a player's ships+planets hit 0) — the LB failure mode our old
margin-only evals never measured.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import (
    fleet_owner,
    fleet_ships,
    planet_owner,
    planet_ships,
)
from python.orbit_wars_gym.observation import to_official_observation
from scripts.league_agents import make


def _totals(state, num_players):
    tot = [0.0] * num_players
    for p in state.get("planets", []):
        o = planet_owner(p)
        if 0 <= o < num_players:
            tot[o] += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if 0 <= o < num_players:
            tot[o] += fleet_ships(f)
    return tot


def play_batch(names_by_seat, seeds, steps, decision_ms):
    """names_by_seat: list of agent names, one per seat. Returns per-seed games."""
    n, np_ = len(seeds), len(names_by_seat)
    backend = RustBatchBackend(num_envs=n, num_players=np_, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=True))
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [[make(name) for name in names_by_seat] for _ in range(n)]
    died_at = [[None] * np_ for _ in range(n)]
    for t in range(steps):
        rows = []
        for i in range(n):
            for seat, name in enumerate(names_by_seat):
                if died_at[i][seat] is not None:
                    continue
                obs = to_official_observation(states[i], seat)
                t0 = time.perf_counter()
                try:
                    moves = agents[i][seat](obs) or []
                except Exception:
                    moves = []  # a crashing agent just passes (recorded via weakness)
                decision_ms.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)
                for m in moves:
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
        for i in range(n):
            tot = _totals(states[i], np_)
            for seat in range(np_):
                if died_at[i][seat] is None and tot[seat] <= 0.0:
                    died_at[i][seat] = t + 1
    games = []
    for i in range(n):
        tot = _totals(states[i], np_)
        winner = int(np.argmax(tot)) if max(tot) > 0 else -1
        games.append({
            "seed": int(seeds[i]),
            "seats": list(names_by_seat),
            "final_ships": [round(x, 1) for x in tot],
            "winner_seat": winner,
            "winner": names_by_seat[winner] if winner >= 0 else None,
            "died_at": died_at[i],
        })
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True, help="comma list: 2 names (pair) or 4 (composition)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    names = [s.strip() for s in args.agents.split(",")]
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    decision_ms: dict[str, list[float]] = {}
    games = []
    if len(names) == 2:
        games += play_batch(names, seeds, args.steps, decision_ms)
        games += play_batch(names[::-1], seeds, args.steps, decision_ms)
    elif len(names) == 4:
        for r in range(4):
            rot = names[r:] + names[:r]
            batch = [s for j, s in enumerate(seeds) if j % 4 == r]
            if batch:
                games += play_batch(rot, batch, args.steps, decision_ms)
    else:
        raise SystemExit("need 2 or 4 agents")
    out = {
        "agents": names,
        "mode": f"{len(names)}p",
        "games": games,
        "decision_ms_p95": {k: float(np.percentile(v, 95)) for k, v in decision_ms.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"out": args.out, "games": len(games)}))


if __name__ == "__main__":
    main()
