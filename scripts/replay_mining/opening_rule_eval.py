"""G3.1a Phase-3 eval: does the opening capture-and-hold margin help 4p?

Runs the Producer agent in seat 0 of a 4p game vs 3 isolated baseline opponents,
across a seed set (default: the 124 bad_opening_4p map seeds), 500 steps. ONLY
seat 0 carries the opening-hold margin (via ProducerLiteRuntime config_override);
opponents stay baseline, so the comparison is clean. Run twice (--margin 0.0 vs
--margin X) on the same seeds for a paired A/B.

Reports the 96-seed-style gate metric (mean ship margin, win/death rate) AND the
offline opening metrics the detector flagged (neutral share @ step 50, capture
hold-rate, early-PvP proxy) — so a win shows up as BOTH "4p margin improves" and
"we hold our opening captures".

2p is untouched by construction: the margin lives in CONFIG_4P (4p only).

    .venv/bin/python scripts/replay_mining/opening_rule_eval.py --margin 0.0 --out artifacts/replay_mining/eval_m0.json
    .venv/bin/python scripts/replay_mining/opening_rule_eval.py --margin 0.5 --out artifacts/replay_mining/eval_m05.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import bots.producer._upstream as up
from bots.producer._upstream import ProducerLiteRuntime, _run
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import planet_owner, planet_ships
from python.orbit_wars_gym.observation import to_official_observation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "replay_mining"
SEAT = 0
OTHERS = [1, 2, 3]
ACT_TIMEOUT = 1.0
OPENING = 50
SUPPRESS_PVP = False


def _owner_map(state):
    return {i: planet_owner(p) for i, p in enumerate(state["planets"])}


def _own_enemy_ships(state, seat):
    own = sum(planet_ships(p) for p in state["planets"] if planet_owner(p) == seat)
    enemy = sum(planet_ships(p) for p in state["planets"] if planet_owner(p) not in (-1, seat))
    return float(own), float(enemy)


def run(margin: float, until: int, seeds: list[int], steps: int, opponent: str) -> dict:
    n = len(seeds)
    cfg = dataclasses.replace(
        up.CONFIG_4P, opening_hold_margin=float(margin), opening_hold_until_step=int(until),
        opening_suppress_pvp=bool(SUPPRESS_PVP),
    )
    backend = RustBatchBackend(
        num_envs=n, num_players=4, seed=int(seeds[0]),
        config=RustConfig(enable_comets=True, episode_steps=steps, act_timeout=ACT_TIMEOUT),
    )
    # The Rust batch sim generates `n` distinct 4p maps from one base seed; the
    # Kaggle seeds don't map through, but margin=0 vs margin=X share the base seed
    # + env count, so the A/B is paired map-for-map.
    backend.reset(int(seeds[0]))
    states = backend.states()
    seat0 = [ProducerLiteRuntime(config_override=cfg) for _ in range(n)]
    opp = get_isolated_opponents(opponent, n * len(OTHERS))

    init_owner = [_owner_map(states[i]) for i in range(n)]
    captured_at = [dict() for _ in range(n)]   # pid -> step we took from neutral
    prev_owner = [dict(init_owner[i]) for i in range(n)]
    snap50 = [None] * n
    timeouts = 0
    t0 = time.perf_counter()

    for step in range(steps):
        rows = []
        for i in range(n):
            obs = to_official_observation(states[i], SEAT)
            ts = time.perf_counter()
            moves = _run(seat0[i], obs)
            if time.perf_counter() - ts > ACT_TIMEOUT:
                timeouts += 1
                moves = []
            for m in moves:
                if len(m) >= 3:
                    rows.append([float(i), float(SEAT), float(m[0]), float(m[1]), float(m[2])])
            for j, seat in enumerate(OTHERS):
                for m in opp[i * len(OTHERS) + j](states[i], seat):
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, SEAT)
        states = backend.states()

        # ownership tracking for opening metrics (steps 1..OPENING)
        s = step + 1
        if s <= OPENING:
            for i in range(n):
                om = _owner_map(states[i])
                for pid, ow in om.items():
                    if prev_owner[i].get(pid) == -1 and ow == SEAT and pid not in captured_at[i]:
                        captured_at[i][pid] = s
                prev_owner[i] = om
                if s == OPENING:
                    snap50[i] = om

    # opening metrics
    neutral_shares, hold_rates, owned50 = [], [], []
    for i in range(n):
        om = snap50[i] or _owner_map(states[i])
        caps_us = sum(1 for pid in captured_at[i])
        caps_all = sum(1 for pid, ow in om.items()
                       if init_owner[i].get(pid) == -1 and ow != -1)
        # fallback: count all neutral->someone over window via prev tracking not kept per opp;
        # approximate caps_all by planets no longer neutral that started neutral
        neutral_shares.append(caps_us / caps_all if caps_all else 0.0)
        held = sum(1 for pid in captured_at[i] if om.get(pid) == SEAT)
        hold_rates.append(held / caps_us if caps_us else 1.0)
        owned50.append(sum(1 for ow in om.values() if ow == SEAT))

    deaths, margins = 0, []
    for i in range(n):
        own, enemy = _own_enemy_ships(states[i], SEAT)
        if sum(1 for p in states[i]["planets"] if planet_owner(p) == SEAT) == 0:
            deaths += 1
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else -1.0)

    return {
        "margin": float(margin),
        "until": int(until),
        "opponent": opponent,
        "n_seeds": n,
        "steps": steps,
        "mean_margin": float(np.mean(margins)),
        "win_rate": float(np.mean([1.0 if m > 0 else 0.0 for m in margins])),
        "death_rate": deaths / n,
        "neutral_share_50": float(np.mean(neutral_shares)),
        "capture_hold_rate_50": float(np.mean(hold_rates)),
        "owned_50": float(np.mean(owned50)),
        "timeouts": timeouts,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--until", type=int, default=OPENING)
    ap.add_argument("--seeds", default=str(OUT / "bad_opening_4p_seeds.json"))
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suppress-pvp", type=int, default=0,
                    help="literal G3.1a Phase-2 rule: suppress opening PvP while a neutral is in range")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global SUPPRESS_PVP
    SUPPRESS_PVP = bool(args.suppress_pvp)

    seeds = json.loads(Path(args.seeds).read_text())
    if args.limit:
        seeds = seeds[: args.limit]
    res = run(args.margin, args.until, seeds, args.steps, args.opponent)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
