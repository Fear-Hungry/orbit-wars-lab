"""H7 E2 — self-play dataset for the value net: (state_obs, outcome) pairs.

Plays league matchups (the diverse pool = positions the net must value), and at
sub-sampled steps records ``encode_state(state, seat)`` for each ALIVE seat,
labelled with that seat's FINAL outcome (+1 win / -1 loss / 0 tie-or-wipe). The
schedule mirrors the real field (>=54% 4p, DB id=168) so the net learns where we
are weak. Saves shards to artifacts/h7/value_ds/shard_*.npz (resumable).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/collect_value_dataset.py --games 300 --p4-frac 0.54
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)  # critical for parallel workers: avoid CPU thrashing

# imports below MUST come after set_num_threads (they pull in torch-heavy modules)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig  # noqa: E402
from python.orbit_wars_gym.encoding import encode_state  # noqa: E402
from python.orbit_wars_gym.entities import (  # noqa: E402
    fleet_owner,
    fleet_ships,
    planet_owner,
    planet_ships,
)
from scripts.league_agents import make  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/h7/value_ds"

# diverse pool (positions to value) — excludes rusher (0-win junk, adds no states worth learning)
POOL = ["producer", "oep", "brep", "pgs_hold", "pgs_holdwave", "pgs_wave_s100",
        "pgs_bigwave", "pgs_allscripts", "ext_lb1050", "ext_hellburner"]


def _totals(state, np_):
    tot = [0.0] * np_
    for p in state.get("planets", []):
        o = planet_owner(p)
        if 0 <= o < np_:
            tot[o] += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if 0 <= o < np_:
            tot[o] += fleet_ships(f)
    return tot


def play_and_capture(names, seed, steps, subsample, rng):
    """Returns (obs, labels, crashes) captured; labels filled after the game ends.

    A game where ANY agent crashed contributes NO rows: its outcome labels would
    describe a pass-bot, not the agent (silent dataset contamination otherwise).
    """
    np_ = len(names)
    backend = RustBatchBackend(num_envs=1, num_players=np_, seed=int(seed),
                               config=RustConfig(enable_comets=True))
    backend.reset(int(seed))
    states = backend.states()
    agents = [make(n) for n in names]
    died = [None] * np_
    cap_obs, cap_seat = [], []
    crashes: dict[str, int] = {}
    from python.orbit_wars_gym.observation import to_official_observation
    for t in range(steps):
        rows = []
        capture = (t % subsample == 0)
        for seat in range(np_):
            if died[seat] is not None:
                continue
            obs_off = to_official_observation(states[0], seat)
            try:
                moves = agents[seat](obs_off) or []
            except Exception as e:
                name = names[seat]
                if crashes.get(name, 0) == 0:
                    print(f"[crash] {name} seat={seat} seed={seed} step={t}: {e!r}",
                          file=sys.stderr, flush=True)
                crashes[name] = crashes.get(name, 0) + 1
                moves = []
            if capture:
                cap_obs.append(encode_state(states[0], seat).astype(np.float32))
                cap_seat.append(seat)
            for m in moves:
                if len(m) >= 3:
                    rows.append([0.0, float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
        tot = _totals(states[0], np_)
        for seat in range(np_):
            if died[seat] is None and tot[seat] <= 0.0:
                died[seat] = t + 1
    tot = _totals(states[0], np_)
    top = max(tot)
    winners = [s for s in range(np_) if tot[s] == top]
    def outcome(seat):
        if len(winners) != 1 or top <= 0:
            return 0.0
        return 1.0 if seat == winners[0] else -1.0
    if crashes:
        empty = np.zeros((0, encode_state(states[0], 0).shape[0]), np.float32)
        return empty, np.zeros((0,), np.float32), crashes
    labels = np.array([outcome(s) for s in cap_seat], dtype=np.float32)
    obs = np.stack(cap_obs).astype(np.float32) if cap_obs else \
        np.zeros((0, encode_state(states[0], 0).shape[0]), np.float32)
    return obs, labels, crashes


def pick_matchup(rng, p4_frac):
    if rng.random() < p4_frac:
        return rng.sample(POOL, 4)
    return rng.sample(POOL, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--subsample", type=int, default=10, help="capture every K steps")
    ap.add_argument("--p4-frac", type=float, default=0.54)
    ap.add_argument("--seed-base", type=int, default=60000)
    ap.add_argument("--shard-every", type=int, default=40, help="flush a shard every N games")
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--tag", default="w0", help="worker tag (parallel-safe shard naming)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng)
    existing = len(glob.glob(str(OUT / f"shard_{args.tag}_*.npz")))
    buf_obs, buf_lab = [], []
    n2 = n4 = 0
    crash_totals: dict[str, int] = {}
    dropped_games = 0
    seed = args.seed_base + existing * 100000
    t0 = time.perf_counter()
    shard = existing
    for g in range(args.games):
        names = pick_matchup(rng, args.p4_frac)
        if len(names) == 2:
            n2 += 1
        else:
            n4 += 1
        obs, lab, game_crashes = play_and_capture(names, seed, args.steps, args.subsample, rng)
        seed += 1
        for name, c in game_crashes.items():
            crash_totals[name] = crash_totals.get(name, 0) + c
        dropped_games += 1 if game_crashes else 0
        if obs.shape[0]:
            buf_obs.append(obs)
            buf_lab.append(lab)
        if (g + 1) % args.shard_every == 0 or g == args.games - 1:
            if buf_obs:
                obs_arr = np.concatenate(buf_obs)
                L = np.concatenate(buf_lab)
                np.savez_compressed(OUT / f"shard_{args.tag}_{shard:04d}.npz", obs=obs_arr, label=L)
                shard += 1
                buf_obs, buf_lab = [], []
            tot_states = sum(len(np.load(f)["label"]) for f in glob.glob(str(OUT / f"shard_{args.tag}_*.npz")))
            p4 = n4 / max(n2 + n4, 1)
            print(f"[{g+1}/{args.games}] games(2p={n2},4p={n4} {p4:.0%}) states~{tot_states} "
                  f"crashes={crash_totals or 0} dropped_games={dropped_games} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if crash_totals:
        print(f"ATENCAO: {dropped_games} jogo(s) DESCARTADO(s) por crash de agente: {crash_totals} "
              f"(dataset NAO contem esses jogos)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
