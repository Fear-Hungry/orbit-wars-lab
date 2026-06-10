"""Validate the packaged BReP submission tarball end-to-end (pre-submit gate).

Extracts the tar.gz, imports its main.py, and runs the REAL submission agent
(official Kaggle obs in, moves out) vs Producer at 500 steps, both seats. Reports
margin (must match the torch BReP ~+0.10) AND submission stats — fallbacks/illegal
MUST be 0 (no silent degradation, per docs/SUBMISSION.md)."""
from __future__ import annotations
import argparse
import importlib.util
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from python.train.train_ppo import _moves_to_flat_rows
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.agents.registry import make_isolated_opponent
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


def _load_submission(tarball: str):
    tmp = tempfile.mkdtemp(prefix="brep_sub_")
    with tarfile.open(tarball) as t:
        t.extractall(tmp)
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location("brep_submission_main", str(Path(tmp) / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["brep_submission_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def _play(sub, agent_seat, seeds, steps, enable_comets):
    opp_seat = 1 - agent_seat
    margins = []
    for seed in seeds:
        backend = RustBatchBackend(num_envs=1, num_players=2, seed=int(seed),
                                   config=RustConfig(enable_comets=enable_comets))
        states = backend.reset(int(seed))
        opp = make_isolated_opponent("producer")
        for _ in range(steps):
            state = states[0]
            obs = to_official_observation(state, agent_seat)
            amoves = sub.agent(obs)
            omoves = opp(state, opp_seat)
            rows = _moves_to_flat_rows(0, agent_seat, amoves) + _moves_to_flat_rows(0, opp_seat, omoves)
            flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            backend.step_flat_with_encoded_states(flat, agent_seat)
            states = backend.states()
        own, enemy = _ships(states[0], agent_seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", default="artifacts/submission_brep.tar.gz")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--no-comets", action="store_true")
    args = ap.parse_args()

    sub = _load_submission(args.tarball)
    seeds = list(range(1000, 1000 + args.seeds))
    enable_comets = not args.no_comets
    m0 = _play(sub, 0, seeds, args.episode_steps, enable_comets)
    m1 = _play(sub, 1, seeds, args.episode_steps, enable_comets)
    stats = dict(getattr(sub, "SUBMISSION_STATS", {}))
    overall = float(np.mean(m0 + m1))
    print(f"submission margin vs Producer (both seats, {args.seeds} seeds, {args.episode_steps} steps):")
    print(f"  mean={overall:+.4f}  seat0={float(np.mean(m0)):+.4f}  seat1={float(np.mean(m1)):+.4f}")
    print(f"  SUBMISSION_STATS={stats}")
    fallbacks = int(stats.get("fallbacks", 0)) + int(stats.get("fallback_errors", 0))
    illegal = int(stats.get("illegal_moves", 0))
    ok = overall > 0 and fallbacks == 0 and illegal == 0
    print(f"  GATE: margin>0={overall>0} fallbacks={fallbacks} illegal={illegal} -> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
