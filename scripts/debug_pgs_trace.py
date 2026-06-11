"""Trace ownership/production/ships over a PGS-vs-Producer game vs the Producer mirror."""
from __future__ import annotations

import argparse

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


def snapshot(state):
    out = {}
    for pl in (0, 1):
        planets = [p for p in state.get("planets", []) if planet_owner(p) == pl]
        prod = sum(float(p.get("production", p.get("prod", 0)) if isinstance(p, dict) else 0) for p in planets)
        garrison = sum(planet_ships(p) for p in planets)
        inflight = sum(fleet_ships(f) for f in state.get("fleets", []) if fleet_owner(f) == pl)
        out[pl] = (len(planets), prod, garrison + inflight)
    return out


def run(agent_fn, steps, seed, label, marks):
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(enable_comets=True))
    backend.reset(seed)
    opp = get_isolated_opponents("producer", 1)[0]
    states = backend.states()
    for step in range(steps + 1):
        if step in marks:
            s = snapshot(states[0])
            print(f"{label} step={step}: p0 planets={s[0][0]} prod={s[0][1]:.1f} ships={s[0][2]:.0f} | "
                  f"p1 planets={s[1][0]} prod={s[1][1]:.1f} ships={s[1][2]:.0f}")
        rows = []
        for m in agent_fn(states[0]):
            if len(m) >= 3:
                rows.append([0.0, 0.0, float(m[0]), float(m[1]), float(m[2])])
        for m in opp(states[0], 1):
            if len(m) >= 3:
                rows.append([0.0, 1.0, float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(1)
    marks = {0, 20, 40, 60, 90, 120, 200, 300, 400, 500} & set(range(args.steps + 1))

    from bots.pgs.planner import make_runtime
    pgs = make_runtime()
    run(lambda st: pgs.act(to_official_observation(st, 0)), args.steps, args.seed, "PGS", marks)

    # one call with count=2: distinct instances (two count=1 calls would hand the
    # SAME pool slot to both sides and cross-contaminate the mirror)
    pair = get_isolated_opponents("producer", 2)
    run(lambda st: pair[1](st, 0), args.steps, args.seed, "PRODMIRROR", marks)


if __name__ == "__main__":
    main()
