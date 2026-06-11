"""Check PGS floor fidelity: does the all-PRODUCER assignment reproduce the REAL
(stateful) Producer move-for-move along a real game trajectory?

Plays Producer-vs-Producer (isolated instances) and, at every step, also asks the
PGS floor (max_deviations=0, fresh ProducerLiteRuntime per turn) for player 0's
moves on the same state. Reports divergence rate and first divergence step —
the fidelity gap of the stateless-per-turn base-plan generator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.agents.registry import get_isolated_opponents  # noqa: E402
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig  # noqa: E402
from python.orbit_wars_gym.observation import to_official_observation  # noqa: E402


def norm_moves(moves):
    return sorted(tuple(float(x) for x in m[:3]) for m in moves if len(m) >= 3)


def classify(real, mine, angle_tol=1e-4):
    """Severity of a divergence: count > src_ships > angle_big > angle_tiny > none."""
    if real == mine:
        return None
    if len(real) != len(mine):
        return "count"
    if [(m[0], m[2]) for m in real] != [(m[0], m[2]) for m in mine]:
        return "src_ships"
    worst = max(abs(r[1] - m[1]) for r, m in zip(real, mine, strict=False))
    return "angle_big" if worst > angle_tol else "angle_tiny"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(1)

    from bots.pgs.planner import PGSConfig, make_runtime

    floor = make_runtime(PGSConfig(max_deviations=0))
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=args.seed,
                               config=RustConfig(enable_comets=True))
    backend.reset(args.seed)
    pols = get_isolated_opponents("producer", 2)
    states = backend.states()
    by_class = {"count": 0, "src_ships": 0, "angle_big": 0, "angle_tiny": 0}
    first_div = None
    for step in range(args.steps):
        state = states[0]
        real = norm_moves(pols[0](state, 0))
        mine = norm_moves(floor.act(to_official_observation(state, 0)))
        sev = classify(real, mine)
        if sev is not None:
            by_class[sev] += 1
            if first_div is None:
                first_div = step
                print(f"first divergence at step {step} [{sev}]:")
                print(f"  real : {real[:6]}")
                print(f"  floor: {mine[:6]}")
        rows = [[0.0, 0.0, *m] for m in real]
        rows += [[0.0, 1.0, *[float(x) for x in m[:3]]] for m in pols[1](state, 1) if len(m) >= 3]
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
    diverged = sum(by_class.values())
    print(f"divergence: {diverged}/{args.steps} steps ({100.0*diverged/args.steps:.1f}%), "
          f"first at {first_div} | by class: {by_class}")


if __name__ == "__main__":
    main()
