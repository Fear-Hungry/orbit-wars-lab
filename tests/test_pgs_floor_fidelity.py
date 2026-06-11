"""Regression: the PGS floor (max_deviations=0) must reproduce the REAL stateful
Producer move-for-move along a real trajectory.

Root cause locked down here (2026-06-10): _producer_entries used to build a
fresh ProducerLiteRuntime per turn, losing the rolling PlanetMovement memory
(planned-launch ledger reconciled against the next obs). The fresh rebuild
re-estimated in-flight arrivals from the obs alone, drifting garrison_status
and diverging from the real Producer (~45% of steps; 3/80 with different move
COUNTS). With per-owner persistent runtimes the floor is exact by induction:
identical memory at step 0 + identical obs => identical plan every step.
"""
from __future__ import annotations

import numpy as np
import torch


def _norm(moves):
    return sorted(tuple(float(x) for x in m[:3]) for m in moves if len(m) >= 3)


def test_pgs_floor_reproduces_real_producer_exactly():
    torch.set_num_threads(1)
    from bots.pgs.planner import PGSConfig, make_runtime
    from python.agents.registry import get_isolated_opponents
    from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
    from python.orbit_wars_gym.observation import to_official_observation

    seed, steps = 1000, 60
    floor = make_runtime(PGSConfig(max_deviations=0))
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=seed,
                               config=RustConfig(enable_comets=True))
    backend.reset(seed)
    pols = get_isolated_opponents("producer", 2)
    states = backend.states()
    for step in range(steps):
        state = states[0]
        real = _norm(pols[0](state, 0))
        mine = _norm(floor.act(to_official_observation(state, 0)))
        assert mine == real, (
            f"floor diverged from real Producer at step {step}:\n"
            f"  real : {real[:6]}\n  floor: {mine[:6]}"
        )
        rows = [[0.0, 0.0, *m] for m in real]
        rows += [[0.0, 1.0, *[float(x) for x in m[:3]]]
                 for m in pols[1](state, 1) if len(m) >= 3]
        flat = (np.asarray(rows, dtype=np.float64) if rows
                else np.zeros((0, 5), dtype=np.float64))
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
