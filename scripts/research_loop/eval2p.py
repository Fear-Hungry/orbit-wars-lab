"""2-player evaluation — the 2p mirror of scripts/h9_4p_gate.run_config.

The 4p gate collapsed the whole PGS hold-family to a tied floor (the falsified
fitness, 2026-06-13). The six-hats decision picked 2p as the first route to test,
because the project believes (unverified) that PGS SEPARATES in 2p where it
collapses in 4p ([field_is_majority_4p]; the field is 46% 2p anyway).

This reuses the exact 4p code path (same backend, same helpers, same seat-0
candidate vs isolated opponent) with num_players=2 and a single opponent seat, so
any difference vs the 4p gate is the seat count, not the harness.
"""
from __future__ import annotations

import time

import numpy as np

from bots.pgs.planner import PGSConfig, make_runtime
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from scripts.h9_4p_gate import SEAT, ACT_TIMEOUT, _own_enemy_ships, _own_planets


def run_config_2p(name, pgs_config, seeds, steps, enable_comets, opponent="producer"):
    """1v1: seat 0 = candidate PGS(genome); seat 1 = isolated opponent. 500 steps."""
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=2, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=enable_comets, episode_steps=steps,
                                                 act_timeout=ACT_TIMEOUT))
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [make_runtime(PGSConfig(**pgs_config)) for _ in range(n)]
    others = [1]
    opp = get_isolated_opponents(opponent, n * len(others))
    timeouts = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        rows = []
        for i in range(n):
            obs = to_official_observation(states[i], SEAT)
            ts = time.perf_counter()
            moves = agents[i].act(obs)
            if time.perf_counter() - ts > ACT_TIMEOUT:
                timeouts += 1
                moves = []
            for m in moves:
                if len(m) >= 3:
                    rows.append([float(i), float(SEAT), float(m[0]), float(m[1]), float(m[2])])
            for j, seat in enumerate(others):
                for m in opp[i * len(others) + j](states[i], seat):
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, SEAT)
        states = backend.states()

    deaths, margins, planets = 0, [], []
    for i in range(n):
        own, enemy = _own_enemy_ships(states[i], SEAT)
        npl = _own_planets(states[i], SEAT)
        if npl == 0:
            deaths += 1
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else -1.0)
        planets.append(npl)
    return {
        "name": name,
        "death_rate": deaths / n,
        "mean_margin": float(np.mean(margins)),
        "mean_final_planets": float(np.mean(planets)),
        "timeouts": timeouts,
        "elapsed_s": time.perf_counter() - t0,
    }
