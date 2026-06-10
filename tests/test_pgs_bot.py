"""PGS bot (heuristic front): legality + parity-floor fidelity (todo H-P1)."""
from __future__ import annotations

import numpy as np
import pytest

from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal


def _advance(backend, states, policies, steps: int):
    for _ in range(steps):
        rows = []
        for player, pol in enumerate(policies):
            for m in pol(states[0], player):
                if len(m) >= 3:
                    rows.append([0.0, float(player), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
    return states


def _states_along_game(num_players: int, capture: list[int], seed: int = 7):
    backend = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed, config=RustConfig(enable_comets=True)
    )
    backend.reset(seed)
    states = backend.states()
    policies = get_isolated_opponents("producer", num_players)
    out, last = [], 0
    for step in sorted(capture):
        states = _advance(backend, states, policies, step - last)
        out.append(states[0])
        last = step
    return out


@pytest.mark.parametrize("num_players", [2, 4])
def test_pgs_returns_legal_moves(num_players: int) -> None:
    from bots.pgs.planner import make_runtime

    states = _states_along_game(num_players, capture=[0, 25, 60])
    runtime = make_runtime()
    for state in states:
        for player in range(num_players):
            moves = runtime.act(to_official_observation(state, player))
            assert isinstance(moves, list)
            for move in moves:
                assert len(move) == 3 and move[2] > 0
            assert moves_are_legal(state, player, moves)


def test_pgs_floor_matches_producer_modulo_float() -> None:
    """All-PRODUCER assignment ≈ the real Producer plan: same (source, ships)
    multiset; angles within float tolerance (the runtime is stateless per turn)."""
    from bots.pgs.planner import PGSConfig, make_runtime

    states = _states_along_game(2, capture=[0, 10])
    floor = make_runtime(PGSConfig(max_deviations=0))
    producer = get_isolated_opponents("producer", 1)[0]
    for state in states:
        real = sorted((round(m[0]), round(m[2]), round(float(m[1]), 3)) for m in producer(state, 0))
        mine = sorted(
            (round(m[0]), round(m[2]), round(float(m[1]), 3))
            for m in floor.act(to_official_observation(state, 0))
        )
        assert mine == real
