from __future__ import annotations

import pytest

from python.agents.registry import (
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal


def _initial_state(*, num_players: int, seed: int = 0) -> dict:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=int(num_players),
        seed=int(seed),
        config=RustConfig(episode_steps=32, enable_comets=False, act_timeout=1.0),
    )
    return backend.reset(seed)[0]


def test_strong_opponents_are_registered() -> None:
    policies = get_heuristic_policies()
    assert "producer" in policies
    assert "oep" in policies
    assert STATEFUL_SINGLETON_OPPONENTS == frozenset({"producer", "oep", "pgs"})


@pytest.mark.parametrize("name", ["producer", "oep"])
@pytest.mark.parametrize("num_players", [2, 4])
def test_strong_opponent_returns_legal_moves(name: str, num_players: int) -> None:
    policy = get_heuristic_policies()[name]
    state = _initial_state(num_players=num_players)

    for player in range(num_players):
        moves = policy(state, player)
        assert isinstance(moves, list)
        for move in moves:
            assert len(move) == 3
            assert move[2] > 0
        assert moves_are_legal(state, player, moves)
