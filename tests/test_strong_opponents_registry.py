from __future__ import annotations

import pytest
from python.agents.registry import (
    HEURISTIC_NAMES,
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
    get_isolated_opponents,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal

# Handicapped Producer rungs for the curriculum ladder. They wrap the singleton
# Producer runtime (so they are stateful) and broke training twice before being
# wired through both paths: train_ppo's PHASE0 validation ("unknown phase-0
# opponents") and the batched-rollout isolation ("is stateless").
_HANDICAPPED = ["producer_h30", "producer_h50", "producer_h70"]


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
    for name in ("producer", "oep", *_HANDICAPPED):
        assert name in policies, name
        assert name in HEURISTIC_NAMES, name
    # pgs is registered as a policy/singleton but deliberately NOT in
    # HEURISTIC_NAMES: that tuple seeds default league/curriculum lists and the
    # PGS planner (~180ms/step) is too slow to be a default opponent.
    assert "pgs" in policies
    assert "pgs" not in HEURISTIC_NAMES
    assert STATEFUL_SINGLETON_OPPONENTS == frozenset(
        {"producer", "producer_h30", "producer_h50", "producer_h70", "oep", "pgs"}
    )


def test_handicapped_producers_pass_phase0_validation() -> None:
    # Regression: train_ppo validates --opponents against PHASE0_OPPONENTS; the
    # handicapped names raised "unknown phase-0 opponents" before being added.
    from python.train.train_ppo import PHASE0_OPPONENTS, _parse_opponents

    for name in _HANDICAPPED:
        assert name in PHASE0_OPPONENTS, name
    parsed = _parse_opponents("producer_h30,producer_h50,greedy")
    assert "producer_h30" in parsed and "producer_h50" in parsed


@pytest.mark.parametrize("name", _HANDICAPPED)
@pytest.mark.parametrize("num_players", [2, 4])
def test_handicapped_producer_isolated_returns_legal_moves(name: str, num_players: int) -> None:
    # Regression: batched rollout (rollout_num_envs > 1) calls get_isolated_opponents
    # for stateful singletons; the handicapped names raised "is stateless" before
    # _make_isolated_policy learned to build a fresh handicapped Producer.
    opponents = get_isolated_opponents(name, count=2)
    assert len(opponents) == 2
    state = _initial_state(num_players=num_players)
    for policy in opponents:
        for player in range(num_players):
            moves = policy(state, player)
            assert isinstance(moves, list)
            for move in moves:
                assert len(move) == 3
                assert move[2] > 0
            assert moves_are_legal(state, player, moves)


@pytest.mark.parametrize("name", ["producer", "oep", *_HANDICAPPED])
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
