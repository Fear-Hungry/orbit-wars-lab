from __future__ import annotations

import numpy as np
import pytest
from python.agents.registry import (
    _make_isolated_policy,
    get_heuristic_policies,
    get_isolated_opponents,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig


def _flat(player: int, moves) -> list[list[float]]:
    return [[0.0, float(player), float(m[0]), float(m[1]), float(m[2])] for m in moves]


def _moves_key(moves) -> tuple:
    return tuple((int(m[0]), round(float(m[1]), 6), int(m[2])) for m in moves)


def _backend(seed: int) -> RustBatchBackend:
    return RustBatchBackend(
        num_envs=1, num_players=2, seed=seed,
        config=RustConfig(episode_steps=40, enable_comets=False, act_timeout=1.0),
    )


def _flat_array(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)


def _play_alone(policy, seed: int, steps: int) -> list[tuple]:
    """Record player-0 (policy) moves over a game; player 1 is fixed greedy."""
    greedy = get_heuristic_policies()["greedy"]
    backend = _backend(seed)
    state = backend.reset(seed)[0]
    recorded = []
    for _ in range(steps):
        m0 = policy(state, 0)
        recorded.append(_moves_key(m0))
        outcomes, states = backend.step_flat_with_states(_flat_array(_flat(0, m0) + _flat(1, greedy(state, 1))))
        state = states[0]
        if bool(outcomes[0].get("done", False)):
            break
    return recorded


def _play_interleaved(policy_a, policy_b, seed_a: int, seed_b: int, steps: int) -> list[tuple]:
    """Step two games in lockstep, calling B between A's consecutive calls.

    Returns A's player-0 move sequence on game A. If A and B shared per-game
    memory, B's interleaved calls would perturb A's decisions.
    """
    greedy = get_heuristic_policies()["greedy"]
    ba, bb = _backend(seed_a), _backend(seed_b)
    sa, sb = ba.reset(seed_a)[0], bb.reset(seed_b)[0]
    recorded = []
    for _ in range(steps):
        a0 = policy_a(sa, 0)
        b0 = policy_b(sb, 0)  # interleave B between A's consecutive calls
        recorded.append(_moves_key(a0))
        oa, ssa = ba.step_flat_with_states(_flat_array(_flat(0, a0) + _flat(1, greedy(sa, 1))))
        _ob, ssb = bb.step_flat_with_states(_flat_array(_flat(0, b0) + _flat(1, greedy(sb, 1))))
        sa, sb = ssa[0], ssb[0]
        if bool(oa[0].get("done", False)):
            break
    return recorded


@pytest.mark.parametrize("name", ["producer", "oep"])
def test_isolated_instance_has_no_cross_game_contamination(name: str) -> None:
    reference = _play_alone(_make_isolated_policy(name), seed=0, steps=24)
    inst_a = _make_isolated_policy(name)
    inst_b = _make_isolated_policy(name)
    interleaved = _play_interleaved(inst_a, inst_b, seed_a=0, seed_b=7, steps=24)
    assert interleaved == reference  # B's interleaved calls did not perturb A


def test_get_isolated_opponents_distinct_and_pooled() -> None:
    a = get_isolated_opponents("producer", 4)
    assert len({id(p) for p in a}) == 4
    # pool is cached and grown on demand: requesting fewer returns a prefix
    assert get_isolated_opponents("producer", 2) == a[:2]
    # stateless heuristics share one callable
    g = get_isolated_opponents("greedy", 3)
    assert len({id(p) for p in g}) == 1
