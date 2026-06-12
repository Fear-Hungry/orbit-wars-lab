from __future__ import annotations

import pytest
from bots.oep.candidate_factory import available_families
from bots.oep.family_h import (
    defensive_reinforce,
    hammer_multiprong,
    make_eval_policy,
    production_projected_attack,
    regroup_dominance,
    timeline_risk,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal

H_FAMILIES = {
    "production_projected_attack": production_projected_attack,
    "timeline_risk": timeline_risk,
    "hammer_multiprong": hammer_multiprong,
    "regroup_dominance": regroup_dominance,
    "defensive_reinforce": defensive_reinforce,
    "eval_function": make_eval_policy(None),
}


def _obs_sequence(*, seed: int, num_players: int, steps: int) -> list[tuple[dict, int]]:
    """Drive a real game forward and yield (obs, player) at several steps."""

    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=steps + 1, enable_comets=True, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    out: list[tuple[dict, int]] = []
    for _ in range(steps):
        for player in range(num_players):
            out.append((to_official_observation(state, player=player), player))
        actions = [[] for _ in range(num_players)]
        _, states = backend.step_with_states([actions])
        state = states[0]
    return out


def test_h_families_registered() -> None:
    for name in H_FAMILIES:
        assert name in available_families()


@pytest.mark.parametrize("name", list(H_FAMILIES))
@pytest.mark.parametrize("num_players", [2, 4])
def test_family_emits_legal_moves(name: str, num_players: int) -> None:
    generator = H_FAMILIES[name]
    samples = _obs_sequence(seed=7, num_players=num_players, steps=8)
    assert samples
    for obs, player in samples:
        moves = generator(obs)
        assert isinstance(moves, list)
        assert moves_are_legal(obs, player, moves), (
            f"{name} (p{player}) emitted illegal moves: {moves}"
        )


@pytest.mark.parametrize("name", list(H_FAMILIES))
def test_family_returns_list_on_empty_board(name: str) -> None:
    empty = {"planets": [], "fleets": [], "player": 0, "comet_planet_ids": []}
    assert H_FAMILIES[name](empty) == []
