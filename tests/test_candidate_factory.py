from __future__ import annotations

import pytest
from bots.oep.candidate_factory import (
    DEFAULT_FAMILIES,
    CandidateFactory,
    PlanCandidate,
    available_families,
    register_family,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal


def _initial_official_obs(*, seed: int = 0, player: int = 0) -> dict:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(episode_steps=32, enable_comets=False, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    return to_official_observation(state, player=int(player))


def test_default_families_present() -> None:
    assert "producer" in available_families()
    assert "oep" in available_families()
    # Producer + OEP archetypes lead; Family-H generators follow.
    assert DEFAULT_FAMILIES[:2] == ("producer", "oep")
    for name in ("production_projected_attack", "timeline_risk", "hammer_multiprong"):
        assert name in DEFAULT_FAMILIES


def test_factory_yields_one_candidate_per_family() -> None:
    obs = _initial_official_obs()
    factory = CandidateFactory()
    candidates = factory.candidates(obs)

    assert [c.family for c in candidates] == list(DEFAULT_FAMILIES)
    assert all(isinstance(c, PlanCandidate) for c in candidates)


def test_every_candidate_emits_legal_moves() -> None:
    obs = _initial_official_obs(player=0)
    factory = CandidateFactory()

    for candidate in factory.candidates(obs):
        # [] (pass) is always legal; otherwise every move must be legal for p0.
        assert moves_are_legal(obs, 0, candidate.moves), (
            f"family {candidate.family} emitted illegal moves: {candidate.moves}"
        )


def test_fresh_mode_rebuilds_generators() -> None:
    obs = _initial_official_obs()
    factory = CandidateFactory(families=("producer",))

    # Cached vs fresh must agree on the same state (no stale memory on step 0).
    cached = factory.candidates(obs)[0].moves
    fresh = factory.candidates(obs, fresh=True)[0].moves
    assert moves_are_legal(obs, 0, cached)
    assert moves_are_legal(obs, 0, fresh)


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValueError):
        CandidateFactory(families=("producer", "does_not_exist"))


def test_register_duplicate_rejected() -> None:
    with pytest.raises(ValueError):
        register_family("producer", _producer_again)


def _producer_again():  # pragma: no cover - only used to trigger the guard
    from bots.producer.agent import make_agent

    return make_agent()


def test_graded_verdict_bands() -> None:
    from scripts.oracle_candidates import _graded_verdict

    b, eps = -0.045, 0.02
    assert _graded_verdict(-0.10, b, eps) == "HARD_GENERATOR_BOTTLENECK"
    assert _graded_verdict(-0.045, b, eps) == "HARD_GENERATOR_BOTTLENECK"
    assert _graded_verdict(-0.03, b, eps) == "WEAK_BLOOD"
    assert _graded_verdict(-0.01, b, eps) == "USEFUL_SIGNAL"
    assert _graded_verdict(0.05, b, eps) == "CROSSES_FLOOR"


def test_grafted_policy_plays_candidate_then_incumbent() -> None:
    from scripts.oracle_candidates import _grafted_policy

    calls = []

    def incumbent(state, idx):
        calls.append(idx)
        return [[9.0, 0.0, 3.0]]

    graft = _grafted_policy([[1.0, 0.5, 2.0]], incumbent)
    assert graft({}, 0) == [[1.0, 0.5, 2.0]]  # first = candidate
    assert graft({}, 0) == [[9.0, 0.0, 3.0]]  # then = incumbent
    assert calls == [0]
