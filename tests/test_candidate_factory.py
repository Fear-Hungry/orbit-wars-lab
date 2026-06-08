"""Tests for the Frente B candidate factory (B1 keystone)."""
from __future__ import annotations

from python.agents.candidate_factory import (
    CANDIDATE_NAMES,
    NUM_CANDIDATES,
    CandidateFactory,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal


def _mid_game_state(seed: int = 0, warmup: int = 40):
    """A non-trivial 2p state a few dozen steps in (experts have something to do)."""
    backend = RustBatchBackend(
        num_envs=1, num_players=2, seed=seed,
        config=RustConfig(episode_steps=256, enable_comets=True, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    factory = CandidateFactory()
    for _ in range(warmup):
        cands = factory.candidates(state, 0)
        outs, states = backend.step_with_states([[cands[1]["moves"], []]])  # p0 plays producer
        state = states[0]
        if outs[0]["done"]:
            break
    return state


def test_panel_shape_and_no_op_first():
    state = _mid_game_state()
    factory = CandidateFactory()
    cands = factory.candidates(state, 0)
    assert len(cands) == NUM_CANDIDATES
    assert [c["name"] for c in cands] == list(CANDIDATE_NAMES)
    assert cands[0]["name"] == "no_op"
    assert cands[0]["moves"] == []  # no-op is always an empty (pass) move-set
    assert cands[0]["legal"] is True


def test_every_candidate_is_legal_to_play():
    """No candidate ever hands the env an illegal move-set (fail-safe to pass)."""
    state = _mid_game_state(seed=1)
    factory = CandidateFactory()
    cands = factory.candidates(state, 0)
    for c in cands:
        assert moves_are_legal(state, 0, c["moves"]), c["name"]


def test_experts_actually_move_midgame():
    """At least one expert (producer/oep) produces a real plan mid-game — the panel
    is not a no-op-only degenerate set."""
    state = _mid_game_state(seed=2)
    factory = CandidateFactory()
    cands = {c["name"]: c for c in factory.candidates(state, 0)}
    assert len(cands["producer"]["moves"]) > 0 or len(cands["oep"]["moves"]) > 0


def test_mask_all_selectable_v1():
    state = _mid_game_state(seed=3)
    factory = CandidateFactory()
    cands = factory.candidates(state, 0)
    mask = factory.mask(cands)
    assert len(mask) == NUM_CANDIDATES
    assert all(mask)


def test_isolated_instances_are_independent():
    """Two factories must hold independent stateful-expert instances so per-env
    rollouts don't cross-contaminate (producer/oep carry memory)."""
    f1, f2 = CandidateFactory(), CandidateFactory()
    assert f1._experts["producer"] is not f2._experts["producer"]
    assert f1._experts["oep"] is not f2._experts["oep"]
