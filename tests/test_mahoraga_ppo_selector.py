from __future__ import annotations

from dataclasses import dataclass

from bots.mahoraga_ppo.selector import (
    MAHORAGA,
    PPO,
    roles_from_runtime,
    select_policy,
)

_KW = dict(min_confidence=0.65, ppo_ready=True)


def test_4p_always_stays_with_mahoraga() -> None:
    roles = {1: ("expander", 0.9)}
    assert select_policy(num_players=4, roles=roles, **_KW) == MAHORAGA


def test_dangerous_roles_stay_with_mahoraga() -> None:
    for role in ("rusher", "wave", "sprayer", "unknown"):
        assert select_policy(num_players=2, roles={1: (role, 0.99)}, **_KW) == MAHORAGA


def test_low_confidence_stays_with_mahoraga() -> None:
    assert select_policy(num_players=2, roles={1: ("expander", 0.5)}, **_KW) == MAHORAGA


def test_no_profiles_yet_stays_with_mahoraga() -> None:
    assert select_policy(num_players=2, roles={}, **_KW) == MAHORAGA


def test_ppo_not_ready_stays_with_mahoraga() -> None:
    roles = {1: ("expander", 0.9)}
    assert select_policy(num_players=2, roles=roles, min_confidence=0.65, ppo_ready=False) == MAHORAGA


def test_favourable_confident_2p_hands_to_ppo() -> None:
    for role in ("expander", "producer-like", "turtle"):
        assert select_policy(num_players=2, roles={1: (role, 0.8)}, **_KW) == PPO


def test_any_hostile_opponent_vetoes_ppo() -> None:
    roles = {1: ("expander", 0.9), 2: ("rusher", 0.9)}
    assert select_policy(num_players=2, roles=roles, **_KW) == MAHORAGA


@dataclass
class _FakeStat:
    role: str
    confidence: float


class _FakeRuntimePrivate:
    _opp_profiles = {1: _FakeStat("expander", 0.8)}


class _FakeRuntimePublic:
    def opponent_roles(self) -> dict[int, tuple[str, float]]:
        return {2: ("wave", 0.7)}


def test_roles_from_runtime_prefers_public_accessor() -> None:
    assert roles_from_runtime(_FakeRuntimePublic()) == {2: ("wave", 0.7)}


def test_roles_from_runtime_falls_back_to_profiles() -> None:
    assert roles_from_runtime(_FakeRuntimePrivate()) == {1: ("expander", 0.8)}


def test_hybrid_agent_runs_one_turn_as_pure_mahoraga() -> None:
    # With the ppo_policy stub (ready() False) the hybrid must behave exactly
    # like Mahoraga and report the selector counters.
    from bots.mahoraga_ppo.agent import MahoragaPPORuntime
    from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
    from python.orbit_wars_gym.observation import to_official_observation

    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=0,
        config=RustConfig(episode_steps=8, enable_comets=False, act_timeout=1.0),
    )
    state = backend.reset(0)[0]
    runtime = MahoragaPPORuntime()
    moves = runtime.act(to_official_observation(state, player=0))
    assert isinstance(moves, list)
    stats = runtime.runtime_stats()
    assert stats["calls"] == 1
    assert stats["mahoraga_turns"] == 1
    assert stats["ppo_turns"] == 0
