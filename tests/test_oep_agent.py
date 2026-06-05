from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from bots.oep import planner as oep_planner
from bots.oep.planner import OEPLiteConfig, OEPLiteRuntime
from bots.producer.agent import agent as producer_agent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from scripts.benchmark_submission import benchmark_two_player_spec


def _initial_official_obs(*, seed: int = 0, player: int = 0) -> dict:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(episode_steps=32, enable_comets=False, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    return to_official_observation(state, player=int(player))


def test_oep_agent_runs_legal_smoke_against_producer() -> None:
    report = benchmark_two_player_spec(
        Path("bots/oep/agent.py"),
        "producer",
        seeds=[0],
        episode_steps=32,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )
    records = report["records"]
    assert len(records) == 2
    assert all(record["crashes"] == 0.0 for record in records)
    assert all(record["timeouts"] == 0.0 for record in records)
    assert all(record["invalid_actions"] == 0.0 for record in records)


def test_oep_runtime_has_no_time_budget_fallback_config() -> None:
    config_fields = {field.name for field in fields(OEPLiteConfig)}

    assert "time_budget_ms" not in config_fields
    assert not hasattr(OEPLiteRuntime, "_deadline")
    assert not hasattr(OEPLiteRuntime, "_deadline_expired")


def test_oep_runtime_fails_fast_when_oep_planner_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_from_oep(**_kwargs):
        raise RuntimeError("forced OEP planner failure")

    monkeypatch.setattr(oep_planner, "plan_oep_waves", _raise_from_oep)
    runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(),
    )

    with pytest.raises(RuntimeError, match="forced OEP planner failure"):
        runtime.act(_initial_official_obs())


def test_oep_env_config_overrides_search_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OEP_FRACTIONS", "0.25,0.5")
    monkeypatch.setenv("OEP_MIN_ADVANTAGE", "0.1")
    monkeypatch.setenv("OEP_MAX_SOURCES_PER_LANE", "4")
    monkeypatch.setenv("OEP_MAX_OFFENSIVE_TARGETS", "5")
    monkeypatch.setenv("OEP_MAX_DEFENSIVE_TARGETS", "1")
    monkeypatch.setenv("OEP_MAX_WAVES_PER_TURN", "3")
    monkeypatch.setenv("OEP_OPPONENT_RESPONSE_MODE", "producer")
    monkeypatch.setenv("OEP_PRODUCER_PLAN_MODE", "inline")

    config = oep_planner._env_config()

    assert config.fractions == (0.25, 0.5)
    assert config.min_advantage == 0.1
    assert config.max_sources_per_lane == 4
    assert config.max_offensive_targets == 5
    assert config.max_defensive_targets == 1
    assert config.max_waves_per_turn == 3
    assert config.opponent_response_mode == "producer"
    assert config.producer_plan_mode == "inline"


def test_oep_env_config_rejects_empty_fraction_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OEP_FRACTIONS", ",,,")

    with pytest.raises(ValueError, match="OEP_FRACTIONS"):
        oep_planner._env_config()


def test_oep_inline_producer_plan_matches_policy_on_initial_obs() -> None:
    obs = _initial_official_obs(seed=0, player=0)
    policy_runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(producer_plan_mode="policy"),
    )
    inline_runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(producer_plan_mode="inline"),
    )

    assert inline_runtime.act(obs) == policy_runtime.act(obs)
