from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch
from bots.oep import planner as oep_planner
from bots.oep.planner import OEPLiteConfig, OEPLiteRuntime
from bots.producer import _upstream as producer_upstream
from bots.producer.agent import agent as producer_agent
from orbit_lite.adapter import single_obs_to_tensor
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


def test_oep_memory_selection_summary_resets_with_profile() -> None:
    memory = oep_planner.OEPLiteMemory()
    empty_entries = oep_planner._empty_entries(torch.device("cpu"), torch.float32)

    memory.record_selection(
        step=12,
        chose_oep=True,
        oep_fitness=2.5,
        producer_fitness=1.0,
        oep_entries=empty_entries,
        producer_entries=empty_entries,
        ordinal_win_rate=2.0 / 3.0,
        ordinal_wins=2,
        ordinal_variants=3,
    )
    memory.record_plan_memory_variant(candidate_count=2, chosen_index=1)
    memory.record_beam_first(candidate_count=4, chosen_index=2)
    memory.record_beam_pair(candidate_count=9, chosen_index=3)

    summary = memory.selection_summary()
    assert summary["decisions"] == 1.0
    assert summary["oep_choices"] == 1.0
    assert summary["producer_choices"] == 0.0
    assert summary["mean_fitness_delta_oep_minus_producer"] == 1.5
    assert summary["mean_ordinal_win_rate_oep_vs_producer"] == 2.0 / 3.0
    assert summary["min_ordinal_win_rate_oep_vs_producer"] == 2.0 / 3.0
    assert summary["max_ordinal_win_rate_oep_vs_producer"] == 2.0 / 3.0
    assert summary["ordinal_wins_oep_vs_producer"] == 2.0
    assert summary["ordinal_variants_oep_vs_producer"] == 3.0
    assert summary["plan_memory_variant_calls"] == 1.0
    assert summary["plan_memory_variant_choices"] == 1.0
    assert summary["plan_memory_variant_candidates"] == 2.0
    assert summary["plan_memory_variant_choice_rate"] == 1.0
    assert summary["beam_first_calls"] == 1.0
    assert summary["beam_first_choices"] == 1.0
    assert summary["beam_first_candidates"] == 4.0
    assert summary["beam_first_choice_rate"] == 1.0
    assert summary["beam_pair_calls"] == 1.0
    assert summary["beam_pair_choices"] == 1.0
    assert summary["beam_pair_candidates"] == 9.0
    assert summary["beam_pair_choice_rate"] == 1.0
    assert summary["early_decisions"] == 1.0
    assert summary["early_oep_choice_rate"] == 1.0
    assert summary["early_mean_fitness_delta_oep_minus_producer"] == 1.5

    memory.reset_profile()

    reset_summary = memory.selection_summary()
    assert reset_summary["decisions"] == 0.0
    assert reset_summary["oep_choices"] == 0.0
    assert reset_summary["mean_fitness_delta_oep_minus_producer"] == 0.0
    assert reset_summary["mean_ordinal_win_rate_oep_vs_producer"] == 0.0
    assert reset_summary["ordinal_variants_oep_vs_producer"] == 0.0
    assert reset_summary["plan_memory_variant_calls"] == 0.0
    assert reset_summary["plan_memory_variant_choice_rate"] == 0.0
    assert reset_summary["beam_first_calls"] == 0.0
    assert reset_summary["beam_first_choice_rate"] == 0.0
    assert reset_summary["beam_pair_calls"] == 0.0
    assert reset_summary["beam_pair_choice_rate"] == 0.0
    assert reset_summary["early_decisions"] == 0.0


def test_oep_ordinal_win_stats_are_thresholded_by_rank() -> None:
    chose, wins, variants, win_rate = oep_planner._ordinal_win_stats(
        oep_scores=(1.0, 3.0, -1.0, 2.0, 3.0),
        producer_scores=(0.0, 4.0, -2.0, 2.0, 2.0),
        threshold=0.6,
    )

    assert chose is True
    assert wins == 3
    assert variants == 5
    assert win_rate == 0.6


def test_oep_opponent_launch_variants_perturb_base_prediction() -> None:
    base = oep_planner.LaunchSet(
        source_slots=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        target_slots=torch.tensor([[4, 5, 6, 7]], dtype=torch.long),
        ships=torch.tensor([[10.0, 8.0, 6.0, 4.0]]),
        eta=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        owner=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        valid=torch.tensor([[True, True, True, True]]),
    )

    variants = oep_planner._opponent_launch_variants(base, variant_count=5)

    assert len(variants) == 5
    assert variants[0] is base
    assert variants[1] is not None
    assert torch.equal(variants[1].ships, torch.tensor([[7.0, 6.0, 4.0, 3.0]]))
    assert variants[2] is not None
    assert torch.equal(variants[2].ships, torch.tensor([[5.0, 4.0, 3.0, 2.0]]))
    assert variants[3] is not None
    assert torch.equal(variants[3].eta, torch.tensor([[2.0, 3.0, 4.0, 5.0]]))
    assert variants[4] is not None
    assert int(variants[4].valid.sum().item()) == 3


def test_oep_select_best_plan_variant_chooses_stronger_entry() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    movement = oep_planner.PlanetMovement.from_obs_tensors(
        obs_tensors,
        config=oep_planner._movement_config(oep_planner.OEPPlannerConfig(), player_count=2),
    )
    parsed = oep_planner.parse_obs(obs_tensors)
    cache = oep_planner.build_distance_cache(movement, max_k=18)
    status = movement.garrison_status(max_horizon=18)
    alive_by_step = movement.alive_by_step[:19]
    base_entries = oep_planner.plan_oep_waves(
        movement=movement,
        obs=parsed,
        obs_tensors=obs_tensors,
        cache=cache,
        status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=oep_planner.OEPPlannerConfig(),
        fractions=(0.5, 1.0),
        player_count=2,
        opponent_entries=None,
    )
    lanes = oep_planner._lane_intents_from_entries(
        base_entries,
        movement=movement,
        status=status,
        obs=parsed,
        player_id=0,
    )
    assert lanes
    rebuilt_entries = oep_planner._entries_from_lane_intents(
        lanes,
        movement=movement,
        obs=parsed,
        status=status,
        config=oep_planner.OEPPlannerConfig(),
        player_id=0,
    )
    assert bool(rebuilt_entries.valid.any().item())

    chosen, fitness, index = oep_planner._select_best_plan_variant(
        (oep_planner._empty_entries(parsed.device, parsed.ships.dtype), rebuilt_entries),
        opponent_launch_set=None,
        status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        player_count=2,
        player_id=0,
    )

    assert index == 1
    assert fitness > 0.0
    assert bool(chosen.valid.any().item())


def test_oep_beam_first_generates_plan_variants() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    movement = oep_planner.PlanetMovement.from_obs_tensors(
        obs_tensors,
        config=oep_planner._movement_config(oep_planner.OEPPlannerConfig(), player_count=2),
    )
    parsed = oep_planner.parse_obs(obs_tensors)
    cache = oep_planner.build_distance_cache(movement, max_k=18)
    status = movement.garrison_status(max_horizon=18)
    alive_by_step = movement.alive_by_step[:19]

    entries, candidate_count, chosen_index = oep_planner.plan_oep_beam_first_waves(
        movement=movement,
        obs=parsed,
        obs_tensors=obs_tensors,
        cache=cache,
        status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=oep_planner.OEPPlannerConfig(),
        fractions=(0.5, 1.0),
        player_count=2,
        opponent_entries=None,
        opponent_launch_set=None,
        beam_width=3,
    )

    assert candidate_count >= 2
    assert chosen_index >= 0
    assert bool(entries.valid.any().item())


def test_oep_beam_pair_generates_plan_variants() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    movement = oep_planner.PlanetMovement.from_obs_tensors(
        obs_tensors,
        config=oep_planner._movement_config(oep_planner.OEPPlannerConfig(), player_count=2),
    )
    parsed = oep_planner.parse_obs(obs_tensors)
    cache = oep_planner.build_distance_cache(movement, max_k=18)
    status = movement.garrison_status(max_horizon=18)
    alive_by_step = movement.alive_by_step[:19]

    entries, candidate_count, chosen_index = oep_planner.plan_oep_beam_pair_waves(
        movement=movement,
        obs=parsed,
        obs_tensors=obs_tensors,
        cache=cache,
        status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=oep_planner.OEPPlannerConfig(),
        fractions=(0.5, 1.0),
        player_count=2,
        opponent_entries=None,
        opponent_launch_set=None,
        beam_width=2,
    )

    assert candidate_count >= 2
    assert chosen_index >= 0
    assert bool(entries.valid.any().item())


def test_oep_reactive_movement_clone_debits_sources_without_mutating_live_cache() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    movement = oep_planner.PlanetMovement.from_obs_tensors(
        obs_tensors,
        config=oep_planner._movement_config(oep_planner.OEPPlannerConfig(), player_count=2),
    )
    source_slot = int(torch.where(movement.planet_ships >= 3.0)[0][0].item())
    original_ships = movement.planet_ships.clone()
    entries = oep_planner.LaunchEntries(
        source_slots=torch.tensor([source_slot], dtype=torch.long),
        target_slots=torch.tensor([0], dtype=torch.long),
        ships=torch.tensor([3.0], dtype=movement.dtype),
        angle=torch.tensor([0.0], dtype=movement.dtype),
        eta=torch.tensor([1.0], dtype=movement.dtype),
        valid=torch.tensor([True]),
    )

    cloned = oep_planner._clone_movement(movement)
    oep_planner._debit_entry_sources(cloned, entries)

    assert torch.equal(movement.planet_ships, original_ships)
    assert cloned.planet_ships[source_slot] == original_ships[source_slot] - 3.0


def test_producer_runtime_resets_cached_movement_on_step_zero() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    runtime = producer_upstream.ProducerLiteRuntime()

    runtime.tensor_action(obs_tensors)
    first_movement = runtime.memory.movement
    assert first_movement is not None
    assert runtime.memory.last_sparse_action_row is not None

    runtime.tensor_action(obs_tensors)

    assert runtime.memory.movement is not first_movement


def test_effective_horizon_never_crosses_spawn() -> None:
    cfg = oep_planner.OEPPlannerConfig(horizon=18)

    for spawn in oep_planner.COMET_SPAWN_STEPS:
        for step in range(max(0, spawn - 5), spawn + 6):
            effective = oep_planner._effective_config(cfg, step=step)
            future = [
                future_spawn - step
                for future_spawn in oep_planner.COMET_SPAWN_STEPS
                if future_spawn > step
            ]
            if future:
                assert effective.horizon <= min(future)
            else:
                assert effective.horizon == cfg.horizon
            assert effective.horizon >= 1


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
    monkeypatch.setenv("OEP_OPPONENT_RESPONSE_MODE", "producer_inline")
    monkeypatch.setenv("OEP_PRODUCER_PLAN_MODE", "tensor")
    monkeypatch.setenv("OEP_ENABLE_REGROUP", "0")
    monkeypatch.setenv("OEP_LATE_CONFIG_STEP", "120")
    monkeypatch.setenv("OEP_LATE_MAX_SOURCES_PER_LANE", "3")
    monkeypatch.setenv("OEP_ORDINAL_OPPONENT_VARIANTS", "7")
    monkeypatch.setenv("OEP_ORDINAL_WIN_THRESHOLD", "0.75")
    monkeypatch.setenv("OEP_PLAN_MEMORY_VARIANTS", "4")
    monkeypatch.setenv("OEP_BEAM_FIRST_WIDTH", "3")
    monkeypatch.setenv("OEP_BEAM_PAIR_WIDTH", "2")

    config = oep_planner._env_config()

    assert config.fractions == (0.25, 0.5)
    assert config.min_advantage == 0.1
    assert config.max_sources_per_lane == 4
    assert config.max_offensive_targets == 5
    assert config.max_defensive_targets == 1
    assert config.max_waves_per_turn == 3
    assert config.opponent_response_mode == "producer_inline"
    assert config.producer_plan_mode == "tensor"
    assert config.enable_regroup is False
    assert config.late_config_step == 120
    assert config.late_max_sources_per_lane == 3
    assert config.ordinal_opponent_variants == 7
    assert config.ordinal_win_threshold == 0.75
    assert config.plan_memory_variants == 4
    assert config.beam_first_width == 3
    assert config.beam_pair_width == 2


def test_oep_env_config_overrides_reactive_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OEP_REACTIVE_REPLY", "1")
    monkeypatch.setenv("OEP_REACTIVE_REPLY_PRUNE_ADVANTAGE", "2.5")

    config = oep_planner._env_config()

    assert config.reactive_reply is True
    assert config.reactive_reply_prune_advantage == 2.5


def test_oep_config_rejects_reactive_reply_with_ordinal_variants() -> None:
    with pytest.raises(ValueError, match="OEP_REACTIVE_REPLY"):
        OEPLiteConfig(ordinal_opponent_variants=2, reactive_reply=True)


def test_oep_config_rejects_negative_plan_memory_variants() -> None:
    with pytest.raises(ValueError, match="OEP_PLAN_MEMORY_VARIANTS"):
        OEPLiteConfig(plan_memory_variants=-1)


def test_oep_rollout_search_generates_variant_list() -> None:
    obs = producer_upstream._to_list_observation(_initial_official_obs(seed=0, player=0))
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    movement = oep_planner.PlanetMovement.from_obs_tensors(
        obs_tensors,
        config=oep_planner._movement_config(oep_planner.OEPPlannerConfig(), player_count=2),
    )
    parsed = oep_planner.parse_obs(obs_tensors)
    cache = oep_planner.build_distance_cache(movement, max_k=18)
    status = movement.garrison_status(max_horizon=18)
    alive_by_step = movement.alive_by_step[:19]

    variants = oep_planner._oep_plan_variant_list(
        movement=movement,
        obs=parsed,
        obs_tensors=obs_tensors,
        cache=cache,
        status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=oep_planner.OEPPlannerConfig(),
        fractions=(0.5, 1.0),
        player_count=2,
        opponent_entries=None,
        beam_width=3,
    )

    assert len(variants) >= 2
    assert all(isinstance(v, oep_planner.LaunchEntries) for v in variants)
    assert bool(variants[0].valid.any().item())


def test_oep_config_rejects_negative_rollout_search_width() -> None:
    with pytest.raises(ValueError, match="OEP_ROLLOUT_SEARCH_WIDTH"):
        OEPLiteConfig(rollout_search_width=-1)


def test_oep_config_rejects_negative_beam_first_width() -> None:
    with pytest.raises(ValueError, match="OEP_BEAM_FIRST_WIDTH"):
        OEPLiteConfig(beam_first_width=-1)


def test_oep_config_rejects_negative_beam_pair_width() -> None:
    with pytest.raises(ValueError, match="OEP_BEAM_PAIR_WIDTH"):
        OEPLiteConfig(beam_pair_width=-1)


def test_oep_env_config_rejects_reactive_reply_with_ordinal_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OEP_ORDINAL_OPPONENT_VARIANTS", "2")
    monkeypatch.setenv("OEP_REACTIVE_REPLY", "1")

    with pytest.raises(ValueError, match="OEP_REACTIVE_REPLY"):
        oep_planner._env_config()


def test_oep_late_config_caps_sources_after_step() -> None:
    runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(
            max_sources_per_lane=6,
            late_config_step=100,
            late_max_sources_per_lane=4,
        ),
    )
    base = oep_planner.OEPPlannerConfig(max_sources_per_lane=12)

    early = runtime._oep_config(base, step=99)
    late = runtime._oep_config(base, step=100)

    assert early.max_sources_per_lane == 6
    assert late.max_sources_per_lane == 4


def test_oep_env_config_rejects_empty_fraction_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OEP_FRACTIONS", ",,,")

    with pytest.raises(ValueError, match="OEP_FRACTIONS"):
        oep_planner._env_config()


def test_oep_env_config_rejects_invalid_bool_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OEP_ENABLE_REGROUP", "maybe")

    with pytest.raises(ValueError, match="OEP_ENABLE_REGROUP"):
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


def test_oep_tensor_producer_plan_matches_policy_on_initial_obs() -> None:
    obs = _initial_official_obs(seed=0, player=0)
    policy_runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(producer_plan_mode="policy"),
    )
    tensor_runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=OEPLiteConfig(producer_plan_mode="tensor"),
    )

    assert tensor_runtime.act(obs) == policy_runtime.act(obs)
