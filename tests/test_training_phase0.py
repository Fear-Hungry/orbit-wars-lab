from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from python.train.train_ppo import (
    Phase0TrainingConfig,
    _parse_opponents,
    build_phase0_env,
    build_phase5_4p_config,
    four_player_shaping_scales,
    shaping_scales,
    train_phase0,
    train_phase5_4p,
)


def test_phase0_env_is_two_player_and_uses_requested_opponent():
    env = build_phase0_env(seed=0, opponent_name="defensive", enable_comets=False)

    obs, _ = env.reset(seed=123)

    assert env.num_players == 2
    assert env.backend.num_players == 2
    assert obs.shape[0] > 0
    assert env.opponent_policy.__name__ == "defensive_agent"


def test_phase0_env_can_be_built_for_four_players():
    env = build_phase0_env(seed=0, num_players=4, opponent_name="greedy", enable_comets=True)

    obs, _ = env.reset(seed=123)

    assert env.num_players == 4
    assert env.backend.num_players == 4
    assert obs.shape[0] > 0


def test_phase0_env_can_run_with_comets_past_official_spawn_step():
    env = build_phase0_env(seed=0, opponent_name="greedy", enable_comets=True)
    env.reset(seed=123)

    for _ in range(60):
        _, states = env.backend.step_with_states([[[], []]])
        state = states[0]
    assert state["comets"] != []
    assert state["comet_planet_ids"] != []


def test_phase0_env_tracks_sun_and_border_losses_in_shaping():
    env = build_phase0_env(seed=0, opponent_name="greedy", enable_comets=False)
    sun_previous_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 38.0, "y": 50.0, "radius": 2.0, "ships": 30, "production": 2},
        ],
        "fleets": [],
    }
    border_previous_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 1.5, "y": 50.0, "radius": 1.0, "ships": 30, "production": 2},
        ],
        "fleets": [],
    }
    sun_next_state = {"planets": sun_previous_state["planets"], "fleets": []}
    border_next_state = {"planets": border_previous_state["planets"], "fleets": []}
    sun_moves = [[0, 0.0, 10]]
    border_moves = [[0, math.pi, 10]]

    sun_losses, border_losses = env._loss_counts(sun_previous_state, sun_next_state, player=0, player_moves=sun_moves)
    border_only_sun, border_only_border = env._loss_counts(
        border_previous_state,
        border_next_state,
        player=0,
        player_moves=border_moves,
    )

    assert sun_losses == 1
    assert border_losses == 0
    assert border_only_sun == 0
    assert border_only_border == 1


def test_phase0_env_comet_auxiliary_reward_tracks_capture_swing():
    env = build_phase0_env(seed=0, opponent_name="greedy", enable_comets=True)
    previous_state = {
        "planets": [{"id": 9, "owner": -1, "x": 40.0, "y": 50.0, "radius": 1.0, "ships": 6, "production": 1}],
        "comet_planet_ids": [9],
    }
    captured_state = {
        "planets": [{"id": 9, "owner": 0, "x": 42.0, "y": 50.0, "radius": 1.0, "ships": 3, "production": 1}],
        "comet_planet_ids": [9],
    }
    lost_state = {
        "planets": [{"id": 9, "owner": 1, "x": 44.0, "y": 50.0, "radius": 1.0, "ships": 4, "production": 1}],
        "comet_planet_ids": [9],
    }

    assert env._comet_auxiliary_reward(previous_state, captured_state, player=0) > 0.0
    assert env._comet_auxiliary_reward(captured_state, lost_state, player=0) < 0.0


def test_phase0_env_ship_margin_reward_tracks_score_delta():
    env = build_phase0_env(seed=0, opponent_name="greedy", enable_comets=False, ship_margin_scale=0.01)
    previous_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }
    improved_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 30, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 15, "production": 2},
        ],
        "fleets": [],
    }
    worse_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 12, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 24, "production": 2},
        ],
        "fleets": [],
    }

    assert env._ship_margin_reward(previous_state, improved_state, player=0) > 0.0
    assert env._ship_margin_reward(previous_state, worse_state, player=0) < 0.0


def test_phase0_training_anneals_base_and_comet_shaping():
    cfg = Phase0TrainingConfig(
        enable_comets=True,
        base_shaping_scale_start=1.0,
        base_shaping_scale_end=0.15,
        comet_shaping_scale_start=0.08,
        comet_shaping_scale_end=0.0,
    )

    start_base, start_comet = shaping_scales(cfg, 0.0)
    end_base, end_comet = shaping_scales(cfg, 1.0)

    assert start_base > end_base
    assert start_comet > end_comet
    assert end_comet == 0.0


def test_phase0_training_anneals_four_player_strategic_shaping():
    cfg = Phase0TrainingConfig(
        num_players=4,
        four_player_vulnerability_scale_start=0.06,
        four_player_vulnerability_scale_end=0.02,
        four_player_leader_scale_start=0.05,
        four_player_leader_scale_end=0.02,
        four_player_third_player_scale_start=0.04,
        four_player_third_player_scale_end=0.015,
    )

    start_vulnerability, start_leader, start_third_player = four_player_shaping_scales(cfg, 0.0)
    end_vulnerability, end_leader, end_third_player = four_player_shaping_scales(cfg, 1.0)

    assert start_vulnerability > end_vulnerability
    assert start_leader > end_leader
    assert start_third_player > end_third_player


def test_phase0_training_rejects_single_bot_regime():
    with pytest.raises(ValueError, match="at least two distinct opponents"):
        _parse_opponents(["greedy"])


def test_phase0_training_accepts_gate_opponent_pool():
    assert _parse_opponents(["greedy", "defensive", "rush", "anti_meta", "weak_random"]) == (
        "greedy",
        "defensive",
        "rush",
        "anti_meta",
        "weak_random",
    )


def test_phase0_env_four_player_strategic_reward_improves_when_exposure_and_pressure_gaps_drop():
    env = build_phase0_env(seed=0, num_players=4, opponent_name="greedy", enable_comets=True)
    previous_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 50.0, "y": 50.0, "radius": 2.0, "ships": 10, "production": 2},
            {"id": 1, "owner": 1, "x": 53.0, "y": 50.0, "radius": 2.0, "ships": 40, "production": 2},
            {"id": 2, "owner": 2, "x": 70.0, "y": 70.0, "radius": 2.0, "ships": 60, "production": 2},
            {"id": 3, "owner": 3, "x": 30.0, "y": 30.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }
    current_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 50.0, "y": 50.0, "radius": 2.0, "ships": 22, "production": 2},
            {"id": 1, "owner": 1, "x": 58.0, "y": 50.0, "radius": 2.0, "ships": 18, "production": 2},
            {"id": 2, "owner": 2, "x": 70.0, "y": 70.0, "radius": 2.0, "ships": 45, "production": 2},
            {"id": 3, "owner": 3, "x": 30.0, "y": 30.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }

    vulnerability_reward, leader_reward, third_player_reward = env._four_player_strategic_reward(
        previous_state,
        current_state,
        player=0,
    )

    assert vulnerability_reward > 0.0
    assert leader_reward > 0.0
    assert third_player_reward > 0.0


def test_phase0_training_runs_real_ppo_loop_and_emits_metrics(tmp_path: Path):
    checkpoint = tmp_path / "phase0.pt"
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=7,
            total_timesteps=48,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            checkpoint_out=str(checkpoint),
            enable_comets=True,
        )
    )

    assert summary["algorithm"] == "ppo"
    assert summary["timesteps"] == 48
    assert summary["updates"] >= 1
    assert set(summary["opponents"]) == {"greedy", "defensive", "rush", "anti_meta", "weak_random"}
    assert all(summary["opponent_segments"][name] >= 1 for name in summary["opponents"])
    assert summary["enable_comets"] is True
    assert summary["reward_shaping"] == "annealed_base_plus_temporal_comet_auxiliary"
    assert summary["ship_margin_scale"] == 0.0
    assert summary["decoder"]["max_moves_per_turn"] == 8
    assert summary["decoder"]["min_ships_to_launch"] == 2
    assert summary["decoder"]["reserve_home_ships"] == 8
    assert summary["base_shaping_scale_end"] < summary["base_shaping_scale_start"]
    assert summary["comet_shaping_scale_end"] < summary["comet_shaping_scale_start"]
    assert summary["episodes_observed"] > 0
    assert summary["mean_neutral_capture_rate"] >= 0.0
    assert 0.0 <= summary["mean_early_survival_rate"] <= 1.0
    assert checkpoint.exists()


def test_phase0_training_can_collect_batched_rust_rollouts(tmp_path: Path):
    checkpoint = tmp_path / "phase0_batched.pt"
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=13,
            total_timesteps=32,
            rollout_steps=4,
            rollout_num_envs=4,
            update_epochs=1,
            minibatch_size=8,
            opponents=("greedy", "rush"),
            checkpoint_out=str(checkpoint),
            enable_comets=False,
        )
    )

    assert summary["algorithm"] == "ppo"
    assert summary["timesteps"] == 32
    assert summary["rollout_num_envs"] == 4
    assert summary["rollout_backend"] == "rust_batch"
    assert summary["training_wall_seconds"] > 0.0
    assert summary["env_steps_per_second"] > 0.0
    assert set(summary["opponents"]) == {"greedy", "rush"}
    assert checkpoint.exists()


def test_phase0_training_can_resume_from_checkpoint(tmp_path: Path):
    initial_checkpoint = tmp_path / "phase0_initial.pt"
    resumed_checkpoint = tmp_path / "phase0_resumed.pt"
    train_phase0(
        Phase0TrainingConfig(
            seed=7,
            total_timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            checkpoint_out=str(initial_checkpoint),
            enable_comets=False,
        )
    )

    summary = train_phase0(
        Phase0TrainingConfig(
            seed=8,
            total_timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            checkpoint_in=str(initial_checkpoint),
            checkpoint_out=str(resumed_checkpoint),
            enable_comets=False,
        )
    )

    assert summary["checkpoint_in"] == str(initial_checkpoint)
    assert resumed_checkpoint.exists()
    resumed_payload = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    assert resumed_payload["config"]["checkpoint_in"] == str(initial_checkpoint)
    assert resumed_payload["summary"]["checkpoint_in"] == str(initial_checkpoint)
    assert resumed_payload["summary"]["decoder"]["max_moves_per_turn"] == 8


def test_phase0_training_runs_real_ppo_loop_for_four_players(tmp_path: Path):
    checkpoint = tmp_path / "phase5_4p.pt"
    summary = train_phase5_4p(
        build_phase5_4p_config(
            seed=11,
            total_timesteps=32,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            checkpoint_out=str(checkpoint),
        )
    )

    assert summary["algorithm"] == "ppo"
    assert summary["policy_track"] == "phase5_4p"
    assert summary["num_players"] == 4
    assert summary["enable_comets"] is True
    assert summary["four_player_vulnerability_scale_end"] < summary["four_player_vulnerability_scale_start"]
    assert summary["four_player_leader_scale_end"] < summary["four_player_leader_scale_start"]
    assert summary["four_player_third_player_scale_end"] < summary["four_player_third_player_scale_start"]
    assert checkpoint.exists()
