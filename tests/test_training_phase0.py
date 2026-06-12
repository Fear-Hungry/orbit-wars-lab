from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from python.agents.policy import FlatActorCritic
from python.orbit_wars_gym.encoding import encode_state, observation_dim
from python.orbit_wars_gym.rules import normalized_margin
from python.train.train_ppo import (
    Phase0TrainingConfig,
    _batched_rollout_supported,
    _evaluate_margin,
    _parse_opponents,
    bc_anchor_coef,
    build_phase0_env,
    build_phase5_4p_config,
    four_player_shaping_scales,
    normalized_margin_scale,
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
    assert env.opponent_policy.__name__ == "defensive"


def test_phase0_env_can_be_built_for_four_players():
    env = build_phase0_env(seed=0, num_players=4, opponent_name="greedy", enable_comets=True)

    obs, _ = env.reset(seed=123)

    assert env.num_players == 4
    assert env.backend.num_players == 4
    assert obs.shape[0] > 0


def test_phase0_env_can_train_from_nonzero_learner_seat():
    env = build_phase0_env(
        seed=0,
        num_players=4,
        learner_player=2,
        opponent_name="greedy+rush+defensive",
        enable_comets=True,
    )

    obs, _ = env.reset(seed=123)
    expected = encode_state(env.state, player=2)
    _next_obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0, 0])

    assert env.learner_player == 2
    assert obs.tolist() == pytest.approx(expected.tolist(), abs=1e-6)
    assert env.opponent_policy.__name__ == "greedy+rush+defensive"
    assert len(info["scores"]) == 4


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


def test_phase0_env_normalized_margin_reward_tracks_league_margin():
    env = build_phase0_env(
        seed=0,
        num_players=4,
        opponent_name="greedy",
        enable_comets=False,
        normalized_margin_scale=0.2,
    )
    previous_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 2, "owner": 2, "x": 80.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 3, "owner": 3, "x": 20.0, "y": 80.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }
    improved_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 30, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 15, "production": 2},
            {"id": 2, "owner": 2, "x": 80.0, "y": 20.0, "radius": 2.0, "ships": 15, "production": 2},
            {"id": 3, "owner": 3, "x": 20.0, "y": 80.0, "radius": 2.0, "ships": 15, "production": 2},
        ],
        "fleets": [],
    }
    worse_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 12, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 24, "production": 2},
            {"id": 2, "owner": 2, "x": 80.0, "y": 20.0, "radius": 2.0, "ships": 24, "production": 2},
            {"id": 3, "owner": 3, "x": 20.0, "y": 80.0, "radius": 2.0, "ships": 24, "production": 2},
        ],
        "fleets": [],
    }

    assert env._normalized_margin_reward(previous_state, improved_state, player=0) > 0.0
    assert env._normalized_margin_reward(previous_state, worse_state, player=0) < 0.0


def test_phase0_env_elimination_reward_fires_once_on_player_death():
    env = build_phase0_env(seed=0, num_players=4, opponent_name="greedy", elimination_penalty=0.35)
    alive_state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 10, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }
    dead_state = {
        "planets": [
            {"id": 0, "owner": 1, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 4, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 80.0, "radius": 2.0, "ships": 20, "production": 2},
        ],
        "fleets": [],
    }

    assert env._elimination_reward(alive_state, dead_state, player=0) == pytest.approx(-0.35)
    assert env._elimination_reward(dead_state, dead_state, player=0) == 0.0
    reward, info = env.transition_reward(alive_state, dead_state, player=0, player_moves=[], done=False)
    assert info["elimination_reward"] == pytest.approx(-0.35)
    assert reward < 0.0


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


def test_phase5_4p_config_enables_normalized_margin_curriculum():
    cfg = build_phase5_4p_config()

    assert normalized_margin_scale(cfg, 0.0) > normalized_margin_scale(cfg, 1.0)
    assert normalized_margin_scale(cfg, 1.0) > 0.0
    assert cfg.elimination_penalty > 0.0
    assert cfg.learner_seat_rotation is True


def test_phase0_batched_rollout_disabled_when_learner_seat_rotates():
    cfg = Phase0TrainingConfig(num_players=2, rollout_num_envs=8, learner_seat_rotation=True)

    assert not _batched_rollout_supported(cfg, "producer")


def test_eval_margin_uses_all_scores_in_four_player(monkeypatch):
    class FakeEnv:
        state = {"planets": [], "fleets": []}

        def reset(self, *, seed=None):
            return np.zeros(observation_dim(), dtype=np.float32), {}

        def step(self, _action):
            return (
                np.zeros(observation_dim(), dtype=np.float32),
                0.0,
                True,
                False,
                {"scores": [10.0, 0.0, 100.0, 100.0]},
            )

    monkeypatch.setattr("python.train.train_ppo.build_phase0_env", lambda **_kwargs: FakeEnv())
    model = FlatActorCritic(observation_dim())
    cfg = build_phase5_4p_config(eval_max_steps=1)

    margin = _evaluate_margin(model, cfg, opponent_name="greedy", seeds=1, device=torch.device("cpu"))

    scores = [10.0, 0.0, 100.0, 100.0]
    expected = sum(normalized_margin(scores, player) for player in range(4)) / 4.0
    assert margin == pytest.approx(expected)
    assert margin < 0.0


def test_phase0_training_rejects_single_bot_regime():
    with pytest.raises(ValueError, match="at least two distinct opponents"):
        _parse_opponents(["greedy"])


def test_phase0_training_accepts_gate_opponent_pool():
    assert _parse_opponents(["greedy", "defensive", "rush", "pgs_holdwave", "pgs_bigwave"]) == (
        "greedy",
        "defensive",
        "rush",
        "pgs_holdwave",
        "pgs_bigwave",
    )


def test_phase0_training_accepts_composite_four_player_lineups():
    assert _parse_opponents(["producer+oep+pgs_holdwave", "producer+brep+pgs_bigwave"]) == (
        "producer+oep+pgs_holdwave",
        "producer+brep+pgs_bigwave",
    )


def test_phase0_training_disables_batched_rollout_for_composite_lineups():
    cfg = Phase0TrainingConfig(num_players=2, rollout_num_envs=8)

    assert not _batched_rollout_supported(cfg, "producer+oep+pgs_holdwave")


def test_phase0_training_disables_batched_rollout_for_league_opponents():
    cfg = Phase0TrainingConfig(num_players=2, rollout_num_envs=8)

    assert _batched_rollout_supported(cfg, "producer")
    assert not _batched_rollout_supported(cfg, "pgs_holdwave")
    assert not _batched_rollout_supported(cfg, "pgs_bigwave")
    assert not _batched_rollout_supported(cfg, "brep")


def test_phase0_env_can_use_pgs_holdwave_training_opponent():
    env = build_phase0_env(seed=0, num_players=2, opponent_name="pgs_holdwave", enable_comets=True)
    env.reset(seed=123)

    _obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0, 0])

    assert "scores" in info


def test_phase0_env_can_use_composite_four_player_training_lineup():
    env = build_phase0_env(
        seed=0,
        num_players=4,
        opponent_name="producer+brep+pgs_bigwave",
        enable_comets=True,
    )
    env.reset(seed=123)

    _obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0, 0])

    assert env.opponent_policy.__name__ == "producer+brep+pgs_bigwave"
    assert len(info["scores"]) == 4


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
    assert summary["reward_shaping"] == "annealed_base_plus_normalized_margin_plus_temporal_comet_auxiliary"
    assert summary["ship_margin_scale"] == 0.0
    assert summary["normalized_margin_scale_start"] == 0.0
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


def test_phase0_training_short_episode_curriculum_observes_terminal_rewards(tmp_path: Path):
    checkpoint = tmp_path / "phase0_short_episode.pt"
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=23,
            total_timesteps=16,
            episode_steps=2,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=8,
            opponents=("greedy", "rush"),
            checkpoint_out=str(checkpoint),
            enable_comets=False,
        )
    )

    assert summary["episode_steps"] == 2
    assert summary["completed_episodes"] > 0
    assert checkpoint.exists()


def test_phase0_training_bc_anchor_collects_teacher_labels(tmp_path: Path):
    checkpoint = tmp_path / "phase0_bc_anchor.pt"
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=29,
            total_timesteps=16,
            episode_steps=8,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            opponents=("greedy", "rush"),
            checkpoint_out=str(checkpoint),
            enable_comets=False,
            bc_anchor_coef=0.05,
            bc_anchor_coef_end=0.01,
            bc_anchor_teacher="greedy",
        )
    )

    assert summary["bc_anchor_coef"] == 0.05
    assert summary["bc_anchor_coef_end"] == 0.01
    assert summary["bc_anchor_teacher"] == "greedy"
    assert summary["last_bc_anchor_examples"] > 0
    assert summary["last_bc_anchor_loss"] > 0.0
    assert summary["last_bc_anchor_effective_coef"] == pytest.approx(0.01)
    assert checkpoint.exists()


def test_bc_anchor_coef_linearly_decays_to_configured_end():
    cfg = Phase0TrainingConfig(bc_anchor_coef=0.08, bc_anchor_coef_end=0.02)

    assert bc_anchor_coef(cfg, 0.0) == pytest.approx(0.08)
    assert bc_anchor_coef(cfg, 0.5) == pytest.approx(0.05)
    assert bc_anchor_coef(cfg, 1.0) == pytest.approx(0.02)


def test_phase0_training_requires_teacher_when_bc_anchor_enabled():
    with pytest.raises(ValueError, match="bc_anchor_teacher"):
        train_phase0(
            Phase0TrainingConfig(
                seed=31,
                total_timesteps=8,
                rollout_steps=4,
                update_epochs=1,
                minibatch_size=4,
                opponents=("greedy", "rush"),
                enable_comets=False,
                bc_anchor_coef=0.05,
            )
        )


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


def test_phase0_training_inherits_checkpoint_decoder(tmp_path: Path):
    initial_checkpoint = tmp_path / "phase0_single_action_initial.pt"
    resumed_checkpoint = tmp_path / "phase0_single_action_resumed.pt"
    train_phase0(
        Phase0TrainingConfig(
            seed=17,
            total_timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            decoder_max_moves_per_turn=1,
            checkpoint_out=str(initial_checkpoint),
            enable_comets=False,
        )
    )

    summary = train_phase0(
        Phase0TrainingConfig(
            seed=18,
            total_timesteps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            checkpoint_in=str(initial_checkpoint),
            checkpoint_out=str(resumed_checkpoint),
            enable_comets=False,
        )
    )

    assert summary["decoder"]["max_moves_per_turn"] == 1
    resumed_payload = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    assert resumed_payload["config"]["decoder_max_moves_per_turn"] == 1
    assert resumed_payload["summary"]["decoder"]["max_moves_per_turn"] == 1


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
    assert summary["learner_seat_rotation"] is True
    assert sum(1 for count in summary["learner_seat_segments"].values() if count > 0) > 1
    assert summary["enable_comets"] is True
    assert summary["four_player_vulnerability_scale_end"] < summary["four_player_vulnerability_scale_start"]
    assert summary["four_player_leader_scale_end"] < summary["four_player_leader_scale_start"]
    assert summary["four_player_third_player_scale_end"] < summary["four_player_third_player_scale_start"]
    assert summary["normalized_margin_scale_end"] < summary["normalized_margin_scale_start"]
    assert summary["normalized_margin_scale_end"] > 0.0
    assert checkpoint.exists()
