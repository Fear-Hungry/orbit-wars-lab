"""Plumbing test for the candidate-selector PPO integration (B1 train_ppo wiring).

This is a unit-scale smoke (48 timesteps, single env, no checkpoint) — it proves the
candidate arch flows through build -> rollout -> update without crashing and with the
right shapes. It is NOT the campaign that answers "does it train without collapse";
that needs a real (longer) run and a paired eval.
"""
from __future__ import annotations

from python.train.train_ppo import Phase0TrainingConfig, train_phase0


def test_candidate_selector_ppo_plumbing_smoke():
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=3,
            policy_arch="candidate_selector",
            opponents="greedy,defensive",  # 2 cheap stateless opponents (parser needs >=2)
            total_timesteps=48,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=8,
            enable_comets=True,
        )
    )
    assert summary["arch"] == "candidate_selector"
    assert summary["timesteps"] == 48
    assert summary["updates"] >= 1
    # candidate arch forces the single-env path (batched mask is 50-dim, incompatible)
    assert summary["rollout_backend"] == "gym_single_env"
    # the PPO update actually ran (finite internal stats present)
    assert "last_value_loss" in summary
    assert len(summary["update_series"]) >= 1
    # episodes were observed (rollout stepped the env)
    assert summary["episodes_observed"] > 0


def test_candidate_selector_batched_config_still_uses_single_env():
    """Even with rollout_num_envs>1, the candidate arch must fall back to single-env
    (it cannot use the 50-dim batched mask) — guards against a silent shape crash."""
    summary = train_phase0(
        Phase0TrainingConfig(
            seed=4,
            policy_arch="candidate_selector",
            opponents="greedy,defensive",
            total_timesteps=32,
            rollout_steps=8,
            rollout_num_envs=4,  # would normally trigger the batched path
            update_epochs=1,
            minibatch_size=8,
            enable_comets=True,
        )
    )
    assert summary["arch"] == "candidate_selector"
    assert summary["rollout_backend"] == "gym_single_env"
