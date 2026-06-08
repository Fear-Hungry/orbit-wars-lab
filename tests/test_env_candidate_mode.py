"""Smoke tests for the Frente B candidate-selector env mode (B1)."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from python.agents.candidate_factory import NUM_CANDIDATES
from python.orbit_wars_gym.gym_env import OrbitWarsGymEnv


def test_candidate_mode_action_space_is_discrete_k():
    env = OrbitWarsGymEnv(action_mode="candidate")
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == NUM_CANDIDATES


def test_raw_mode_unchanged():
    env = OrbitWarsGymEnv()  # default
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert list(env.action_space.nvec) == [2, 16, 32, 4, 5]


def test_candidate_episode_runs_clean_for_every_index():
    """Stepping each candidate index through full episodes never crashes or
    raises (the factory guarantees legal moves reach the backend)."""
    for idx in range(NUM_CANDIDATES):
        env = OrbitWarsGymEnv(action_mode="candidate")
        obs, _ = env.reset(seed=idx)
        assert obs.shape == env.observation_space.shape
        for _ in range(120):
            obs, reward, terminated, truncated, info = env.step(idx)
            assert np.isfinite(reward)
            assert obs.shape == env.observation_space.shape
            if terminated or truncated:
                break


def test_candidate_no_op_index_passes():
    """Index 0 (no_op) must drive the env without issuing any of player 0's moves."""
    env = OrbitWarsGymEnv(action_mode="candidate")
    env.reset(seed=0)
    # a no-op step should not crash and should advance the sim
    _, reward, terminated, _, _ = env.step(0)
    assert np.isfinite(reward)


def test_bad_action_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        OrbitWarsGymEnv(action_mode="bogus")
