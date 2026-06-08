"""Tests for the B3 dense potential-based reward (reward_mode='dense_potential')."""
from __future__ import annotations

import math

import numpy as np
import pytest

from python.orbit_wars_gym.gym_env import OrbitWarsGymEnv


def test_potential_even_near_start():
    env = OrbitWarsGymEnv(reward_mode="dense_potential")
    env.reset(seed=0)
    phi, comps = env._dense_potential(env.state, player=0)
    assert 0.0 <= phi <= 1.0
    assert 0.3 < phi < 0.7  # symmetric start -> roughly even
    assert set(comps) == {"prod_share", "ship_share", "planet_share", "potential"}


def test_pbrs_reward_equals_gamma_phi_next_minus_phi_prev():
    """Non-terminal shaped reward must be exactly F = γ·Φ(s') − Φ(s) (policy-invariant)."""
    env = OrbitWarsGymEnv(reward_mode="dense_potential", action_mode="candidate")
    env.reset(seed=1)
    prev_phi, _ = env._dense_potential(env.state, player=0)
    _, reward, terminated, _, info = env.step(1)  # play the producer candidate
    assert not terminated
    next_phi = info["dense_potential"]["potential"]
    expected = env.reward_gamma * next_phi - prev_phi
    assert math.isclose(reward, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_dense_episode_runs_finite():
    env = OrbitWarsGymEnv(reward_mode="dense_potential", action_mode="candidate")
    env.reset(seed=2)
    for _ in range(150):
        _, reward, terminated, truncated, info = env.step(1)
        assert np.isfinite(reward)
        assert 0.0 <= info["dense_potential"]["potential"] <= 1.0
        if terminated or truncated:
            break


def test_collapse_floors_potential():
    """A wiped-out player (no owned planets) gets Φ = 0 (collapse_penalty)."""
    env = OrbitWarsGymEnv(reward_mode="dense_potential")
    env.reset(seed=3)
    state = env.state
    # force every planet to the opponent so player 0 owns nothing
    for p in state.get("planets", []):
        p["owner"] = 1
    phi, comps = env._dense_potential(state, player=0)
    assert phi == 0.0
    assert comps["planet_share"] == 0.0


def test_legacy_mode_unchanged():
    env = OrbitWarsGymEnv()  # default reward_mode == "legacy"
    assert env.reward_mode == "legacy"
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0, 0, 0, 0, 0]))
    assert np.isfinite(reward)
    assert "dense_potential" not in info


def test_bad_reward_mode_raises():
    with pytest.raises(ValueError):
        OrbitWarsGymEnv(reward_mode="bogus")
