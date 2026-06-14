"""Regression tests for potential-based reward shaping (Movement 1).

The base shaping term must be in the potential-difference form
``F = γ·Φ(s') − Φ(s)`` (Ng, Harada & Russell 1999), NOT the raw ``Φ(s')`` value
that previously biased the optimum toward the production-greedy (Producer-like)
basin. These tests pin:

  1. the difference form is what the env actually returns,
  2. the terminal state takes ``Φ ≡ 0`` by convention,
  3. the discounted shaping telescopes to a boundary term — i.e. PBRS
     policy-invariance of the return.
"""

from __future__ import annotations

import numpy as np
import pytest
from python.train.train_ppo import build_phase0_env

# action layout is [launch, source_rank, target_rank, fraction_idx, offset_idx];
# launch == 0 passes, giving a deterministic no-op for the learner.
PASS_ACTION = np.array([0, 0, 0, 0, 0], dtype=np.int64)


def _potential_part(env, info) -> float:
    """Recover ``γ·Φ(s') − Φ(s)`` from the returned base shaping reward by adding
    back the (event-based, non-potential) sun/border loss penalties."""
    penalty = (
        env.sun_loss_penalty * info["sun_losses"]
        + env.border_loss_penalty * info["border_losses"]
    )
    return info["base_shaping_reward"] + penalty


def test_base_shaping_is_potential_difference():
    gamma = 0.97
    env = build_phase0_env(seed=0, opponent_name="greedy", enable_comets=False, shaping_gamma=gamma)
    env.reset(seed=0)
    prev = env.state
    phi_prev = env._state_potential(prev, player=0)
    _, _, terminated, _, info = env.step(PASS_ACTION)
    phi_next = 0.0 if terminated else env._state_potential(env.state, player=0)
    expected = gamma * phi_next - phi_prev
    assert _potential_part(env, info) == pytest.approx(expected, abs=1e-9)


def test_terminal_state_potential_is_zeroed():
    # done=True must drop the next-state potential (Φ(terminal) ≡ 0), so the only
    # difference vs done=False is exactly the γ·Φ(s') term (penalties cancel).
    gamma = 0.99
    env = build_phase0_env(seed=3, opponent_name="greedy", enable_comets=False, shaping_gamma=gamma)
    env.reset(seed=3)
    prev = env.state
    env.step(PASS_ACTION)
    nxt = env.state
    r_open = env._base_shaping_reward(prev, nxt, player=0, player_moves=[], done=False)
    r_done = env._base_shaping_reward(prev, nxt, player=0, player_moves=[], done=True)
    assert (r_open - r_done) == pytest.approx(gamma * env._state_potential(nxt, player=0), abs=1e-9)


def test_discounted_shaping_telescopes_to_boundary():
    # Σ_t γ^t (γ·Φ(s_{t+1}) − Φ(s_t)) == γ^N·Φ(s_N) − Φ(s_0)  (Φ(terminal)=0).
    # The collapse to a pure boundary term is PBRS policy-invariance of the
    # discounted return; if the shaping still carried a standing bias it would
    # not telescope.
    gamma = 0.95
    env = build_phase0_env(seed=7, opponent_name="greedy", enable_comets=False, shaping_gamma=gamma)
    env.reset(seed=7)
    phi0 = env._state_potential(env.state, player=0)
    disc_sum = 0.0
    steps = 0
    terminated = False
    for t in range(40):
        _, _, terminated, _, info = env.step(PASS_ACTION)
        disc_sum += (gamma**t) * _potential_part(env, info)
        steps += 1
        if terminated:
            break
    phi_n = 0.0 if terminated else env._state_potential(env.state, player=0)
    expected = (gamma**steps) * phi_n - phi0
    assert disc_sum == pytest.approx(expected, abs=1e-9)
