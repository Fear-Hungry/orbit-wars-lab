"""Tests for CandidateSelectorActorCritic (Frente B selector head)."""
from __future__ import annotations

import torch

from python.agents.policy import (
    FLEET_F,
    FLEET_N,
    GLOBAL_F,
    PLANET_F,
    PLANET_N,
    CandidateSelectorActorCritic,
)

OBS_DIM = GLOBAL_F + PLANET_N * PLANET_F + FLEET_N * FLEET_F  # 3912
K = 6


def _policy():
    torch.manual_seed(0)
    return CandidateSelectorActorCritic(OBS_DIM, num_candidates=K)


def test_forward_shapes():
    pol = _policy()
    obs = torch.zeros(4, OBS_DIM)
    out = pol.forward(obs)
    assert out["candidate"].shape == (4, K)
    assert out["value"].shape == (4,)


def test_get_action_and_value_shapes():
    pol = _policy()
    obs = torch.randn(8, OBS_DIM)
    action, logprob, entropy, value = pol.get_action_and_value(obs)
    assert action.shape == (8,)
    assert logprob.shape == (8,)
    assert entropy.shape == (8,)
    assert value.shape == (8,)
    assert action.min() >= 0 and action.max() < K


def test_mask_excludes_candidates():
    """A masked candidate must never be sampled and carries ~zero probability."""
    pol = _policy()
    obs = torch.randn(16, OBS_DIM)
    # allow only candidate 0 (no_op) and 2 (oep); forbid the rest
    mask = torch.zeros(16, K, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 2] = True
    masks = {"candidate": mask}
    action, _, entropy, _ = pol.get_action_and_value(obs, masks=masks)
    assert set(action.tolist()).issubset({0, 2})
    # entropy bounded by log(2) when only two candidates allowed
    assert (entropy <= torch.log(torch.tensor(2.0)) + 1e-4).all()


def test_single_allowed_candidate_zero_entropy():
    pol = _policy()
    obs = torch.randn(5, OBS_DIM)
    mask = torch.zeros(5, K, dtype=torch.bool)
    mask[:, 3] = True  # only candidate 3 allowed
    action, logprob, entropy, _ = pol.get_action_and_value(obs, masks={"candidate": mask})
    assert (action == 3).all()
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-4)
    assert torch.allclose(logprob, torch.zeros_like(logprob), atol=1e-4)  # log(1) == 0


def test_same_mask_reproduces_logprob_for_ppo_ratio():
    """Sampling then re-scoring the same action with the same mask reproduces logprob
    (the PPO importance ratio must be exact)."""
    pol = _policy()
    obs = torch.randn(10, OBS_DIM)
    mask = torch.ones(10, K, dtype=torch.bool)
    mask[:, 5] = False  # forbid one to exercise masking
    masks = {"candidate": mask}
    action, lp_sample, _, _ = pol.get_action_and_value(obs, masks=masks)
    _, lp_rescore, _, _ = pol.get_action_and_value(obs, action=action, masks=masks)
    assert torch.allclose(lp_sample, lp_rescore, atol=1e-6)
