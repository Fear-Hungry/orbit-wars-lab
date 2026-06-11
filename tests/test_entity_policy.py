from __future__ import annotations

import pytest
import torch
from python.agents.policy import EntityActorCritic
from python.orbit_wars_gym.action_masks import LAUNCH_N, SOURCE_N, TARGET_N
from python.orbit_wars_gym.encoding import observation_dim


def test_entity_forward_shapes() -> None:
    model = EntityActorCritic(observation_dim())
    obs = torch.randn(8, observation_dim())
    out = model.forward(obs)
    assert out["launch"].shape == (8, 2)
    assert out["source"].shape == (8, 16)
    assert out["target"].shape == (8, 32)
    assert out["frac"].shape == (8, 4)
    assert out["offset"].shape == (8, 5)
    assert out["value"].shape == (8,)


def test_entity_rejects_wrong_obs_dim() -> None:
    with pytest.raises(ValueError, match="flat obs_dim"):
        EntityActorCritic(123)


def test_entity_action_value_and_masking() -> None:
    torch.manual_seed(0)
    model = EntityActorCritic(observation_dim())
    batch = 64
    obs = torch.randn(batch, observation_dim())
    src = torch.zeros(batch, SOURCE_N, dtype=torch.bool)
    src[:, 1] = True
    tgt = torch.zeros(batch, TARGET_N, dtype=torch.bool)
    tgt[:, 3] = True
    masks = {
        "launch": torch.ones(batch, LAUNCH_N, dtype=torch.bool),
        "source": src,
        "target": tgt,
    }
    action, logprob, entropy, value = model.get_action_and_value(obs, masks=masks)
    assert action.shape == (batch, 5)
    launched = action[:, 0] == 1
    assert bool((action[launched, 1] == 1).all())
    assert bool((action[launched, 2] == 3).all())
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()

    # replaying the same action under the same mask reproduces the log-prob
    _, lp_replay, _, _ = model.get_action_and_value(obs, action, masks=masks)
    assert torch.allclose(logprob, lp_replay, atol=1e-6)


def test_entity_is_permutation_invariant_over_planets() -> None:
    # Shuffling planet slots must not change the pooled output (slot-order invariance).
    torch.manual_seed(1)
    model = EntityActorCritic(observation_dim())
    model.eval()
    from python.agents.policy import GLOBAL_F, PLANET_F, PLANET_N

    obs = torch.randn(1, observation_dim())
    # mark first 5 planet slots present, rest absent, so a permutation is meaningful
    planets = obs[:, GLOBAL_F : GLOBAL_F + PLANET_N * PLANET_F].reshape(1, PLANET_N, PLANET_F)
    planets[:, :, 0] = 0.0
    planets[:, :5, 0] = 1.0
    obs = obs.clone()
    obs[:, GLOBAL_F : GLOBAL_F + PLANET_N * PLANET_F] = planets.reshape(1, -1)

    perm = torch.randperm(PLANET_N)
    permuted = planets[:, perm, :]
    obs2 = obs.clone()
    obs2[:, GLOBAL_F : GLOBAL_F + PLANET_N * PLANET_F] = permuted.reshape(1, -1)

    with torch.no_grad():
        a = model.forward(obs)
        b = model.forward(obs2)
    for head in ("launch", "source", "target", "frac", "offset"):
        assert torch.allclose(a[head], b[head], atol=1e-5)
