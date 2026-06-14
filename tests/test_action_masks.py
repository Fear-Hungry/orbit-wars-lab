from __future__ import annotations

import torch
from python.agents.policy import FlatActorCritic
from python.orbit_wars_gym.action_masks import (
    LAUNCH_N,
    MASK_DIM,
    SOURCE_N,
    TARGET_N,
    build_action_masks,
    split_masks,
)
from python.orbit_wars_gym.encoding import observation_dim

_STATE = {
    "planets": [
        {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 30, "production": 2},
        {"id": 1, "owner": 0, "x": 24.0, "y": 22.0, "radius": 2.0, "ships": 16, "production": 2},
        {"id": 2, "owner": -1, "x": 36.0, "y": 25.0, "radius": 2.0, "ships": 5, "production": 3},
        {"id": 3, "owner": 1, "x": 80.0, "y": 75.0, "radius": 2.0, "ships": 18, "production": 3},
    ],
    "fleets": [],
    "step": 7,
}


def test_build_masks_for_launchable_state() -> None:
    mask = build_action_masks(_STATE, 0, min_ships_to_launch=2)
    assert mask.shape == (MASK_DIM,)
    parts = split_masks(mask)
    # player 0 has 2 launchable planets, 4 planets total.
    assert parts["launch"].tolist() == [True, True]
    assert int(parts["source"].sum()) == 2  # exactly the 2 own launchable planets
    assert int(parts["target"].sum()) == 3  # planet_count - 1


def test_build_masks_blocks_launch_when_nothing_to_send() -> None:
    state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 1, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 75.0, "radius": 2.0, "ships": 18, "production": 3},
        ],
        "fleets": [],
        "step": 0,
    }
    parts = split_masks(build_action_masks(state, 0, min_ships_to_launch=2))
    assert parts["launch"].tolist() == [True, False]  # launch forbidden
    assert bool(parts["source"].all())  # fallback all-valid (ignored since launch==0)


def test_policy_never_samples_masked_entries_and_is_finite() -> None:
    torch.manual_seed(0)
    model = FlatActorCritic(observation_dim())
    batch = 256
    obs = torch.randn(batch, observation_dim())
    # launch always allowed; force a single valid source (idx 2) and target (idx 7).
    src = torch.zeros(batch, SOURCE_N, dtype=torch.bool)
    src[:, 2] = True
    tgt = torch.zeros(batch, TARGET_N, dtype=torch.bool)
    tgt[:, 7] = True
    masks = {
        "launch": torch.ones(batch, LAUNCH_N, dtype=torch.bool),
        "source": src,
        "target": tgt,
    }
    action, logprob, entropy, _ = model.get_action_and_value(obs, masks=masks)
    launched = action[:, 0] == 1
    assert bool((action[launched, 1] == 2).all())  # only the unmasked source
    assert bool((action[launched, 2] == 7).all())  # only the unmasked target
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()


def test_masked_launch_forces_pass() -> None:
    torch.manual_seed(1)
    model = FlatActorCritic(observation_dim())
    batch = 128
    obs = torch.randn(batch, observation_dim())
    masks = {
        "launch": torch.tensor([[True, False]]).expand(batch, LAUNCH_N).contiguous(),
        "source": torch.ones(batch, SOURCE_N, dtype=torch.bool),
        "target": torch.ones(batch, TARGET_N, dtype=torch.bool),
    }
    action, logprob, _, _ = model.get_action_and_value(obs, masks=masks)
    assert bool((action[:, 0] == 0).all())  # launch disabled -> always pass
    assert torch.isfinite(logprob).all()


def test_same_mask_reproduces_logprob_for_ppo_ratio() -> None:
    torch.manual_seed(2)
    model = FlatActorCritic(observation_dim())
    batch = 64
    obs = torch.randn(batch, observation_dim())
    mask_flat = torch.ones(batch, MASK_DIM, dtype=torch.bool)
    # Restrict source to the first 4 ranks and target to the first 6, keeping at
    # least one valid entry per head (an all-False head would be NaN by design).
    mask_flat[:, LAUNCH_N + 4 : LAUNCH_N + SOURCE_N] = False
    mask_flat[:, LAUNCH_N + SOURCE_N + 6 :] = False
    masks = split_masks(mask_flat)

    action, lp_sample, _, _ = model.get_action_and_value(obs, masks=masks)
    _, lp_replay, _, _ = model.get_action_and_value(obs, action, masks=masks)
    assert torch.allclose(lp_sample, lp_replay, atol=1e-6)
