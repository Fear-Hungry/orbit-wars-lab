"""Regression: the H7 value-net must see the REAL angular_velocity at inference.

Locked-down bug (2026-06-10): _value_net_plan_value built its synthetic
post-launch state as {"planets", "fleets", "step"} only, so encode_state fell
back to angular_velocity=0.0 (global feature 1). The generator draws
angular_velocity from [0.025, 0.05) — never 0 — and the training set
(collect_value_dataset) encoded real backend states, so EVERY inference call
fed the net an out-of-distribution global feature. The planner now snapshots
obs["angular_velocity"] in act() alongside the board and passes it through.
"""
from __future__ import annotations

import pytest
import torch


def _empty_entries():
    from orbit_lite.movement_step import LaunchEntries

    return LaunchEntries(
        source_slots=torch.zeros(0, dtype=torch.long),
        target_slots=torch.zeros(0, dtype=torch.long),
        ships=torch.zeros(0),
        angle=torch.zeros(0),
        eta=torch.zeros(0),
        valid=torch.zeros(0, dtype=torch.bool),
    )


def test_value_net_synthetic_state_carries_real_angular_velocity():
    from bots.pgs.planner import PGSConfig, make_runtime
    from orbit_lite.adapter import single_obs_to_tensor
    from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
    from python.orbit_wars_gym.observation import to_official_observation

    seed = 7
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=seed,
                               config=RustConfig(enable_comets=True))
    backend.reset(seed)
    state = backend.states()[0]
    obs = to_official_observation(state, 0)
    angular = float(state["angular_velocity"])
    assert angular > 0.0, "generator draws angular_velocity from [0.025, 0.05)"

    captured: list[torch.Tensor] = []

    class _SpyNet:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            captured.append(x.detach().clone())
            return torch.zeros(x.shape[0])

    runtime = make_runtime(PGSConfig(max_deviations=0))
    runtime._value_net = _SpyNet()
    # real glue path: act() must snapshot the board AND angular_velocity
    runtime.act(obs)

    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    runtime._value_net_plan_value(obs_tensors, _empty_entries(), [], 0)

    assert captured, "value net was never consulted"
    # encode_state global feature layout: [step, angular_velocity, ...]
    got = float(captured[-1][0, 1])
    assert got == pytest.approx(angular), (
        f"value net saw angular_velocity={got}, real obs has {angular} — "
        f"out-of-distribution input (training never saw 0.0)"
    )
