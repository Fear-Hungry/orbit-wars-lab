"""Engine-accurate fleet aiming for Family H.

The homebrewed ``geometry.orbital_intercept`` mis-modelled the engine: it used a
constant ship speed (real speed is size-dependent ``fleet_speed``), ignored the
launch/target surface-radius gap, and had NO viability check — so fleets aimed at
a chosen planet routinely missed it or hit a different planet, and the bot wasted
ships on unreachable targets (diagnosed: never expands -> -1.0 vs Producer).

This wraps the engine's exact ``intercept_angle`` (orbit_lite.intercept_aim,
which reproduces the first-contact verdict) behind a per-observation aimer:
build the PlanetMovement once, then aim any (source, target) pair with the
correct angle/eta and a hard ``viable`` flag. orbit_lite is pure Python, so this
is submission-legal (no Rust crate import, D10/D11 intact).
"""

from __future__ import annotations

from typing import Any

import torch
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import ensure_planet_movement

Obs = dict[str, Any]

_HORIZON = 24


class Aimer:
    """Per-observation engine aimer. Build once per turn, query many pairs."""

    def __init__(self, obs: Obs, horizon: int = _HORIZON) -> None:
        me = int(obs.get("player", 0))
        owners = [int(r[1]) for r in obs.get("planets", [])]
        player_count = max(2, (max(owners) + 1) if owners else 2)
        obs_tensors = single_obs_to_tensor(obs, player_id=me)
        cfg = MovementConfig(
            movement_horizon=int(horizon),
            drift_epsilon=1e-3,
            track_fleets=True,
            player_count=player_count,
            max_tracked_fleets=128,
        )
        self._mv = ensure_planet_movement(
            obs_tensors=obs_tensors, expected_cfg=cfg, cached_movement=None
        )
        self._id2slot = {int(p): i for i, p in enumerate(self._mv.planet_ids.tolist())}

    def aim(self, source_id: int, target_id: int, ships: float) -> tuple[float, float] | None:
        """Return ``(angle, eta)`` that actually hits ``target_id``, or None if the
        shot is not viable (unreachable / blocked) for ``ships`` ships."""

        ssl = self._id2slot.get(int(source_id))
        tsl = self._id2slot.get(int(target_id))
        if ssl is None or tsl is None:
            return None
        out = intercept_angle(
            self._mv,
            torch.tensor([ssl]),
            torch.tensor([tsl]),
            torch.tensor([float(max(1.0, ships))]),
        )
        if not bool(out["viable"][0].item()):
            return None
        return float(out["angle"][0].item()), float(out["eta"][0].item())
