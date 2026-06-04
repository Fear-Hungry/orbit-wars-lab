"""Orbit-phase helper used by the movement forecaster."""
from __future__ import annotations

from torch import Tensor


def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Convert the observation ``step`` counter into the engine orbit phase index.

    The Rust engine rotates orbiting planets once during each ``step()`` call and
    then increments the public ``state.step``. A reset observation has
    ``step == 0`` and the initial phase; the first post-step observation has
    ``step == 1`` and one applied rotation. Therefore the public step counter is
    already the orbit phase index used by the movement projector.
    """
    s = obs_step.float()
    return s.clamp(min=0.0)
