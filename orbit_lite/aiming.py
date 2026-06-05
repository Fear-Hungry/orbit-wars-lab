"""Orbit-phase helper used by the movement forecaster."""
from __future__ import annotations

from torch import Tensor


def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Convert the observation ``step`` counter into the engine orbit phase index.

    The official engine rotates orbiting planets from their initial coordinates
    using the pre-increment step inside ``step()``. A reset observation has
    ``step == 0`` and the first post-step observation has ``step == 1``; both
    expose the initial orbital phase. Public step ``N > 0`` therefore maps to
    phase ``N - 1``.
    """
    s = obs_step.float()
    return (s - 1.0).clamp(min=0.0)
