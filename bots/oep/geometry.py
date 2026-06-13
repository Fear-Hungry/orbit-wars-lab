"""Shared orbital geometry for Family H.

Targeting MUST be a common dependency, not a per-family detail: if each family
rolls its own geometry, ETA / projected production / overkill / hammer-sync all
drift silently and the oracle becomes unauditable. Every family computes aim and
eta through :func:`orbital_intercept` here.

The engine moves planets on circular orbits about ``CENTER``:
``pos(t) = CENTER + orb_r * (cos, sin)(a0 + angvel * t)`` (orbit_lite.movement).
A fleet leaves the source's *current* position and flies straight at
``SHIP_SPEED``; the intercept is the fixed point of
``t = ||pos(t) - source|| / SHIP_SPEED``.
"""

from __future__ import annotations

import math

CENTER = 50.0  # orbit_lite.constants.CENTER (sun / board center)
SHIP_SPEED = 6.0


def orbital_intercept(
    sx: float, sy: float, tx: float, ty: float, angvel: float, *, iters: int = 4
) -> tuple[float, float]:
    """Return ``(angle, eta)`` to hit an orbiting target from ``(sx, sy)``.

    Static targets (``orb_r ~= 0`` or no board rotation) reduce to straight aim.
    eta is clamped to ``>= 1`` (a launch takes at least one step).
    """

    orb_r = math.hypot(tx - CENTER, ty - CENTER)
    if orb_r < 1e-6 or abs(angvel) < 1e-9:
        eta = math.hypot(tx - sx, ty - sy) / SHIP_SPEED
        return math.atan2(ty - sy, tx - sx), max(1.0, eta)
    a0 = math.atan2(ty - CENTER, tx - CENTER)
    t = math.hypot(tx - sx, ty - sy) / SHIP_SPEED
    px, py = tx, ty
    for _ in range(iters):
        ang = a0 + angvel * t
        px = CENTER + orb_r * math.cos(ang)
        py = CENTER + orb_r * math.sin(ang)
        t = math.hypot(px - sx, py - sy) / SHIP_SPEED
    return math.atan2(py - sy, px - sx), max(1.0, t)
