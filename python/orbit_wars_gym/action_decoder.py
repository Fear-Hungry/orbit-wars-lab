from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .entities import (
    planet_id,
    planet_owner,
    planet_production,
    planet_radius,
    planet_ships,
    planet_x,
    planet_y,
)

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0


@dataclass(frozen=True)
class DecoderConfig:
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    angle_offsets: tuple[float, ...] = (-0.261799, -0.130899, 0.0, 0.130899, 0.261799)
    max_moves_per_turn: int = 8
    min_ships_to_launch: int = 2
    reserve_home_ships: int = 8


DEFAULT_DECODER_CONFIG = DecoderConfig()


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _segment_distance_to_sun(start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = CENTER, CENTER
    vx, vy = start
    wx, wy = end
    dx = wx - vx
    dy = wy - vy
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - vx, py - vy)
    t = max(0.0, min(1.0, ((px - vx) * dx + (py - vy) * dy) / l2))
    proj = (vx + t * dx, vy + t * dy)
    return math.hypot(px - proj[0], py - proj[1])


def _sun_safe_angle(source: tuple[float, float], target: tuple[float, float], base_angle: float) -> float:
    if _segment_distance_to_sun(source, target) >= SUN_RADIUS + 1.0:
        return base_angle
    # Simple detour: bias perpendicular to line-to-center. More exact tangent
    # geometry should replace this during decoder tuning.
    to_center = _angle(source, (CENTER, CENTER))
    candidates = [to_center + math.pi / 2.0, to_center - math.pi / 2.0]
    return min(candidates, key=lambda a: abs(math.atan2(math.sin(a - base_angle), math.cos(a - base_angle))))


def _fleet_speed(ships: int, ship_speed: float = SHIP_SPEED) -> float:
    scale = math.log(max(ships, 1)) / math.log(1000.0)
    speed = 1.0 + (ship_speed - 1.0) * scale**1.5
    return min(ship_speed, max(1.0, speed))


def _rotate_about_center(point: tuple[float, float], angle: float) -> tuple[float, float]:
    dx = point[0] - CENTER
    dy = point[1] - CENTER
    c = math.cos(angle)
    s = math.sin(angle)
    return (CENTER + dx * c - dy * s, CENTER + dx * s + dy * c)


def _is_rotating_planet(planet: Any) -> bool:
    orbital_radius = math.hypot(planet_x(planet) - CENTER, planet_y(planet) - CENTER)
    return orbital_radius + planet_radius(planet) < ROTATION_RADIUS_LIMIT


def _predict_target_xy(
    state: dict[str, Any],
    source_xy: tuple[float, float],
    target: Any,
    ships: int,
) -> tuple[float, float]:
    target_xy = (planet_x(target), planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    distance = math.hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1])
    travel_steps = max(1, math.ceil(distance / _fleet_speed(ships)))
    return _rotate_about_center(target_xy, float(state.get("angular_velocity", 0.0)) * travel_steps)


def decode_discrete_action(
    state: dict[str, Any],
    player: int,
    action: np.ndarray | list[int],
    cfg: DecoderConfig = DEFAULT_DECODER_CONFIG,
) -> list[list[float]]:
    """Decode one compact discrete action into official move list.

    action layout (canonical, 5 fields):
        [launch, source_rank, target_rank, fraction_idx, offset_idx]

    ``launch == 0`` is an explicit pass (no moves) — see todo P1.5. The four
    trailing ranks refer to sorted candidate lists, not raw planet ids, keeping
    the neural action space fixed across variable maps.

    Temporary backward compatibility: a 4-field action
    ``[source_rank, target_rank, fraction_idx, offset_idx]`` is still accepted and
    treated as an implicit launch (legacy behaviour, also used by the inverse
    grid-search in ``action_inverse``). Remove once all call sites pass 5 fields.
    """

    a = np.asarray(action, dtype=np.int64).tolist()
    if len(a) >= 5:
        launch, source_rank, target_rank, fraction_idx, offset_idx = a[:5]
        if launch == 0:
            return []
    elif len(a) == 4:
        source_rank, target_rank, fraction_idx, offset_idx = a[:4]
    else:
        return []

    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player and planet_ships(p) >= cfg.min_ships_to_launch]
    if not own:
        return []

    # Sort own planets by ships descending, then production. This gives stable
    # ranks and tends to expose high-leverage sources first.
    own.sort(key=lambda p: (planet_ships(p), planet_production(p)), reverse=True)
    ranked_sources = own[source_rank % len(own) :] + own[: source_rank % len(own)]

    moves: list[list[float]] = []
    used_targets: set[int] = set()
    for src in ranked_sources:
        if len(moves) >= cfg.max_moves_per_turn:
            break
        candidates = [p for p in planets if planet_id(p) != planet_id(src)]
        if not candidates:
            continue

        sx, sy = planet_x(src), planet_y(src)
        max_launch_ships = max(1, int(planet_ships(src) * cfg.fractions[-1]))

        source_xy = (sx, sy)

        def target_score(p: Any, source_xy: tuple[float, float] = source_xy, max_ships: int = max_launch_ships) -> float:
            tx, ty = _predict_target_xy(state, source_xy, p, max_ships)
            dist = math.hypot(tx - source_xy[0], ty - source_xy[1])
            owner = planet_owner(p)
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            repeat_penalty = 3.0 if planet_id(p) in used_targets else 0.0
            return (
                float(planet_production(p)) * 10.0
                + enemy_bonus
                + neutral_bonus
                - repeat_penalty
                - 0.15 * dist
                - 0.12 * float(planet_ships(p))
            )

        candidates.sort(key=target_score, reverse=True)
        target = candidates[target_rank % len(candidates)]

        frac = cfg.fractions[fraction_idx % len(cfg.fractions)]
        ships = int(max(0, math.floor(float(planet_ships(src)) * frac)))
        if ships <= 0:
            continue
        if planet_ships(src) - ships < cfg.reserve_home_ships and len(own) <= 2:
            ships = max(0, planet_ships(src) - cfg.reserve_home_ships)
        if ships <= 0:
            continue

        target_xy = _predict_target_xy(state, source_xy, target, ships)
        base = _angle(source_xy, target_xy)
        base = _sun_safe_angle(source_xy, target_xy, base)
        offset = cfg.angle_offsets[offset_idx % len(cfg.angle_offsets)]
        angle = base + offset

        moves.append([planet_id(src), float(angle), int(ships)])
        used_targets.add(planet_id(target))

    return moves


PLANET_SLOTS = 96  # must match encoding.max_planets / policy.PLANET_N


def gridnet_planet_mask(
    state: dict[str, Any], player: int, cfg: DecoderConfig = DEFAULT_DECODER_CONFIG
) -> "np.ndarray":
    """Per-planet launch mask aligned to encode_state's slot order (planets[i] -> i).

    Slot ``i`` is active iff ``planets[i]`` is the player's and has enough ships to
    launch. Inactive slots are forced to no-op by the policy — this is the launch
    half of the full GridNet mask (target/frac stay decode-legal via modulo)."""
    planets = state.get("planets", [])
    mask = np.zeros(PLANET_SLOTS, dtype=bool)
    for i, p in enumerate(planets[:PLANET_SLOTS]):
        if planet_owner(p) == player and planet_ships(p) >= cfg.min_ships_to_launch:
            mask[i] = True
    return mask


def decode_gridnet_action(
    state: dict[str, Any],
    player: int,
    action: "np.ndarray | list",
    cfg: DecoderConfig = DEFAULT_DECODER_CONFIG,
) -> list[list[float]]:
    """Decode a per-planet GridNet action into official moves.

    ``action`` is ``(PLANET_SLOTS, 4)``: row ``i`` = ``[launch, target_rank,
    frac_idx, offset_idx]`` for ``planets[i]``. Each OWN planet with launch==1 and
    enough ships emits ONE move toward its OWN heuristically-ranked target — sources
    are INDEPENDENT (no shared rank), so multi-source plans are representable. Each
    target may be hit once per turn (used_targets) to avoid degenerate stacking."""
    a = np.asarray(action, dtype=np.int64)
    planets = state.get("planets", [])
    moves: list[list[float]] = []
    used_targets: set[int] = set()
    for i, src in enumerate(planets[:PLANET_SLOTS]):
        if i >= a.shape[0] or int(a[i, 0]) != 1:
            continue
        if planet_owner(src) != player or planet_ships(src) < cfg.min_ships_to_launch:
            continue
        candidates = [p for p in planets if planet_id(p) != planet_id(src)]
        if not candidates:
            continue
        sx, sy = planet_x(src), planet_y(src)
        max_launch_ships = max(1, int(planet_ships(src) * cfg.fractions[-1]))
        source_xy = (sx, sy)

        def target_score(p: Any, source_xy: tuple[float, float] = source_xy, max_ships: int = max_launch_ships) -> float:
            tx, ty = _predict_target_xy(state, source_xy, p, max_ships)
            dist = math.hypot(tx - source_xy[0], ty - source_xy[1])
            owner = planet_owner(p)
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            repeat_penalty = 3.0 if planet_id(p) in used_targets else 0.0
            return (
                float(planet_production(p)) * 10.0
                + enemy_bonus + neutral_bonus - repeat_penalty
                - 0.15 * dist - 0.12 * float(planet_ships(p))
            )

        candidates.sort(key=target_score, reverse=True)
        target = candidates[int(a[i, 1]) % len(candidates)]
        frac = cfg.fractions[int(a[i, 2]) % len(cfg.fractions)]
        ships = int(max(0, math.floor(float(planet_ships(src)) * frac)))
        if ships <= 0:
            continue
        target_xy = _predict_target_xy(state, source_xy, target, ships)
        base = _angle(source_xy, target_xy)
        base = _sun_safe_angle(source_xy, target_xy, base)
        angle = base + cfg.angle_offsets[int(a[i, 3]) % len(cfg.angle_offsets)]
        moves.append([planet_id(src), float(angle), int(ships)])
        used_targets.add(planet_id(target))
    return moves


def greedy_moves(
    state: dict[str, Any],
    player: int,
    cfg: DecoderConfig = DEFAULT_DECODER_CONFIG,
) -> list[list[float]]:
    """Safe baseline used for fallback and opponent pool."""
    moves: list[list[float]] = []
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player and planet_ships(p) > cfg.reserve_home_ships + cfg.min_ships_to_launch]
    targets = [p for p in planets if planet_owner(p) != player]
    for src in sorted(own, key=planet_ships, reverse=True):
        if len(moves) >= cfg.max_moves_per_turn:
            break
        sx, sy = planet_x(src), planet_y(src)
        ranked = sorted(
            targets,
            key=lambda p: (float(planet_ships(p)) + 1.0) / (float(planet_production(p)) + 1.0) + 0.05 * math.hypot(planet_x(p) - sx, planet_y(p) - sy),
        )
        if not ranked:
            continue
        tgt = ranked[0]
        ships = max(0, planet_ships(src) - cfg.reserve_home_ships)
        if ships <= 0:
            continue
        source_xy = (sx, sy)
        target_xy = _predict_target_xy(state, source_xy, tgt, ships)
        base = _angle(source_xy, target_xy)
        angle = _sun_safe_angle(source_xy, target_xy, base)
        moves.append([planet_id(src), float(angle), int(ships)])
    return moves
