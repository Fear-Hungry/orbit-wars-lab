from __future__ import annotations

import math
import random
from typing import Any

from orbit_wars_gym.action_decoder import SHIP_SPEED, greedy_moves
from orbit_wars_gym.entities import (
    planet_id,
    planet_owner,
    planet_production,
    planet_radius,
    planet_ships,
    planet_x,
    planet_y,
)

CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS = 10.0
MIN_SHIPS_TO_LAUNCH = 2


def greedy_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    return greedy_moves(state, player)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _fleet_speed(ships: int) -> float:
    scale = math.log(max(int(ships), 1)) / math.log(1000.0)
    speed = 1.0 + (SHIP_SPEED - 1.0) * scale**1.5
    return min(SHIP_SPEED, max(1.0, speed))


def _rotate_about_center(point: tuple[float, float], angle: float) -> tuple[float, float]:
    dx = point[0] - CENTER
    dy = point[1] - CENTER
    c = math.cos(angle)
    s = math.sin(angle)
    return (CENTER + dx * c - dy * s, CENTER + dx * s + dy * c)


def _is_rotating_planet(planet: Any) -> bool:
    return _distance((planet_x(planet), planet_y(planet)), (CENTER, CENTER)) + planet_radius(planet) < ROTATION_RADIUS_LIMIT


def _predict_target_xy(state: dict[str, Any], source_xy: tuple[float, float], target: Any, ships: int) -> tuple[float, float]:
    target_xy = (planet_x(target), planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    travel_steps = max(1, math.ceil(_distance(source_xy, target_xy) / _fleet_speed(ships)))
    return _rotate_about_center(target_xy, float(state.get("angular_velocity", 0.0)) * travel_steps)


def _point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    vx, vy = start
    wx, wy = end
    dx = wx - vx
    dy = wy - vy
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - vx) * dx + (point[1] - vy) * dy) / l2))
    proj = (vx + t * dx, vy + t * dy)
    return _distance(point, proj)


def _sun_safe_angle(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    base = _angle(source_xy, target_xy)
    if _point_to_segment_distance((CENTER, CENTER), source_xy, target_xy) >= SUN_RADIUS + 1.0:
        return base
    to_center = _angle(source_xy, (CENTER, CENTER))
    candidates = [to_center + math.pi / 2.0, to_center - math.pi / 2.0]
    return min(candidates, key=lambda a: abs(math.atan2(math.sin(a - base), math.cos(a - base))))


def _leader_owner(planets: list[Any], player: int) -> int | None:
    owners = sorted({planet_owner(planet) for planet in planets if planet_owner(planet) not in (-1, player)})
    if not owners:
        return None
    return max(
        owners,
        key=lambda owner: (
            sum(planet_production(planet) for planet in planets if planet_owner(planet) == owner),
            sum(planet_ships(planet) for planet in planets if planet_owner(planet) == owner),
        ),
    )


def _reserve_for_source(source: Any, own_count: int, enemies: list[Any], *, defensive: bool, ffa: bool) -> int:
    reserve = 10
    if own_count <= 2:
        reserve += 6 if defensive else 2
    if defensive:
        reserve += 4
    if ffa:
        reserve += 3
    if planet_production(source) >= 4:
        reserve += 2
    if enemies:
        source_xy = (planet_x(source), planet_y(source))
        nearest_enemy = min(_distance(source_xy, (planet_x(enemy), planet_y(enemy))) for enemy in enemies)
        if nearest_enemy < 20.0:
            reserve += 5
        elif nearest_enemy < 30.0:
            reserve += 2
    return reserve


def _target_value(
    state: dict[str, Any],
    player: int,
    source: Any,
    target: Any,
    ships: int,
    *,
    leader_owner: int | None,
    prefer_neutral: bool,
    pressure_leader: bool,
) -> tuple[float, tuple[float, float]]:
    source_xy = (planet_x(source), planet_y(source))
    target_xy = _predict_target_xy(state, source_xy, target, ships)
    dist = _distance(source_xy, target_xy)
    owner = planet_owner(target)
    production = float(planet_production(target))
    defenders = float(planet_ships(target))
    comet_ids = set(int(pid) for pid in state.get("comet_planet_ids", []))

    value = production * (16.0 if owner == -1 else 18.0)
    if owner == -1:
        value += 10.0 if prefer_neutral else 4.0
    elif owner == leader_owner:
        value += 10.0 if pressure_leader else 5.0
    else:
        value += 3.0
    if planet_id(target) in comet_ids:
        value += 4.0 if dist < 30.0 else -4.0

    required = defenders + (2.0 if owner == -1 else 5.0)
    if owner not in (-1, player):
        required += max(0.0, dist / max(_fleet_speed(ships), 1.0)) * production
    roi = value / max(required, 1.0)
    score = value + 18.0 * roi - 0.18 * dist - 0.26 * defenders
    if prefer_neutral and owner != -1:
        score -= 10.0
    return score, target_xy


def _ranked_targets(
    state: dict[str, Any],
    player: int,
    source: Any,
    targets: list[Any],
    ships: int,
    *,
    prefer_neutral: bool,
    pressure_leader: bool,
) -> list[tuple[float, Any, tuple[float, float]]]:
    leader = _leader_owner(state.get("planets", []), player)
    ranked = [
        (*_target_value(state, player, source, target, ships, leader_owner=leader, prefer_neutral=prefer_neutral, pressure_leader=pressure_leader), target)
        for target in targets
        if planet_id(target) != planet_id(source)
    ]
    return sorted(((score, target, xy) for score, xy, target in ranked), key=lambda item: item[0], reverse=True)


def defensive_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    targets = [p for p in planets if planet_owner(p) == -1]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    if not own or not targets:
        return []
    moves = []
    ffa = len({planet_owner(p) for p in enemies}) >= 2
    for src in sorted(own, key=lambda p: (planet_ships(p), planet_production(p)), reverse=True):
        reserve = _reserve_for_source(src, len(own), enemies, defensive=True, ffa=ffa)
        available = planet_ships(src) - reserve
        if available < MIN_SHIPS_TO_LAUNCH:
            continue
        ships = max(MIN_SHIPS_TO_LAUNCH, int(available * 0.55))
        ranked = _ranked_targets(state, player, src, targets, ships, prefer_neutral=True, pressure_leader=False)
        if not ranked or ranked[0][0] <= 0.0:
            continue
        _, _target, target_xy = ranked[0]
        source_xy = (planet_x(src), planet_y(src))
        moves.append([planet_id(src), float(_sun_safe_angle(source_xy, target_xy)), int(ships)])
        if len(moves) >= 4:
            break
    return moves


def rush_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    if not own or not enemies:
        return greedy_moves(state, player)
    leader = _leader_owner(planets, player)
    focus_pool = [p for p in enemies if planet_owner(p) == leader] or enemies
    enemy_home = max(focus_pool, key=lambda p: (planet_production(p), planet_ships(p)))
    moves = []
    for src in sorted(own, key=planet_ships, reverse=True)[:2]:
        reserve = _reserve_for_source(src, len(own), enemies, defensive=False, ffa=False)
        ships = max(0, planet_ships(src) - max(4, reserve - 5))
        if ships < MIN_SHIPS_TO_LAUNCH:
            continue
        source_xy = (planet_x(src), planet_y(src))
        target_xy = _predict_target_xy(state, source_xy, enemy_home, ships)
        angle = _sun_safe_angle(source_xy, target_xy)
        moves.append([planet_id(src), float(angle), int(ships)])
    return moves


def anti_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    neutrals = [p for p in planets if planet_owner(p) == -1]
    if not own:
        return []
    leader = _leader_owner(planets, player)
    enemy_owners = {planet_owner(p) for p in enemies}
    ffa = len(enemy_owners) >= 2
    targets = [p for p in enemies if planet_owner(p) == leader] if ffa and leader is not None else []
    if not targets:
        targets = neutrals if len(own) <= 3 and neutrals else enemies or neutrals
    if not targets:
        return []
    moves = []
    for src in sorted(own, key=lambda p: (planet_ships(p), planet_production(p)), reverse=True)[:3]:
        reserve = _reserve_for_source(src, len(own), enemies, defensive=False, ffa=ffa)
        available = planet_ships(src) - reserve
        if available < MIN_SHIPS_TO_LAUNCH:
            continue
        ships = max(MIN_SHIPS_TO_LAUNCH, int(available * (0.72 if ffa else 0.62)))
        ranked = _ranked_targets(
            state,
            player,
            src,
            targets,
            ships,
            prefer_neutral=not enemies or (len(own) <= 3 and bool(neutrals)),
            pressure_leader=True,
        )
        if not ranked or ranked[0][0] <= -4.0:
            continue
        _, _target, target_xy = ranked[0]
        source_xy = (planet_x(src), planet_y(src))
        moves.append([planet_id(src), float(_sun_safe_angle(source_xy, target_xy)), int(ships)])
    return moves


def weak_random_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    targets = [p for p in planets if planet_owner(p) != player]
    if not own or not targets:
        return []
    rng = random.Random(int(state.get("step", 0)) + 997 * player + len(planets))
    src = rng.choice(sorted(own, key=planet_ships, reverse=True)[: max(1, min(3, len(own)))])
    tgt = rng.choice(targets)
    ships = max(0, int((planet_ships(src) - 6) * rng.uniform(0.25, 0.55)))
    if ships <= 0:
        return []
    source_xy = (planet_x(src), planet_y(src))
    target_xy = _predict_target_xy(state, source_xy, tgt, ships)
    angle = _sun_safe_angle(source_xy, target_xy)
    return [[planet_id(src), float(angle), ships]]
