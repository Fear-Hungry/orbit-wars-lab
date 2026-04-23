from __future__ import annotations

import math
import random
from typing import Any

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.entities import (
    fleet_owner,
    fleet_ships,
    fleet_x,
    fleet_y,
    planet_id,
    planet_owner,
    planet_production,
    planet_radius,
    planet_ships,
    planet_x,
    planet_y,
)

BOARD_CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0
MAX_FIELD_CONTROL_MOVES = 4
_OPENING_STYLE_CACHE: dict[str, dict[int, str | None]] = {
    "opening_gate_rush_meta": {},
    "opening_gate_anti_meta_meta": {},
}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _fleet_speed(ships: int, ship_speed: float = SHIP_SPEED) -> float:
    scale = math.log(max(ships, 1)) / math.log(1000.0)
    speed = 1.0 + (ship_speed - 1.0) * scale**1.5
    return min(ship_speed, max(1.0, speed))


def _rotate_about_center(point: tuple[float, float], angle: float) -> tuple[float, float]:
    dx = point[0] - BOARD_CENTER
    dy = point[1] - BOARD_CENTER
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        BOARD_CENTER + dx * c - dy * s,
        BOARD_CENTER + dx * s + dy * c,
    )


def _is_rotating_planet(planet: Any) -> bool:
    orbital_radius = math.hypot(planet_x(planet) - BOARD_CENTER, planet_y(planet) - BOARD_CENTER)
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
    distance = _distance(source_xy, target_xy)
    travel_steps = max(1, math.ceil(distance / _fleet_speed(ships)))
    angular_velocity = float(state.get("angular_velocity", 0.0))
    return _rotate_about_center(target_xy, angular_velocity * travel_steps)


def _point_to_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    l2 = vx * vx + vy * vy
    if l2 == 0.0:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / l2))
    projection = (start[0] + t * vx, start[1] + t * vy)
    return _distance(point, projection)


def _sun_safe_angle(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    base_angle = math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])
    if (
        _point_to_segment_distance((BOARD_CENTER, BOARD_CENTER), source_xy, target_xy)
        >= SUN_RADIUS + 1.0
    ):
        return base_angle
    to_center = math.atan2(BOARD_CENTER - source_xy[1], BOARD_CENTER - source_xy[0])
    candidates = [to_center + math.pi / 2.0, to_center - math.pi / 2.0]
    return min(
        candidates,
        key=lambda angle: abs(
            math.atan2(math.sin(angle - base_angle), math.cos(angle - base_angle))
        ),
    )


def _planet_pressure(planet: Any, neighbors: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    pressure = 0.0
    for other in neighbors:
        ox, oy = planet_x(other), planet_y(other)
        dist = max(4.0, _distance((px, py), (ox, oy)))
        mass = planet_ships(other) + 5.0 * planet_production(other)
        pressure += mass / (dist**1.15)
    return pressure


def _fleet_pressure(planet: Any, fleets: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    pressure = 0.0
    for fleet in fleets:
        fx, fy = fleet_x(fleet), fleet_y(fleet)
        dist = max(4.0, _distance((px, py), (fx, fy)))
        pressure += fleet_ships(fleet) / (dist**1.2)
    return pressure


def _frontline_bias(planet: Any, enemies: list[Any], own: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    enemy_dist = min(
        (_distance((px, py), (planet_x(enemy), planet_y(enemy))) for enemy in enemies), default=80.0
    )
    own_dist = min(
        (
            _distance((px, py), (planet_x(friend), planet_y(friend)))
            for friend in own
            if planet_id(friend) != planet_id(planet)
        ),
        default=80.0,
    )
    return (own_dist - enemy_dist) * 0.08


def _target_value(
    state: dict[str, Any],
    target: Any,
    player: int,
    own: list[Any],
    enemies: list[Any],
    friendly_fleets: list[Any],
    enemy_fleets: list[Any],
) -> float:
    owner = planet_owner(target)
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(
        target, friendly_fleets
    )
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    production_value = 8.0 * planet_production(target)
    ship_penalty = 1.15 * planet_ships(target)
    centrality = 3.0 * (
        1.0
        - min(
            1.0,
            _distance((planet_x(target), planet_y(target)), (BOARD_CENTER, BOARD_CENTER)) / 60.0,
        )
    )
    frontier = _frontline_bias(target, enemies, own)
    if owner == -1:
        denial = max(0.0, enemy_pressure - friendly_pressure) * 0.7
        return (
            production_value
            - ship_penalty
            + 0.35 * friendly_pressure
            - 0.2 * enemy_pressure
            + centrality
            + frontier
            + denial
        )
    if owner == player:
        threat = max(0.0, enemy_pressure - friendly_pressure - 0.35 * planet_ships(target))
        return 11.0 * threat + 4.5 * planet_production(target) + frontier
    vulnerability = max(0.0, friendly_pressure - enemy_pressure)
    return production_value + 6.0 + 0.8 * vulnerability - ship_penalty + centrality + frontier


def _source_reserve(source: Any, enemies: list[Any], enemy_fleets: list[Any]) -> int:
    local_threat = _planet_pressure(source, enemies) + 0.8 * _fleet_pressure(source, enemy_fleets)
    frontier_tax = max(0.0, _frontline_bias(source, enemies, [source]))
    reserve = 8 + int(0.45 * planet_production(source) + 0.35 * local_threat + frontier_tax)
    return max(8, min(24, reserve))


def _coalition_source_reserve(source: Any, enemies: list[Any], enemy_fleets: list[Any]) -> int:
    local_threat = _planet_pressure(source, enemies) + 0.9 * _fleet_pressure(source, enemy_fleets)
    frontier_tax = max(0.0, _frontline_bias(source, enemies, [source]))
    reserve = 12 + int(0.55 * planet_production(source) + 0.45 * local_threat + 1.25 * frontier_tax)
    return max(12, min(30, reserve))


def _leader_owner(enemies: list[Any]) -> int | None:
    owner_strength: dict[int, float] = {}
    for enemy in enemies:
        owner = planet_owner(enemy)
        owner_strength[owner] = (
            owner_strength.get(owner, 0.0) + planet_ships(enemy) + 6.0 * planet_production(enemy)
        )
    if not owner_strength:
        return None
    return max(owner_strength, key=owner_strength.get)


def _nearest_neutral(source: Any, neutrals: list[Any]) -> Any | None:
    if not neutrals:
        return None
    source_xy = (planet_x(source), planet_y(source))
    return min(
        neutrals,
        key=lambda planet: _distance(source_xy, (planet_x(planet), planet_y(planet))),
    )


def _has_backup_low_cost_neutral(
    source: Any, neutrals: list[Any], *, exclude_id: int | None = None
) -> bool:
    source_xy = (planet_x(source), planet_y(source))
    return any(
        planet_id(planet) != exclude_id
        and _distance(source_xy, (planet_x(planet), planet_y(planet))) <= 26.0
        and planet_ships(planet) <= 10
        and planet_production(planet) <= 2
        for planet in neutrals
    )


def _should_localized_4p_rush(own: list[Any], enemies: list[Any]) -> bool:
    if len(own) > 1 or not enemies:
        return False
    home = own[0]
    nearest_enemy = min(
        enemies,
        key=lambda planet: _distance(
            (planet_x(home), planet_y(home)), (planet_x(planet), planet_y(planet))
        ),
    )
    dx = abs(planet_x(nearest_enemy) - planet_x(home))
    dy = abs(planet_y(nearest_enemy) - planet_y(home))
    return (
        _distance(
            (planet_x(home), planet_y(home)), (planet_x(nearest_enemy), planet_y(nearest_enemy))
        )
        <= 20.0
        and min(dx, dy) <= 5.0
    )


def _should_localized_2p_anti_rush(
    own: list[Any],
    neutrals: list[Any],
    *,
    angular_velocity: float,
) -> bool:
    if not own or not neutrals or angular_velocity < 0.045:
        return False
    nearest_neutral = _nearest_neutral(own[0], neutrals)
    if nearest_neutral is None:
        return False
    distance = _distance(
        (planet_x(own[0]), planet_y(own[0])),
        (planet_x(nearest_neutral), planet_y(nearest_neutral)),
    )
    return (
        planet_ships(nearest_neutral) <= 14
        and planet_production(nearest_neutral) >= 4
        and distance <= 20.0
    )


def _opening_style_signature_2p(
    own: list[Any],
    neutrals: list[Any],
    *,
    angular_velocity: float,
    player: int = 0,
) -> str | None:
    if not own or not neutrals:
        return None
    home = own[0]
    home_xy = (planet_x(home), planet_y(home))
    nearest_neutral = _nearest_neutral(home, neutrals)
    if nearest_neutral is None:
        return None
    nearest_distance = _distance(home_xy, (planet_x(nearest_neutral), planet_y(nearest_neutral)))

    if (
        angular_velocity <= 0.039
        and nearest_distance <= 13.5
        and planet_ships(nearest_neutral) >= 40
        and planet_production(nearest_neutral) >= 3
    ):
        return "rush_then_greedy"

    if (
        angular_velocity >= 0.043
        and nearest_distance <= 16.0
        and planet_ships(nearest_neutral) >= 40
        and planet_production(nearest_neutral) >= 3
    ):
        return "greedy"

    if (
        angular_velocity >= 0.041
        and nearest_distance >= 15.0
        and planet_ships(nearest_neutral) <= 10
        and planet_production(nearest_neutral) >= 3
    ):
        return "field_control"

    if (
        angular_velocity < 0.037
        and nearest_distance <= 11.0
        and planet_ships(nearest_neutral) >= 30
        and planet_production(nearest_neutral) <= 1
    ):
        return "greedy"

    if (
        0.037 <= angular_velocity <= 0.0405
        and nearest_distance <= 13.5
        and 18 <= planet_ships(nearest_neutral) <= 22
        and planet_production(nearest_neutral) >= 5
    ):
        return "anti_meta"

    if (
        angular_velocity >= 0.04
        and nearest_distance <= 18.0
        and planet_ships(nearest_neutral) >= 20
        and planet_production(nearest_neutral) <= 1
    ):
        for neutral in neutrals:
            if planet_id(neutral) == planet_id(nearest_neutral):
                continue
            neutral_distance = _distance(home_xy, (planet_x(neutral), planet_y(neutral)))
            if (
                neutral_distance <= 22.0
                and planet_production(neutral) >= 4
                and planet_ships(neutral) <= 18
            ):
                return "field_control"
    return None


def _response_opening_style_signature_2p(
    own: list[Any],
    neutrals: list[Any],
    *,
    angular_velocity: float,
    player: int = 0,
) -> str | None:
    if not own or not neutrals:
        return None
    home = own[0]
    home_xy = (planet_x(home), planet_y(home))
    nearest_neutral = _nearest_neutral(home, neutrals)
    if nearest_neutral is None:
        return None
    nearest_distance = _distance(home_xy, (planet_x(nearest_neutral), planet_y(nearest_neutral)))
    ships = planet_ships(nearest_neutral)
    production = planet_production(nearest_neutral)

    if (
        player == 1
        and angular_velocity >= 0.039
        and ships >= 25
        and production <= 1
        and nearest_distance <= 14.5
    ):
        return "rush_then_defensive_short"
    if (
        player == 1
        and angular_velocity >= 0.041
        and ships <= 8
        and production <= 1
        and nearest_distance >= 16.0
    ):
        return "greedy"
    if (
        player == 1
        and 0.0395 <= angular_velocity <= 0.0405
        and ships <= 12
        and production >= 5
        and nearest_distance <= 13.0
    ):
        return "defensive"
    if (
        player == 1
        and 0.041 <= angular_velocity <= 0.043
        and ships >= 26
        and production >= 4
        and nearest_distance <= 15.0
    ):
        return "defensive_then_field_control_short"
    if angular_velocity >= 0.045 and ships >= 30 and production <= 1 and nearest_distance <= 14.5:
        return "field_control"
    if angular_velocity < 0.033 and ships <= 15 and production == 2 and nearest_distance <= 13.0:
        return "greedy"
    if (
        0.038 <= angular_velocity <= 0.039
        and ships <= 16
        and production == 2
        and nearest_distance <= 13.5
    ):
        return "rush_then_greedy_short"
    if (
        0.038 <= angular_velocity <= 0.039
        and ships >= 20
        and production >= 4
        and nearest_distance <= 13.5
    ):
        return "greedy"
    if angular_velocity >= 0.042 and ships >= 20 and production >= 4 and nearest_distance <= 15.0:
        return "defensive"
    return _opening_style_signature_2p(own, neutrals, angular_velocity=angular_velocity)


def _rush_meta_opening_style_signature_2p(
    own: list[Any],
    neutrals: list[Any],
    *,
    angular_velocity: float,
    player: int = 0,
) -> str | None:
    if not own or not neutrals:
        return None
    home = own[0]
    home_xy = (planet_x(home), planet_y(home))
    nearest_neutral = _nearest_neutral(home, neutrals)
    if nearest_neutral is None:
        return None
    nearest_distance = _distance(home_xy, (planet_x(nearest_neutral), planet_y(nearest_neutral)))
    ships = planet_ships(nearest_neutral)
    production = planet_production(nearest_neutral)

    if (
        player == 1
        and angular_velocity < 0.033
        and nearest_distance <= 13.5
        and ships >= 40
        and production >= 3
    ):
        return "field_control"
    if (
        0.042 <= angular_velocity <= 0.044
        and nearest_distance <= 13.5
        and 10 <= ships <= 14
        and production >= 4
    ):
        return "field_control"
    if (
        player == 0
        and 0.039 <= angular_velocity <= 0.041
        and ships >= 25
        and production <= 1
        and nearest_distance <= 11.0
    ):
        return "rush_then_defensive_short"
    if (
        player == 0
        and angular_velocity >= 0.041
        and ships <= 8
        and production <= 1
        and nearest_distance >= 16.0
    ):
        if _has_backup_low_cost_neutral(home, neutrals, exclude_id=planet_id(nearest_neutral)):
            return "rush_then_greedy_one"
        return "rush_then_field_control_one"
    if (
        player == 0
        and 0.0395 <= angular_velocity <= 0.0405
        and ships <= 12
        and production >= 5
        and nearest_distance <= 13.0
    ):
        return "greedy"
    if (
        player == 0
        and 0.041 <= angular_velocity <= 0.043
        and ships >= 26
        and production >= 4
        and nearest_distance <= 15.0
    ):
        return "field_control"
    return _opening_style_signature_2p(
        own, neutrals, angular_velocity=angular_velocity, player=player
    )


def _locked_opening_style(
    cache_key: str,
    state: dict[str, Any],
    player: int,
    own: list[Any],
    enemies: list[Any],
    neutrals: list[Any],
    *,
    signature_fn=_opening_style_signature_2p,
) -> str | None:
    cache = _OPENING_STYLE_CACHE[cache_key]
    enemy_owner_count = len({planet_owner(enemy) for enemy in enemies})
    if state.get("step", 0) == 0:
        cache[player] = (
            signature_fn(
                own,
                neutrals,
                angular_velocity=float(state.get("angular_velocity", 0.0)),
                player=player,
            )
            if enemy_owner_count == 1
            else None
        )
    if enemy_owner_count != 1:
        cache.pop(player, None)
        return None
    return cache.get(player)


def _required_commitment(
    state: dict[str, Any],
    source: Any,
    target: Any,
    player: int,
    own: list[Any],
    enemies: list[Any],
    friendly_fleets: list[Any],
    enemy_fleets: list[Any],
    committed: int,
) -> int:
    source_xy = (planet_x(source), planet_y(source))
    guess_ships = max(6, int(max(planet_ships(source) - 8, 0) * 0.55))
    target_xy = _predict_target_xy(state, source_xy, target, guess_ships)
    travel_steps = max(
        1, math.ceil(_distance(source_xy, target_xy) / _fleet_speed(max(guess_ships, 2)))
    )
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(
        target, friendly_fleets
    )
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    owner = planet_owner(target)
    if owner == player:
        vulnerability = max(0.0, enemy_pressure - friendly_pressure - 0.3 * planet_ships(target))
        return max(0, int(math.ceil(vulnerability + 2.0)) - committed)
    growth = max(0.0, float(planet_production(target)) * travel_steps)
    if owner == -1:
        need = (
            planet_ships(target)
            + 1
            + 0.2 * growth
            + 0.15 * max(0.0, enemy_pressure - friendly_pressure)
        )
    else:
        need = (
            planet_ships(target)
            + 1
            + 0.55 * growth
            + 0.4 * max(0.0, enemy_pressure - friendly_pressure)
        )
    return max(0, int(math.ceil(need)) - committed)


def _build_move(state: dict[str, Any], source: Any, target: Any, ships: int) -> list[float]:
    source_xy = (planet_x(source), planet_y(source))
    target_xy = _predict_target_xy(state, source_xy, target, ships)
    angle = _sun_safe_angle(source_xy, target_xy)
    return [planet_id(source), float(angle), int(ships)]


def greedy_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    return greedy_moves(state, player)


def defensive_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    targets = [p for p in planets if planet_owner(p) == -1]
    if not own or not targets:
        return []
    moves = []
    for src in sorted(own, key=planet_ships, reverse=True):
        reserve = 18 if len(own) <= 2 else 10
        ships = planet_ships(src) - reserve
        if ships <= 0:
            continue
        sx, sy = planet_x(src), planet_y(src)
        tgt = min(
            targets,
            key=lambda p: (
                float(planet_ships(p)) + 0.08 * math.hypot(planet_x(p) - sx, planet_y(p) - sy)
            ),
        )
        angle = math.atan2(planet_y(tgt) - sy, planet_x(tgt) - sx)
        launched = int(ships * 0.5)
        if launched < 2:
            continue
        moves.append([planet_id(src), float(angle), launched])
        if len(moves) >= 4:
            break
    return moves


def rush_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    if not own or not enemies:
        return greedy_moves(state, player)
    enemy_home = max(enemies, key=planet_ships)
    moves = []
    for src in sorted(own, key=planet_ships, reverse=True)[:2]:
        ships = max(0, planet_ships(src) - 5)
        if ships <= 0:
            continue
        angle = math.atan2(
            planet_y(enemy_home) - planet_y(src), planet_x(enemy_home) - planet_x(src)
        )
        moves.append([planet_id(src), float(angle), int(ships)])
    return moves


def anti_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    neutrals = [p for p in planets if planet_owner(p) == -1]
    if not own:
        return []
    if enemies:
        targets = sorted(
            enemies,
            key=lambda p: (
                -planet_ships(p),
                -sum(
                    1
                    for n in neutrals
                    if math.hypot(planet_x(n) - planet_x(p), planet_y(n) - planet_y(p)) < 20.0
                ),
            ),
        )
        focus = targets[0]
    elif neutrals:
        focus = max(neutrals, key=lambda p: (planet_production(p), -planet_ships(p)))
    else:
        return greedy_moves(state, player)

    moves = []
    for src in sorted(own, key=planet_ships, reverse=True)[:3]:
        ships = max(0, planet_ships(src) - 7)
        if ships <= 0:
            continue
        angle = math.atan2(planet_y(focus) - planet_y(src), planet_x(focus) - planet_x(src))
        moves.append([planet_id(src), float(angle), int(max(1, ships * 0.6))])
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
    angle = math.atan2(planet_y(tgt) - planet_y(src), planet_x(tgt) - planet_x(src))
    return [[planet_id(src), float(angle), ships]]


def field_control_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    fleets = state.get("fleets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []

    friendly_fleets = [fleet for fleet in fleets if fleet_owner(fleet) == player]
    enemy_fleets = [fleet for fleet in fleets if fleet_owner(fleet) not in (-1, player)]
    own_ships = sum(planet_ships(planet) for planet in own)
    enemy_ships = sum(planet_ships(planet) for planet in enemies)
    own_prod = sum(planet_production(planet) for planet in own)
    enemy_prod = sum(planet_production(planet) for planet in enemies)

    threatened = sorted(
        (planet for planet in own if planet_ships(planet) > 0),
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )
    neutral_targets = sorted(
        neutrals,
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )
    enemy_targets = sorted(
        enemies,
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )

    top_threat = (
        _target_value(state, threatened[0], player, own, enemies, friendly_fleets, enemy_fleets)
        if threatened
        else 0.0
    )
    want_expand = bool(neutral_targets) and (
        own_prod <= enemy_prod + 1
        or own_ships <= enemy_ships + 10
        or len(own) < max(3, len(enemies))
    )
    if top_threat > 8.0:
        target_priority = threatened + neutral_targets[:2] + enemy_targets[:2]
    elif want_expand:
        target_priority = neutral_targets + threatened[:2] + enemy_targets[:2]
    else:
        target_priority = enemy_targets + neutral_targets[:2] + threatened[:2]

    if not target_priority:
        return greedy_moves(state, player)

    moves: list[list[float]] = []
    committed: dict[int, int] = {}
    sources = sorted(
        own,
        key=lambda planet: (
            planet_ships(planet) - _source_reserve(planet, enemies, enemy_fleets),
            planet_production(planet),
        ),
        reverse=True,
    )

    for source in sources:
        if len(moves) >= MAX_FIELD_CONTROL_MOVES:
            break
        reserve = _source_reserve(source, enemies, enemy_fleets)
        surplus = planet_ships(source) - reserve
        if surplus < 2:
            continue

        best_target = None
        best_score = -1e9
        best_need = 0
        for target in target_priority:
            if planet_id(target) == planet_id(source):
                continue
            need = _required_commitment(
                state,
                source,
                target,
                player,
                own,
                enemies,
                friendly_fleets,
                enemy_fleets,
                committed.get(planet_id(target), 0),
            )
            strategic = _target_value(
                state, target, player, own, enemies, friendly_fleets, enemy_fleets
            )
            source_xy = (planet_x(source), planet_y(source))
            target_xy = _predict_target_xy(
                state, source_xy, target, max(2, min(surplus, max(need, 6)))
            )
            dist_penalty = 0.12 * _distance(source_xy, target_xy)
            score = strategic - dist_penalty - 0.25 * max(0, need - surplus)
            if score > best_score:
                best_score = score
                best_target = target
                best_need = need

        if best_target is None:
            continue

        owner = planet_owner(best_target)
        if owner == player:
            ships = min(surplus, max(2, best_need))
        elif owner == -1:
            ships = min(surplus, max(2, int(math.ceil(max(best_need, surplus * 0.45)))))
        else:
            ships = min(surplus, max(2, int(math.ceil(max(best_need * 1.1, surplus * 0.6)))))
        if ships <= 0:
            continue

        moves.append(_build_move(state, source, best_target, ships))
        committed[planet_id(best_target)] = committed.get(planet_id(best_target), 0) + ships

    return moves or greedy_moves(state, player)


def coalition_field_control_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    fleets = state.get("fleets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []
    if len({planet_owner(enemy) for enemy in enemies}) < 2:
        return field_control_agent(state, player)

    friendly_fleets = [fleet for fleet in fleets if fleet_owner(fleet) == player]
    enemy_fleets = [fleet for fleet in fleets if fleet_owner(fleet) not in (-1, player)]
    own_ships = sum(planet_ships(planet) for planet in own)
    enemy_ships = sum(planet_ships(planet) for planet in enemies)
    own_prod = sum(planet_production(planet) for planet in own)
    enemy_prod = sum(planet_production(planet) for planet in enemies)
    leader_owner = _leader_owner(enemies)

    threatened = sorted(
        (planet for planet in own if planet_ships(planet) > 0),
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )
    neutral_targets = sorted(
        neutrals,
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )
    leader_targets = sorted(
        (planet for planet in enemies if planet_owner(planet) == leader_owner),
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )
    other_enemy_targets = sorted(
        (planet for planet in enemies if planet_owner(planet) != leader_owner),
        key=lambda planet: _target_value(
            state, planet, player, own, enemies, friendly_fleets, enemy_fleets
        ),
        reverse=True,
    )

    top_threat = (
        _target_value(state, threatened[0], player, own, enemies, friendly_fleets, enemy_fleets)
        if threatened
        else 0.0
    )
    want_expand = bool(neutral_targets) and (
        own_prod <= enemy_prod or own_ships <= enemy_ships + 12 or len(own) < 4
    )
    if top_threat > 8.0:
        target_priority = (
            threatened + leader_targets[:2] + neutral_targets[:2] + other_enemy_targets[:1]
        )
    elif want_expand:
        target_priority = (
            neutral_targets + threatened[:2] + leader_targets[:2] + other_enemy_targets[:1]
        )
    else:
        target_priority = (
            leader_targets + threatened[:2] + neutral_targets[:2] + other_enemy_targets[:1]
        )

    if not target_priority:
        return field_control_agent(state, player)

    moves: list[list[float]] = []
    committed: dict[int, int] = {}
    sources = sorted(
        own,
        key=lambda planet: (
            planet_ships(planet) - _coalition_source_reserve(planet, enemies, enemy_fleets),
            planet_production(planet),
        ),
        reverse=True,
    )

    for source in sources:
        if len(moves) >= 3:
            break
        reserve = _coalition_source_reserve(source, enemies, enemy_fleets)
        surplus = planet_ships(source) - reserve
        if surplus < 2:
            continue

        best_target = None
        best_score = -1e9
        best_need = 0
        for target in target_priority:
            if planet_id(target) == planet_id(source):
                continue
            need = _required_commitment(
                state,
                source,
                target,
                player,
                own,
                enemies,
                friendly_fleets,
                enemy_fleets,
                committed.get(planet_id(target), 0),
            )
            strategic = _target_value(
                state, target, player, own, enemies, friendly_fleets, enemy_fleets
            )
            if planet_owner(target) == leader_owner:
                strategic += 4.0
            elif planet_owner(target) not in (-1, player):
                strategic -= 4.0
            source_xy = (planet_x(source), planet_y(source))
            target_xy = _predict_target_xy(
                state, source_xy, target, max(2, min(surplus, max(need, 6)))
            )
            dist_penalty = 0.15 * _distance(source_xy, target_xy)
            score = strategic - dist_penalty - 0.35 * max(0, need - surplus)
            if score > best_score:
                best_score = score
                best_target = target
                best_need = need

        if best_target is None:
            continue

        owner = planet_owner(best_target)
        if owner == player:
            ships = min(surplus, max(2, best_need))
        elif owner == -1:
            ships = min(surplus, max(2, int(math.ceil(max(best_need, surplus * 0.38)))))
        elif owner == leader_owner:
            ships = min(surplus, max(2, int(math.ceil(max(best_need * 1.05, surplus * 0.48)))))
        else:
            ships = min(surplus, max(2, int(math.ceil(max(best_need, surplus * 0.32)))))
        if ships <= 0:
            continue

        moves.append(_build_move(state, source, best_target, ships))
        committed[planet_id(best_target)] = committed.get(planet_id(best_target), 0) + ships

    return moves or field_control_agent(state, player)


def opening_gate_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []

    enemy_owner_count = len({planet_owner(enemy) for enemy in enemies})
    if enemy_owner_count >= 2:
        return coalition_field_control_agent(state, player)

    nearest_neutral = _nearest_neutral(own[0], neutrals)
    angular_velocity = float(state.get("angular_velocity", 0.0))
    if (
        nearest_neutral is not None
        and planet_ships(nearest_neutral) >= 12
        and planet_production(nearest_neutral) <= 1
        and angular_velocity > 0.037
    ):
        return defensive_agent(state, player)
    return coalition_field_control_agent(state, player)


def opening_gate_rush_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []

    enemy_owner_count = len({planet_owner(enemy) for enemy in enemies})
    if enemy_owner_count >= 2:
        if _should_localized_4p_rush(own, enemies):
            return rush_agent(state, player)
        return coalition_field_control_agent(state, player)

    opening_style = _locked_opening_style(
        "opening_gate_rush_meta",
        state,
        player,
        own,
        enemies,
        neutrals,
        signature_fn=_rush_meta_opening_style_signature_2p,
    )
    if opening_style == "field_control":
        return field_control_agent(state, player)
    if opening_style == "anti_meta":
        return anti_meta_agent(state, player)
    if opening_style == "greedy":
        return greedy_agent(state, player)
    if opening_style == "rush_then_greedy":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 5
            else greedy_agent(state, player)
        )
    if opening_style == "rush_then_greedy_one":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 1
            else greedy_agent(state, player)
        )
    if opening_style == "rush_then_field_control_one":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 1
            else field_control_agent(state, player)
        )
    if opening_style == "rush_then_defensive_short":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 1
            else defensive_agent(state, player)
        )

    nearest_neutral = _nearest_neutral(own[0], neutrals)
    angular_velocity = float(state.get("angular_velocity", 0.0))
    if (
        nearest_neutral is not None
        and planet_ships(nearest_neutral) >= 12
        and planet_production(nearest_neutral) <= 1
        and angular_velocity > 0.037
    ):
        return defensive_agent(state, player)
    if _should_localized_2p_anti_rush(own, neutrals, angular_velocity=angular_velocity):
        return anti_meta_agent(state, player)
    return coalition_field_control_agent(state, player)


def opening_gate_anti_meta_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []

    enemy_owner_count = len({planet_owner(enemy) for enemy in enemies})
    if enemy_owner_count >= 2:
        if _should_localized_4p_rush(own, enemies):
            return anti_meta_agent(state, player)
        return coalition_field_control_agent(state, player)

    opening_style = _locked_opening_style(
        "opening_gate_anti_meta_meta",
        state,
        player,
        own,
        enemies,
        neutrals,
        signature_fn=_response_opening_style_signature_2p,
    )
    if opening_style == "field_control":
        return field_control_agent(state, player)
    if opening_style == "anti_meta":
        return anti_meta_agent(state, player)
    if opening_style == "greedy":
        return greedy_agent(state, player)
    if opening_style == "defensive":
        return defensive_agent(state, player)
    if opening_style == "rush_then_greedy":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 5
            else greedy_agent(state, player)
        )
    if opening_style == "rush_then_greedy_short":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 3
            else greedy_agent(state, player)
        )
    if opening_style == "rush_then_defensive_short":
        return (
            rush_agent(state, player)
            if int(state.get("step", 0)) < 1
            else defensive_agent(state, player)
        )
    if opening_style == "defensive_then_field_control_short":
        return (
            defensive_agent(state, player)
            if int(state.get("step", 0)) < 1
            else field_control_agent(state, player)
        )

    nearest_neutral = _nearest_neutral(own[0], neutrals)
    angular_velocity = float(state.get("angular_velocity", 0.0))
    if (
        nearest_neutral is not None
        and planet_ships(nearest_neutral) >= 12
        and planet_production(nearest_neutral) <= 1
        and angular_velocity > 0.037
    ):
        return defensive_agent(state, player)
    if _should_localized_2p_anti_rush(own, neutrals, angular_velocity=angular_velocity):
        return anti_meta_agent(state, player)
    return coalition_field_control_agent(state, player)
