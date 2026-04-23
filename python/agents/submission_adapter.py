from __future__ import annotations

import math
from typing import Any

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
MAX_MOVES_PER_TURN = 4


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
    obs: dict[str, Any],
    source_xy: tuple[float, float],
    target: Any,
    ships: int,
) -> tuple[float, float]:
    target_xy = (planet_x(target), planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    distance = _distance(source_xy, target_xy)
    travel_steps = max(1, math.ceil(distance / _fleet_speed(ships)))
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    return _rotate_about_center(target_xy, angular_velocity * travel_steps)


def _point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
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
    if _point_to_segment_distance((BOARD_CENTER, BOARD_CENTER), source_xy, target_xy) >= SUN_RADIUS + 1.0:
        return base_angle
    to_center = math.atan2(BOARD_CENTER - source_xy[1], BOARD_CENTER - source_xy[0])
    candidates = [to_center + math.pi / 2.0, to_center - math.pi / 2.0]
    return min(candidates, key=lambda angle: abs(math.atan2(math.sin(angle - base_angle), math.cos(angle - base_angle))))


def _planet_pressure(planet: Any, neighbors: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    pressure = 0.0
    for other in neighbors:
        dist = max(4.0, _distance((px, py), (planet_x(other), planet_y(other))))
        pressure += (planet_ships(other) + 5.0 * planet_production(other)) / (dist**1.15)
    return pressure


def _fleet_pressure(planet: Any, fleets: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    pressure = 0.0
    for fleet in fleets:
        dist = max(4.0, _distance((px, py), (fleet_x(fleet), fleet_y(fleet))))
        pressure += fleet_ships(fleet) / (dist**1.2)
    return pressure


def _frontline_bias(planet: Any, enemies: list[Any], own: list[Any]) -> float:
    px, py = planet_x(planet), planet_y(planet)
    enemy_dist = min((_distance((px, py), (planet_x(enemy), planet_y(enemy))) for enemy in enemies), default=80.0)
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
    target: Any,
    player: int,
    own: list[Any],
    enemies: list[Any],
    friendly_fleets: list[Any],
    enemy_fleets: list[Any],
) -> float:
    owner = planet_owner(target)
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(target, friendly_fleets)
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    production_value = 8.0 * planet_production(target)
    ship_penalty = 1.15 * planet_ships(target)
    centrality = 3.0 * (1.0 - min(1.0, _distance((planet_x(target), planet_y(target)), (BOARD_CENTER, BOARD_CENTER)) / 60.0))
    frontier = _frontline_bias(target, enemies, own)
    if owner == -1:
        denial = max(0.0, enemy_pressure - friendly_pressure) * 0.7
        return production_value - ship_penalty + 0.35 * friendly_pressure - 0.2 * enemy_pressure + centrality + frontier + denial
    if owner == player:
        threat = max(0.0, enemy_pressure - friendly_pressure - 0.35 * planet_ships(target))
        return 11.0 * threat + 4.5 * planet_production(target) + frontier
    vulnerability = max(0.0, friendly_pressure - enemy_pressure)
    return production_value + 6.0 + 0.8 * vulnerability - ship_penalty + centrality + frontier


def _source_reserve(source: Any, enemies: list[Any], enemy_fleets: list[Any]) -> int:
    local_threat = _planet_pressure(source, enemies) + 0.8 * _fleet_pressure(source, enemy_fleets)
    reserve = 8 + int(0.45 * planet_production(source) + 0.35 * local_threat)
    return max(8, min(24, reserve))


def _required_commitment(
    obs: dict[str, Any],
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
    target_xy = _predict_target_xy(obs, source_xy, target, guess_ships)
    travel_steps = max(1, math.ceil(_distance(source_xy, target_xy) / _fleet_speed(max(guess_ships, 2))))
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(target, friendly_fleets)
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    owner = planet_owner(target)
    if owner == player:
        vulnerability = max(0.0, enemy_pressure - friendly_pressure - 0.3 * planet_ships(target))
        return max(0, int(math.ceil(vulnerability + 2.0)) - committed)
    growth = max(0.0, float(planet_production(target)) * travel_steps)
    if owner == -1:
        need = planet_ships(target) + 1 + 0.2 * growth + 0.15 * max(0.0, enemy_pressure - friendly_pressure)
    else:
        need = planet_ships(target) + 1 + 0.55 * growth + 0.4 * max(0.0, enemy_pressure - friendly_pressure)
    return max(0, int(math.ceil(need)) - committed)


def _build_move(obs: dict[str, Any], source: Any, target: Any, ships: int) -> list[float]:
    source_xy = (planet_x(source), planet_y(source))
    target_xy = _predict_target_xy(obs, source_xy, target, ships)
    angle = _sun_safe_angle(source_xy, target_xy)
    return [planet_id(source), float(angle), int(ships)]


def encode_submission(obs: dict[str, Any], player: int) -> dict[str, Any]:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    friendly_fleets = [fleet for fleet in fleets if fleet_owner(fleet) == player]
    enemy_fleets = [fleet for fleet in fleets if fleet_owner(fleet) not in (-1, player)]
    return {
        "player": player,
        "planets": planets,
        "fleets": fleets,
        "own_count": len(own),
        "enemy_count": len(enemies),
        "neutral_count": len(neutrals),
        "own_ships": sum(planet_ships(planet) for planet in own),
        "enemy_ships": sum(planet_ships(planet) for planet in enemies),
        "own_prod": sum(planet_production(planet) for planet in own),
        "enemy_prod": sum(planet_production(planet) for planet in enemies),
        "friendly_fleets": len(friendly_fleets),
        "enemy_fleets": len(enemy_fleets),
        "top_neutral_value": max((_target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets) for planet in neutrals), default=-1e9),
        "top_enemy_value": max((_target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets) for planet in enemies), default=-1e9),
        "top_own_threat": max((_target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets) for planet in own), default=0.0),
    }


def policy_forward(features: dict[str, Any]) -> dict[str, Any]:
    top_own_threat = float(features["top_own_threat"])
    top_neutral_value = float(features["top_neutral_value"])
    top_enemy_value = float(features["top_enemy_value"])
    want_expand = int(features["neutral_count"]) > 0 and (
        int(features["own_prod"]) <= int(features["enemy_prod"]) + 1
        or int(features["own_ships"]) <= int(features["enemy_ships"]) + 10
        or int(features["own_count"]) < max(3, int(features["enemy_count"]))
    )
    if top_own_threat > 8.0:
        mode = "stabilize"
    elif want_expand and top_neutral_value >= top_enemy_value - 3.0:
        mode = "expand"
    else:
        mode = "pressure"
    return {
        "mode": mode,
        "max_moves": MAX_MOVES_PER_TURN,
        "aggression": 0.62 if mode == "pressure" else 0.48,
        "reinforce_bias": 1.15 if mode == "stabilize" else 1.0,
    }


def decode_submission_action(obs: dict[str, Any], player: int, action: dict[str, Any]) -> list[list[float]]:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    own = [planet for planet in planets if planet_owner(planet) == player]
    enemies = [planet for planet in planets if planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if planet_owner(planet) == -1]
    if not own:
        return []

    friendly_fleets = [fleet for fleet in fleets if fleet_owner(fleet) == player]
    enemy_fleets = [fleet for fleet in fleets if fleet_owner(fleet) not in (-1, player)]
    threatened = sorted(
        own,
        key=lambda planet: _target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )
    neutral_targets = sorted(
        neutrals,
        key=lambda planet: _target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )
    enemy_targets = sorted(
        enemies,
        key=lambda planet: _target_value(planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )

    mode = str(action.get("mode", "pressure"))
    if mode == "stabilize":
        target_priority = threatened + neutral_targets[:2] + enemy_targets[:2]
    elif mode == "expand":
        target_priority = neutral_targets + threatened[:2] + enemy_targets[:2]
    else:
        target_priority = enemy_targets + neutral_targets[:2] + threatened[:2]
    if not target_priority:
        return fallback_greedy(obs)

    moves: list[list[float]] = []
    committed: dict[int, int] = {}
    max_moves = max(1, int(action.get("max_moves", MAX_MOVES_PER_TURN)))
    aggression = float(action.get("aggression", 0.55))
    reinforce_bias = float(action.get("reinforce_bias", 1.0))
    sources = sorted(
        own,
        key=lambda planet: (
            planet_ships(planet) - _source_reserve(planet, enemies, enemy_fleets),
            planet_production(planet),
        ),
        reverse=True,
    )

    for source in sources:
        if len(moves) >= max_moves:
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
                obs,
                source,
                target,
                player,
                own,
                enemies,
                friendly_fleets,
                enemy_fleets,
                committed.get(planet_id(target), 0),
            )
            strategic = _target_value(target, player, own, enemies, friendly_fleets, enemy_fleets)
            if planet_owner(target) == player:
                strategic *= reinforce_bias
            source_xy = (planet_x(source), planet_y(source))
            target_xy = _predict_target_xy(obs, source_xy, target, max(2, min(surplus, max(need, 6))))
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
            ships = min(surplus, max(2, int(math.ceil(max(best_need * 1.1, surplus * aggression)))))
        if ships <= 0:
            continue

        moves.append(_build_move(obs, source, best_target, ships))
        committed[planet_id(best_target)] = committed.get(planet_id(best_target), 0) + ships

    return moves or fallback_greedy(obs)


def _moves_are_legal(obs: dict[str, Any], player: int, moves: list[list[float]]) -> bool:
    own_ids = {planet_id(planet) for planet in obs.get("planets", []) if planet_owner(planet) == player}
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return False
        from_id, angle, ships = move
        if int(from_id) not in own_ids:
            return False
        if not math.isfinite(float(angle)):
            return False
        if int(ships) <= 0:
            return False
    return True


def fallback_greedy(obs: dict[str, Any]) -> list[list[float]]:
    try:
        player = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        own = [planet for planet in planets if planet_owner(planet) == player and planet_ships(planet) >= 2]
        if not own:
            return []
        own.sort(key=lambda planet: (planet_ships(planet), planet_production(planet)), reverse=True)
        src = own[0]
        candidates = [planet for planet in planets if planet_id(planet) != planet_id(src)]
        if not candidates:
            return []

        sx, sy = planet_x(src), planet_y(src)

        def target_score(planet: Any) -> float:
            tx, ty = planet_x(planet), planet_y(planet)
            dist = math.hypot(tx - sx, ty - sy)
            owner = planet_owner(planet)
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            return float(planet_production(planet)) * 10.0 + enemy_bonus + neutral_bonus - 0.15 * dist - 0.12 * float(planet_ships(planet))

        target = max(candidates, key=target_score)
        ships = max(2, min(planet_ships(src) - 8, planet_ships(src) // 4))
        if ships <= 0:
            return []
        moves = [_build_move(obs, src, target, ships)]
        return moves if _moves_are_legal(obs, player, moves) else []
    except Exception:
        return []


def safe_submission_agent(obs: dict[str, Any]) -> list[list[float]]:
    try:
        player = int(obs.get("player", 0))
        features = encode_submission(obs, player)
        action = policy_forward(features)
        moves = decode_submission_action(obs, player, action)
        if not _moves_are_legal(obs, player, moves):
            raise ValueError(f"submission policy produced illegal moves for player={player}: {moves!r}")
        return list(moves)
    except Exception:
        return fallback_greedy(obs)
