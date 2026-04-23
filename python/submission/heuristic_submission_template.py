# Kaggle Orbit Wars heuristic submission template.
# Generated agent is self-contained and does not depend on the local Rust stack.

from math import atan2, ceil, cos, hypot, isfinite, log, pi, sin
from random import Random

HEURISTIC_POLICY = "__HEURISTIC_POLICY__"
SUN_RADIUS = 10.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0
MIN_SHIPS_TO_LAUNCH = 2


def _field(entity, index, key):
    if isinstance(entity, dict):
        return entity[key]
    return entity[index]


def _planet_id(planet):
    return int(_field(planet, 0, "id"))


def _planet_owner(planet):
    return int(_field(planet, 1, "owner"))


def _planet_x(planet):
    return float(_field(planet, 2, "x"))


def _planet_y(planet):
    return float(_field(planet, 3, "y"))


def _planet_radius(planet):
    return float(_field(planet, 4, "radius"))


def _planet_ships(planet):
    return int(_field(planet, 5, "ships"))


def _planet_production(planet):
    return int(_field(planet, 6, "production"))


def _angle(a, b):
    return atan2(b[1] - a[1], b[0] - a[0])


def _fleet_speed(ships):
    scale = log(max(int(ships), 1)) / log(1000.0)
    speed = 1.0 + (SHIP_SPEED - 1.0) * scale**1.5
    return min(SHIP_SPEED, max(1.0, speed))


def _point_to_segment_distance(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    l2 = vx * vx + vy * vy
    if l2 == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / l2))
    proj = (start[0] + t * vx, start[1] + t * vy)
    return hypot(point[0] - proj[0], point[1] - proj[1])


def _sun_safe_angle(source, target, base_angle):
    if _point_to_segment_distance((CENTER, CENTER), source, target) >= SUN_RADIUS + 1.0:
        return base_angle
    to_center = _angle(source, (CENTER, CENTER))
    candidates = [to_center + pi / 2.0, to_center - pi / 2.0]
    return min(candidates, key=lambda a: abs(atan2(sin(a - base_angle), cos(a - base_angle))))


def _rotate_about_center(point, angle):
    dx = point[0] - CENTER
    dy = point[1] - CENTER
    c = cos(angle)
    s = sin(angle)
    return (CENTER + dx * c - dy * s, CENTER + dx * s + dy * c)


def _is_rotating_planet(planet):
    orbital_radius = hypot(_planet_x(planet) - CENTER, _planet_y(planet) - CENTER)
    return orbital_radius + _planet_radius(planet) < ROTATION_RADIUS_LIMIT


def _predict_target_xy(obs, source_xy, target, ships):
    target_xy = (_planet_x(target), _planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    distance = hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1])
    travel_steps = max(1, ceil(distance / _fleet_speed(ships)))
    return _rotate_about_center(target_xy, float(obs.get("angular_velocity", 0.0)) * travel_steps)


def _build_move(obs, source, target, ships):
    if ships < MIN_SHIPS_TO_LAUNCH:
        return None
    source_xy = (_planet_x(source), _planet_y(source))
    target_xy = _predict_target_xy(obs, source_xy, target, ships)
    base = _angle(source_xy, target_xy)
    angle = _sun_safe_angle(source_xy, target_xy, base)
    return [_planet_id(source), float(angle), int(ships)]


def _moves_are_legal(obs, player, moves):
    own_ids = {_planet_id(planet) for planet in obs.get("planets", []) if _planet_owner(planet) == player}
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return False
        from_id, angle, ships = move
        if int(from_id) not in own_ids:
            return False
        if not isfinite(float(angle)):
            return False
        if int(ships) <= 0:
            return False
    return True


def fallback_greedy(obs):
    try:
        player = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        own = [planet for planet in planets if _planet_owner(planet) == player]
        targets = [planet for planet in planets if _planet_owner(planet) != player]
        if not own or not targets:
            return []

        moves = []
        for source in sorted(own, key=lambda planet: (_planet_ships(planet), _planet_production(planet)), reverse=True):
            if len(moves) >= 3:
                break
            reserve = 8 + (4 if len(own) <= 2 else 0)
            ships = _planet_ships(source) - reserve
            if ships < MIN_SHIPS_TO_LAUNCH:
                continue
            source_xy = (_planet_x(source), _planet_y(source))
            target = min(
                targets,
                key=lambda planet: (
                    (_planet_ships(planet) + 1.0) / (_planet_production(planet) + 1.0),
                    hypot(source_xy[0] - _planet_x(planet), source_xy[1] - _planet_y(planet)),
                ),
            )
            move = _build_move(obs, source, target, ships)
            if move is not None:
                moves.append(move)
        return moves if _moves_are_legal(obs, player, moves) else []
    except Exception:
        return []


def _defensive_policy(obs, player, own, neutrals):
    if not own or not neutrals:
        return []
    moves = []
    for source in sorted(own, key=_planet_ships, reverse=True):
        reserve = 18 if len(own) <= 2 else 10
        ships = _planet_ships(source) - reserve
        if ships <= 0:
            continue
        sx, sy = _planet_x(source), _planet_y(source)
        target = min(
            neutrals,
            key=lambda planet: float(_planet_ships(planet)) + 0.08 * hypot(_planet_x(planet) - sx, _planet_y(planet) - sy),
        )
        move = _build_move(obs, source, target, int(ships * 0.5))
        if move is not None:
            moves.append(move)
        if len(moves) >= 4:
            break
    return moves


def _rush_policy(obs, own, enemies):
    if not own or not enemies:
        return []
    enemy_home = max(enemies, key=_planet_ships)
    moves = []
    for source in sorted(own, key=_planet_ships, reverse=True)[:2]:
        ships = max(0, _planet_ships(source) - 5)
        move = _build_move(obs, source, enemy_home, ships)
        if move is not None:
            moves.append(move)
    return moves


def _anti_meta_policy(obs, player, own, enemies, neutrals):
    if not own:
        return []
    if enemies:
        focus = sorted(
            enemies,
            key=lambda planet: (
                -_planet_ships(planet),
                -sum(1 for neutral in neutrals if hypot(_planet_x(neutral) - _planet_x(planet), _planet_y(neutral) - _planet_y(planet)) < 20.0),
            ),
        )[0]
    elif neutrals:
        focus = max(neutrals, key=lambda planet: (_planet_production(planet), -_planet_ships(planet)))
    else:
        return fallback_greedy(obs)

    moves = []
    for source in sorted(own, key=_planet_ships, reverse=True)[:3]:
        ships = max(0, _planet_ships(source) - 7)
        move = _build_move(obs, source, focus, max(1, int(ships * 0.6)))
        if move is not None:
            moves.append(move)
    return moves


def _weak_random_policy(obs, own, targets, player):
    if not own or not targets:
        return []
    rng = Random(int(obs.get("step", 0)) + 997 * player + len(obs.get("planets", [])))
    source = rng.choice(sorted(own, key=_planet_ships, reverse=True)[: max(1, min(3, len(own)))])
    target = rng.choice(targets)
    ships = max(0, int((_planet_ships(source) - 6) * rng.uniform(0.25, 0.55)))
    move = _build_move(obs, source, target, ships)
    return [move] if move is not None else []


def agent(obs):
    try:
        player = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        own = [planet for planet in planets if _planet_owner(planet) == player]
        enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
        neutrals = [planet for planet in planets if _planet_owner(planet) == -1]
        targets = [planet for planet in planets if _planet_owner(planet) != player]

        if HEURISTIC_POLICY == "defensive":
            moves = _defensive_policy(obs, player, own, neutrals)
        elif HEURISTIC_POLICY == "rush":
            moves = _rush_policy(obs, own, enemies)
        elif HEURISTIC_POLICY == "anti_meta":
            moves = _anti_meta_policy(obs, player, own, enemies, neutrals)
        elif HEURISTIC_POLICY == "weak_random":
            moves = _weak_random_policy(obs, own, targets, player)
        else:
            moves = fallback_greedy(obs)

        if not _moves_are_legal(obs, player, moves):
            raise ValueError(f"submission policy produced illegal moves for player={player}: {moves!r}")
        return list(moves)
    except Exception:
        return fallback_greedy(obs)
