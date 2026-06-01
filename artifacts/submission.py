# Kaggle Orbit Wars submission template.
# Generated agent is self-contained and does not depend on the local Rust stack.

from math import atan2, ceil, cos, hypot, isfinite, log, pi, sin

SUN_RADIUS = 10.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0
RESERVE_HOME_SHIPS = 8
MIN_SHIPS_TO_LAUNCH = 2
MAX_MOVES_PER_TURN = 6
MIN_CAPTURE_MARGIN = 2


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


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _reserve_for_source(source, own_count, enemies, action):
    reserve = RESERVE_HOME_SHIPS
    if own_count <= 2:
        reserve += 4
    if action.get("ffa"):
        reserve += 4
    if action.get("pressure"):
        reserve += 2
    if _planet_production(source) >= 4:
        reserve += 2
    if enemies:
        source_xy = (_planet_x(source), _planet_y(source))
        nearest_enemy = min(_distance(source_xy, (_planet_x(enemy), _planet_y(enemy))) for enemy in enemies)
        if nearest_enemy < 18.0:
            reserve += 4
        elif nearest_enemy < 28.0:
            reserve += 2
    return reserve


def _source_priority(source, own_count, enemies, action):
    reserve = _reserve_for_source(source, own_count, enemies, action)
    surplus = max(0, _planet_ships(source) - reserve)
    return (
        surplus,
        _planet_production(source),
        -abs(_planet_x(source) - CENTER) - abs(_planet_y(source) - CENTER),
    )


def encode(obs):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player]
    enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if _planet_owner(planet) == -1]
    enemy_owners = sorted({_planet_owner(planet) for planet in enemies})

    owner_totals = {}
    owner_prod = {}
    for planet in planets:
        owner = _planet_owner(planet)
        owner_totals[owner] = owner_totals.get(owner, 0) + _planet_ships(planet)
        owner_prod[owner] = owner_prod.get(owner, 0) + _planet_production(planet)

    leader_owner = None
    if enemies:
        leader_owner = max(
            sorted({_planet_owner(planet) for planet in enemies}),
            key=lambda owner: (owner_prod.get(owner, 0), owner_totals.get(owner, 0)),
        )

    return {
        "player": player,
        "own_count": len(own),
        "enemy_count": len(enemies),
        "enemy_players": len(enemy_owners),
        "neutral_count": len(neutrals),
        "own_ships": sum(_planet_ships(planet) for planet in own),
        "enemy_ships": sum(_planet_ships(planet) for planet in enemies),
        "own_prod": sum(_planet_production(planet) for planet in own),
        "enemy_prod": sum(_planet_production(planet) for planet in enemies),
        "leader_owner": leader_owner,
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
    }


def policy_forward(features):
    ffa = features["enemy_players"] >= 2
    pressure = features["enemy_ships"] >= max(features["own_ships"] - 4, 1)
    behind_on_econ = features["enemy_prod"] > features["own_prod"]
    neutrals_open = features["neutral_count"] > 0
    expand = neutrals_open and (
        features["own_count"] <= 3
        or ffa
        or not (pressure or behind_on_econ)
    )
    return {
        "expand": bool(expand),
        "ffa": bool(ffa),
        "pressure": bool(pressure),
        "behind_on_econ": bool(behind_on_econ),
        "leader_owner": features["leader_owner"],
        "neutral_count": int(features["neutral_count"]),
    }


def _required_ships(obs, source, target, committed, action):
    source_xy = (_planet_x(source), _planet_y(source))
    estimate = max(MIN_SHIPS_TO_LAUNCH, _planet_ships(target) + MIN_CAPTURE_MARGIN)
    target_xy = _predict_target_xy(obs, source_xy, target, estimate)
    travel_steps = max(1, ceil(_distance(source_xy, target_xy) / _fleet_speed(estimate)))
    owner = _planet_owner(target)
    growth = 0
    if owner != -1:
        growth = travel_steps * _planet_production(target)
    need = _planet_ships(target) + growth + MIN_CAPTURE_MARGIN - int(committed)
    if owner == action.get("leader_owner"):
        need += 1
    return max(MIN_SHIPS_TO_LAUNCH, need), target_xy


def _target_value(obs, source, target, committed, action, own, enemies):
    required, target_xy = _required_ships(obs, source, target, committed, action)
    source_xy = (_planet_x(source), _planet_y(source))
    distance = _distance(source_xy, target_xy)
    owner = _planet_owner(target)
    production = _planet_production(target)
    ships = _planet_ships(target)
    ffa = bool(action.get("ffa"))

    own_proximity = min(
        (_distance((_planet_x(planet), _planet_y(planet)), target_xy) for planet in own if _planet_id(planet) != _planet_id(source)),
        default=distance,
    )
    enemy_proximity = min(
        (_distance((_planet_x(planet), _planet_y(planet)), target_xy) for planet in enemies),
        default=distance,
    )

    value = production * (14.0 if owner == -1 else 17.0)
    if owner == -1:
        value += 8.0
        if ffa:
            value += 4.0
    else:
        value += 8.0
    if owner == action.get("leader_owner"):
        value += 5.0
    if action.get("expand") and owner == -1:
        value += 8.0
    if action.get("pressure") and owner not in (-1, int(obs.get("player", 0))):
        value += 3.0
    if action.get("expand") and action.get("neutral_count", 0) > 0 and owner != -1:
        value -= 10.0
    if ffa and owner not in (-1, action.get("leader_owner")):
        value -= 6.0
    if ffa and owner == action.get("leader_owner"):
        value += 3.0
    if enemy_proximity < 18.0:
        value += 2.5 if owner == -1 else 1.5
    if own_proximity < 16.0:
        value += 2.5 if owner == -1 else 1.0

    roi = value / max(required, 1)
    distance_penalty = 0.16 * distance
    if ffa:
        distance_penalty += 0.06 * distance
    if action.get("expand") and owner != -1:
        distance_penalty += 0.08 * distance
    return value + 24.0 * roi - distance_penalty - 0.22 * ships, required, target_xy


def decode(action, obs):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player]
    enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
    targets = [planet for planet in planets if _planet_owner(planet) != player]
    if not own or not targets:
        return []

    committed_by_target = {}
    launched_by_source = {}
    used_targets = set()
    moves = []

    max_moves = 4 if action.get("ffa") else MAX_MOVES_PER_TURN
    sources = sorted(own, key=lambda planet: _source_priority(planet, len(own), enemies, action), reverse=True)

    for source in sources:
        if len(moves) >= max_moves:
            break
        source_id = _planet_id(source)
        reserve = _reserve_for_source(source, len(own), enemies, action)
        available = _planet_ships(source) - reserve - launched_by_source.get(source_id, 0)
        if available < MIN_SHIPS_TO_LAUNCH:
            continue

        best = None
        source_xy = (_planet_x(source), _planet_y(source))
        for target in targets:
            target_id = _planet_id(target)
            committed = committed_by_target.get(target_id, 0)
            score, required, target_xy = _target_value(obs, source, target, committed, action, own, enemies)
            ships = min(available, required)
            if target_id in used_targets and required <= MIN_SHIPS_TO_LAUNCH:
                continue
            if required > available:
                score -= 6.0 + 0.35 * (required - available)
            if ships < MIN_SHIPS_TO_LAUNCH:
                continue
            if best is None or score > best["score"]:
                best = {
                    "target_id": target_id,
                    "ships": ships,
                    "score": score,
                    "target_xy": target_xy,
                    "source_xy": source_xy,
                }

        if best is None or best["score"] <= 0.0:
            continue

        angle = _sun_safe_angle(best["source_xy"], best["target_xy"], _angle(best["source_xy"], best["target_xy"]))
        moves.append([source_id, float(angle), int(best["ships"])])
        launched_by_source[source_id] = launched_by_source.get(source_id, 0) + int(best["ships"])
        committed_by_target[best["target_id"]] = committed_by_target.get(best["target_id"], 0) + int(best["ships"])
        used_targets.add(best["target_id"])

    return moves


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
            reserve = RESERVE_HOME_SHIPS + (4 if len(own) <= 2 else 0)
            ships = _planet_ships(source) - reserve
            if ships < MIN_SHIPS_TO_LAUNCH:
                continue
            source_xy = (_planet_x(source), _planet_y(source))
            target = min(
                targets,
                key=lambda planet: (
                    (_planet_ships(planet) + 1.0) / (_planet_production(planet) + 1.0),
                    _distance(source_xy, (_planet_x(planet), _planet_y(planet))),
                ),
            )
            target_xy = _predict_target_xy(obs, source_xy, target, ships)
            angle = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
            moves.append([_planet_id(source), float(angle), int(ships)])
        return moves if _moves_are_legal(obs, player, moves) else []
    except Exception:
        return []


def agent(obs):
    try:
        player = int(obs.get("player", 0))
        features = encode(obs)
        action = policy_forward(features)
        moves = decode(action, obs)
        if not _moves_are_legal(obs, player, moves):
            raise ValueError(f"submission policy produced illegal moves for player={player}: {moves!r}")
        return list(moves)
    except Exception:
        return fallback_greedy(obs)
