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
PROFILE_DECAY = 0.82
PROFILE_RAY_MAX_ANGLE = 0.36
PROFILE_CAPTURE_TTL = 18
FSM_OPENING_TURNS = 55

_PROFILE_STATE = {}


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


def _fleet_id(fleet):
    return int(_field(fleet, 0, "id"))


def _fleet_owner(fleet):
    return int(_field(fleet, 1, "owner"))


def _fleet_x(fleet):
    return float(_field(fleet, 2, "x"))


def _fleet_y(fleet):
    return float(_field(fleet, 3, "y"))


def _fleet_angle(fleet):
    return float(_field(fleet, 4, "angle"))


def _fleet_from_planet_id(fleet):
    return int(_field(fleet, 5, "from_planet_id"))


def _fleet_ships(fleet):
    return int(_field(fleet, 6, "ships"))


def _angle(a, b):
    return atan2(b[1] - a[1], b[0] - a[0])


def _angle_delta(left, right):
    return abs(atan2(sin(left - right), cos(left - right)))


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


def _ray_score_to_planet(origin, angle, planet):
    target_xy = (_planet_x(planet), _planet_y(planet))
    distance = _distance(origin, target_xy)
    if distance <= 0.0:
        return 999.0
    delta = _angle_delta(angle, _angle(origin, target_xy))
    if delta > PROFILE_RAY_MAX_ANGLE:
        return 999.0
    return delta + 0.004 * distance


def _estimate_fleet_target(obs, fleet):
    origin = (_fleet_x(fleet), _fleet_y(fleet))
    angle = _fleet_angle(fleet)
    best = None
    for planet in obs.get("planets", []):
        score = _ray_score_to_planet(origin, angle, planet)
        if score >= 999.0:
            continue
        if best is None or score < best[0]:
            best = (score, planet)
    return None if best is None else best[1]


def _empty_profile_state(step):
    return {
        "step": int(step),
        "seen_fleets": set(),
        "to_neutral": 0.0,
        "to_me": 0.0,
        "to_leader": 0.0,
        "to_other": 0.0,
        "owners": {},
        "recent_captures": {},
    }


def _decay_profile(state):
    for key in ("to_neutral", "to_me", "to_leader", "to_other"):
        state[key] = float(state.get(key, 0.0)) * PROFILE_DECAY


def _profile_ratios(state):
    total = (
        float(state.get("to_neutral", 0.0))
        + float(state.get("to_me", 0.0))
        + float(state.get("to_leader", 0.0))
        + float(state.get("to_other", 0.0))
    )
    if total <= 0.0:
        return {
            "profile_total": 0.0,
            "to_neutral_ratio": 0.0,
            "to_me_ratio": 0.0,
            "to_leader_ratio": 0.0,
            "to_other_ratio": 0.0,
        }
    return {
        "profile_total": total,
        "to_neutral_ratio": float(state.get("to_neutral", 0.0)) / total,
        "to_me_ratio": float(state.get("to_me", 0.0)) / total,
        "to_leader_ratio": float(state.get("to_leader", 0.0)) / total,
        "to_other_ratio": float(state.get("to_other", 0.0)) / total,
    }


def _update_recent_captures(state, player, planets):
    previous_owners = state.get("owners", {})
    recent = {
        int(planet_id): int(ttl) - 1
        for planet_id, ttl in state.get("recent_captures", {}).items()
        if int(ttl) > 1
    }
    owners = {}
    for planet in planets:
        planet_id = _planet_id(planet)
        owner = _planet_owner(planet)
        old_owner = previous_owners.get(planet_id, owner)
        if old_owner in (-1, player) and owner not in (-1, player) and _planet_ships(planet) <= 14:
            recent[planet_id] = PROFILE_CAPTURE_TTL
        owners[planet_id] = owner
    state["owners"] = owners
    state["recent_captures"] = recent
    return set(recent)


def _update_opponent_profile(obs, player, planets, leader_owner):
    global _PROFILE_STATE
    step = int(obs.get("step", obs.get("turn", 0)))
    state = _PROFILE_STATE.get(player)
    if state is None or step <= int(state.get("step", -1)):
        state = _empty_profile_state(step)
    else:
        state["step"] = step
        _decay_profile(state)

    recent_captures = _update_recent_captures(state, player, planets)
    seen = state.setdefault("seen_fleets", set())
    for fleet in obs.get("fleets", []):
        owner = _fleet_owner(fleet)
        if owner in (-1, player):
            continue
        fleet_key = (owner, _fleet_id(fleet))
        if fleet_key in seen:
            continue
        seen.add(fleet_key)
        target = _estimate_fleet_target(obs, fleet)
        if target is None:
            continue
        ships = max(0, _fleet_ships(fleet))
        target_owner = _planet_owner(target)
        if target_owner == -1:
            state["to_neutral"] = float(state.get("to_neutral", 0.0)) + ships
        elif target_owner == player:
            state["to_me"] = float(state.get("to_me", 0.0)) + ships
        elif target_owner == leader_owner:
            state["to_leader"] = float(state.get("to_leader", 0.0)) + ships
        else:
            state["to_other"] = float(state.get("to_other", 0.0)) + ships

    _PROFILE_STATE[player] = state
    ratios = _profile_ratios(state)
    ratios["recent_enemy_captures"] = recent_captures
    return ratios


def _fsm_state(features):
    step = int(features.get("step", 0))
    if features.get("enemy_players", 0) >= 2 and features.get("to_me_ratio", 0.0) >= 0.58:
        return "DEFEND_UNDER_PRESSURE"
    if features.get("recent_enemy_captures"):
        return "PUNISH_WEAK_CAPTURE"
    if step <= FSM_OPENING_TURNS and features.get("neutral_count", 0) > 0 and features.get("own_count", 0) <= 4:
        return "OPENING_EXPAND"
    if features.get("enemy_prod", 0) > features.get("own_prod", 0) and features.get("neutral_count", 0) > 0:
        return "ECON_CONSOLIDATE"
    return "BASELINE"


def _reserve_for_source(source, own_count, enemies, action):
    reserve = RESERVE_HOME_SHIPS
    if own_count <= 2:
        reserve += 4
    if action.get("ffa"):
        reserve += 4
    if action.get("pressure"):
        reserve += 2
    if action.get("ffa") and action.get("fsm_state") == "DEFEND_UNDER_PRESSURE":
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


def _outgoing_by_source(obs, player):
    outgoing = {}
    for fleet in obs.get("fleets", []):
        if _fleet_owner(fleet) != player:
            continue
        source_id = _fleet_from_planet_id(fleet)
        item = outgoing.setdefault(source_id, [0, 0])
        item[0] += 1
        item[1] += _fleet_ships(fleet)
    return outgoing


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
    profile = _update_opponent_profile(obs, player, planets, leader_owner)

    return {
        "player": player,
        "step": int(obs.get("step", obs.get("turn", 0))),
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
        **profile,
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
    state = _fsm_state(features)
    if ffa and state == "DEFEND_UNDER_PRESSURE" and features.get("profile_total", 0.0) >= 16.0:
        pressure = True
        expand = False
    return {
        "expand": bool(expand),
        "ffa": bool(ffa),
        "pressure": bool(pressure),
        "behind_on_econ": bool(behind_on_econ),
        "leader_owner": features["leader_owner"],
        "neutral_count": int(features["neutral_count"]),
        "fsm_state": state,
        "recent_enemy_captures": set(features.get("recent_enemy_captures", set())),
        "profile_total": float(features.get("profile_total", 0.0)),
        "to_neutral_ratio": float(features.get("to_neutral_ratio", 0.0)),
        "to_me_ratio": float(features.get("to_me_ratio", 0.0)),
        "to_leader_ratio": float(features.get("to_leader_ratio", 0.0)),
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
    if ffa and action.get("fsm_state") == "DEFEND_UNDER_PRESSURE" and owner == -1:
        value -= 1.5
    if ffa and _planet_id(target) in action.get("recent_enemy_captures", set()):
        value += 2.0
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
    if action.get("fsm_state") == "OPENING_EXPAND" and owner == -1:
        distance_penalty += 0.12 * distance
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
    outgoing_by_source = _outgoing_by_source(obs, player)
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
        outgoing_count = outgoing_by_source.get(source_id, [0, 0])[0]
        min_launch = MIN_SHIPS_TO_LAUNCH + min(12, 4 * outgoing_count)

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
            if ships < min_launch:
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
