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
MAX_GAME_TURNS = 500
FUTURE_FLEET_HORIZON = 90

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


def _comet_planet_ids(group):
    return list(_field(group, 0, "planet_ids"))


def _comet_paths(group):
    return list(_field(group, 1, "paths"))


def _comet_path_index(group):
    return int(_field(group, 2, "path_index"))


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


def _comet_path_for_planet(obs, planet_id):
    for group in obs.get("comets", []):
        planet_ids = _comet_planet_ids(group)
        try:
            offset = planet_ids.index(int(planet_id))
        except ValueError:
            continue
        paths = _comet_paths(group)
        if offset < len(paths):
            return paths[offset], _comet_path_index(group)
    return None, None


def _is_comet(obs, planet):
    return _planet_id(planet) in set(int(pid) for pid in obs.get("comet_planet_ids", []))


def _comet_turns_remaining(obs, planet):
    path, path_index = _comet_path_for_planet(obs, _planet_id(planet))
    if path is None or path_index is None:
        return None
    return max(0, len(path) - max(0, int(path_index)))


def _comet_xy_at(obs, planet, turns):
    path, path_index = _comet_path_for_planet(obs, _planet_id(planet))
    if path is None or path_index is None:
        return None
    index = int(path_index) + max(0, int(turns))
    if index < 0 or index >= len(path):
        return None
    point = path[index]
    return (float(point[0]), float(point[1]))


def _predict_target_xy(obs, source_xy, target, ships):
    target_xy = (_planet_x(target), _planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    distance = hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1])
    travel_steps = max(1, ceil(distance / _fleet_speed(ships)))
    return _rotate_about_center(target_xy, float(obs.get("angular_velocity", 0.0)) * travel_steps)


def _planet_xy_at(obs, planet, turns):
    if _is_comet(obs, planet):
        comet_xy = _comet_xy_at(obs, planet, turns)
        if comet_xy is not None:
            return comet_xy
    point = (_planet_x(planet), _planet_y(planet))
    if not _is_rotating_planet(planet):
        return point
    return _rotate_about_center(point, float(obs.get("angular_velocity", 0.0)) * max(0, int(turns)))


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


def _fleet_eta_to_planet(obs, fleet, target, max_turns):
    origin = (_fleet_x(fleet), _fleet_y(fleet))
    angle = _fleet_angle(fleet)
    speed = _fleet_speed(_fleet_ships(fleet))
    radius = max(0.5, _planet_radius(target))
    limit = max(1, int(max_turns))
    target_xy = (_planet_x(target), _planet_y(target))
    estimate = max(1, min(limit, ceil(max(0.0, _distance(origin, target_xy) - radius - 0.75) / max(speed, 0.1))))
    for _ in range(4):
        target_xy = _planet_xy_at(obs, target, estimate)
        next_estimate = max(1, min(limit, ceil(max(0.0, _distance(origin, target_xy) - radius - 0.75) / max(speed, 0.1))))
        if abs(next_estimate - estimate) <= 1:
            estimate = next_estimate
            break
        estimate = next_estimate
    for turns in range(max(1, estimate - 3), min(limit, estimate + 4) + 1):
        target_xy = _planet_xy_at(obs, target, turns)
        if _angle_delta(angle, _angle(origin, target_xy)) > PROFILE_RAY_MAX_ANGLE:
            continue
        if _distance(origin, target_xy) <= speed * turns + radius + 0.75:
            return turns
    return None


def _resolve_arrivals(owner, garrison, arrivals):
    if not arrivals:
        return owner, max(0, int(garrison))

    attackers = [(int(fleet_owner), int(ships)) for fleet_owner, ships in arrivals.items() if int(ships) > 0]
    if not attackers:
        return owner, max(0, int(garrison))
    attackers.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    survivor_owner, survivor_ships = attackers[0]
    if len(attackers) > 1:
        second_ships = attackers[1][1]
        if survivor_ships == second_ships:
            return owner, max(0, int(garrison))
        survivor_ships -= second_ships

    if survivor_owner == owner:
        return owner, max(0, int(garrison) + survivor_ships)
    if survivor_ships > garrison:
        return survivor_owner, int(survivor_ships - garrison)
    if survivor_ships == garrison:
        return -1, 0
    return owner, int(garrison - survivor_ships)


def _project_planet_state(obs, target, horizon, extra_arrivals=None, cache=None):
    horizon = max(0, min(FUTURE_FLEET_HORIZON, int(horizon)))
    extra_key = tuple((int(eta), int(owner), int(ships)) for eta, owner, ships in (extra_arrivals or ()) if int(ships) > 0)
    cache_key = (_planet_id(target), horizon, extra_key)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    owner = _planet_owner(target)
    garrison = _planet_ships(target)
    production = _planet_production(target)
    target_id = _planet_id(target)
    arrivals_by_turn = {}

    if horizon > 0:
        fleet_eta_cache = None if cache is None else cache.setdefault(("fleet_etas",), {})
        for fleet in _fleets_by_target(obs, cache).get(target_id, []):
            fleet_key = (_fleet_owner(fleet), _fleet_id(fleet), _fleet_x(fleet), _fleet_y(fleet), _fleet_angle(fleet))
            eta_key = (fleet_key, target_id, horizon)
            if fleet_eta_cache is not None and eta_key in fleet_eta_cache:
                eta = fleet_eta_cache[eta_key]
            else:
                eta = _fleet_eta_to_planet(obs, fleet, target, horizon)
                if fleet_eta_cache is not None:
                    fleet_eta_cache[eta_key] = eta
            if eta is None:
                continue
            turn_arrivals = arrivals_by_turn.setdefault(eta, {})
            fleet_owner = _fleet_owner(fleet)
            turn_arrivals[fleet_owner] = turn_arrivals.get(fleet_owner, 0) + _fleet_ships(fleet)

    for arrival in extra_arrivals or ():
        eta, arrival_owner, ships = arrival
        eta = max(1, min(horizon, int(eta))) if horizon > 0 else 0
        if eta <= 0 or int(ships) <= 0:
            continue
        turn_arrivals = arrivals_by_turn.setdefault(eta, {})
        arrival_owner = int(arrival_owner)
        turn_arrivals[arrival_owner] = turn_arrivals.get(arrival_owner, 0) + int(ships)

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += production
        owner, garrison = _resolve_arrivals(owner, garrison, arrivals_by_turn.get(turn, {}))
    result = (owner, garrison)
    if cache is not None:
        cache[cache_key] = result
    return result


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
    if step <= FSM_OPENING_TURNS and features.get("neutral_count", 0) > 0 and features.get("own_count", 0) <= 4:
        return "OPENING_EXPAND"
    if (
        features.get("recent_enemy_captures")
        and not (
            features.get("neutral_count", 0) >= 8
            and features.get("enemy_prod", 0) > features.get("own_prod", 0)
        )
    ):
        return "PUNISH_WEAK_CAPTURE"
    if features.get("enemy_prod", 0) > features.get("own_prod", 0) and features.get("neutral_count", 0) > 0:
        return "ECON_CONSOLIDATE"
    return "BASELINE"


def _fleet_target_id(obs, fleet, cache=None):
    fleet_key = (_fleet_owner(fleet), _fleet_id(fleet), _fleet_x(fleet), _fleet_y(fleet), _fleet_angle(fleet))
    if cache is not None and fleet_key in cache:
        return cache[fleet_key]
    estimated_target = _estimate_fleet_target(obs, fleet)
    target_id = None if estimated_target is None else _planet_id(estimated_target)
    if cache is not None:
        cache[fleet_key] = target_id
    return target_id


def _fleets_by_target(obs, cache=None):
    index_key = ("fleets_by_target",)
    if cache is not None and index_key in cache:
        return cache[index_key]
    fleet_target_cache = None if cache is None else cache.setdefault(("fleet_targets",), {})
    grouped = {}
    for fleet in obs.get("fleets", []):
        target_id = _fleet_target_id(obs, fleet, fleet_target_cache)
        if target_id is not None:
            grouped.setdefault(target_id, []).append(fleet)
    if cache is not None:
        cache[index_key] = grouped
    return grouped


def _incoming_threats_by_target(obs, player, horizon, cache=None):
    if obs is None or player is None:
        return {}
    threat_key = ("incoming_threats_by_target", int(player), int(horizon))
    if cache is not None and threat_key in cache:
        return cache[threat_key]
    fleet_target_cache = None if cache is None else cache.setdefault(("fleet_targets",), {})
    fleet_eta_cache = None if cache is None else cache.setdefault(("fleet_etas",), {})
    planets_by_id = {_planet_id(planet): planet for planet in obs.get("planets", [])}
    threats = {}
    for fleet in obs.get("fleets", []):
        if _fleet_owner(fleet) in (-1, player):
            continue
        target_id = _fleet_target_id(obs, fleet, fleet_target_cache)
        if target_id is None:
            continue
        target = planets_by_id.get(target_id)
        if target is None:
            continue
        fleet_key = (_fleet_owner(fleet), _fleet_id(fleet), _fleet_x(fleet), _fleet_y(fleet), _fleet_angle(fleet))
        eta_key = (fleet_key, target_id, int(horizon))
        if fleet_eta_cache is not None and eta_key in fleet_eta_cache:
            eta = fleet_eta_cache[eta_key]
        else:
            eta = _fleet_eta_to_planet(obs, fleet, target, horizon)
            if fleet_eta_cache is not None:
                fleet_eta_cache[eta_key] = eta
        if eta is not None and eta <= horizon:
            threats[target_id] = threats.get(target_id, 0) + _fleet_ships(fleet)
    if cache is not None:
        cache[threat_key] = threats
    return threats


def _incoming_threat_before(obs, source, player, horizon, cache=None):
    return _incoming_threats_by_target(obs, player, horizon, cache).get(_planet_id(source), 0)


def _reserve_for_source(source, own_count, enemies, action, obs=None, player=None):
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
    cache = None if action is None else action.setdefault("_projection_cache", {})
    incoming_threat = _incoming_threat_before(obs, source, player, 40, cache)
    step = int((obs or {}).get("step", (obs or {}).get("turn", 0)))
    if step >= 30 and action.get("fsm_state") == "OPENING_EXPAND" and incoming_threat <= 0 and _planet_production(source) <= 1:
        reserve = min(reserve, RESERVE_HOME_SHIPS)
    if incoming_threat >= _planet_ships(source) and incoming_threat > 0:
        reserve = min(reserve, max(MIN_CAPTURE_MARGIN, _planet_ships(source) // 3))
    else:
        reserve = max(reserve, incoming_threat + MIN_CAPTURE_MARGIN)
    return reserve


def _source_priority(source, own_count, enemies, action, obs=None, player=None):
    reserve = _reserve_for_source(source, own_count, enemies, action, obs, player)
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


def _incoming_by_target(obs, owner, cache=None):
    incoming_key = ("incoming_by_target", int(owner))
    if cache is not None and incoming_key in cache:
        return cache[incoming_key]
    fleet_target_cache = None if cache is None else cache.setdefault(("fleet_targets",), {})
    incoming = {}
    for fleet in obs.get("fleets", []):
        if _fleet_owner(fleet) != owner:
            continue
        target_id = _fleet_target_id(obs, fleet, fleet_target_cache)
        if target_id is not None:
            incoming[target_id] = incoming.get(target_id, 0) + _fleet_ships(fleet)
    if cache is not None:
        cache[incoming_key] = incoming
    return incoming


def _incoming_ships_to_target(obs, target, owner, cache=None):
    return _incoming_by_target(obs, owner, cache).get(_planet_id(target), 0)


def _nearest_enemy_distance(planet, enemies):
    if not enemies:
        return 999.0
    source_xy = (_planet_x(planet), _planet_y(planet))
    return min(_distance(source_xy, (_planet_x(enemy), _planet_y(enemy))) for enemy in enemies)


def _frontier_reinforcement_moves(obs, own, enemies, action, launched_by_source, max_moves_left):
    step = int(obs.get("step", obs.get("turn", 0)))
    if max_moves_left <= 0 or len(own) < 5 or not enemies or step <= FSM_OPENING_TURNS:
        return []
    ffa = bool(action.get("ffa"))
    if not ffa and (len(own) < 4 or step < 55):
        return []
    player = int(obs.get("player", 0))
    own_with_pressure = [(planet, _nearest_enemy_distance(planet, enemies)) for planet in own]
    frontier_limit = 24.0 if ffa else 20.0
    source_min_distance = 44.0 if ffa else 52.0
    min_available = MIN_SHIPS_TO_LAUNCH + (18 if ffa else 28)
    keep_after_send = 12 if ffa else 20
    frontier = [
        planet
        for planet, enemy_distance in own_with_pressure
        if enemy_distance <= frontier_limit
    ]
    if not frontier:
        return []

    moves = []
    for source, source_enemy_distance in sorted(own_with_pressure, key=lambda item: item[1], reverse=True):
        if len(moves) >= max_moves_left:
            break
        source_id = _planet_id(source)
        if source in frontier:
            continue
        if source_enemy_distance < source_min_distance:
            continue
        reserve = _reserve_for_source(source, len(own), enemies, action, obs, player)
        available = _planet_ships(source) - reserve - launched_by_source.get(source_id, 0)
        if available < min_available:
            continue
        source_xy = (_planet_x(source), _planet_y(source))
        target = min(
            frontier,
            key=lambda planet: (
                _nearest_enemy_distance(planet, enemies),
                _distance(source_xy, (_planet_x(planet), _planet_y(planet))),
                -_planet_production(planet),
            ),
        )
        target_enemy_distance = _nearest_enemy_distance(target, enemies)
        if target_enemy_distance >= source_enemy_distance:
            continue
        ships = max(MIN_SHIPS_TO_LAUNCH, min(available // (2 if ffa else 3), available - keep_after_send))
        target_xy = _predict_target_xy(obs, source_xy, target, ships)
        angle = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
        moves.append([source_id, float(angle), int(ships)])
        launched_by_source[source_id] = launched_by_source.get(source_id, 0) + int(ships)
        break
    return moves


def _expiring_comet_evacuation_moves(obs, own, action, launched_by_source, max_moves_left):
    if max_moves_left <= 0 or len(own) < 2:
        return []
    durable_own = [planet for planet in own if not _is_comet(obs, planet)]
    if not durable_own:
        return []

    moves = []
    for source in sorted(own, key=lambda planet: (_comet_turns_remaining(obs, planet) or 999, -_planet_ships(planet))):
        if len(moves) >= max_moves_left:
            break
        source_life = _comet_turns_remaining(obs, source)
        if source_life is None or source_life > 12:
            continue
        source_id = _planet_id(source)
        available = _planet_ships(source) - 1 - launched_by_source.get(source_id, 0)
        if available < MIN_SHIPS_TO_LAUNCH:
            continue
        source_xy = (_planet_x(source), _planet_y(source))
        target = min(
            durable_own,
            key=lambda planet: (
                _distance(source_xy, (_planet_x(planet), _planet_y(planet))),
                -_planet_production(planet),
                -_planet_ships(planet),
            ),
        )
        ships = int(available)
        target_xy = _predict_target_xy(obs, source_xy, target, ships)
        angle = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
        moves.append([source_id, float(angle), ships])
        launched_by_source[source_id] = launched_by_source.get(source_id, 0) + ships
    return moves


def encode(obs):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player]
    enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if _planet_owner(planet) == -1]
    enemy_owners = sorted({_planet_owner(planet) for planet in enemies})

    owner_totals = {}
    owner_prod = {}
    owner_fleet_ships = {}
    for planet in planets:
        owner = _planet_owner(planet)
        owner_totals[owner] = owner_totals.get(owner, 0) + _planet_ships(planet)
        owner_prod[owner] = owner_prod.get(owner, 0) + _planet_production(planet)
    for fleet in obs.get("fleets", []):
        owner = _fleet_owner(fleet)
        owner_fleet_ships[owner] = owner_fleet_ships.get(owner, 0) + _fleet_ships(fleet)

    leader_owner = None
    if enemies:
        leader_owner = max(
            sorted({_planet_owner(planet) for planet in enemies}),
            key=lambda owner: (owner_prod.get(owner, 0), owner_totals.get(owner, 0)),
        )
    profile = _update_opponent_profile(obs, player, planets, leader_owner)
    enemy_planet_ships = sum(_planet_ships(planet) for planet in enemies)
    enemy_fleet_ships = sum(owner_fleet_ships.get(owner, 0) for owner in enemy_owners)
    enemy_total_ships = enemy_planet_ships + enemy_fleet_ships

    return {
        "player": player,
        "step": int(obs.get("step", obs.get("turn", 0))),
        "own_count": len(own),
        "enemy_count": len(enemies),
        "enemy_players": len(enemy_owners),
        "neutral_count": len(neutrals),
        "own_ships": sum(_planet_ships(planet) for planet in own),
        "enemy_ships": enemy_planet_ships,
        "own_fleet_ships": owner_fleet_ships.get(player, 0),
        "enemy_fleet_ships": enemy_fleet_ships,
        "enemy_fleet_ratio": enemy_fleet_ships / max(1, enemy_total_ships),
        "own_prod": sum(_planet_production(planet) for planet in own),
        "enemy_prod": sum(_planet_production(planet) for planet in enemies),
        "leader_owner": leader_owner,
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        **profile,
    }


def policy_forward(features):
    ffa = features["enemy_players"] >= 2
    pressure = features["enemy_ships"] >= max(features["own_ships"] - 4, 1)
    fleet_pressure = features.get("enemy_fleet_ratio", 0.0) >= 0.70 and (
        features.get("to_me_ratio", 0.0) >= 0.95 and features.get("enemy_fleet_ships", 0) >= 0.85 * max(1, features["own_ships"])
    )
    pressure = pressure or fleet_pressure
    behind_on_econ = features["enemy_prod"] > features["own_prod"]
    neutrals_open = features["neutral_count"] > 0
    production_ratio = features["own_prod"] / max(1, features["enemy_prod"])
    adaptive_opening_expand = (
        15 <= features["step"] <= 80
        and neutrals_open
        and not pressure
        and production_ratio < 1.0
    )
    expand = neutrals_open and (
        features["own_count"] <= 3
        or ffa
        or adaptive_opening_expand
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
        "own_count": int(features["own_count"]),
        "neutral_count": int(features["neutral_count"]),
        "fsm_state": state,
        "recent_enemy_captures": set(features.get("recent_enemy_captures", set())),
        "profile_total": float(features.get("profile_total", 0.0)),
        "production_ratio": float(production_ratio),
        "adaptive_opening_expand": bool(adaptive_opening_expand),
        "enemy_fleet_ratio": float(features.get("enemy_fleet_ratio", 0.0)),
        "fleet_pressure": bool(fleet_pressure),
        "to_neutral_ratio": float(features.get("to_neutral_ratio", 0.0)),
        "to_me_ratio": float(features.get("to_me_ratio", 0.0)),
        "to_leader_ratio": float(features.get("to_leader_ratio", 0.0)),
        "enemy_overextended": bool(
            features.get("enemy_fleet_ships", 0) > 1.15 * max(1, features.get("enemy_ships", 0))
            and features.get("to_neutral_ratio", 0.0) >= 0.80
        ),
    }


def _required_ships(obs, source, target, committed, action):
    player = int(obs.get("player", 0))
    projection_cache = action.setdefault("_projection_cache", {})
    source_xy = (_planet_x(source), _planet_y(source))
    estimate = max(MIN_SHIPS_TO_LAUNCH, _planet_ships(target) + MIN_CAPTURE_MARGIN)
    projected_owner = _planet_owner(target)
    projected_ships = _planet_ships(target)
    travel_steps = 1
    target_xy = _predict_target_xy(obs, source_xy, target, estimate)
    for _ in range(3):
        target_xy = _predict_target_xy(obs, source_xy, target, estimate)
        travel_steps = max(1, ceil(_distance(source_xy, target_xy) / _fleet_speed(estimate)))
        projected_owner, projected_ships = _project_planet_state(
            obs,
            target,
            travel_steps,
            cache=projection_cache,
        )
        current_owner = _planet_owner(target)
        if projected_owner == player and current_owner == player:
            need = MIN_SHIPS_TO_LAUNCH
        else:
            if projected_owner == player:
                need = _planet_ships(target) + MIN_CAPTURE_MARGIN - int(committed)
            else:
                need = max(_planet_ships(target), projected_ships) + MIN_CAPTURE_MARGIN - int(committed)
            if projected_owner == action.get("leader_owner"):
                need += 1
        next_estimate = max(MIN_SHIPS_TO_LAUNCH, int(need))
        if abs(next_estimate - estimate) <= 1:
            estimate = next_estimate
            break
        estimate = next_estimate
    return max(MIN_SHIPS_TO_LAUNCH, int(estimate)), target_xy, travel_steps, projected_owner, projected_ships


def _target_value(obs, source, target, committed, action, own, enemies):
    player = int(obs.get("player", 0))
    projection_cache = action.setdefault("_projection_cache", {})
    step = int(obs.get("step", obs.get("turn", 0)))
    required, target_xy, travel_steps, projected_owner, projected_ships = _required_ships(obs, source, target, committed, action)
    source_xy = (_planet_x(source), _planet_y(source))
    distance = _distance(source_xy, target_xy)
    owner = _planet_owner(target)
    production = _planet_production(target)
    ships = _planet_ships(target)
    ffa = bool(action.get("ffa"))
    game_turns_remaining = max(0, MAX_GAME_TURNS - step - travel_steps)
    comet_turns_remaining = _comet_turns_remaining(obs, target)
    if comet_turns_remaining is not None and comet_turns_remaining <= 1:
        return -999.0, required, target_xy
    turns_remaining = game_turns_remaining
    own_incoming = _incoming_ships_to_target(obs, target, player, projection_cache)

    after_owner, after_ships = _project_planet_state(
        obs,
        target,
        travel_steps,
        extra_arrivals=[(travel_steps, player, int(committed) + int(required))],
        cache=projection_cache,
    )

    own_proximity = min(
        (_distance((_planet_x(planet), _planet_y(planet)), target_xy) for planet in own if _planet_id(planet) != _planet_id(source)),
        default=distance,
    )
    enemy_proximity = min(
        (_distance((_planet_x(planet), _planet_y(planet)), target_xy) for planet in enemies),
        default=distance,
    )

    projected_gain = 0.0
    if projected_owner != player and after_owner == player:
        projected_gain += after_ships + production * min(40, turns_remaining)
    elif owner != player and projected_owner == player and after_owner == player:
        projected_gain += 0.45 * after_ships + production * min(24, turns_remaining)
    elif projected_owner == player and after_owner == player:
        projected_gain += 0.35 * required
    elif after_owner not in (-1, player):
        projected_gain -= 12.0

    if owner == -1 and production * max(1, turns_remaining) <= required:
        return -999.0, required, target_xy
    value = projected_gain + production * (8.0 if owner == -1 else 11.0)
    if owner == -1:
        value += 8.0
        if ffa:
            value += 4.0
        if not ffa and action.get("fsm_state") == "OPENING_EXPAND" and step <= 10:
            safe_opening_neutral = production >= 4 and distance <= 30.0 and enemy_proximity >= distance * 0.9
            if safe_opening_neutral:
                value += 36.0 + 6.0 * production
        if action.get("fsm_state") == "OPENING_EXPAND" and own_incoming > 0 and ships >= 12:
            value += min(48.0, 1.4 * own_incoming + 8.0 * production)
            overcommit = own_incoming - 2.2 * max(1, ships + MIN_CAPTURE_MARGIN)
            if overcommit > 0:
                value -= 1.6 * overcommit
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
        if action.get("enemy_overextended") and owner not in (-1, player):
            value += 18.0 + 8.0 * production - 0.18 * ships
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
    future_penalty = 0.16 * max(0, projected_ships - ships)
    return value + 24.0 * roi - distance_penalty - 0.22 * ships - future_penalty, required, target_xy


def _opponent_response_penalty(obs, source, target, ships, target_xy, action, enemies, remaining_after_launch):
    if action.get("ffa") or _planet_owner(target) == int(obs.get("player", 0)):
        return 0.0
    source_production = _planet_production(source)
    if source_production <= 1 and action.get("own_count", 0) > 3:
        return 0.0

    source_xy = (_planet_x(source), _planet_y(source))
    my_travel = max(1, ceil(_distance(source_xy, target_xy) / _fleet_speed(ships)))
    best_penalty = 0.0
    for enemy in enemies:
        enemy_ships = _planet_ships(enemy)
        enemy_attack = max(0, enemy_ships - RESERVE_HOME_SHIPS)
        if enemy_attack < MIN_SHIPS_TO_LAUNCH:
            continue
        enemy_xy = (_planet_x(enemy), _planet_y(enemy))
        enemy_eta = max(1, ceil(_distance(enemy_xy, source_xy) / _fleet_speed(enemy_attack)))
        if enemy_eta > my_travel + 8:
            continue
        source_defense = remaining_after_launch + source_production * min(enemy_eta, 12)
        exposure = enemy_attack - source_defense - MIN_CAPTURE_MARGIN
        if exposure <= 0:
            continue
        penalty = 10.0 + 4.0 * source_production + 0.45 * exposure + 0.8 * max(0, my_travel - enemy_eta)
        best_penalty = max(best_penalty, penalty)
    return best_penalty


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
    moves.extend(_expiring_comet_evacuation_moves(obs, own, action, launched_by_source, max_moves - len(moves)))
    sources = sorted(
        own,
        key=lambda planet: _source_priority(planet, len(own), enemies, action, obs, player),
        reverse=True,
    )

    for source in sources:
        if len(moves) >= max_moves:
            break
        source_id = _planet_id(source)
        source_life = _comet_turns_remaining(obs, source)
        if source_life is not None and source_life <= 1:
            continue
        reserve = _reserve_for_source(source, len(own), enemies, action, obs, player)
        outgoing_count = outgoing_by_source.get(source_id, [0, 0])[0]
        step = int(obs.get("step", obs.get("turn", 0)))
        if step > FSM_OPENING_TURNS and outgoing_count >= 3 and (action.get("pressure") or action.get("behind_on_econ") or len(own) >= 3):
            reserve += min(18, 3 * (outgoing_count - 2))
        available = _planet_ships(source) - reserve - launched_by_source.get(source_id, 0)
        if available < MIN_SHIPS_TO_LAUNCH:
            continue
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
            remaining_after_launch = _planet_ships(source) - launched_by_source.get(source_id, 0) - ships
            score -= _opponent_response_penalty(obs, source, target, ships, target_xy, action, enemies, remaining_after_launch)
            if best is None or score > best["score"]:
                best = {
                    "target_id": target_id,
                    "target_owner": _planet_owner(target),
                    "ships": ships,
                    "score": score,
                    "target_xy": target_xy,
                    "source_xy": source_xy,
                }

        if best is None or (best["target_owner"] == -1 and best["score"] <= 0.0):
            continue

        angle = _sun_safe_angle(best["source_xy"], best["target_xy"], _angle(best["source_xy"], best["target_xy"]))
        moves.append([source_id, float(angle), int(best["ships"])])
        launched_by_source[source_id] = launched_by_source.get(source_id, 0) + int(best["ships"])
        committed_by_target[best["target_id"]] = committed_by_target.get(best["target_id"], 0) + int(best["ships"])
        used_targets.add(best["target_id"])

    if len(moves) < max_moves:
        moves.extend(
            _frontier_reinforcement_moves(
                obs,
                own,
                enemies,
                action,
                launched_by_source,
                max_moves - len(moves),
            )
        )

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
