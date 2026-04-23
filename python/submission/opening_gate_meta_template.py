# Kaggle Orbit Wars opening-gate meta submission template.
# Generated agent is self-contained and mirrors the local heuristic stack.

from math import atan2, ceil, cos, hypot, isfinite, log, pi, sin

BOARD_CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0
MAX_FIELD_CONTROL_MOVES = 4
MAX_GREEDY_MOVES = 8
RESERVE_HOME_SHIPS = 8
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


def _fleet_owner(fleet):
    return int(_field(fleet, 1, "owner"))


def _fleet_x(fleet):
    return float(_field(fleet, 2, "x"))


def _fleet_y(fleet):
    return float(_field(fleet, 3, "y"))


def _fleet_ships(fleet):
    return int(_field(fleet, 6, "ships"))


def _distance(a, b):
    return hypot(b[0] - a[0], b[1] - a[1])


def _fleet_speed(ships):
    scale = log(max(int(ships), 1)) / log(1000.0)
    speed = 1.0 + (SHIP_SPEED - 1.0) * scale**1.5
    return min(SHIP_SPEED, max(1.0, speed))


def _rotate_about_center(point, angle):
    dx = point[0] - BOARD_CENTER
    dy = point[1] - BOARD_CENTER
    c = cos(angle)
    s = sin(angle)
    return (
        BOARD_CENTER + dx * c - dy * s,
        BOARD_CENTER + dx * s + dy * c,
    )


def _is_rotating_planet(planet):
    orbital_radius = hypot(_planet_x(planet) - BOARD_CENTER, _planet_y(planet) - BOARD_CENTER)
    return orbital_radius + _planet_radius(planet) < ROTATION_RADIUS_LIMIT


def _predict_target_xy(state, source_xy, target, ships):
    target_xy = (_planet_x(target), _planet_y(target))
    if not _is_rotating_planet(target):
        return target_xy
    distance = _distance(source_xy, target_xy)
    travel_steps = max(1, ceil(distance / _fleet_speed(ships)))
    angular_velocity = float(state.get("angular_velocity", 0.0))
    return _rotate_about_center(target_xy, angular_velocity * travel_steps)


def _point_to_segment_distance(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    l2 = vx * vx + vy * vy
    if l2 == 0.0:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / l2))
    projection = (start[0] + t * vx, start[1] + t * vy)
    return _distance(point, projection)


def _sun_safe_angle(source_xy, target_xy):
    base_angle = atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])
    if _point_to_segment_distance((BOARD_CENTER, BOARD_CENTER), source_xy, target_xy) >= SUN_RADIUS + 1.0:
        return base_angle
    to_center = atan2(BOARD_CENTER - source_xy[1], BOARD_CENTER - source_xy[0])
    candidates = [to_center + pi / 2.0, to_center - pi / 2.0]
    return min(candidates, key=lambda angle: abs(atan2(sin(angle - base_angle), cos(angle - base_angle))))


def _planet_pressure(planet, neighbors):
    px, py = _planet_x(planet), _planet_y(planet)
    pressure = 0.0
    for other in neighbors:
        dist = max(4.0, _distance((px, py), (_planet_x(other), _planet_y(other))))
        mass = _planet_ships(other) + 5.0 * _planet_production(other)
        pressure += mass / (dist**1.15)
    return pressure


def _fleet_pressure(planet, fleets):
    px, py = _planet_x(planet), _planet_y(planet)
    pressure = 0.0
    for fleet in fleets:
        dist = max(4.0, _distance((px, py), (_fleet_x(fleet), _fleet_y(fleet))))
        pressure += _fleet_ships(fleet) / (dist**1.2)
    return pressure


def _frontline_bias(planet, enemies, own):
    px, py = _planet_x(planet), _planet_y(planet)
    enemy_dist = min((_distance((px, py), (_planet_x(enemy), _planet_y(enemy))) for enemy in enemies), default=80.0)
    own_dist = min(
        (
            _distance((px, py), (_planet_x(friend), _planet_y(friend)))
            for friend in own
            if _planet_id(friend) != _planet_id(planet)
        ),
        default=80.0,
    )
    return (own_dist - enemy_dist) * 0.08


def _target_value(state, target, player, own, enemies, friendly_fleets, enemy_fleets):
    owner = _planet_owner(target)
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(target, friendly_fleets)
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    production_value = 8.0 * _planet_production(target)
    ship_penalty = 1.15 * _planet_ships(target)
    centrality = 3.0 * (1.0 - min(1.0, _distance((_planet_x(target), _planet_y(target)), (BOARD_CENTER, BOARD_CENTER)) / 60.0))
    frontier = _frontline_bias(target, enemies, own)
    if owner == -1:
        denial = max(0.0, enemy_pressure - friendly_pressure) * 0.7
        return production_value - ship_penalty + 0.35 * friendly_pressure - 0.2 * enemy_pressure + centrality + frontier + denial
    if owner == player:
        threat = max(0.0, enemy_pressure - friendly_pressure - 0.35 * _planet_ships(target))
        return 11.0 * threat + 4.5 * _planet_production(target) + frontier
    vulnerability = max(0.0, friendly_pressure - enemy_pressure)
    return production_value + 6.0 + 0.8 * vulnerability - ship_penalty + centrality + frontier


def _source_reserve(source, enemies, enemy_fleets):
    local_threat = _planet_pressure(source, enemies) + 0.8 * _fleet_pressure(source, enemy_fleets)
    frontier_tax = max(0.0, _frontline_bias(source, enemies, [source]))
    reserve = 8 + int(0.45 * _planet_production(source) + 0.35 * local_threat + frontier_tax)
    return max(8, min(24, reserve))


def _coalition_source_reserve(source, enemies, enemy_fleets):
    local_threat = _planet_pressure(source, enemies) + 0.9 * _fleet_pressure(source, enemy_fleets)
    frontier_tax = max(0.0, _frontline_bias(source, enemies, [source]))
    reserve = 12 + int(0.55 * _planet_production(source) + 0.45 * local_threat + 1.25 * frontier_tax)
    return max(12, min(30, reserve))


def _leader_owner(enemies):
    owner_strength = {}
    for enemy in enemies:
        owner = _planet_owner(enemy)
        owner_strength[owner] = owner_strength.get(owner, 0.0) + _planet_ships(enemy) + 6.0 * _planet_production(enemy)
    if not owner_strength:
        return None
    return max(owner_strength, key=owner_strength.get)


def _nearest_neutral(source, neutrals):
    if not neutrals:
        return None
    source_xy = (_planet_x(source), _planet_y(source))
    return min(neutrals, key=lambda planet: _distance(source_xy, (_planet_x(planet), _planet_y(planet))))


def _required_commitment(state, source, target, player, own, enemies, friendly_fleets, enemy_fleets, committed):
    source_xy = (_planet_x(source), _planet_y(source))
    guess_ships = max(6, int(max(_planet_ships(source) - 8, 0) * 0.55))
    target_xy = _predict_target_xy(state, source_xy, target, guess_ships)
    travel_steps = max(1, ceil(_distance(source_xy, target_xy) / _fleet_speed(max(guess_ships, 2))))
    friendly_pressure = _planet_pressure(target, own) + 0.6 * _fleet_pressure(target, friendly_fleets)
    enemy_pressure = _planet_pressure(target, enemies) + 0.7 * _fleet_pressure(target, enemy_fleets)
    owner = _planet_owner(target)
    if owner == player:
        vulnerability = max(0.0, enemy_pressure - friendly_pressure - 0.3 * _planet_ships(target))
        return max(0, int(ceil(vulnerability + 2.0)) - committed)
    growth = max(0.0, float(_planet_production(target)) * travel_steps)
    if owner == -1:
        need = _planet_ships(target) + 1 + 0.2 * growth + 0.15 * max(0.0, enemy_pressure - friendly_pressure)
    else:
        need = _planet_ships(target) + 1 + 0.55 * growth + 0.4 * max(0.0, enemy_pressure - friendly_pressure)
    return max(0, int(ceil(need)) - committed)


def _build_move(state, source, target, ships):
    if ships < MIN_SHIPS_TO_LAUNCH:
        return None
    source_xy = (_planet_x(source), _planet_y(source))
    target_xy = _predict_target_xy(state, source_xy, target, ships)
    angle = _sun_safe_angle(source_xy, target_xy)
    return [_planet_id(source), float(angle), int(ships)]


def _moves_are_legal(state, player, moves):
    own_ids = {_planet_id(planet) for planet in state.get("planets", []) if _planet_owner(planet) == player}
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


def fallback_greedy(state):
    try:
        player = int(state.get("player", 0))
        planets = state.get("planets", [])
        own = [
            planet
            for planet in planets
            if _planet_owner(planet) == player and _planet_ships(planet) > RESERVE_HOME_SHIPS + MIN_SHIPS_TO_LAUNCH
        ]
        targets = [planet for planet in planets if _planet_owner(planet) != player]
        moves = []
        for source in sorted(own, key=_planet_ships, reverse=True):
            if len(moves) >= MAX_GREEDY_MOVES:
                break
            sx, sy = _planet_x(source), _planet_y(source)
            ranked = sorted(
                targets,
                key=lambda planet: (
                    (float(_planet_ships(planet)) + 1.0) / (float(_planet_production(planet)) + 1.0)
                    + 0.05 * hypot(_planet_x(planet) - sx, _planet_y(planet) - sy)
                ),
            )
            if not ranked:
                continue
            target = ranked[0]
            ships = max(0, _planet_ships(source) - RESERVE_HOME_SHIPS)
            if ships <= 0:
                continue
            move = _build_move(state, source, target, ships)
            if move is not None:
                moves.append(move)
        return moves if _moves_are_legal(state, player, moves) else []
    except Exception:
        return []


def _defensive_policy(state, player, own, neutrals):
    if not own or not neutrals:
        return []
    moves = []
    for source in sorted(own, key=_planet_ships, reverse=True):
        reserve = 18 if len(own) <= 2 else 10
        ships = _planet_ships(source) - reserve
        if ships <= 0:
            continue
        sx, sy = _planet_x(source), _planet_y(source)
        target = min(neutrals, key=lambda planet: float(_planet_ships(planet)) + 0.08 * hypot(_planet_x(planet) - sx, _planet_y(planet) - sy))
        move = _build_move(state, source, target, int(ships * 0.5))
        if move is not None:
            moves.append(move)
        if len(moves) >= 4:
            break
    return moves


def _field_control_policy(state, player, coalition_mode=False):
    planets = state.get("planets", [])
    fleets = state.get("fleets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player]
    enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if _planet_owner(planet) == -1]
    if not own:
        return []
    if coalition_mode and len({_planet_owner(enemy) for enemy in enemies}) < 2:
        return _field_control_policy(state, player, coalition_mode=False)

    friendly_fleets = [fleet for fleet in fleets if _fleet_owner(fleet) == player]
    enemy_fleets = [fleet for fleet in fleets if _fleet_owner(fleet) not in (-1, player)]
    own_ships = sum(_planet_ships(planet) for planet in own)
    enemy_ships = sum(_planet_ships(planet) for planet in enemies)
    own_prod = sum(_planet_production(planet) for planet in own)
    enemy_prod = sum(_planet_production(planet) for planet in enemies)
    leader_owner = _leader_owner(enemies) if coalition_mode else None

    threatened = sorted(
        (planet for planet in own if _planet_ships(planet) > 0),
        key=lambda planet: _target_value(state, planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )
    neutral_targets = sorted(
        neutrals,
        key=lambda planet: _target_value(state, planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )
    enemy_targets = sorted(
        enemies,
        key=lambda planet: _target_value(state, planet, player, own, enemies, friendly_fleets, enemy_fleets),
        reverse=True,
    )

    if coalition_mode:
        leader_targets = [planet for planet in enemy_targets if _planet_owner(planet) == leader_owner]
        other_enemy_targets = [planet for planet in enemy_targets if _planet_owner(planet) != leader_owner]
        top_threat = _target_value(state, threatened[0], player, own, enemies, friendly_fleets, enemy_fleets) if threatened else 0.0
        want_expand = bool(neutral_targets) and (own_prod <= enemy_prod or own_ships <= enemy_ships + 12 or len(own) < 4)
        if top_threat > 8.0:
            target_priority = threatened + leader_targets[:2] + neutral_targets[:2] + other_enemy_targets[:1]
        elif want_expand:
            target_priority = neutral_targets + threatened[:2] + leader_targets[:2] + other_enemy_targets[:1]
        else:
            target_priority = leader_targets + threatened[:2] + neutral_targets[:2] + other_enemy_targets[:1]
    else:
        top_threat = _target_value(state, threatened[0], player, own, enemies, friendly_fleets, enemy_fleets) if threatened else 0.0
        want_expand = bool(neutral_targets) and (
            own_prod <= enemy_prod + 1 or own_ships <= enemy_ships + 10 or len(own) < max(3, len(enemies))
        )
        if top_threat > 8.0:
            target_priority = threatened + neutral_targets[:2] + enemy_targets[:2]
        elif want_expand:
            target_priority = neutral_targets + threatened[:2] + enemy_targets[:2]
        else:
            target_priority = enemy_targets + neutral_targets[:2] + threatened[:2]

    if not target_priority:
        if coalition_mode:
            return _field_control_policy(state, player, coalition_mode=False)
        return fallback_greedy(state)

    moves = []
    committed = {}
    reserve_fn = _coalition_source_reserve if coalition_mode else _source_reserve
    max_moves = 3 if coalition_mode else MAX_FIELD_CONTROL_MOVES
    sources = sorted(
        own,
        key=lambda planet: (
            _planet_ships(planet) - reserve_fn(planet, enemies, enemy_fleets),
            _planet_production(planet),
        ),
        reverse=True,
    )

    for source in sources:
        if len(moves) >= max_moves:
            break
        reserve = reserve_fn(source, enemies, enemy_fleets)
        surplus = _planet_ships(source) - reserve
        if surplus < 2:
            continue

        best_target = None
        best_score = -1e9
        best_need = 0
        for target in target_priority:
            if _planet_id(target) == _planet_id(source):
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
                committed.get(_planet_id(target), 0),
            )
            strategic = _target_value(state, target, player, own, enemies, friendly_fleets, enemy_fleets)
            if coalition_mode and _planet_owner(target) == leader_owner:
                strategic += 4.0
            elif coalition_mode and _planet_owner(target) not in (-1, player):
                strategic -= 4.0
            source_xy = (_planet_x(source), _planet_y(source))
            target_xy = _predict_target_xy(state, source_xy, target, max(2, min(surplus, max(need, 6))))
            dist_penalty = (0.15 if coalition_mode else 0.12) * _distance(source_xy, target_xy)
            score = strategic - dist_penalty - ((0.35 if coalition_mode else 0.25) * max(0, need - surplus))
            if score > best_score:
                best_score = score
                best_target = target
                best_need = need

        if best_target is None:
            continue

        owner = _planet_owner(best_target)
        if owner == player:
            ships = min(surplus, max(2, best_need))
        elif owner == -1:
            ships = min(surplus, max(2, int(ceil(max(best_need, surplus * (0.38 if coalition_mode else 0.45))))))
        elif coalition_mode and owner == leader_owner:
            ships = min(surplus, max(2, int(ceil(max(best_need * 1.05, surplus * 0.48)))))
        elif coalition_mode:
            ships = min(surplus, max(2, int(ceil(max(best_need, surplus * 0.32)))))
        else:
            ships = min(surplus, max(2, int(ceil(max(best_need * 1.1, surplus * 0.6)))))
        if ships <= 0:
            continue

        move = _build_move(state, source, best_target, ships)
        if move is None:
            continue
        moves.append(move)
        committed[_planet_id(best_target)] = committed.get(_planet_id(best_target), 0) + ships

    if moves:
        return moves
    if coalition_mode:
        return _field_control_policy(state, player, coalition_mode=False)
    return fallback_greedy(state)


def _opening_gate_meta_agent(state, player):
    planets = state.get("planets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player]
    enemies = [planet for planet in planets if _planet_owner(planet) not in (-1, player)]
    neutrals = [planet for planet in planets if _planet_owner(planet) == -1]
    if not own:
        return []

    if len({_planet_owner(enemy) for enemy in enemies}) >= 2:
        return _field_control_policy(state, player, coalition_mode=True)

    nearest_neutral = _nearest_neutral(own[0], neutrals)
    angular_velocity = float(state.get("angular_velocity", 0.0))
    if (
        nearest_neutral is not None
        and _planet_ships(nearest_neutral) >= 12
        and _planet_production(nearest_neutral) <= 1
        and angular_velocity > 0.037
    ):
        return _defensive_policy(state, player, own, neutrals)
    return _field_control_policy(state, player, coalition_mode=True)


def agent(obs):
    try:
        player = int(obs.get("player", 0))
        moves = _opening_gate_meta_agent(obs, player)
        if not _moves_are_legal(obs, player, moves):
            raise ValueError(f"submission policy produced illegal moves for player={player}: {moves!r}")
        return list(moves)
    except Exception:
        return fallback_greedy(obs)
