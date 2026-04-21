# Kaggle Orbit Wars submission template.
# Generated agent is self-contained and does not depend on the local Rust stack.

from math import atan2, cos, floor, hypot, isfinite, pi, sin

FRACTIONS = (0.10, 0.25, 0.50, 0.75)
ANGLE_OFFSETS = (-0.261799, -0.130899, 0.0, 0.130899, 0.261799)
SUN_RADIUS = 10.0
CENTER = 50.0
RESERVE_HOME_SHIPS = 8
MIN_SHIPS_TO_LAUNCH = 2
MAX_MOVES_PER_TURN = 8


def _angle(a, b):
    return atan2(b[1] - a[1], b[0] - a[0])


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


def encode(obs):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    own = [p for p in planets if int(p[1]) == player]
    enemies = [p for p in planets if int(p[1]) not in (-1, player)]
    neutrals = [p for p in planets if int(p[1]) == -1]
    return {
        "player": player,
        "planets": planets,
        "own_count": len(own),
        "enemy_count": len(enemies),
        "neutral_count": len(neutrals),
        "own_ships": sum(int(p[5]) for p in own),
        "enemy_ships": sum(int(p[5]) for p in enemies),
        "own_prod": sum(int(p[6]) for p in own),
        "enemy_prod": sum(int(p[6]) for p in enemies),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
    }


def policy_forward(features):
    source_rank = 0
    target_rank = 0
    offset_idx = 2

    if features["enemy_prod"] > features["own_prod"] or features["enemy_ships"] >= max(features["own_ships"] - 4, 1):
        fraction_idx = 3
    elif features["neutral_count"] > 0 and features["own_count"] <= 2:
        fraction_idx = 2
    else:
        fraction_idx = 1

    if features["angular_velocity"] > 0.04:
        offset_idx = 1
    elif features["angular_velocity"] < -0.04:
        offset_idx = 3

    return [source_rank, target_rank, fraction_idx, offset_idx]


def decode(action, obs):
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    own = [p for p in planets if int(p[1]) == player and int(p[5]) >= MIN_SHIPS_TO_LAUNCH]
    if not own:
        return []

    own.sort(key=lambda p: (p[5], p[6]), reverse=True)
    source_rank, target_rank, fraction_idx, offset_idx = [int(x) for x in action[:4]]
    src = own[source_rank % len(own)]

    candidates = [p for p in planets if int(p[0]) != int(src[0])]
    if not candidates:
        return []

    sx, sy = float(src[2]), float(src[3])

    def target_score(planet):
        tx, ty = float(planet[2]), float(planet[3])
        dist = hypot(tx - sx, ty - sy)
        owner = int(planet[1])
        enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
        neutral_bonus = 4.0 if owner == -1 else 0.0
        return float(planet[6]) * 10.0 + enemy_bonus + neutral_bonus - 0.15 * dist - 0.12 * float(planet[5])

    candidates.sort(key=target_score, reverse=True)
    target = candidates[target_rank % len(candidates)]

    frac = FRACTIONS[fraction_idx % len(FRACTIONS)]
    ships = int(max(0, floor(float(src[5]) * frac)))
    if int(src[5]) - ships < RESERVE_HOME_SHIPS and len(own) <= 2:
        ships = max(0, int(src[5]) - RESERVE_HOME_SHIPS)
    if ships <= 0:
        return []

    source_xy = (sx, sy)
    target_xy = (float(target[2]), float(target[3]))
    base = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
    angle = base + ANGLE_OFFSETS[offset_idx % len(ANGLE_OFFSETS)]
    return [[int(src[0]), float(angle), int(ships)]]


def _moves_are_legal(obs, player, moves):
    own_ids = {int(p[0]) for p in obs.get("planets", []) if int(p[1]) == player}
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
        own = [p for p in planets if int(p[1]) == player and int(p[5]) >= MIN_SHIPS_TO_LAUNCH]
        if not own:
            return []
        own.sort(key=lambda p: (p[5], p[6]), reverse=True)
        src = own[0]
        candidates = [p for p in planets if int(p[0]) != int(src[0])]
        if not candidates:
            return []

        sx, sy = float(src[2]), float(src[3])

        def target_score(planet):
            tx, ty = float(planet[2]), float(planet[3])
            dist = hypot(tx - sx, ty - sy)
            owner = int(planet[1])
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            return float(planet[6]) * 10.0 + enemy_bonus + neutral_bonus - 0.15 * dist - 0.12 * float(planet[5])

        target = max(candidates, key=target_score)
        ships = max(MIN_SHIPS_TO_LAUNCH, min(int(src[5]) - RESERVE_HOME_SHIPS, int(floor(float(src[5]) * 0.25))))
        if ships <= 0:
            return []
        source_xy = (sx, sy)
        target_xy = (float(target[2]), float(target[3]))
        angle = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
        moves = [[int(src[0]), float(angle), int(ships)]]
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
