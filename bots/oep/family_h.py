"""Family H — candidate generators (H2-H5).

Diverse heuristic plans, each an ``obs -> moves`` function (Kaggle contract:
``[] == launch nothing``). They are the "different weapons on the table" the
oracle (H1) measures: the saturated line only ever perturbed the greedy plan, so
these aim for *structurally different* plans, not threshold tweaks.

* H2 ``production_projected_attack`` — capture by ETA-projected economics.
* H3 ``timeline_risk``             — H2 minus comet/overkill/source-drain risk.
* H4 ``hammer_multiprong``         — synchronized multi-source pressure.
* H5 ``regroup_dominance``         — reinforce the frontier; bias the 4p leader.

All emit legal ``[from_planet_id, direction_angle, num_ships]`` moves (source is
owned, angle finite, ships int > 0). Aim is at the target's current position —
good enough for the oracle gate (legality + signal), not a parity claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from python.orbit_wars_gym.entities import (
    fleet_angle,
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

from bots.oep.engine_aim import Aimer
from bots.oep.geometry import orbital_intercept

Obs = dict[str, Any]
Move = list[float]
Moves = list[Move]

RESERVE = 1  # ships kept home on a source planet
CAPTURE_MARGIN = 1.0  # extra ships required over projected defense
VALUE_HORIZON = 60.0  # steps over which captured production is valued

# Defensive threat model (H6 ``defensive_reinforce``). These are PROXY weights for
# *prioritisation* only — every launch is still aimed by the engine-exact ``Aimer``.
# The proxy is the surface Lever A's metaheuristic tuner calibrates; it is NOT an
# engine-parity claim about enemy-fleet targets (those are not in the obs).
THREAT_CONE = 0.3  # cos(bearing - fleet_angle) above this => fleet "bearing toward" P
THREAT_DIST_SCALE = 30.0  # distance falloff for incoming pressure
PLANET_PRESSURE = 0.5  # weight of a static enemy planet's garrison as latent threat
RECAPTURE_RANGE = 35.0  # only retake enemy planets this close to my territory


@dataclass
class _Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


def _planets(obs: Obs) -> list[_Planet]:
    return [
        _Planet(
            planet_id(p),
            planet_owner(p),
            planet_x(p),
            planet_y(p),
            planet_radius(p),
            planet_ships(p),
            planet_production(p),
        )
        for p in obs.get("planets", [])
    ]


@dataclass
class _Fleet:
    owner: int
    x: float
    y: float
    angle: float
    ships: int


def _fleets(obs: Obs) -> list[_Fleet]:
    return [
        _Fleet(
            fleet_owner(f),
            fleet_x(f),
            fleet_y(f),
            fleet_angle(f),
            fleet_ships(f),
        )
        for f in obs.get("fleets", [])
    ]


def _player_count(planets: list[_Planet]) -> int:
    return len({p.owner for p in planets if p.owner >= 0})


def _dist(a: _Planet, b: _Planet) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _aim(source: _Planet, target: _Planet, angvel: float) -> tuple[float, float]:
    """Orbital interception: angle + eta to hit a target that ORBITS the sun.

    The target orbits ``CENTER`` per the engine model
    ``pos(t) = CENTER + orb_r * (cos, sin)(a0 + angvel * t)``
    (orbit_lite.movement). Aiming at the current position (static geometry)
    plays a false game — the planet has moved by arrival. A cheap fixed-point on
    ``t = ||pos(t) - source|| / ship_speed`` converges fast and feeds the real
    eta into defense/production/comet/hammer reasoning.
    """

    return orbital_intercept(source.x, source.y, target.x, target.y, angvel)


def _projected_defense(target: _Planet, eta: float) -> float:
    """Defenders the target will have when our fleet arrives (conservative)."""

    growth = max(0, target.production) * eta if target.owner != -1 else 0.0
    return float(target.ships) + growth


def _move(source: _Planet, angle: float, ships: int) -> Move:
    return [float(source.id), float(angle), float(int(ships))]


def _split_planets(obs: Obs) -> tuple[int, list[_Planet], list[_Planet]]:
    me = int(obs.get("player", 0))
    planets = _planets(obs)
    own = [p for p in planets if p.owner == me]
    others = [p for p in planets if p.owner != me]
    return me, own, others


# --------------------------------------------------------------------------- H2
def production_projected_attack(obs: Obs) -> Moves:
    me, own, targets = _split_planets(obs)
    if not own or not targets:
        return []
    aimer = Aimer(obs)
    scored: list[tuple[float, _Planet, _Planet, int, float]] = []
    for source in own:
        avail = source.ships - RESERVE
        if avail < 1:
            continue
        for target in targets:
            aim = aimer.aim(source.id, target.id, avail)
            if aim is None:  # engine says the shot is not viable -> skip
                continue
            angle, eta = aim
            defense = _projected_defense(target, eta)
            need = int(math.ceil(defense + CAPTURE_MARGIN))
            if avail < need:
                continue
            value = max(0, target.production) * VALUE_HORIZON - defense
            scored.append((value, source, target, need, angle))
    scored.sort(key=lambda item: -item[0])
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    moves: Moves = []
    for _value, source, target, need, angle in scored:
        if source.id in used_sources or target.id in used_targets:
            continue
        ships = min(source.ships - RESERVE, max(need, 1))
        if ships < 1:
            continue
        moves.append(_move(source, angle, ships))
        used_sources.add(source.id)
        used_targets.add(target.id)
    return moves


# --------------------------------------------------------------------------- H3
def timeline_risk(obs: Obs) -> Moves:
    me, own, targets = _split_planets(obs)
    if not own or not targets:
        return []
    comet_ids = {int(pid) for pid in obs.get("comet_planet_ids", [])}
    aimer = Aimer(obs)
    scored: list[tuple[float, _Planet, _Planet, int, float]] = []
    for source in own:
        avail = source.ships - RESERVE
        if avail < 1:
            continue
        for target in targets:
            aim = aimer.aim(source.id, target.id, avail)
            if aim is None:  # engine: shot not viable -> skip
                continue
            angle, eta = aim
            defense = _projected_defense(target, eta)
            need = int(math.ceil(defense + CAPTURE_MARGIN))
            if avail < need:
                continue
            value = max(0, target.production) * VALUE_HORIZON - defense
            # Risk penalties: comet target, late arrival, draining the source.
            if target.id in comet_ids:
                value -= max(0, target.production) * VALUE_HORIZON  # avoid comets
            value -= eta * 0.5  # prefer near, low-latency captures
            overkill = max(0, avail - need)
            value -= 0.25 * overkill  # do not dump the whole stack
            scored.append((value, source, target, need, angle))
    scored.sort(key=lambda item: -item[0])
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    moves: Moves = []
    for value, source, target, need, angle in scored:
        if value <= 0:
            continue
        if source.id in used_sources or target.id in used_targets:
            continue
        # Send just enough (plus margin), never the whole garrison.
        ships = min(source.ships - RESERVE, need + max(1, need // 4))
        if ships < 1:
            continue
        moves.append(_move(source, angle, ships))
        used_sources.add(source.id)
        used_targets.add(target.id)
    return moves


# --------------------------------------------------------------------------- H4
def hammer_multiprong(obs: Obs) -> Moves:
    me, own, targets = _split_planets(obs)
    if not own or not targets:
        return []
    sources = [p for p in own if p.ships - RESERVE >= 1]
    if not sources:
        return []
    aimer = Aimer(obs)

    def eta_to(source: _Planet, target: _Planet) -> float:
        aim = aimer.aim(source.id, target.id, source.ships - RESERVE)
        return aim[1] if aim is not None else math.inf

    def angle_to(source: _Planet, target: _Planet) -> float | None:
        aim = aimer.aim(source.id, target.id, source.ships - RESERVE)
        return aim[0] if aim is not None else None

    def target_value(target: _Planet) -> float:
        nearest = min(eta_to(s, target) for s in sources)
        if math.isinf(nearest):
            return -math.inf
        defense = _projected_defense(target, nearest)
        return max(0, target.production) * VALUE_HORIZON - defense

    reachable = [t for t in targets if any(not math.isinf(eta_to(s, t)) for s in sources)]
    if not reachable:
        return []
    ranked = sorted(reachable, key=target_value, reverse=True)
    primary = ranked[0]
    # Hammer: every source that can reach the primary piles on if the synchronized
    # force can plausibly overwhelm its projected defense.
    contributions = sorted(
        (s for s in sources if angle_to(s, primary) is not None), key=lambda s: eta_to(s, primary)
    )
    if not contributions:
        return []
    total = sum(s.ships - RESERVE for s in contributions)
    nearest_eta = eta_to(contributions[0], primary)
    if total > _projected_defense(primary, nearest_eta) + CAPTURE_MARGIN:
        return [_move(s, angle_to(s, primary), s.ships - RESERVE) for s in contributions]
    # Multiprong fallback: split nearest sources across the two best targets so
    # the opponent must divide its response.
    moves: Moves = []
    for idx, source in enumerate(contributions):
        target = ranked[min(idx % 2, len(ranked) - 1)]
        ang = angle_to(source, target)
        ships = source.ships - RESERVE
        if ang is not None and ships >= 1:
            moves.append(_move(source, ang, ships))
    return moves


# --------------------------------------------------------------------------- H5
def regroup_dominance(obs: Obs) -> Moves:
    me, own, others = _split_planets(obs)
    if not own:
        return []
    enemies = [p for p in others if p.owner >= 0]
    if not enemies:
        return []
    four_player = _player_count(_planets(obs)) >= 3
    aimer = Aimer(obs)

    # Frontier = own planet closest to any enemy. Reinforce it from the safest
    # (most enemy-distant) interior planet that has spare ships.
    def enemy_proximity(planet: _Planet) -> float:
        return min(_dist(planet, e) for e in enemies)

    frontier = min(own, key=enemy_proximity)
    interior = sorted(
        (p for p in own if p.id != frontier.id and p.ships - RESERVE >= 1),
        key=enemy_proximity,
        reverse=True,
    )
    moves: Moves = []
    for source in interior[:2]:
        ships = source.ships - RESERVE
        aim = aimer.aim(source.id, frontier.id, ships)
        if ships >= 1 and aim is not None:
            moves.append(_move(source, aim[0], ships))

    # 4p bias: from the frontier itself, pressure the production leader so we do
    # not kingmake by hitting a convenient weak neighbour.
    if four_player:
        leader = max(enemies, key=lambda e: e.production)
        spare = frontier.ships - RESERVE
        aim = aimer.aim(frontier.id, leader.id, spare)
        if spare >= 1 and aim is not None:
            moves.append(_move(frontier, aim[0], spare))
    return moves


# --------------------------------------------------------------------------- H6
def _incoming_threat(obs: Obs, me: int, own: list[_Planet], enemies: list[_Planet]) -> dict[int, float]:
    """Proxy threat each owned planet faces, in ships.

    Two sources: (1) committed in-flight ENEMY fleets whose heading aligns with the
    planet (a fleet flies straight; planets orbit, so we use current positions — a
    near-term approximation, honest for a heuristic), discounted by alignment and
    distance; (2) latent pressure from nearby enemy planet garrisons. The absolute
    scale is arbitrary — only the *ranking* matters, and Lever A tunes the weights.
    """

    hostile_fleets = [f for f in _fleets(obs) if f.owner != me and f.owner >= 0]
    threat: dict[int, float] = {}
    for p in own:
        t = 0.0
        for f in hostile_fleets:
            dx, dy = p.x - f.x, p.y - f.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                t += f.ships
                continue
            align = math.cos(math.atan2(dy, dx) - f.angle)  # 1 => flying straight at P
            if align > THREAT_CONE:
                t += f.ships * align / (1.0 + dist / THREAT_DIST_SCALE)
        for e in enemies:
            t += max(0, e.ships) * PLANET_PRESSURE / (1.0 + _dist(p, e) / THREAT_DIST_SCALE)
        threat[p.id] = t
    return threat


def defensive_reinforce(obs: Obs) -> Moves:
    """H6 — survive the counterattack: reinforce, recapture, redistribute.

    Every other Family-H generator is pure offence; the diagnosed dominant cause of
    H's -1.0 standing is the absence of any response to the opponent's counterattack
    (commit c30b5d3). This family fills that gap with three missions, in priority
    order, never double-spending a source:

    1. reinforce-under-threat — pull spare ships from the safest interior planets to
       owned planets whose projected incoming threat exceeds their garrison;
    2. recapture — retake weak enemy planets adjacent to my territory while affordable;
    3. redistribute — push stranded interior garrisons toward the frontier.
    """

    me, own, others = _split_planets(obs)
    if not own:
        return []
    enemies = [p for p in others if p.owner >= 0]
    neutrals = [p for p in others if p.owner == -1]
    hostile_fleets = [f for f in _fleets(obs) if f.owner != me and f.owner >= 0]
    if not enemies and not hostile_fleets:
        return []  # uncontested board: nothing to defend

    aimer = Aimer(obs)
    threat = _incoming_threat(obs, me, own, enemies)
    deficit = {p.id: threat[p.id] - float(p.ships) for p in own}

    used_sources: set[int] = set()
    used_targets: set[int] = set()
    moves: Moves = []

    # --- Mission 1: reinforce planets that will not survive on their own garrison.
    threatened = sorted((p for p in own if deficit[p.id] > 0), key=lambda p: -deficit[p.id])
    safe_sources = sorted(
        (p for p in own if deficit[p.id] <= 0 and p.ships - RESERVE >= 1),
        key=lambda p: threat[p.id],  # spend from the calmest planets first
    )
    for target in threatened:
        need = deficit[target.id]
        for source in safe_sources:
            if need <= 0:
                break
            if source.id in used_sources or source.id == target.id:
                continue
            spare = source.ships - RESERVE
            if spare < 1:
                continue
            aim = aimer.aim(source.id, target.id, spare)
            if aim is None:  # engine: reinforcement cannot reach -> try another source
                continue
            send = max(1, min(spare, int(math.ceil(need))))
            moves.append(_move(source, aim[0], send))
            used_sources.add(source.id)
            used_targets.add(target.id)
            need -= send

    # --- Mission 2: recapture weak enemy planets pressing on my frontier.
    retake_targets = [
        e
        for e in enemies
        if e.id not in used_targets
        and min((_dist(o, e) for o in own), default=math.inf) <= RECAPTURE_RANGE
    ]
    retake_targets.sort(key=lambda e: max(0, e.production) * VALUE_HORIZON - e.ships, reverse=True)
    free_sources = [p for p in own if p.id not in used_sources and p.ships - RESERVE >= 1]
    for target in retake_targets:
        best: tuple[float, _Planet, float, int] | None = None
        for source in free_sources:
            if source.id in used_sources:
                continue
            avail = source.ships - RESERVE
            aim = aimer.aim(source.id, target.id, avail)
            if aim is None:
                continue
            angle, eta = aim
            need = int(math.ceil(_projected_defense(target, eta) + CAPTURE_MARGIN))
            if avail < need:
                continue
            eta_rank = -eta
            if best is None or eta_rank > best[0]:
                best = (eta_rank, source, angle, need)
        if best is not None:
            _rank, source, angle, need = best
            moves.append(_move(source, angle, min(source.ships - RESERVE, need)))
            used_sources.add(source.id)
            used_targets.add(target.id)

    # --- Mission 3: redistribute stranded interior garrisons toward the frontier.
    targets_for_front = enemies or neutrals
    if targets_for_front:
        def front_proximity(planet: _Planet) -> float:
            return min(_dist(planet, t) for t in targets_for_front)

        frontier = min(own, key=front_proximity)
        interior = sorted(
            (
                p
                for p in own
                if p.id not in used_sources
                and p.id != frontier.id
                and p.ships - RESERVE >= 1
                and threat[p.id] <= 0.0
            ),
            key=front_proximity,
            reverse=True,  # farthest-from-front (most stranded) first
        )
        for source in interior[:2]:
            spare = source.ships - RESERVE
            aim = aimer.aim(source.id, frontier.id, spare)
            if aim is not None and spare >= 1:
                moves.append(_move(source, aim[0], spare))
                used_sources.add(source.id)

    return moves


# --------------------------------------------------------------------------- H7
# Lever A: ONE parameterised evaluation function over every candidate move
# (capture + reinforce), combining offence and the H6 defence terms. The weight
# vector IS the genome a metaheuristic (CMA-ES / N-Tuple Bandit EA) tunes offline
# against Producer — the realisable form of the "learned value function" the PPO
# line called for, reached by evolution over hand-designed features instead of RL.
# Reference: Gaina, Devlin, Lucas, Perez-Liebana 2020 (arXiv:2003.12331) — offline
# parameter optimisation of one rich agent beats runtime selection over fixed bots.

#: Named weight layout. Order is the genome order the tuner perturbs.
EVAL_WEIGHT_NAMES: tuple[str, ...] = (
    "prod",            # 0  capture value per unit production
    "defense",         # 1  penalty per projected defender at arrival
    "eta",             # 2  latency penalty per step to arrival
    "enemy_denial",    # 3  bonus (× production) for taking an ENEMY (not neutral) planet
    "comet",           # 4  penalty (× production) for a comet target
    "overkill",        # 5  penalty per ship sent beyond what's needed
    "reinforce",       # 6  gain (× deficit) for defending a threatened owned planet
    "overextend",      # 7  penalty per unit distance from my nearest planet
    "consolidate",     # 8  bonus for targets near my territory
    "reserve",         # 9  ships kept home on a source (param, clamped int >= 0)
    "capture_margin",  # 10 extra ships required over projected defense (param)
    "min_score",       # 11 candidates scoring below this are not launched (param)
    "threat_reserve",  # 12 per-source ships held back ∝ that source's own incoming threat
)

#: Hand-tuned baseline genome (roughly timeline_risk + H6 defence). The tuner
#: starts here; with these weights ``eval_function`` is a sane standalone bot.
EVAL_DEFAULT_WEIGHTS: tuple[float, ...] = (
    60.0,  # prod
    1.0,   # defense
    0.5,   # eta
    30.0,  # enemy_denial
    60.0,  # comet
    0.25,  # overkill
    5.0,   # reinforce
    0.1,   # overextend
    5.0,   # consolidate
    1.0,   # reserve
    1.0,   # capture_margin
    0.0,   # min_score
    1.0,   # threat_reserve (per-source threat-aware hold-back: the anti-over-commit gate)
)


def make_eval_policy(weights: Any = None):
    """Build an ``obs -> moves`` generator scored by ``weights`` (default baseline).

    Greedy one-shot allocator: score every viable capture/reinforce candidate by a
    weighted sum of features, then assign best-first without reusing a source or
    target. ``weights`` may be any sequence of len ``EVAL_WEIGHT_NAMES``.
    """

    w = tuple(float(x) for x in (EVAL_DEFAULT_WEIGHTS if weights is None else weights))
    if len(w) != len(EVAL_WEIGHT_NAMES):
        raise ValueError(f"expected {len(EVAL_WEIGHT_NAMES)} weights, got {len(w)}")
    (
        w_prod, w_def, w_eta, w_enemy, w_comet, w_overkill, w_reinforce,
        w_overextend, w_consolidate, w_reserve, w_margin, w_min_score,
        w_threat_reserve,
    ) = w
    reserve = max(0, int(round(w_reserve)))
    margin = max(0.0, w_margin)
    threat_reserve = max(0.0, w_threat_reserve)

    def policy(obs: Obs) -> Moves:
        me, own, others = _split_planets(obs)
        if not own:
            return []
        enemies = [p for p in others if p.owner >= 0]
        non_owned = others  # neutrals + enemies are capture targets
        comet_ids = {int(pid) for pid in obs.get("comet_planet_ids", [])}
        aimer = Aimer(obs)
        threat = _incoming_threat(obs, me, own, enemies)

        def dist_to_own(t: _Planet) -> float:
            return min((_dist(o, t) for o in own), default=0.0)

        # (score, source, target, angle, send_ships)
        scored: list[tuple[float, _Planet, _Planet, float, int]] = []

        # Capture candidates: take a neutral/enemy planet. The per-source hold-back
        # is ``reserve + threat_reserve * incoming_threat(source)`` — a safe interior
        # planet (threat ≈ 0) commits fully, a threatened frontier planet keeps enough
        # to survive the counterattack. This is the anti-over-commit gate a uniform
        # reserve cannot express (verified failure mode: send-then-get-recaptured).
        for source in own:
            avail = source.ships - reserve - threat_reserve * threat[source.id]
            if avail < 1:
                continue
            for target in non_owned:
                aim = aimer.aim(source.id, target.id, avail)
                if aim is None:
                    continue
                angle, eta = aim
                defense = _projected_defense(target, eta)
                need = int(math.ceil(defense + margin))
                if avail < need:
                    continue
                prod = max(0, target.production)
                score = (
                    w_prod * prod
                    - w_def * defense
                    - w_eta * eta
                    - w_overextend * dist_to_own(target)
                    + w_consolidate / (1.0 + dist_to_own(target))
                )
                if target.owner >= 0:
                    score += w_enemy * prod  # denial: hurts the opponent too
                if target.id in comet_ids:
                    score -= w_comet * prod
                overkill = max(0, avail - need)
                score -= w_overkill * overkill
                send = min(avail, need + max(1, need // 4))
                scored.append((score, source, target, angle, max(1, send)))

        # Reinforce candidates: shore up a threatened owned planet.
        for target in own:
            deficit = threat[target.id] - float(target.ships)
            if deficit <= 0:
                continue
            for source in own:
                if source.id == target.id:
                    continue
                avail = source.ships - reserve
                if avail < 1 or threat[source.id] > threat[target.id]:
                    continue  # don't strip a more-threatened planet
                aim = aimer.aim(source.id, target.id, avail)
                if aim is None:
                    continue
                send = max(1, min(avail, int(math.ceil(deficit))))
                score = w_reinforce * deficit - w_eta * aim[1]
                scored.append((score, source, target, aim[0], send))

        scored.sort(key=lambda item: -item[0])
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        moves: Moves = []
        for score, source, target, angle, send in scored:
            if score < w_min_score:
                break
            if source.id in used_sources or target.id in used_targets:
                continue
            send = min(send, source.ships - reserve)
            if send < 1:
                continue
            moves.append(_move(source, angle, send))
            used_sources.add(source.id)
            used_targets.add(target.id)
        return moves

    return policy


# Registration of these generators is centralized in
# candidate_factory._register_builtin_families (keeps this module free of any
# candidate_factory import → no import cycle).
