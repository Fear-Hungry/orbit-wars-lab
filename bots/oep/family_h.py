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


# Registration of these generators is centralized in
# candidate_factory._register_builtin_families (keeps this module free of any
# candidate_factory import → no import cycle).
