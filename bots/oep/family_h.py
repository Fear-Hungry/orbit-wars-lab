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

from bots.oep.candidate_factory import register_family

Obs = dict[str, Any]
Move = list[float]
Moves = list[Move]

SHIP_SPEED = 6.0
CENTER = 50.0  # orbit_lite.constants.CENTER (sun / board center; planets orbit it)
RESERVE = 5  # ships kept home on a source planet
CAPTURE_MARGIN = 2.0  # extra ships required over projected defense
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

    orb_r = math.hypot(target.x - CENTER, target.y - CENTER)
    if orb_r < 1e-6 or abs(angvel) < 1e-9:  # static target (or no rotation)
        eta = _dist(source, target) / SHIP_SPEED
        return math.atan2(target.y - source.y, target.x - source.x), max(1.0, eta)
    a0 = math.atan2(target.y - CENTER, target.x - CENTER)
    t = _dist(source, target) / SHIP_SPEED
    px, py = target.x, target.y
    for _ in range(4):
        ang = a0 + angvel * t
        px = CENTER + orb_r * math.cos(ang)
        py = CENTER + orb_r * math.sin(ang)
        t = math.hypot(px - source.x, py - source.y) / SHIP_SPEED
    return math.atan2(py - source.y, px - source.x), max(1.0, t)


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
    angvel = float(obs.get("angular_velocity", 0.03))
    scored: list[tuple[float, _Planet, _Planet, int, float]] = []
    for source in own:
        avail = source.ships - RESERVE
        if avail < 1:
            continue
        for target in targets:
            angle, eta = _aim(source, target, angvel)
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
    angvel = float(obs.get("angular_velocity", 0.03))
    scored: list[tuple[float, _Planet, _Planet, int, float]] = []
    for source in own:
        avail = source.ships - RESERVE
        if avail < 1:
            continue
        for target in targets:
            angle, eta = _aim(source, target, angvel)
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
    angvel = float(obs.get("angular_velocity", 0.03))

    def target_value(target: _Planet) -> float:
        nearest = min(_aim(s, target, angvel)[1] for s in sources)
        defense = _projected_defense(target, nearest)
        return max(0, target.production) * VALUE_HORIZON - defense

    ranked = sorted(targets, key=target_value, reverse=True)
    if not ranked:
        return []
    primary = ranked[0]
    # Hammer: every source with spare ships piles onto the primary target if the
    # synchronized force can plausibly overwhelm its projected defense.
    contributions = sorted(sources, key=lambda s: _aim(s, primary, angvel)[1])
    total = sum(s.ships - RESERVE for s in contributions)
    nearest_eta = _aim(contributions[0], primary, angvel)[1]
    if total > _projected_defense(primary, nearest_eta) + CAPTURE_MARGIN:
        return [
            _move(s, _aim(s, primary, angvel)[0], s.ships - RESERVE) for s in contributions
        ]
    # Multiprong fallback: split nearest sources across the two best targets so
    # the opponent must divide its response.
    moves: Moves = []
    for idx, source in enumerate(contributions):
        target = ranked[min(idx % 2, len(ranked) - 1)]
        ships = source.ships - RESERVE
        if ships >= 1:
            moves.append(_move(source, _aim(source, target, angvel)[0], ships))
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
    angvel = float(obs.get("angular_velocity", 0.03))

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
        if ships >= 1:
            moves.append(_move(source, _aim(source, frontier, angvel)[0], ships))

    # 4p bias: from the frontier itself, pressure the production leader so we do
    # not kingmake by hitting a convenient weak neighbour.
    if four_player:
        leader = max(enemies, key=lambda e: e.production)
        spare = frontier.ships - RESERVE
        if spare >= 1:
            moves.append(_move(frontier, _aim(frontier, leader, angvel)[0], spare))
    return moves


def register_family_h() -> None:
    """Register H2-H5 (idempotent-safe: only call once at import)."""

    register_family("production_projected_attack", lambda: production_projected_attack)
    register_family("timeline_risk", lambda: timeline_risk)
    register_family("hammer_multiprong", lambda: hammer_multiprong)
    register_family("regroup_dominance", lambda: regroup_dominance)


register_family_h()
