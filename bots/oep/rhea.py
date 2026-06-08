"""Family H — H6 Rolling Horizon Evolutionary Algorithm over MACRO-actions.

RHEA (Gaina et al.) evolves a short sequence of actions each tick and plays the
first action of the best sequence. Here a *gene* is a choice of candidate FAMILY
(H2-H5 or pass); a *genome* is a horizon-length sequence of such choices.

Evaluating a genome with the real Rust sim + a Producer continuation would blow
``actTimeout`` (Producer ~80ms/call x population x horizon). So genomes are ranked
by a **lightweight pure-Python forward model** (production growth + fleet arrival
+ simple combat). The model only RANKS genomes — the move actually played is the
first family's *real* legal moves on the real obs, so legality/parity are intact;
only ranking quality depends on the model.

Budget-bounded by wall-clock; the previous best genome is carried over (shifted),
as RHEA prescribes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
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
from bots.oep.family_h import (
    SHIP_SPEED,
    hammer_multiprong,
    production_projected_attack,
    regroup_dominance,
    timeline_risk,
)

Obs = dict[str, Any]
Moves = list[list[float]]

#: Macro-action vocabulary (index 0 = pass / launch nothing).
_VOCAB: list[Any] = [
    None,
    production_projected_attack,
    timeline_risk,
    hammer_multiprong,
    regroup_dominance,
]

HORIZON = 4
MACRO_STEP = 8  # sim steps advanced per macro-decision
POP = 12
MAX_GENS = 8
BUDGET_MS = 120.0
MUTATION_RATE = 0.3


@dataclass
class _Planet:
    pid: int
    owner: int
    ships: float
    prod: int
    x: float
    y: float
    radius: float


@dataclass
class _LightState:
    planets: dict[int, _Planet]
    fleets: list[tuple[int, int, float, int]] = field(default_factory=list)  # target,owner,ships,eta


def _light_from_obs(obs: Obs) -> _LightState:
    planets: dict[int, _Planet] = {}
    for p in obs.get("planets", []):
        pid = planet_id(p)
        planets[pid] = _Planet(
            pid=pid,
            owner=planet_owner(p),
            ships=float(planet_ships(p)),
            prod=planet_production(p),
            x=planet_x(p),
            y=planet_y(p),
            radius=planet_radius(p),
        )
    return _LightState(planets=planets)


def _obs_from_light(state: _LightState, me: int) -> Obs:
    rows = [
        [p.pid, p.owner, p.x, p.y, p.radius, int(round(p.ships)), p.prod]
        for p in state.planets.values()
    ]
    return {"planets": rows, "fleets": [], "player": int(me), "comet_planet_ids": []}


def _reconstruct_target(state: _LightState, source_id: int, angle: float) -> int | None:
    """Map a ``[from, angle, ships]`` move back to the planet it best aims at."""

    source = state.planets.get(source_id)
    if source is None:
        return None
    best_id: int | None = None
    best_diff = math.inf
    for pid, planet in state.planets.items():
        if pid == source_id:
            continue
        want = math.atan2(planet.y - source.y, planet.x - source.x)
        diff = abs(math.atan2(math.sin(want - angle), math.cos(want - angle)))
        if diff < best_diff:
            best_diff = diff
            best_id = pid
    return best_id


def _apply_moves(state: _LightState, moves: Moves, owner: int) -> None:
    for move in moves:
        from_id, angle, ships = int(move[0]), float(move[1]), float(move[2])
        source = state.planets.get(from_id)
        if source is None or source.owner != owner or ships < 1 or source.ships < ships:
            continue
        target_id = _reconstruct_target(state, from_id, angle)
        if target_id is None:
            continue
        source.ships -= ships
        target = state.planets[target_id]
        eta = max(1, int(math.ceil(math.hypot(target.x - source.x, target.y - source.y) / SHIP_SPEED)))
        state.fleets.append((target_id, owner, ships, eta))


def _advance(state: _LightState, steps: int) -> None:
    for _ in range(steps):
        for planet in state.planets.values():
            if planet.owner >= 0:
                planet.ships += planet.prod
        remaining: list[tuple[int, int, float, int]] = []
        for target_id, owner, ships, eta in state.fleets:
            eta -= 1
            if eta > 0:
                remaining.append((target_id, owner, ships, eta))
                continue
            target = state.planets.get(target_id)
            if target is None:
                continue
            if target.owner == owner:
                target.ships += ships
            elif ships > target.ships:
                target.owner = owner
                target.ships = ships - target.ships
            else:
                target.ships -= ships
        state.fleets = remaining


def _territory_value(state: _LightState, me: int) -> float:
    mine = sum(p.prod for p in state.planets.values() if p.owner == me)
    enemy = sum(p.prod for p in state.planets.values() if p.owner >= 0 and p.owner != me)
    return float(mine - enemy)


def _evaluate(genome: list[int], obs: Obs, me: int) -> float:
    state = _light_from_obs(obs)
    for gene in genome:
        family = _VOCAB[gene]
        if family is not None:
            moves = family(_obs_from_light(state, me))
            _apply_moves(state, moves, me)
        _advance(state, MACRO_STEP)
    return _territory_value(state, me)


def _moves_for_first_gene(gene: int, obs: Obs) -> Moves:
    family = _VOCAB[gene]
    if family is None:
        return []
    moves = family(obs)
    return moves if isinstance(moves, list) else []


def rhea_plan(obs: Obs, *, _carry: list[list[int]] | None = None) -> Moves:
    me = int(obs.get("player", 0))
    if not obs.get("planets"):
        return []
    rng = random.Random(1009 * len(obs.get("planets", [])) + 7 * len(obs.get("fleets", [])))
    n = len(_VOCAB)

    def random_genome() -> list[int]:
        return [rng.randrange(n) for _ in range(HORIZON)]

    # Seed population with single-family genomes (each family for the whole
    # horizon) + random genomes — covers "commit to one weapon" and mixes.
    population: list[list[int]] = [[g] * HORIZON for g in range(n)]
    while len(population) < POP:
        population.append(random_genome())

    start = perf_counter()
    # HARD deadline: bound the worst-case tail, not just p95. Default fallback is
    # production_projected_attack (gene 1) — a fast, legal real move — so even if
    # the deadline trips before a single genome is scored we never emit nothing
    # or an illegal move. (A deployable agent would fall back to Producer/OEP.)
    best_genome = [1] * HORIZON
    best_score = -math.inf

    def expired() -> bool:
        return (perf_counter() - start) * 1000.0 > BUDGET_MS

    for _gen in range(MAX_GENS):
        scored: list[tuple[float, list[int]]] = []
        for genome in population:
            scored.append((_evaluate(genome, obs, me), genome))
            if expired():
                break  # per-genome deadline check bounds the tail
        scored.sort(key=lambda item: -item[0])
        if scored and scored[0][0] > best_score:
            best_score, best_genome = scored[0][0], list(scored[0][1])
        if expired():
            break
        # Elitism + mutation of the top half.
        survivors = [g for _s, g in scored[: max(2, POP // 2)]]
        children: list[list[int]] = [list(best_genome)]
        while len(children) < POP:
            parent = list(rng.choice(survivors))
            for i in range(HORIZON):
                if rng.random() < MUTATION_RATE:
                    parent[i] = rng.randrange(n)
            children.append(parent)
        population = children

    return _moves_for_first_gene(best_genome[0], obs)


register_family("rhea_macro", lambda: rhea_plan)
