"""Family H — H7 contextual hyper-heuristic (family selector).

Instead of tuning ``OEP_MIN_ADVANTAGE``, the hyper-heuristic decides *which
generator earns compute* in a given state — i.e. it searches the space of
heuristic components, not the raw solution space (Burke et al., hyper-heuristics
survey). It is cheap at runtime: read a few context features, pick a family,
return that family's real moves.

The mapping is a small, **offline-tunable** weight table (``FAMILY_WEIGHTS``):
each context bucket holds a per-family score; the runtime takes the argmax. A
metaheuristic (random search / NTBEA / bandit) can overwrite the table offline
without touching this code. The shipped defaults are a sensible rule-of-thumb so
the selector is runnable before any offline tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bots.oep.family_h import (
    _planets,
    _player_count,
    hammer_multiprong,
    production_projected_attack,
    regroup_dominance,
    timeline_risk,
)

Obs = dict[str, Any]
Moves = list[list[float]]

_FAMILIES: dict[str, Callable[[Obs], Moves]] = {
    "production_projected_attack": production_projected_attack,
    "timeline_risk": timeline_risk,
    "hammer_multiprong": hammer_multiprong,
    "regroup_dominance": regroup_dominance,
}

#: Context buckets -> per-family score. ``argmax`` selects the family. Defaults
#: are hand-set; an offline optimizer may replace the inner dicts wholesale.
FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "expand": {"production_projected_attack": 1.0, "timeline_risk": 0.6},
    "comet": {"timeline_risk": 1.0, "production_projected_attack": 0.3},
    "ahead": {"hammer_multiprong": 1.0, "production_projected_attack": 0.5},
    "behind": {"regroup_dominance": 1.0, "timeline_risk": 0.5},
    "four_player": {"regroup_dominance": 1.0, "timeline_risk": 0.6},
}

_FALLBACK = "production_projected_attack"


def context_bucket(obs: Obs) -> str:
    """Classify the state into one of the FAMILY_WEIGHTS buckets."""

    me = int(obs.get("player", 0))
    planets = _planets(obs)
    if not planets:
        return "expand"
    if _player_count(planets) >= 3:
        return "four_player"
    if obs.get("comet_planet_ids"):
        return "comet"

    total = len(planets)
    neutral = sum(1 for p in planets if p.owner < 0)
    if total and neutral / total >= 0.4:
        return "expand"  # early: land grab still open

    my_ships = sum(p.ships for p in planets if p.owner == me)
    enemy_ships = sum(p.ships for p in planets if p.owner >= 0 and p.owner != me)
    return "ahead" if my_ships >= enemy_ships else "behind"


def select_family(obs: Obs) -> str:
    """Pick the family name for ``obs`` via the (tunable) weight table."""

    weights = FAMILY_WEIGHTS.get(context_bucket(obs), {})
    if not weights:
        return _FALLBACK
    return max(weights, key=lambda name: weights[name])


def hyperheuristic_plan(obs: Obs) -> Moves:
    moves = _FAMILIES[select_family(obs)](obs)
    return moves if isinstance(moves, list) else []


# Registration centralized in candidate_factory._register_builtin_families.
