"""Genome ops: mutation over the search space, (de)serialisation, fitness.

A genome is a flat dict of PGSConfig knobs (search-space knobs + fixed regime
knobs from BASELINE). ``mutate`` perturbs 1-2 search-space knobs of a parent
(gaussian, scale-relative to the knob range), clamped to bounds — the classic
FunSearch/EA local move. Fitness collapses the multi-opponent gate metrics into
one scalar so parents can be ranked.
"""
from __future__ import annotations

import json
import random

from scripts.research_loop.search_space import BASELINE, SEARCH_SPACE


def baseline_genome() -> dict:
    """Gen-0 parent = the shipped config, restricted to declared knobs + regime."""
    return dict(BASELINE)


def mutate(parent: dict, rng: random.Random, *, n_knobs: int = 2, sigma_frac: float = 0.25) -> dict:
    """Perturb ``n_knobs`` (1..n_knobs) search-space knobs of ``parent``.

    Each perturbation is gaussian with std = ``sigma_frac`` * (hi-lo) of the knob,
    clamped to [lo, hi]. ints are rounded. Regime/string knobs are copied as-is.
    """
    child = dict(parent)
    names = list(SEARCH_SPACE.keys())
    k = rng.randint(1, max(1, min(n_knobs, len(names))))
    for name in rng.sample(names, k):
        knob = SEARCH_SPACE[name]
        cur = float(child.get(name, knob.lo))
        delta = rng.gauss(0.0, sigma_frac * (knob.hi - knob.lo))
        child[name] = knob.clamp(cur + delta)
    return child


def to_pgs_config(genome: dict) -> dict:
    """Genome -> kwargs for PGSConfig(**...). Already 1:1; copy defensively."""
    return dict(genome)


def serialize(genome: dict) -> str:
    """Stable JSON for the DuckDB ``idea`` column (sorted keys, compact)."""
    return json.dumps(genome, sort_keys=True, separators=(",", ":"))


def deserialize(s: str) -> dict:
    return json.loads(s)


def fitness(metrics: dict) -> float:
    """Collapse multi-opponent gate metrics into one scalar (higher = better).

    ``metrics`` carries pool-averaged ``mean_margin`` (higher better, ~[-1, 1])
    and ``death_rate`` (lower better, [0, 1]). Survival dominates in 4p (memory:
    field_is_majority_4p; the whole point of H9 is cutting the death rate), so we
    weight death heavily and use margin as a secondary signal:

        fitness = mean_margin - 2.0 * death_rate

    A hard timeout penalty guards against configs that blow the 1s act budget
    (those would fall back to Producer on the real Kaggle agent anyway).
    """
    margin = float(metrics.get("mean_margin", -1.0))
    death = float(metrics.get("death_rate", 1.0))
    timeout_pen = 1.0 if int(metrics.get("timeouts", 0)) > 0 else 0.0
    return margin - 2.0 * death - timeout_pen
