from __future__ import annotations

from collections.abc import Callable
from typing import Any

Policy = Callable[[dict[str, Any], int], list[list[float]]]

HEURISTIC_NAMES: tuple[str, ...] = ("greedy", "defensive", "rush", "anti_meta", "weak_random")


def get_heuristic_policies() -> dict[str, Policy]:
    from .heuristics import (
        anti_meta_agent,
        defensive_agent,
        greedy_agent,
        rush_agent,
        weak_random_agent,
    )

    return {
        "greedy": greedy_agent,
        "defensive": defensive_agent,
        "rush": rush_agent,
        "anti_meta": anti_meta_agent,
        "weak_random": weak_random_agent,
    }
