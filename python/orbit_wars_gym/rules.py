from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from .entities import planet_id, planet_owner, planet_ships


def moves_are_legal(state: dict[str, Any], player: int, moves: list[list[float]]) -> bool:
    own_budget = {
        planet_id(planet): max(0, planet_ships(planet))
        for planet in state.get("planets", [])
        if planet_owner(planet) == player
    }
    spent: dict[int, int] = {}
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return False
        try:
            from_id = int(move[0])
            angle = float(move[1])
            ships = int(move[2])
        except (TypeError, ValueError, OverflowError):
            return False
        if from_id not in own_budget:
            return False
        if not math.isfinite(angle):
            return False
        if ships <= 0:
            return False
        spent[from_id] = spent.get(from_id, 0) + ships
        if spent[from_id] > own_budget[from_id]:
            return False
    return True


def normalized_margin(scores: list[float], player: int) -> float:
    own = float(scores[player])
    others = [float(score) for idx, score in enumerate(scores) if idx != player]
    if not others:
        return 0.0
    reference = fmean(others)
    return (own - reference) / max(abs(own) + abs(reference), 1.0)
