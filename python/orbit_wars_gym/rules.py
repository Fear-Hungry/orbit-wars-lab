from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from .entities import planet_id, planet_owner


def moves_are_legal(state: dict[str, Any], player: int, moves: list[list[float]]) -> bool:
    own_ids = {
        planet_id(planet) for planet in state.get("planets", []) if planet_owner(planet) == player
    }
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return False
        from_id, angle, ships = move
        if int(from_id) not in own_ids:
            return False
        if not math.isfinite(float(angle)):
            return False
        if int(ships) <= 0:
            return False
    return True


def normalized_margin(scores: list[float], player: int) -> float:
    own = float(scores[player])
    others = [float(score) for idx, score in enumerate(scores) if idx != player]
    if not others:
        return 0.0
    reference = fmean(others)
    return (own - reference) / max(abs(own) + abs(reference), 1.0)
