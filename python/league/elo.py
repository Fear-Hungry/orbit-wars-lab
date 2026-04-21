from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloRating:
    rating: float = 1000.0
    games: int = 0


def expected_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def update_elo(a: EloRating, b: EloRating, score_a: float, k: float = 24.0) -> tuple[EloRating, EloRating]:
    ea = expected_score(a.rating, b.rating)
    eb = 1.0 - ea
    score_b = 1.0 - score_a
    return (
        EloRating(a.rating + k * (score_a - ea), a.games + 1),
        EloRating(b.rating + k * (score_b - eb), b.games + 1),
    )
