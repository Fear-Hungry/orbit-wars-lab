"""Pure statistics for the submission selector (no I/O, stdlib only).

Selection cannot be by point win rate: 53-47 vs 51-49 does not separate two
bots. Every comparison here carries uncertainty — Wilson intervals for a single
rate, Beta-posterior Monte Carlo for "P(A beats B)" — and the decision bands
route to RUN_MORE_GAMES instead of pretending noise is signal. The Monte Carlo
is seeded (deterministic to ~±0.003 at 20k draws) so thresholds reproduce.
"""
from __future__ import annotations

import math
import random
from typing import Any

# etapa 14 weights; calibration (league_selector_calibration.py) validates that
# the score built with these weights orders the LB anchors before any use.
DEFAULT_SELECTOR_WEIGHTS = {
    "cal_2p_adv": 0.35,
    "cal_4p_adv": 0.35,
    "h2h_vs_incumbent": 0.15,
    "worst_bucket": 0.10,
    "stability": 0.05,
}

# decision bands (etapa 11): below CONFIDENT the selector must not choose
P_BEATS_INCUMBENT_MIN = 0.80
P_ABOVE_SECOND_MIN = 0.70
TIEBREAK_P_MIN = 0.75


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def beta_p_greater(
    wins_a: int, losses_a: int, wins_b: int, losses_b: int,
    *, samples: int = 20_000, seed: int = 7,
) -> float:
    """P(rate_a > rate_b) under independent Beta(w+1, l+1) posteriors."""
    rng = random.Random(seed)
    hits = 0
    a_alpha, a_beta = wins_a + 1, losses_a + 1
    b_alpha, b_beta = wins_b + 1, losses_b + 1
    for _ in range(samples):
        if rng.betavariate(a_alpha, a_beta) > rng.betavariate(b_alpha, b_beta):
            hits += 1
    return hits / samples


def p_beats_incumbent(h2h_wins: int, h2h_losses: int, *, samples: int = 20_000, seed: int = 7) -> float:
    """P(true H2H win rate > 0.5) from the candidate-vs-incumbent record."""
    rng = random.Random(seed)
    alpha, beta = h2h_wins + 1, h2h_losses + 1
    hits = sum(1 for _ in range(samples) if rng.betavariate(alpha, beta) > 0.5)
    return hits / samples


def selector_score(features: dict[str, float], weights: dict[str, float] | None = None,
                   risk_penalty: float = 0.0) -> float:
    """etapa 14: weighted feature mix minus operational/strategic risk.

    Expected features: cal_2p_adv, cal_4p_adv (normalized advantages),
    h2h_vs_incumbent (decisive win rate, 0..1), worst_bucket (0..1),
    stability (0..1). Missing features contribute 0 — the calibration checks
    catch a feature pipeline that silently stopped producing values."""
    w = weights or DEFAULT_SELECTOR_WEIGHTS
    return sum(float(w[k]) * float(features.get(k, 0.0)) for k in w) - float(risk_penalty)


def selection_uncertainty(
    *,
    h2h_wins_vs_incumbent: int,
    h2h_losses_vs_incumbent: int,
    candidate_record_vs_second: tuple[int, int, int, int] | None = None,
    samples: int = 20_000,
    seed: int = 7,
) -> dict[str, Any]:
    """Decision bands for the selector. candidate_record_vs_second is
    (wins_a, losses_a, wins_b, losses_b) on a COMMON fixed panel."""
    p_inc = p_beats_incumbent(
        h2h_wins_vs_incumbent, h2h_losses_vs_incumbent, samples=samples, seed=seed
    )
    p_second = None
    if candidate_record_vs_second is not None:
        p_second = beta_p_greater(*candidate_record_vs_second, samples=samples, seed=seed)
    confident = p_inc >= P_BEATS_INCUMBENT_MIN and (
        p_second is None or p_second >= P_ABOVE_SECOND_MIN
    )
    return {
        "p_beats_incumbent": p_inc,
        "p_above_second_best": p_second,
        "band": "CONFIDENT" if confident else "RUN_MORE_GAMES",
        "thresholds": {
            "p_beats_incumbent_min": P_BEATS_INCUMBENT_MIN,
            "p_above_second_min": P_ABOVE_SECOND_MIN,
        },
    }
