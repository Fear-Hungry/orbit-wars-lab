"""Pure helpers of the LB calibration meta-validator (scripts/research_loop/calibrate.py).

calibrate.py is the only thing standing between us and trusting a falsified gate.
Its FALSE-PASS guard (competitive_tied) is what catches the "topo plano" failure
that already burned the local league. None of it was tested. These tests pin the
Spearman / inversion / false-pass / verdict logic so a refactor can't quietly
disarm the guard.
"""
from __future__ import annotations

import math

from scripts.research_loop.calibrate import (
    LB_NOISE,
    competitive_tied,
    inversions,
    spearman,
    verdict,
)


def test_spearman_perfect_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_spearman_perfect_inverse():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_spearman_handles_ties_without_crash():
    rho = spearman([1, 1, 2, 3], [5, 5, 6, 7])
    assert -1.0 <= rho <= 1.0


def test_spearman_constant_is_nan():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def _row(name, lb, fit):
    return {"name": name, "lb": lb, "fitness": fit}


def test_inversions_flags_backwards_pair_beyond_noise():
    # b has much higher LB but the gate scored it lower -> a real inversion.
    rows = [_row("a", 1000.0, 5.0), _row("b", 1200.0, 1.0)]
    inv = inversions(rows)
    assert len(inv) == 1
    assert "b" in inv[0] and "ranked BELOW" in inv[0]


def test_inversions_ignores_pairs_inside_noise_band():
    # LB gap < LB_NOISE -> the LB itself can't order them, so not an inversion.
    rows = [_row("a", 1000.0, 5.0), _row("b", 1000.0 + LB_NOISE - 1, 1.0)]
    assert inversions(rows) == []


def test_inversions_none_when_gate_agrees():
    rows = [_row("a", 1000.0, 1.0), _row("b", 1200.0, 5.0)]
    assert inversions(rows) == []


def test_competitive_tied_true_when_only_floor_separated():
    # floor (lowest LB) differs; the 3 competitive anchors share one fitness ->
    # rho would be a FALSE PASS driven solely by the floor outlier.
    rows = [_row("floor", 1000.0, 0.0), _row("a", 1100.0, 2.0),
            _row("b", 1200.0, 2.0), _row("c", 1300.0, 2.0)]
    assert competitive_tied(rows) is True


def test_competitive_tied_false_when_top_separates():
    rows = [_row("floor", 1000.0, 0.0), _row("a", 1100.0, 1.0),
            _row("b", 1200.0, 2.0), _row("c", 1300.0, 3.0)]
    assert competitive_tied(rows) is False


def test_competitive_tied_needs_at_least_three():
    assert competitive_tied([_row("a", 1000.0, 1.0), _row("b", 1200.0, 1.0)]) is False


def test_verdict_false_pass_overrides_high_rho():
    msg = verdict(0.95, n=4, tied=True)
    assert "FALSE PASS" in msg


def test_verdict_bands():
    assert "PREDICTS" in verdict(0.7, 4, tied=False)
    assert "WEAK" in verdict(0.45, 4, tied=False)
    assert "FALSIFIED" in verdict(0.1, 4, tied=False)
    assert "INDETERMINATE" in verdict(float("nan"), 4, tied=False)
