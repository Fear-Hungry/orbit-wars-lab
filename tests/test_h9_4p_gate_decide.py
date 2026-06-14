"""Verdict logic of the H9 4p gate (scripts/h9_4p_gate.decide / _aggregate).

The gate's docstring promised "death < holdwave WITH NO MARGIN REGRESSION", but
the code used to check death only (six-hats validation audit, 2026-06-14). A
death-only verdict green-lights a config that buys survival by shedding margin —
exactly the LB ~1048 stagnation (memory: h9_threat_value_4p_cracks_floor). These
tests pin the corrected rule so it cannot silently regress to death-only again.
"""
from __future__ import annotations

from scripts.h9_4p_gate import _aggregate, decide


def _s(death, margin, timeouts=0, planets=1.0):
    return {"death_rate": death, "mean_margin": margin,
            "mean_final_planets": planets, "timeouts": timeouts}


def test_pass_when_safer_and_margin_held():
    # death down, margin essentially flat -> PASS
    v = decide(_s(0.80, 0.10), _s(0.40, 0.09))
    assert v["passed"] is True
    assert v["reasons"] == []
    assert v["delta_death"] < 0


def test_fail_when_death_down_but_margin_crashes():
    # The exact bug: survival improves but margin regresses past tol -> must FAIL.
    v = decide(_s(0.80, 0.30), _s(0.40, 0.10), margin_tol=0.02)
    assert v["passed"] is False
    assert any("margin regressed" in r for r in v["reasons"])


def test_fail_when_death_not_reduced():
    v = decide(_s(0.40, 0.20), _s(0.40, 0.50))  # tie on death is not "<"
    assert v["passed"] is False
    assert any("death not reduced" in r for r in v["reasons"])


def test_fail_on_timeouts_even_if_otherwise_good():
    v = decide(_s(0.80, 0.10), _s(0.40, 0.20, timeouts=3))
    assert v["passed"] is False
    assert any("timeouts" in r for r in v["reasons"])


def test_margin_tol_governs_the_boundary():
    # a drop within tol is allowed; a drop past tol fails.
    assert decide(_s(0.8, 0.20), _s(0.4, 0.185), margin_tol=0.02)["passed"] is True   # -0.015
    assert decide(_s(0.8, 0.20), _s(0.4, 0.170), margin_tol=0.02)["passed"] is False  # -0.030


def test_aggregate_is_field_mean_and_sums_timeouts():
    agg = _aggregate([_s(0.0, 1.0, timeouts=1), _s(1.0, -1.0, timeouts=2)])
    assert agg["death_rate"] == 0.5
    assert agg["mean_margin"] == 0.0
    assert agg["timeouts"] == 3


def test_aggregate_verdict_not_gameable_per_opponent():
    # Wins big vs producer (death 0.9->0.2) but loses vs rush (margin 0.5->-0.3).
    # Per-opponent tally would show 1/2 PASS; the aggregate must FAIL on margin.
    base = _aggregate([_s(0.90, 0.10), _s(0.00, 0.50)])
    h9 = _aggregate([_s(0.20, 0.05), _s(0.00, -0.30)])
    v = decide(base, h9, margin_tol=0.02)
    assert v["delta_death"] < 0          # aggregate IS safer
    assert v["passed"] is False          # ...but margin regressed -> no free pass
    assert any("margin regressed" in r for r in v["reasons"])
