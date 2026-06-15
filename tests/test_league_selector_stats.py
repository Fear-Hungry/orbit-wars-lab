"""Selector statistics: deterministic, calibrated to known values."""
from __future__ import annotations

import pytest
from scripts.league_selector_stats import (
    DEFAULT_SELECTOR_WEIGHTS,
    beta_p_greater,
    p_beats_incumbent,
    selection_uncertainty,
    selector_score,
    wilson_interval,
)


def test_wilson_matches_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(10, 10)
    assert lo > 0.7 and hi == 1.0


def test_beta_p_greater_dominance_symmetry_and_determinism():
    assert beta_p_greater(50, 0, 0, 50) > 0.999
    assert beta_p_greater(0, 50, 50, 0) < 0.001
    even = beta_p_greater(10, 10, 10, 10)
    assert even == pytest.approx(0.5, abs=0.02)
    assert beta_p_greater(12, 8, 9, 11, seed=7) == beta_p_greater(12, 8, 9, 11, seed=7)


def test_p_beats_incumbent_bands():
    assert p_beats_incumbent(60, 40) > 0.95
    assert p_beats_incumbent(51, 49) < 0.70  # 53-47-style records do not separate
    assert p_beats_incumbent(40, 60) < 0.05


def test_selector_score_formula_and_weights():
    assert sum(DEFAULT_SELECTOR_WEIGHTS.values()) == pytest.approx(1.0)
    features = {
        "cal_2p_adv": 0.2, "cal_4p_adv": 0.1, "h2h_vs_incumbent": 0.6,
        "worst_bucket": 0.5, "stability": 1.0,
    }
    expected = 0.35 * 0.2 + 0.35 * 0.1 + 0.15 * 0.6 + 0.10 * 0.5 + 0.05 * 1.0
    assert selector_score(features) == pytest.approx(expected)
    assert selector_score(features, risk_penalty=0.08) == pytest.approx(expected - 0.08)
    # missing features contribute zero, never crash
    assert selector_score({"cal_2p_adv": 0.2}) == pytest.approx(0.35 * 0.2)


def test_uncertainty_bands_route_to_run_more_games():
    tied = selection_uncertainty(
        h2h_wins_vs_incumbent=51, h2h_losses_vs_incumbent=49,
        candidate_record_vs_second=(53, 47, 51, 49),
    )
    assert tied["band"] == "RUN_MORE_GAMES"

    clear = selection_uncertainty(
        h2h_wins_vs_incumbent=70, h2h_losses_vs_incumbent=30,
        candidate_record_vs_second=(70, 30, 45, 55),
    )
    assert clear["band"] == "CONFIDENT"
    assert clear["p_beats_incumbent"] > 0.99
