from __future__ import annotations

from scripts.oep_promotion_gate import evaluate_promotion


def _report(
    *,
    submission: str,
    margin: float,
    crash: float = 0.0,
    fallback: float = 0.0,
    instrumentation_missing: float = 0.0,
) -> dict:
    records = [
        {
            "seed": 0,
            "submission_player": 0,
            "win_points": 0.0 if margin < 0.0 else 1.0,
            "normalized_margin": margin,
        },
        {
            "seed": 0,
            "submission_player": 1,
            "win_points": 0.0 if margin < 0.0 else 1.0,
            "normalized_margin": margin,
        },
    ]
    return {
        "submission": submission,
        "formats": [
            {
                "format": "2p",
                "opponents": [
                    {
                        "opponent": "producer",
                        "summary": {
                            "games": 2,
                            "win_rate": 0.0 if margin < 0.0 else 1.0,
                            "mean_score_margin": margin,
                            "crash_rate": crash,
                            "timeout_rate": 0.0,
                            "invalid_action_rate": 0.0,
                            "fallback_rate": fallback,
                            "policy_illegal_move_rate": 0.0,
                            "fallback_error_rate": 0.0,
                            "instrumentation_missing_rate": instrumentation_missing,
                        },
                        "records": records,
                    }
                ],
            }
        ],
    }


def test_oep_promotion_gate_accepts_nonnegative_margin_without_regression() -> None:
    report = evaluate_promotion(
        baseline=_report(submission="baseline.py", margin=-1.0),
        candidate=_report(submission="candidate.py", margin=0.1),
        label="2p:producer",
        min_games=2,
        min_margin=0.0,
        alpha=0.05,
        min_effect=0.05,
    )

    assert report["passed"]
    assert report["candidate_summary"]["mean_score_margin"] == 0.1
    assert report["comparison"]["paired_margin_delta"] > 0.0


def test_oep_promotion_gate_rejects_negative_margin() -> None:
    report = evaluate_promotion(
        baseline=_report(submission="baseline.py", margin=-1.0),
        candidate=_report(submission="candidate.py", margin=-0.25),
        label="2p:producer",
        min_games=2,
        min_margin=0.0,
        alpha=0.05,
        min_effect=0.05,
    )

    assert not report["passed"]
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["candidate_margin_floor"]["passed"]


def test_oep_promotion_gate_rejects_technical_failures() -> None:
    report = evaluate_promotion(
        baseline=_report(submission="baseline.py", margin=-1.0),
        candidate=_report(submission="candidate.py", margin=0.1, crash=0.01),
        label="2p:producer",
        min_games=2,
        min_margin=0.0,
        alpha=0.05,
        min_effect=0.05,
    )

    assert not report["passed"]
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["candidate_crash_rate"]["passed"]


def test_oep_promotion_gate_rejects_fallbacks() -> None:
    report = evaluate_promotion(
        baseline=_report(submission="baseline.py", margin=-1.0),
        candidate=_report(submission="candidate.py", margin=0.1, fallback=0.01),
        label="2p:producer",
        min_games=2,
        min_margin=0.0,
        alpha=0.05,
        min_effect=0.05,
    )

    assert not report["passed"]
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["candidate_fallback_rate"]["passed"]


def test_oep_promotion_gate_rejects_missing_instrumentation() -> None:
    report = evaluate_promotion(
        baseline=_report(submission="baseline.py", margin=-1.0),
        candidate=_report(
            submission="candidate.py",
            margin=0.1,
            instrumentation_missing=1.0,
        ),
        label="2p:producer",
        min_games=2,
        min_margin=0.0,
        alpha=0.05,
        min_effect=0.05,
    )

    assert not report["passed"]
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["candidate_instrumentation_missing_rate"]["passed"]
