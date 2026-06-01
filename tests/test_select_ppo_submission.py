from __future__ import annotations

from scripts.select_ppo_submission import score_exported_submission


def test_score_exported_submission_penalizes_runtime_failures():
    clean = {
        "win_rate": 0.50,
        "mean_score_margin": 0.20,
        "crash_rate": 0.0,
        "timeout_rate": 0.0,
        "invalid_action_rate": 0.0,
    }
    broken = {
        **clean,
        "win_rate": 1.0,
        "crash_rate": 0.25,
        "timeout_rate": 0.25,
        "invalid_action_rate": 0.25,
    }

    assert score_exported_submission(clean) > score_exported_submission(broken)
