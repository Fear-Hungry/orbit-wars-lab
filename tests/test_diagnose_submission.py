from __future__ import annotations

from pathlib import Path

from scripts.diagnose_submission import diagnose_match


def test_diagnose_match_records_timeline_and_outcome():
    report = diagnose_match(
        Path("python/submission/submission_template.py"),
        opponent_name="weak_random",
        seed=1,
        submission_player=0,
        episode_steps=20,
        early_turns=3,
        sample_every=10,
    )

    assert report["opponent"] == "weak_random"
    assert report["seed"] == 1
    assert report["submission_player"] == 0
    assert len(report["scores"]) == 2
    assert all(isinstance(score, float) for score in report["scores"])
    assert "normalized_margin" in report
    assert report["timeline"]
    assert report["timeline"][0]["turn"] == 0
    assert "totals_before" in report["timeline"][0]
    assert "submission_actions" in report["timeline"][0]
    assert "submission_action_targets" in report["timeline"][0]
    assert "opponent_action_targets" in report["timeline"][0]
    if report["timeline"][0]["opponent_actions"]:
        target = report["timeline"][0]["opponent_action_targets"][0]
        assert {"action", "target_id", "score"}.issubset(target)
    assert "final_totals" in report
