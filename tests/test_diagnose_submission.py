from __future__ import annotations

from pathlib import Path

from scripts.diagnose_submission import _action_target, diagnose_match


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
    assert "economy_summary" in report
    assert set(report["economy_summary"]) == {"0", "1"}
    assert "launch_events" in report
    assert "capture_events" in report
    if report["launch_events"]:
        assert {
            "turn",
            "actor",
            "player",
            "source_id",
            "ships",
            "target_id",
            "target_owner",
            "target_ships",
            "target_production",
        }.issubset(report["launch_events"][0])
    if report["capture_events"]:
        assert {
            "turn",
            "planet_id",
            "old_owner",
            "new_owner",
            "ships",
            "production",
            "rotating",
        }.issubset(report["capture_events"][0])


def test_action_target_uses_predicted_rotating_planet_position():
    state = {
        "angular_velocity": 0.05,
        "planets": [
            {"id": 0, "owner": 0, "x": 19.0, "y": 34.0, "radius": 2.0, "ships": 30, "production": 4},
            {"id": 1, "owner": -1, "x": 31.0, "y": 69.0, "radius": 2.0, "ships": 6, "production": 5},
            {"id": 2, "owner": -1, "x": 31.0, "y": 31.0, "radius": 2.0, "ships": 6, "production": 5},
        ],
        "fleets": [],
    }

    target = _action_target(state, [0, 1.25, 8])

    assert target["target_id"] == 1
    assert target["score"] is not None
