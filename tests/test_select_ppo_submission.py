from __future__ import annotations

from pathlib import Path

import pytest
from scripts import select_ppo_submission as selector
from scripts.select_ppo_submission import (
    is_technically_eligible,
    score_exported_submission,
    select_exported_submissions,
)


def test_score_exported_submission_penalizes_runtime_failures():
    clean = {
        "win_rate": 0.50,
        "mean_score_margin": 0.20,
        "crash_rate": 0.0,
        "timeout_rate": 0.0,
        "invalid_action_rate": 0.0,
        "fallback_rate": 0.0,
        "policy_illegal_move_rate": 0.0,
        "fallback_error_rate": 0.0,
        "instrumentation_missing_rate": 0.0,
    }
    broken = {
        **clean,
        "win_rate": 1.0,
        "crash_rate": 0.25,
        "timeout_rate": 0.25,
        "invalid_action_rate": 0.25,
        "fallback_rate": 0.25,
        "policy_illegal_move_rate": 0.25,
        "fallback_error_rate": 0.25,
        "instrumentation_missing_rate": 0.25,
    }

    assert score_exported_submission(clean) > score_exported_submission(broken)
    assert is_technically_eligible(clean)
    assert not is_technically_eligible(broken)


def _summary(**overrides):
    base = {
        "games": 2.0,
        "win_rate": 0.5,
        "mean_score_margin": 0.0,
        "crash_rate": 0.0,
        "timeout_rate": 0.0,
        "invalid_action_rate": 0.0,
        "fallback_rate": 0.0,
        "policy_illegal_move_rate": 0.0,
        "fallback_error_rate": 0.0,
        "instrumentation_missing_rate": 0.0,
    }
    base.update(overrides)
    return base


def test_select_exported_submissions_never_promotes_technically_broken(monkeypatch, tmp_path):
    clean = tmp_path / "clean.pt"
    broken = tmp_path / "broken.pt"

    monkeypatch.setattr(selector, "_expand_checkpoints", lambda patterns: [broken, clean])

    def fake_benchmark(checkpoint: Path, *, submission_out: Path, **kwargs):
        if checkpoint == broken:
            summary = _summary(win_rate=1.0, mean_score_margin=1.0, fallback_rate=0.25)
        else:
            summary = _summary(win_rate=0.1, mean_score_margin=-0.5)
        return {"submission": str(submission_out), "summary": summary}

    monkeypatch.setattr(selector, "benchmark_exported_checkpoint", fake_benchmark)

    report = select_exported_submissions(
        ["*.pt"],
        out_dir=tmp_path / "out",
        opponents=["producer"],
        seeds=[0],
        episode_steps=10,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=False,
    )

    assert report["best_checkpoint"] == str(clean)
    assert report["candidates"][0]["eligible"] is True
    assert report["candidates"][1]["eligible"] is False
    assert report["candidates"][1]["technical_failures"] == [
        {"metric": "fallback_rate", "value": 0.25}
    ]


def test_select_exported_submissions_fails_when_all_candidates_are_broken(monkeypatch, tmp_path):
    broken = tmp_path / "broken.pt"
    monkeypatch.setattr(selector, "_expand_checkpoints", lambda patterns: [broken])
    monkeypatch.setattr(
        selector,
        "benchmark_exported_checkpoint",
        lambda checkpoint, *, submission_out, **kwargs: {
            "submission": str(submission_out),
            "summary": _summary(timeout_rate=0.1),
        },
    )

    with pytest.raises(RuntimeError, match="no technically eligible"):
        select_exported_submissions(
            ["*.pt"],
            out_dir=tmp_path / "out",
            opponents=["producer"],
            seeds=[0],
            episode_steps=10,
            enable_comets=True,
            act_timeout=1.0,
            include_4p=False,
        )
