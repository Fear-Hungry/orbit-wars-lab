from __future__ import annotations

import json

from python.agents.registry import HEURISTIC_NAMES
from python.lab.cli import main


def test_lab_doctor_reports_required_files(capsys):
    assert main(["doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["submission_template"] is True
    assert payload["checks"]["quick_eval_config"] is True
    assert payload["checks"]["final_candidate_pool"] is True


def test_lab_heuristics_lists_known_policy_names(capsys):
    assert main(["heuristics"]) == 0

    names = capsys.readouterr().out.strip().splitlines()
    assert names == list(HEURISTIC_NAMES)


def test_lab_quick_dry_run_consolidates_export_and_benchmark(capsys):
    assert main(["quick", "--dry-run", "--seeds", "1", "--episode-steps", "16"]) == 0

    output = capsys.readouterr().out
    assert "scripts.export_submission" in output
    assert "scripts.benchmark_submission" in output
    assert "--submission artifacts/submission.py" in output
    assert "--out artifacts/submission_benchmark.json" in output


def test_lab_eval_and_league_dry_run_use_shared_artifact_paths(capsys):
    assert main(["eval", "--dry-run"]) == 0
    assert main(["league", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "python.train.evaluate_population" in output
    assert "--config configs/eval_quick.yaml" in output
    assert "--output artifacts/evaluation_report.json" in output
    assert "python.train.train_league" in output
    assert "--evaluation-report artifacts/evaluation_report.json" in output
