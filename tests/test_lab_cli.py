from __future__ import annotations

import json

from python.agents.registry import HEURISTIC_NAMES
from python.lab.cli import main


def test_lab_doctor_reports_required_files(capsys):
    assert main(["doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["files"]["submission_template"] is True
    assert payload["files"]["quick_eval_config"] is True
    assert payload["files"]["final_candidate_pool"] is True


def test_lab_doctor_verifies_backend_and_toolchain(capsys):
    # The doctor must not be shallow: it has to actually import the Rust backend
    # and the official engine, and confirm the build toolchain is present.
    assert main(["doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    for module in ("orbit_wars_rs", "kaggle_environments", "duckdb"):
        assert payload["imports"][module]["ok"] is True, payload["imports"][module]
    for tool in ("maturin", "cargo", "rustc"):
        assert payload["tools"][tool]["ok"] is True, payload["tools"][tool]


def test_lab_doctor_fails_when_backend_import_missing(monkeypatch, capsys):
    # If orbit_wars_rs (or any required import) cannot be imported, doctor must
    # return a non-zero exit code instead of silently passing.
    import python.lab.cli as cli

    real_import = cli.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "orbit_wars_rs":
            raise ModuleNotFoundError("No module named 'orbit_wars_rs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)

    assert main(["doctor"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["imports"]["orbit_wars_rs"]["ok"] is False


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


def test_lab_quick_is_2p_only_by_default(capsys):
    # quick is a crash/failure smoke check, not a promotion gate: it must run the
    # seat-rotated 2p path only and never the seat-0-pinned 4p path by default.
    assert main(["quick", "--dry-run", "--seeds", "1", "--episode-steps", "16"]) == 0

    output = capsys.readouterr().out
    assert "--skip-4p" in output
    assert "--ack-seat-biased" not in output


def test_lab_quick_dirty_4p_opts_into_seat_biased_diagnostic(capsys):
    # --dirty-4p runs the seat-0-pinned 4p path as a diagnostic: it drops --skip-4p
    # and acknowledges the seat bias, and warns it is not a promotion gate.
    assert (
        main(["quick", "--dirty-4p", "--dry-run", "--seeds", "1", "--episode-steps", "16"])
        == 0
    )

    captured = capsys.readouterr()
    assert "--skip-4p" not in captured.out
    assert "--ack-seat-biased" in captured.out
    assert "NOT a promotion gate" in captured.err


def test_lab_eval_and_league_dry_run_use_shared_artifact_paths(capsys):
    assert main(["eval", "--dry-run"]) == 0
    assert main(["league", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "python.train.evaluate_population" in output
    assert "--config configs/eval_quick.yaml" in output
    assert "--output artifacts/evaluation_report.json" in output
    assert "python.train.train_league" in output
    assert "--evaluation-report artifacts/evaluation_report.json" in output
