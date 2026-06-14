from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.league_doctor import main, run_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def test_league_doctor_canaries_pass_without_real_smoke(tmp_path):
    report = run_diagnostics(skip_real_smoke=True, existing_glob=str(tmp_path / "*.json"))

    assert report["passed"] is True
    assert report["failed"] == []
    names = {check["name"] for check in report["checks"]}
    assert "crash_becomes_error_and_cannot_win" in names
    assert "timeout_exhausts_bank_and_cannot_win" in names
    assert "tarball_modules_are_isolated" in names
    assert "tarball_reexport_invalidates_cache" in names


def test_league_doctor_existing_artifact_warning_not_failure(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "mode": "2p",
        "games": [{
            "seed": 1,
            "seats": ["a", "b"],
            "final_ships": [2.0, 1.0],
            "winner_seat": 0,
            "winner": "a",
            "tie": False,
            "died_at": [None, None],
        }],
    }))

    report = run_diagnostics(skip_real_smoke=True, existing_glob=str(path))

    assert report["passed"] is True
    assert report["failed"] == []
    assert report["warnings"] == ["existing_artifacts_are_fully_audited"]


def test_league_doctor_strict_existing_turns_warning_into_failure(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "mode": "2p",
        "games": [{
            "seed": 1,
            "seats": ["a", "b"],
            "final_ships": [2.0, 1.0],
            "winner_seat": 0,
            "winner": "a",
            "tie": False,
            "died_at": [None, None],
        }],
    }))

    report = run_diagnostics(skip_real_smoke=True, existing_glob=str(path), strict_existing=True)

    assert report["passed"] is False
    assert report["failed"] == ["existing_artifacts_are_fully_audited"]


def test_league_doctor_cli_json(capsys, tmp_path):
    rc = main(["--skip-real-smoke", "--existing-glob", str(tmp_path / "*.json"), "--json"])

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True


def test_league_scripts_run_directly_without_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    match_out = tmp_path / "match.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/league_match.py"),
            "--agents",
            "producer,producer",
            "--seeds",
            "1",
            "--seed-base",
            "9400",
            "--steps",
            "2",
            "--out",
            str(match_out),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert match_out.exists()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/league_report.py"),
            str(match_out),
            "5",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "artifacts/league/v1/report.json").exists()
