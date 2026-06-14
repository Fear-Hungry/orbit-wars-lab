from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from scripts.validate_pgs_tarball import _nonzero_forbidden_stats


def _write_tarball(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            source = path.parent / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(content, encoding="utf-8")
            tar.add(source, arcname=name, recursive=False)


def test_validate_flat_producer_tarball_without_stats(tmp_path: Path) -> None:
    tarball = tmp_path / "producer.tar.gz"
    _write_tarball(
        tarball,
        {
            "main.py": "def agent(obs):\n    return []\n",
            "_upstream.py": "# marker for flat Producer layout\n",
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_pgs_tarball.py",
            "--tarball",
            str(tarball),
            "--players",
            "2",
            "--seats",
            "0",
            "--skip-pgs-planner-check",
            "--allow-missing-submission-stats",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VALIDATION OK" in proc.stdout
    payload = json.loads(proc.stdout.split("VALIDATION OK")[0])
    assert payload["tarball_sha256"] == hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert payload["tarball_size"] == tarball.stat().st_size


def test_validate_flat_tarball_with_bundled_producer_agent(tmp_path: Path) -> None:
    tarball = tmp_path / "brep_like.tar.gz"
    _write_tarball(
        tarball,
        {
            "main.py": (
                "SUBMISSION_STATS = {'calls': 0, 'fallbacks': 0, "
                "'illegal_moves': 0, 'fallback_errors': 0}\n"
                "def agent(obs, *_):\n"
                "    SUBMISSION_STATS['calls'] += 1\n"
                "    return []\n"
            ),
            "_producer_agent.py": (
                "def make_agent():\n"
                "    def agent(obs):\n"
                "        return []\n"
                "    return agent\n"
            ),
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_pgs_tarball.py",
            "--tarball",
            str(tarball),
            "--players",
            "2",
            "--seats",
            "0",
            "--skip-pgs-planner-check",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VALIDATION OK" in proc.stdout


def test_forbidden_submission_stats_include_dynamic_fallback_counters() -> None:
    assert _nonzero_forbidden_stats({
        "calls": 10,
        "fallbacks": 0,
        "net_fallbacks": 2,
        "timeouts": 1,
        "timeout_thread_blocks": 0,
        "illegal_moves": 3,
        "fallback_errors": 4,
    }) == {
        "net_fallbacks": 2.0,
        "timeouts": 1.0,
        "illegal_moves": 3.0,
        "fallback_errors": 4.0,
    }


def test_validate_tarball_rejects_nonzero_dynamic_fallback_counter(tmp_path: Path) -> None:
    tarball = tmp_path / "brep_like_bad.tar.gz"
    _write_tarball(
        tarball,
        {
            "main.py": (
                "SUBMISSION_STATS = {'calls': 0, 'fallbacks': 0, "
                "'net_fallbacks': 1, 'illegal_moves': 0, 'fallback_errors': 0}\n"
                "def agent(obs, *_):\n"
                "    SUBMISSION_STATS['calls'] += 1\n"
                "    return []\n"
            ),
            "_producer_agent.py": (
                "def make_agent():\n"
                "    def agent(obs):\n"
                "        return []\n"
                "    return agent\n"
            ),
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_pgs_tarball.py",
            "--tarball",
            str(tarball),
            "--players",
            "2",
            "--seats",
            "0",
            "--skip-pgs-planner-check",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode != 0
    assert "net_fallbacks" in proc.stderr
    assert "silent degradation is forbidden" in proc.stderr
