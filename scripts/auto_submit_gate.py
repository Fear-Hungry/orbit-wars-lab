"""Deterministic, fail-closed submission gate — the ONLY path to a Kaggle submit.

Orbit Wars is an AGENT competition, not a tabular/CSV one: the submission is a
CPU-only ``agent(obs)`` Python file (invariants D10/D11), and the deciding metric
is the **paired normalized margin vs the local Producer** — NOT row/column CSV
checks. This gate therefore consumes the JSON written by
``scripts.benchmark_ppo_submission`` (paired eval) and, optionally, a
``scripts.gate_check`` report for the submission invariants + worst-decile.

Policy is owned by ``SUBMISSION_POLICY.md``; thresholds here mirror it and the
user's promotion bar in ``todo.md`` (beat ``-0.7491`` with folga, on the deciding
ruler). Every check FAILS CLOSED: any missing file, unmet threshold, unknown
budget, or duplicate fingerprint rejects the submission. Live submission happens
only when ``AUTO_SUBMIT=1`` AND every gate passes; the default is a dry run that
approves/rejects but never calls Kaggle.

Run dry (default — never submits)::

    KAGGLE_COMPETITION=<slug> python -m scripts.auto_submit_gate \
        --benchmark artifacts/ppo/campaign/eval_c00.json

Arm live submission (Nível 2.5, capped at 1/day)::

    AUTO_SUBMIT=1 KAGGLE_COMPETITION=<slug> python -m scripts.auto_submit_gate \
        --benchmark <paired_eval.json> --submission artifacts/submission.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


# Thresholds mirror SUBMISSION_POLICY.md (env-overridable).
AUTO_SUBMIT = os.environ.get("AUTO_SUBMIT", "0") == "1"
COMPETITION = os.environ.get("KAGGLE_COMPETITION", "")
MAX_AUTO_SUBMISSIONS_PER_DAY = _env_int("MAX_AUTO_SUBMISSIONS_PER_DAY", 1)
PROMOTION_FLOOR = _env_float("PROMOTION_FLOOR", -0.7491)
PROMOTION_FOLGA = _env_float("PROMOTION_FOLGA", 0.02)
MIN_MARGIN_DELTA = _env_float("MIN_MARGIN_DELTA", 0.01)
MIN_SEEDS = _env_int("MIN_SEEDS", 96)


class GateRejected(Exception):
    """Raised on any failed check; the gate exits non-zero and never submits."""


def fail(reason: str) -> None:
    raise GateRejected(reason)


def ok(msg: str) -> None:
    print(f"[GATE] {msg}", flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"could not read JSON {path}: {exc}")
        raise  # unreachable, keeps type-checkers happy


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _resolve_best_margin(spec: str | None) -> float:
    """Best margin to beat: a float literal, a JSON path, or auto-discovery."""
    if spec:
        try:
            return float(spec)
        except ValueError:
            data = _load_json(Path(spec))
            for key in ("best_margin", "mean_score_margin"):
                if key in data:
                    return float(data[key])
                if key in data.get("summary", {}):
                    return float(data["summary"][key])
            fail(f"no best margin found in {spec}")
    # Auto-discovery: latest campaign report, else -inf (any valid candidate beats it).
    report = _ROOT / "artifacts" / "ppo" / "campaign" / "campaign_report.json"
    if report.exists():
        data = _load_json(report)
        if data.get("best_margin") not in (None, float("-inf")):
            return float(data["best_margin"])
    return float("-inf")


def _read_submit_log_today(log_path: Path, today: str) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("date") == today and obj.get("competition") == COMPETITION:
            rows.append(obj)
    return rows


# --- Checks (each fails closed) --------------------------------------------


def check_metrics(bench: dict[str, Any], best_margin: float) -> dict[str, Any]:
    summary = bench.get("summary")
    if not isinstance(summary, dict):
        fail("benchmark JSON has no 'summary' object")

    required = ("mean_score_margin", "win_rate", "invalid_action_rate",
                "crash_rate", "timeout_rate")
    missing = [k for k in required if k not in summary]
    if missing:
        fail(f"benchmark summary missing keys: {missing}")

    # Submission-invariant proxies (D10/D11 + 'never degrade silently').
    for key in ("invalid_action_rate", "crash_rate", "timeout_rate"):
        if float(summary[key]) != 0.0:
            fail(f"{key}={summary[key]} (must be 0)")
    # fallback_rate / worst_decile only exist in richer reports; enforce if present.
    if "fallback_rate" in summary and float(summary["fallback_rate"]) != 0.0:
        fail(f"fallback_rate={summary['fallback_rate']} (silent fallback forbidden)")
    if "worst_decile_score_margin" in summary and \
            float(summary["worst_decile_score_margin"]) < 0.0:
        fail(f"worst_decile_score_margin={summary['worst_decile_score_margin']} < 0")

    seeds = bench.get("seeds")
    n_seeds = len(seeds) if isinstance(seeds, list) else int(bench.get("seeds", 0) or 0)
    if n_seeds < MIN_SEEDS:
        fail(f"only {n_seeds} seeds; deciding ruler needs >= {MIN_SEEDS} "
             f"(16-seed triage never auto-submits)")

    margin = float(summary["mean_score_margin"])
    bar = PROMOTION_FLOOR + PROMOTION_FOLGA
    if margin <= bar:
        fail(f"margin {margin:.4f} <= promotion bar {bar:.4f} "
             f"(floor {PROMOTION_FLOOR} + folga {PROMOTION_FOLGA})")
    if margin < best_margin + MIN_MARGIN_DELTA:
        fail(f"margin {margin:.4f} does not beat best {best_margin:.4f} "
             f"by >= {MIN_MARGIN_DELTA}")

    ok(f"metrics passed: margin={margin:.4f} win={summary['win_rate']:.3f} "
       f"seeds={n_seeds} best={best_margin:.4f}")
    return {"margin": margin, "seeds": n_seeds, "win_rate": float(summary["win_rate"])}


def check_gate_report(path: Path | None) -> None:
    """Optional: the canonical `gate-check-final` report (invariants + worst decile)."""
    if path is None:
        ok("no --gate-report given; relying on benchmark invariant proxies "
           "(run `make gate-check-final` for the full invariant set)")
        return
    data = _load_json(path)
    passed = data.get("passed")
    if passed is not True and not all(
        v.get("passed") is True for v in data.values() if isinstance(v, dict)
    ):
        fail(f"gate-check report {path} did not pass")
    ok(f"gate-check report passed: {path}")


def check_budget_and_fingerprint(submission: Path, log_path: Path, today: str) -> str:
    fingerprint = _file_sha256(submission)
    rows = _read_submit_log_today(log_path, today)
    if len(rows) >= MAX_AUTO_SUBMISSIONS_PER_DAY:
        fail(f"auto-submit budget exhausted: {len(rows)}/{MAX_AUTO_SUBMISSIONS_PER_DAY} today")
    for row in rows:
        if row.get("sha256") == fingerprint:
            fail("this exact submission was already submitted today (duplicate fingerprint)")

    # Team-wide 5/day is shared; query Kaggle best-effort, fail closed if unknown
    # while live submission is armed.
    if AUTO_SUBMIT and COMPETITION:
        team_today = _kaggle_submissions_today(today)
        if team_today is None:
            fail("could not determine team submission count from Kaggle; "
                 "refusing to auto-submit (check `kaggle competitions submissions` manually)")
        if team_today >= 5:
            fail(f"team already at {team_today} submissions today (Kaggle 5/day cap)")
        ok(f"budget ok: auto {len(rows)}/{MAX_AUTO_SUBMISSIONS_PER_DAY}, "
           f"team {team_today}/5")
    else:
        ok(f"budget ok (dry run): auto {len(rows)}/{MAX_AUTO_SUBMISSIONS_PER_DAY}")
    return fingerprint


def _kaggle_submissions_today(today: str) -> int | None:
    try:
        proc = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    lines = proc.stdout.strip().splitlines()
    if len(lines) < 1:
        return None
    header = lines[0].split(",")
    try:
        date_col = next(i for i, c in enumerate(header) if "date" in c.lower())
    except StopIteration:
        return None
    count = 0
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) > date_col and cells[date_col].startswith(today):
            count += 1
    return count


def submit(submission: Path, message: str, fingerprint: str,
           log_path: Path, today: str, result: dict[str, Any]) -> None:
    if not AUTO_SUBMIT:
        ok("APPROVED but AUTO_SUBMIT=0 → candidate NOT submitted (dry run). "
           "Set AUTO_SUBMIT=1 to arm live submission.")
        return
    if not COMPETITION:
        fail("AUTO_SUBMIT=1 but KAGGLE_COMPETITION is unset")

    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION,
           "-f", str(submission), "-m", message]
    print(f"[SUBMIT] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_ROOT), check=False)
    if proc.returncode != 0:
        fail(f"kaggle submit failed (rc={proc.returncode}) — NOT logged")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": today, "competition": COMPETITION, "sha256": fingerprint,
        "submission": str(submission), "message": message, **result,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    ok(f"SUBMITTED and logged: {submission.name} (fingerprint {fingerprint[:12]})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        help="paired-eval JSON from scripts.benchmark_ppo_submission")
    parser.add_argument("--submission", default=None,
                        help="agent file to submit (default: 'submission' field of the benchmark, "
                             "else artifacts/submission.py)")
    parser.add_argument("--best", default=None,
                        help="best margin to beat: float, JSON path, or auto-discover campaign report")
    parser.add_argument("--gate-report", default=None,
                        help="optional scripts.gate_check report JSON (invariants + worst decile)")
    parser.add_argument("--message", default=None, help="Kaggle submission message")
    parser.add_argument("--log", default="artifacts/auto_submit/submit_log.jsonl")
    parser.add_argument("--date", default=None, help="override today's date (YYYY-MM-DD), for tests")
    args = parser.parse_args()

    today = args.date or date.today().isoformat()
    bench_path = (_ROOT / args.benchmark) if not Path(args.benchmark).is_absolute() else Path(args.benchmark)
    log_path = (_ROOT / args.log) if not Path(args.log).is_absolute() else Path(args.log)

    try:
        bench = _load_json(bench_path)
        submission = Path(args.submission) if args.submission else \
            Path(bench.get("submission") or (_ROOT / "artifacts" / "submission.py"))
        if not submission.is_absolute():
            submission = _ROOT / submission
        if not submission.exists():
            fail(f"submission file not found: {submission}")

        best_margin = _resolve_best_margin(args.best)
        result = check_metrics(bench, best_margin)
        check_gate_report(Path(args.gate_report) if args.gate_report else None)
        fingerprint = check_budget_and_fingerprint(submission, log_path, today)
        message = args.message or f"auto-gated: margin={result['margin']:.4f} seeds={result['seeds']}"
        submit(submission, message, fingerprint, log_path, today, result)
    except GateRejected as exc:
        print(f"[GATE REJECTED] {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
