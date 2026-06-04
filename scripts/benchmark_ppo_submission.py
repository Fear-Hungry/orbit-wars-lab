from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.benchmark_submission import (
    HEURISTIC_POLICIES,
    _resolve_opponent,
    benchmark_four_player_spec,
    benchmark_two_player_spec,
)
from scripts.export_submission import render_submission


def _write_exported_submission(checkpoint: Path, out: Path) -> Path:
    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_submission(template, checkpoint=str(checkpoint)), encoding="utf-8")
    return out


def _iter_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in report.get("formats", []):
        if item.get("format") == "2p":
            for opponent in item.get("opponents", []):
                records.extend(opponent.get("records", []))
        elif item.get("format") == "4p":
            records.extend(item.get("records", []))
    return records


def _summary(report: dict[str, Any]) -> dict[str, float]:
    records = _iter_records(report)
    if not records:
        return {
            "games": 0.0,
            "win_rate": 0.0,
            "mean_score_margin": 0.0,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
            "invalid_action_rate": 0.0,
            "mean_decision_ms": 0.0,
        }
    decisions = sum(float(record["decision_turns"]) for record in records)
    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    return {
        "games": float(len(records)),
        "win_rate": float(fmean(float(record["win_points"]) for record in records)),
        "mean_score_margin": float(fmean(float(record["normalized_margin"]) for record in records)),
        "crash_rate": float(sum(float(record["crashes"]) for record in records) / max(decisions, 1.0)),
        "timeout_rate": float(sum(float(record["timeouts"]) for record in records) / max(decisions, 1.0)),
        "invalid_action_rate": float(sum(float(record["invalid_actions"]) for record in records) / max(decisions, 1.0)),
        "mean_decision_ms": float(1000.0 * elapsed / max(decisions, 1.0)),
    }


def benchmark_exported_checkpoint(
    checkpoint: Path,
    *,
    submission_out: Path,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    include_4p: bool,
    jobs: int = 1,
) -> dict[str, Any]:
    for name in opponents:
        _resolve_opponent(name)
    exported = _write_exported_submission(checkpoint, submission_out)
    two_player = [
        benchmark_two_player_spec(
            exported,
            spec,
            seeds=seeds,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
            jobs=jobs,
        )
        for spec in opponents
    ]
    formats: list[dict[str, Any]] = [{"format": "2p", "opponents": two_player}]
    if include_4p:
        formats.append(
            benchmark_four_player_spec(
                exported,
                opponents,
                seeds=seeds,
                episode_steps=episode_steps,
                enable_comets=enable_comets,
                act_timeout=act_timeout,
                jobs=jobs,
            )
        )
    report = {
        "checkpoint": str(checkpoint),
        "submission": str(exported),
        "submission_size_bytes": exported.stat().st_size,
        "seeds": seeds,
        "episode_steps": episode_steps,
        "enable_comets": enable_comets,
        "jobs": jobs,
        "formats": formats,
    }
    report["summary"] = _summary(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission-out", default="artifacts/ppo/exported_submission.py")
    parser.add_argument("--out", default="artifacts/ppo/exported_submission_benchmark.json")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--opponents", nargs="+", default=list(HEURISTIC_POLICIES))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--disable-comets", action="store_true")
    args = parser.parse_args()

    report = benchmark_exported_checkpoint(
        Path(args.checkpoint),
        submission_out=Path(args.submission_out),
        opponents=list(args.opponents),
        seeds=list(range(max(1, int(args.seeds)))),
        episode_steps=int(args.episode_steps),
        enable_comets=not args.disable_comets,
        act_timeout=float(args.act_timeout),
        include_4p=not args.skip_4p,
        jobs=max(1, int(args.jobs)),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "submission": report["submission"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
