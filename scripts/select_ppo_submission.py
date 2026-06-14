from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from python.agents.registry import HEURISTIC_NAMES
from scripts.benchmark_ppo_submission import benchmark_exported_checkpoint
from scripts.benchmark_submission import (
    INSTRUMENTATION_FAILURE_METRIC,
    TECHNICAL_FAILURE_METRICS,
)
from scripts.select_ppo_checkpoint import _expand_checkpoints, checkpoint_id

ZERO_TOLERANCE_METRICS = (*TECHNICAL_FAILURE_METRICS, INSTRUMENTATION_FAILURE_METRIC)


def technical_failures_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for metric in ZERO_TOLERANCE_METRICS:
        value = float(summary.get(metric, 0.0))
        if value > 0.0:
            failures.append({"metric": metric, "value": value})
    return failures


def is_technically_eligible(summary: dict[str, Any]) -> bool:
    return not technical_failures_from_summary(summary)


def score_exported_submission(summary: dict[str, Any]) -> float:
    return (
        2.0 * float(summary.get("win_rate", 0.0))
        + 0.8 * float(summary.get("mean_score_margin", 0.0))
        - 2.0 * float(summary.get("crash_rate", 0.0))
        - 2.0 * float(summary.get("timeout_rate", 0.0))
        - 1.0 * float(summary.get("invalid_action_rate", 0.0))
        - 2.0 * float(summary.get("fallback_rate", 0.0))
        - 1.0 * float(summary.get("policy_illegal_move_rate", 0.0))
        - 2.0 * float(summary.get("fallback_error_rate", 0.0))
        - 2.0 * float(summary.get("instrumentation_missing_rate", 0.0))
    )


def select_exported_submissions(
    checkpoint_patterns: list[str],
    *,
    out_dir: Path,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    include_4p: bool,
) -> dict[str, Any]:
    checkpoints = _expand_checkpoints(checkpoint_patterns)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for checkpoint in checkpoints:
        candidate_id = checkpoint_id(checkpoint)
        report = benchmark_exported_checkpoint(
            checkpoint,
            submission_out=out_dir / f"{candidate_id}_submission.py",
            opponents=opponents,
            seeds=seeds,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
            include_4p=include_4p,
        )
        report_path = out_dir / f"{candidate_id}_submission_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        failures = technical_failures_from_summary(report["summary"])
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "candidate_id": candidate_id,
                "submission": report["submission"],
                "benchmark": str(report_path),
                "score": score_exported_submission(report["summary"]),
                "eligible": not failures,
                "technical_failures": failures,
                "summary": report["summary"],
            }
        )
    candidates.sort(
        key=lambda item: (
            item["eligible"],
            item["score"],
            item["summary"].get("win_rate", 0.0),
            item["summary"].get("mean_score_margin", 0.0),
        ),
        reverse=True,
    )
    if not candidates or not candidates[0]["eligible"]:
        raise RuntimeError("no technically eligible PPO submission candidates")
    return {
        "best_checkpoint": candidates[0]["checkpoint"],
        "best_submission": candidates[0]["submission"],
        "opponents": opponents,
        "seeds": seeds,
        "episode_steps": episode_steps,
        "enable_comets": enable_comets,
        "include_4p": include_4p,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out-dir", default="artifacts/ppo/exported_selection")
    parser.add_argument("--output", default="artifacts/ppo/exported_selection.json")
    parser.add_argument("--opponents", default=",".join(HEURISTIC_NAMES))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--skip-4p", action="store_true")
    args = parser.parse_args()

    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    report = select_exported_submissions(
        args.checkpoints,
        out_dir=Path(args.out_dir),
        opponents=opponents,
        seeds=list(range(max(1, int(args.seeds)))),
        episode_steps=int(args.episode_steps),
        enable_comets=not args.disable_comets,
        act_timeout=float(args.act_timeout),
        include_4p=not args.skip_4p,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"best_checkpoint": report["best_checkpoint"], "best_submission": report["best_submission"]}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
