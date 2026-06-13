from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from python.agents.registry import HEURISTIC_NAMES
from python.league.evaluation import AgentSpec
from python.train.evaluate_population import evaluate_population, load_evaluation_config


def checkpoint_id(path: Path) -> str:
    return path.stem.replace("-", "_").replace(".", "_")


def checkpoint_candidate_id(path: Path) -> str:
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    parent = resolved.parent.name.replace("-", "_").replace(".", "_")
    stem = checkpoint_id(resolved)
    return f"{parent}_{stem}_{digest}" if parent else f"{stem}_{digest}"


def score_metrics(metrics: dict[str, Any]) -> float:
    return (
        2.0 * float(metrics.get("win_rate_vs_league", 0.0))
        + 0.8 * float(metrics.get("normalized_score_margin", 0.0))
        + 0.25 * float(metrics.get("robustness_across_seeds", 0.0))
        - 2.0 * float(metrics.get("crash_rate", 0.0))
        - 2.0 * float(metrics.get("timeout_rate", 0.0))
        - 1.0 * float(metrics.get("invalid_action_rate", 0.0))
    )


def _expand_checkpoints(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern, recursive=True))
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no checkpoint files matched the provided patterns")
    return unique


def _evaluate_checkpoint(path: Path, config_path: str, heuristics: tuple[str, ...]) -> dict[str, Any]:
    spec = AgentSpec(id=checkpoint_id(path), kind="ppo", role="league", checkpoint=str(path))
    manifest = {
        "population": [spec],
        "hall_of_fame": [],
        "heuristics": [
            AgentSpec(id=name, kind="heuristic", role="heuristic", policy=name)
            for name in heuristics
        ],
    }
    report = evaluate_population(manifest, load_evaluation_config(config_path))
    metrics = report["members"][0]["metrics"]
    return {
        "checkpoint": str(path),
        "candidate_id": spec.id,
        "score": score_metrics(metrics),
        "metrics": metrics,
        "summary": report["summary"],
    }


def select_checkpoints(
    checkpoint_patterns: list[str],
    *,
    config_path: str,
    heuristics: tuple[str, ...] = HEURISTIC_NAMES,
) -> dict[str, Any]:
    checkpoints = _expand_checkpoints(checkpoint_patterns)
    candidates = [
        _evaluate_checkpoint(path, config_path, heuristics)
        for path in checkpoints
    ]
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["metrics"].get("win_rate_vs_league", 0.0),
            item["metrics"].get("normalized_score_margin", 0.0),
        ),
        reverse=True,
    )
    return {
        "best_checkpoint": candidates[0]["checkpoint"],
        "config": config_path,
        "heuristics": list(heuristics),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--config", default="configs/eval_quick.yaml")
    parser.add_argument("--heuristics", default=",".join(HEURISTIC_NAMES))
    parser.add_argument("--output", default="artifacts/ppo/checkpoint_selection.json")
    args = parser.parse_args()

    heuristics = tuple(item.strip() for item in args.heuristics.split(",") if item.strip())
    report = select_checkpoints(args.checkpoints, config_path=args.config, heuristics=heuristics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"best_checkpoint": report["best_checkpoint"], "candidates": len(report["candidates"])}, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
