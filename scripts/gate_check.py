from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from python.train.objective_validation import _export_runtime_validation
from scripts.benchmark_submission import (
    HEURISTIC_POLICIES,
    _load_submission_agent,
    _submission_runtime,
    benchmark_four_player,
    benchmark_two_player,
)
from scripts.export_submission import render_submission

DEFAULT_CONFIG = Path("configs/submission_gate.yaml")
DEFAULT_OUT_DIR = Path("artifacts/gate_check")


@dataclass(frozen=True)
class GateConfig:
    episode_steps: int
    enable_comets: bool
    act_timeout: float
    benchmark_seeds: int
    technical_seeds: list[int]
    holdout_seeds: list[int]
    final_seed_start: int
    final_seeds: int
    opponents: list[str]
    floors: dict[str, float]
    regression: dict[str, float]
    min_holdout_worst_decile: float
    critical_matchups: list[dict[str, Any]]


def _load_config(path: Path) -> GateConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = payload["submission_gate"]
    return GateConfig(
        episode_steps=int(cfg.get("episode_steps", 500)),
        enable_comets=bool(cfg.get("enable_comets", True)),
        act_timeout=float(cfg.get("act_timeout", 1.0)),
        benchmark_seeds=int(cfg.get("benchmark_seeds", 8)),
        technical_seeds=[int(seed) for seed in cfg.get("technical_seeds", [0, 1, 2, 3])],
        holdout_seeds=[int(seed) for seed in cfg.get("holdout_seeds", [])],
        final_seed_start=int(cfg.get("final_seed_start", 100)),
        final_seeds=int(cfg.get("final_seeds", 20)),
        opponents=[str(name) for name in cfg.get("opponents", list(HEURISTIC_POLICIES))],
        floors={str(key): float(value) for key, value in cfg.get("floors", {}).items()},
        regression={str(key): float(value) for key, value in cfg.get("regression", {}).items()},
        min_holdout_worst_decile=float(cfg.get("holdout", {}).get("min_worst_decile_score_margin", -0.3)),
        critical_matchups=[dict(item) for item in cfg.get("critical_matchups", [])],
    )


def _export_current_template(path: Path) -> Path:
    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_submission(template, checkpoint=None), encoding="utf-8")
    return path


def _run_pytest() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_submission_pipeline.py"]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=120)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "name": "gate_0_pytest",
            "command": cmd,
            "returncode": None,
            "passed": False,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": (stderr + "\npytest timed out after 120 seconds")[-4000:],
        }
    return {
        "name": "gate_0_pytest",
        "command": cmd,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _final_seed_list(cfg: GateConfig) -> list[int]:
    return list(range(cfg.final_seed_start, cfg.final_seed_start + max(1, cfg.final_seeds)))


def _load_runtime(path: Path):
    return _submission_runtime(_load_submission_agent(path))


def _run_benchmark(path: Path, cfg: GateConfig, seeds: list[int], *, include_4p: bool = True) -> dict[str, Any]:
    runtime = _load_runtime(path)
    unknown = [name for name in cfg.opponents if name not in HEURISTIC_POLICIES]
    if unknown:
        raise ValueError(f"unknown opponents: {unknown}")
    two_player = [
        benchmark_two_player(
            runtime,
            name,
            HEURISTIC_POLICIES[name],
            seeds=seeds,
            episode_steps=cfg.episode_steps,
            enable_comets=cfg.enable_comets,
            act_timeout=cfg.act_timeout,
        )
        for name in cfg.opponents
    ]
    formats: list[dict[str, Any]] = [{"format": "2p", "opponents": two_player}]
    if include_4p:
        formats.append(
            benchmark_four_player(
                runtime,
                cfg.opponents,
                seeds=seeds,
                episode_steps=cfg.episode_steps,
                enable_comets=cfg.enable_comets,
                act_timeout=cfg.act_timeout,
            )
        )
    return {
        "submission": str(path),
        "seeds": seeds,
        "episode_steps": cfg.episode_steps,
        "enable_comets": cfg.enable_comets,
        "formats": formats,
    }


def _iter_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in report.get("formats", []):
        if item.get("format") == "2p":
            for opponent in item.get("opponents", []):
                records.extend(opponent.get("records", []))
        elif item.get("format") == "4p":
            records.extend(item.get("records", []))
    return records


def _worst_decile(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    cutoff = max(1, math.ceil(0.1 * len(ordered)))
    return float(fmean(ordered[:cutoff]))


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    two_player_summaries = []
    for item in report.get("formats", []):
        if item.get("format") == "2p":
            two_player_summaries = [opponent["summary"] for opponent in item.get("opponents", [])]
            break
    margins = [float(record["normalized_margin"]) for record in _iter_records(report)]
    return {
        "win_rate_2p_mean": fmean(float(summary["win_rate"]) for summary in two_player_summaries)
        if two_player_summaries
        else 0.0,
        "mean_score_margin": fmean(margins) if margins else 0.0,
        "worst_decile_score_margin": _worst_decile(margins),
    }


def _gate_technical(path: Path, cfg: GateConfig) -> dict[str, Any]:
    validations = [
        _export_runtime_validation(
            path,
            num_players=2,
            seeds=cfg.technical_seeds,
            episode_steps=cfg.episode_steps,
            enable_comets=cfg.enable_comets,
            act_timeout=cfg.act_timeout,
        ),
        _export_runtime_validation(
            path,
            num_players=4,
            seeds=cfg.technical_seeds,
            episode_steps=cfg.episode_steps,
            enable_comets=cfg.enable_comets,
            act_timeout=cfg.act_timeout,
        ),
    ]
    passed = all(
        validation["crash_rate"] == 0.0
        and validation["timeout_rate"] == 0.0
        and validation["invalid_action_rate"] == 0.0
        and validation["all_done"]
        for validation in validations
    )
    return {"name": "gate_1_technical_robustness", "passed": passed, "validations": validations}


def _gate_floors(report: dict[str, Any], cfg: GateConfig) -> dict[str, Any]:
    checks = []
    for item in report.get("formats", []):
        if item.get("format") == "2p":
            for opponent in item.get("opponents", []):
                name = str(opponent["opponent"])
                floor = cfg.floors.get(name)
                if floor is None:
                    continue
                win_rate = float(opponent["summary"]["win_rate"])
                checks.append({"opponent": name, "win_rate": win_rate, "floor": floor, "passed": win_rate >= floor})
        elif item.get("format") == "4p":
            floor = cfg.floors.get("four_player")
            if floor is not None:
                win_rate = float(item["summary"]["win_rate"])
                checks.append({"opponent": "four_player", "win_rate": win_rate, "floor": floor, "passed": win_rate >= floor})
    return {"name": "gate_2_per_opponent_floors", "passed": all(check["passed"] for check in checks), "checks": checks}


def _gate_critical_matchups(path: Path, cfg: GateConfig) -> dict[str, Any]:
    checks = []
    if not cfg.critical_matchups:
        return {"name": "gate_2b_critical_matchups", "passed": True, "checks": checks}

    runtime = _load_runtime(path)
    for item in cfg.critical_matchups:
        opponent_name = str(item["opponent"])
        if opponent_name not in HEURISTIC_POLICIES:
            raise ValueError(f"unknown critical matchup opponent: {opponent_name}")
        seed = int(item["seed"])
        expected_player = int(item["submission_player"])
        min_win_points = float(item.get("min_win_points", 1.0))
        min_margin = float(item.get("min_normalized_margin", -1.0))
        benchmark = benchmark_two_player(
            runtime,
            opponent_name,
            HEURISTIC_POLICIES[opponent_name],
            seeds=[seed],
            episode_steps=cfg.episode_steps,
            enable_comets=cfg.enable_comets,
            act_timeout=cfg.act_timeout,
        )
        matches = [
            record
            for record in benchmark["records"]
            if int(record["seed"]) == seed and int(record["submission_player"]) == expected_player
        ]
        if len(matches) != 1:
            checks.append(
                {
                    "opponent": opponent_name,
                    "seed": seed,
                    "submission_player": expected_player,
                    "passed": False,
                    "error": f"expected one record, found {len(matches)}",
                }
            )
            continue
        record = matches[0]
        win_points = float(record["win_points"])
        margin = float(record["normalized_margin"])
        checks.append(
            {
                "opponent": opponent_name,
                "seed": seed,
                "submission_player": expected_player,
                "win_points": win_points,
                "min_win_points": min_win_points,
                "normalized_margin": margin,
                "min_normalized_margin": min_margin,
                "passed": win_points >= min_win_points and margin >= min_margin,
            }
        )
    return {"name": "gate_2b_critical_matchups", "passed": all(check["passed"] for check in checks), "checks": checks}


def _gate_regression(candidate: dict[str, Any], baseline: dict[str, Any], cfg: GateConfig) -> dict[str, Any]:
    candidate_metrics = _summary_metrics(candidate)
    baseline_metrics = _summary_metrics(baseline)
    thresholds = {
        "win_rate_2p_mean": cfg.regression.get("max_2p_win_rate_drop", 0.05),
        "mean_score_margin": cfg.regression.get("max_mean_score_margin_drop", 0.10),
        "worst_decile_score_margin": cfg.regression.get("max_worst_decile_score_margin_drop", 0.10),
    }
    checks = []
    for metric, allowed_drop in thresholds.items():
        minimum = baseline_metrics[metric] - allowed_drop
        value = candidate_metrics[metric]
        checks.append(
            {
                "metric": metric,
                "candidate": value,
                "baseline": baseline_metrics[metric],
                "minimum": minimum,
                "passed": value >= minimum,
            }
        )
    return {
        "name": "gate_3_regression_vs_baseline",
        "passed": all(check["passed"] for check in checks),
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "checks": checks,
    }


def _gate_holdout(report: dict[str, Any], cfg: GateConfig) -> dict[str, Any]:
    metrics = _summary_metrics(report)
    return {
        "name": "gate_4_holdout_worst_decile",
        "passed": metrics["worst_decile_score_margin"] >= cfg.min_holdout_worst_decile,
        "minimum": cfg.min_holdout_worst_decile,
        "metrics": metrics,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--submission-out", default=None)
    parser.add_argument("--baseline", default="artifacts/submission.py")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--include-final", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    out_dir = Path(args.out_dir)
    submission_path = Path(args.submission_out) if args.submission_out else out_dir / "current_submission.py"
    submission_path = _export_current_template(submission_path)
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"baseline submission not found: {baseline_path}")

    gates = []
    if not args.skip_pytest:
        gates.append(_run_pytest())
    gates.append(_gate_technical(submission_path, cfg))

    benchmark_seeds = list(range(max(1, cfg.benchmark_seeds)))
    candidate_report = _run_benchmark(submission_path, cfg, benchmark_seeds)
    baseline_report = _run_benchmark(baseline_path, cfg, benchmark_seeds)
    _write_report(out_dir / "candidate_benchmark.json", candidate_report)
    _write_report(out_dir / "baseline_benchmark.json", baseline_report)

    gates.append(_gate_floors(candidate_report, cfg))
    gates.append(_gate_critical_matchups(submission_path, cfg))
    gates.append(_gate_regression(candidate_report, baseline_report, cfg))

    holdout_report = None
    if not args.skip_holdout:
        holdout_report = _run_benchmark(submission_path, cfg, cfg.holdout_seeds)
        _write_report(out_dir / "holdout_benchmark.json", holdout_report)
        gates.append(_gate_holdout(holdout_report, cfg))

    final_report = None
    if args.include_final:
        final_report = _run_benchmark(submission_path, cfg, _final_seed_list(cfg))
        _write_report(out_dir / "final_20_seed_benchmark.json", final_report)
        final_floor_gate = _gate_floors(final_report, cfg)
        final_floor_gate["name"] = "gate_5_final_20_seed_floors"
        gates.append(final_floor_gate)

    report = {
        "passed": all(gate["passed"] for gate in gates),
        "config": str(Path(args.config)),
        "submission": str(submission_path),
        "baseline": str(baseline_path),
        "gates": gates,
        "candidate_summary": _summary_metrics(candidate_report),
        "baseline_summary": _summary_metrics(baseline_report),
        "holdout_summary": _summary_metrics(holdout_report) if holdout_report else None,
        "final_summary": _summary_metrics(final_report) if final_report else None,
    }
    _write_report(out_dir / "gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
