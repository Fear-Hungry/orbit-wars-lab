from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_benchmark_significance import compare_reports

DEFAULT_BASELINE = Path(
    "artifacts/gates/producer_fix_gates/g2_champion_vs_corrected_producer_96seed.json"
)


def _summary_for_label(report: dict[str, Any], label: str) -> dict[str, Any] | None:
    for item in report.get("formats", []):
        game_format = item.get("format", "unknown")
        if game_format == "2p":
            for opponent in item.get("opponents", []):
                candidate_label = f"2p:{opponent.get('opponent', 'unknown')}"
                if candidate_label == label:
                    return dict(opponent.get("summary", {}))
        elif str(game_format) == label:
            return dict(item.get("summary", {}))
    return None


def _comparison_for_label(compare_report: dict[str, Any], label: str) -> dict[str, Any] | None:
    for comparison in compare_report.get("comparisons", []):
        if comparison.get("label") == label:
            return dict(comparison)
    return None


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def evaluate_promotion(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
    min_games: int,
    min_margin: float,
    alpha: float,
    min_effect: float,
) -> dict[str, Any]:
    summary = _summary_for_label(candidate, label)
    compare_report = compare_reports(
        baseline,
        candidate,
        alpha=float(alpha),
        min_games=int(min_games),
        min_effect=float(min_effect),
    )
    comparison = _comparison_for_label(compare_report, label)

    checks: list[dict[str, Any]] = []
    checks.append(_check("candidate_label_present", summary is not None, label=label))
    if summary is None:
        return {
            "passed": False,
            "label": label,
            "min_games": int(min_games),
            "min_margin": float(min_margin),
            "checks": checks,
            "comparison": comparison,
            "compare_report": compare_report,
        }

    games = int(summary.get("games", 0))
    margin = float(summary.get("mean_score_margin", 0.0))
    crash_rate = float(summary.get("crash_rate", 0.0))
    timeout_rate = float(summary.get("timeout_rate", 0.0))
    invalid_action_rate = float(summary.get("invalid_action_rate", 0.0))

    checks.extend(
        [
            _check("candidate_games", games >= int(min_games), games=games, min_games=int(min_games)),
            _check(
                "candidate_margin_floor",
                margin >= float(min_margin),
                mean_score_margin=margin,
                min_margin=float(min_margin),
            ),
            _check("candidate_crash_rate", crash_rate == 0.0, crash_rate=crash_rate),
            _check("candidate_timeout_rate", timeout_rate == 0.0, timeout_rate=timeout_rate),
            _check(
                "candidate_invalid_action_rate",
                invalid_action_rate == 0.0,
                invalid_action_rate=invalid_action_rate,
            ),
        ]
    )

    checks.append(_check("comparison_label_present", comparison is not None, label=label))
    if comparison is not None:
        verdict = str(comparison.get("verdict", "missing"))
        paired_games = float(comparison.get("paired_games", 0.0))
        paired_margin_delta = float(comparison.get("paired_margin_delta", 0.0))
        checks.extend(
            [
                _check(
                    "paired_games",
                    paired_games >= float(min_games),
                    paired_games=paired_games,
                    min_games=int(min_games),
                ),
                _check(
                    "paired_margin_not_below_baseline",
                    paired_margin_delta >= 0.0,
                    paired_margin_delta=paired_margin_delta,
                ),
                _check(
                    "no_significant_regression",
                    not verdict.endswith("_regression"),
                    verdict=verdict,
                ),
            ]
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "label": label,
        "min_games": int(min_games),
        "min_margin": float(min_margin),
        "candidate_summary": summary,
        "checks": checks,
        "comparison": comparison,
        "compare_report": compare_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promotion gate for OEP candidates against the local Producer ruler."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="2p:producer")
    parser.add_argument("--min-games", type=int, default=192)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.05)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = evaluate_promotion(
        baseline=baseline,
        candidate=candidate,
        label=str(args.label),
        min_games=int(args.min_games),
        min_margin=float(args.min_margin),
        alpha=float(args.alpha),
        min_effect=float(args.min_effect),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
