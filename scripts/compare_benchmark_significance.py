from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _two_proportion_test(
    candidate_wins: float, candidate_games: int, baseline_wins: float, baseline_games: int
) -> dict[str, float]:
    if candidate_games <= 0 or baseline_games <= 0:
        return {"z": 0.0, "p_value": 1.0, "pooled_se": 0.0}
    p_candidate = candidate_wins / candidate_games
    p_baseline = baseline_wins / baseline_games
    pooled = (candidate_wins + baseline_wins) / (candidate_games + baseline_games)
    pooled_se = math.sqrt(
        max(pooled * (1.0 - pooled) * (1.0 / candidate_games + 1.0 / baseline_games), 0.0)
    )
    if pooled_se <= 0.0:
        return {"z": 0.0, "p_value": 1.0, "pooled_se": 0.0}
    z = (p_candidate - p_baseline) / pooled_se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"z": z, "p_value": p_value, "pooled_se": pooled_se}


def _extract_win_rates(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for item in report.get("formats", []):
        game_format = item.get("format", "unknown")
        if game_format == "2p":
            for opponent in item.get("opponents", []):
                summary = opponent.get("summary", {})
                label = f"2p:{opponent.get('opponent', 'unknown')}"
                games = int(summary.get("games", 0))
                win_rate = float(summary.get("win_rate", 0.0))
                metrics[label] = {
                    "games": games,
                    "wins": win_rate * games,
                    "win_rate": win_rate,
                    "mean_score_margin": float(summary.get("mean_score_margin", 0.0)),
                    "crash_rate": float(summary.get("crash_rate", 0.0)),
                    "timeout_rate": float(summary.get("timeout_rate", 0.0)),
                    "invalid_action_rate": float(summary.get("invalid_action_rate", 0.0)),
                }
        else:
            summary = item.get("summary", {})
            label = str(game_format)
            games = int(summary.get("games", 0))
            win_rate = float(summary.get("win_rate", 0.0))
            metrics[label] = {
                "games": games,
                "wins": win_rate * games,
                "win_rate": win_rate,
                "mean_score_margin": float(summary.get("mean_score_margin", 0.0)),
                "crash_rate": float(summary.get("crash_rate", 0.0)),
                "timeout_rate": float(summary.get("timeout_rate", 0.0)),
                "invalid_action_rate": float(summary.get("invalid_action_rate", 0.0)),
            }
    return metrics


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    alpha: float,
    min_games: int,
    min_effect: float,
) -> dict[str, Any]:
    baseline_metrics = _extract_win_rates(baseline)
    candidate_metrics = _extract_win_rates(candidate)
    labels = sorted(set(baseline_metrics) & set(candidate_metrics))
    comparisons = []
    for label in labels:
        base = baseline_metrics[label]
        cand = candidate_metrics[label]
        test = _two_proportion_test(
            cand["wins"], int(cand["games"]), base["wins"], int(base["games"])
        )
        candidate_ci = _wilson_interval(cand["wins"], int(cand["games"]))
        baseline_ci = _wilson_interval(base["wins"], int(base["games"]))
        delta = cand["win_rate"] - base["win_rate"]
        underpowered = int(cand["games"]) < min_games or int(base["games"]) < min_games
        significant = test["p_value"] < alpha and abs(delta) >= min_effect
        if underpowered:
            verdict = "underpowered"
        elif significant and delta < 0:
            verdict = "significant_regression"
        elif significant and delta > 0:
            verdict = "significant_improvement"
        else:
            verdict = "inconclusive"
        comparisons.append(
            {
                "label": label,
                "baseline_games": int(base["games"]),
                "candidate_games": int(cand["games"]),
                "baseline_win_rate": base["win_rate"],
                "candidate_win_rate": cand["win_rate"],
                "delta_win_rate": delta,
                "baseline_wilson_95": list(baseline_ci),
                "candidate_wilson_95": list(candidate_ci),
                "z": test["z"],
                "p_value": test["p_value"],
                "min_games": min_games,
                "min_effect": min_effect,
                "verdict": verdict,
            }
        )
    return {
        "baseline_submission": baseline.get("submission"),
        "candidate_submission": candidate.get("submission"),
        "alpha": alpha,
        "min_games": min_games,
        "min_effect": min_effect,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark_submission JSON reports with simple significance checks."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-games", type=int, default=128)
    parser.add_argument("--min-effect", type=float, default=0.05)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_reports(
        baseline,
        candidate,
        alpha=float(args.alpha),
        min_games=int(args.min_games),
        min_effect=float(args.min_effect),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
