from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
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


def _paired_normal_test(deltas: list[float]) -> dict[str, float]:
    if not deltas:
        return {"paired_n": 0.0, "paired_mean_delta": 0.0, "paired_z": 0.0, "paired_p_value": 1.0}
    mean_delta = fmean(deltas)
    if len(deltas) < 2:
        return {
            "paired_n": float(len(deltas)),
            "paired_mean_delta": mean_delta,
            "paired_z": 0.0,
            "paired_p_value": 1.0,
        }
    variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
    standard_error = math.sqrt(variance / len(deltas))
    if standard_error <= 0.0:
        p_value = 1.0 if abs(mean_delta) <= 0.0 else 0.0
        return {
            "paired_n": float(len(deltas)),
            "paired_mean_delta": mean_delta,
            "paired_z": 0.0,
            "paired_p_value": p_value,
        }
    z = mean_delta / standard_error
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {
        "paired_n": float(len(deltas)),
        "paired_mean_delta": mean_delta,
        "paired_z": z,
        "paired_p_value": p_value,
    }


def _welch_margin_test(
    candidate_values: list[float], baseline_values: list[float]
) -> dict[str, float]:
    if not candidate_values or not baseline_values:
        return {
            "margin_n_candidate": float(len(candidate_values)),
            "margin_n_baseline": float(len(baseline_values)),
            "margin_delta": 0.0,
            "margin_welch_t": 0.0,
            "margin_welch_df": 0.0,
            "margin_p_value": 1.0,
        }
    candidate_mean = fmean(candidate_values)
    baseline_mean = fmean(baseline_values)
    delta = candidate_mean - baseline_mean
    if len(candidate_values) < 2 or len(baseline_values) < 2:
        return {
            "margin_n_candidate": float(len(candidate_values)),
            "margin_n_baseline": float(len(baseline_values)),
            "margin_delta": delta,
            "margin_welch_t": 0.0,
            "margin_welch_df": 0.0,
            "margin_p_value": 1.0,
        }

    candidate_var = sum((value - candidate_mean) ** 2 for value in candidate_values) / (
        len(candidate_values) - 1
    )
    baseline_var = sum((value - baseline_mean) ** 2 for value in baseline_values) / (
        len(baseline_values) - 1
    )
    candidate_term = candidate_var / len(candidate_values)
    baseline_term = baseline_var / len(baseline_values)
    standard_error = math.sqrt(candidate_term + baseline_term)
    if standard_error <= 0.0:
        p_value = 1.0 if abs(delta) <= 0.0 else 0.0
        return {
            "margin_n_candidate": float(len(candidate_values)),
            "margin_n_baseline": float(len(baseline_values)),
            "margin_delta": delta,
            "margin_welch_t": 0.0,
            "margin_welch_df": 0.0,
            "margin_p_value": p_value,
        }

    t_stat = delta / standard_error
    df_num = (candidate_term + baseline_term) ** 2
    df_den = 0.0
    if len(candidate_values) > 1:
        df_den += candidate_term**2 / (len(candidate_values) - 1)
    if len(baseline_values) > 1:
        df_den += baseline_term**2 / (len(baseline_values) - 1)
    df = df_num / df_den if df_den > 0.0 else 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return {
        "margin_n_candidate": float(len(candidate_values)),
        "margin_n_baseline": float(len(baseline_values)),
        "margin_delta": delta,
        "margin_welch_t": t_stat,
        "margin_welch_df": df,
        "margin_p_value": p_value,
    }


def _record_key(record: dict[str, Any], *, include_lineup: bool) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (int(record.get("seed", -1)),)
    if "submission_player" in record:
        key += (int(record.get("submission_player", -1)),)
    if include_lineup:
        key += (tuple(record.get("lineup", [])),)
    return key


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


def _extract_records(report: dict[str, Any]) -> dict[str, dict[tuple[Any, ...], dict[str, float]]]:
    records_by_label: dict[str, dict[tuple[Any, ...], dict[str, float]]] = {}
    for item in report.get("formats", []):
        game_format = item.get("format", "unknown")
        if game_format == "2p":
            for opponent in item.get("opponents", []):
                label = f"2p:{opponent.get('opponent', 'unknown')}"
                records_by_label[label] = {
                    _record_key(record, include_lineup=False): record
                    for record in opponent.get("records", [])
                }
        else:
            label = str(game_format)
            records_by_label[label] = {
                _record_key(record, include_lineup=True): record
                for record in item.get("records", [])
            }
    return records_by_label


def _paired_metrics(
    baseline_records: dict[tuple[Any, ...], dict[str, float]],
    candidate_records: dict[tuple[Any, ...], dict[str, float]],
) -> dict[str, float]:
    shared_keys = sorted(set(baseline_records) & set(candidate_records))
    win_deltas = [
        float(candidate_records[key].get("win_points", 0.0))
        - float(baseline_records[key].get("win_points", 0.0))
        for key in shared_keys
    ]
    margin_deltas = [
        float(candidate_records[key].get("normalized_margin", 0.0))
        - float(baseline_records[key].get("normalized_margin", 0.0))
        for key in shared_keys
    ]
    win_test = _paired_normal_test(win_deltas)
    margin_test = _paired_normal_test(margin_deltas)
    return {
        "paired_games": float(len(shared_keys)),
        "paired_win_delta": win_test["paired_mean_delta"],
        "paired_win_z": win_test["paired_z"],
        "paired_win_p_value": win_test["paired_p_value"],
        "paired_margin_delta": margin_test["paired_mean_delta"],
        "paired_margin_z": margin_test["paired_z"],
        "paired_margin_p_value": margin_test["paired_p_value"],
    }


def _margin_values(records: dict[tuple[Any, ...], dict[str, float]]) -> list[float]:
    return [float(record.get("normalized_margin", 0.0)) for record in records.values()]


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
    baseline_records = _extract_records(baseline)
    candidate_records = _extract_records(candidate)
    labels = sorted(set(baseline_metrics) & set(candidate_metrics))
    comparisons = []
    for label in labels:
        base = baseline_metrics[label]
        cand = candidate_metrics[label]
        paired = _paired_metrics(baseline_records.get(label, {}), candidate_records.get(label, {}))
        margin = _welch_margin_test(
            _margin_values(candidate_records.get(label, {})),
            _margin_values(baseline_records.get(label, {})),
        )
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
        elif margin["margin_p_value"] < alpha and margin["margin_delta"] <= -min_effect:
            verdict = "margin_significant_regression"
        elif margin["margin_p_value"] < alpha and margin["margin_delta"] >= min_effect:
            verdict = "margin_significant_improvement"
        elif (
            paired["paired_games"] >= min_games
            and paired["paired_win_p_value"] < alpha
            and paired["paired_win_delta"] <= -min_effect
        ):
            verdict = "paired_significant_regression"
        elif (
            paired["paired_games"] >= min_games
            and paired["paired_win_p_value"] < alpha
            and paired["paired_win_delta"] >= min_effect
        ):
            verdict = "paired_significant_improvement"
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
                **paired,
                **margin,
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
