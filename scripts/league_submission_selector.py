"""Submission decision engine (etapas 17-19): the ONLY component allowed to say
SUBMIT_CANDIDATE.

Reads a selector-profile ruler report (holdout seed split) plus a valid
calibration artifact and applies the objective rule list in order. Every rule
leaves a trace in the output; failing 1-6 is a technical rejection, failing
7-13 routes to RUN_MORE_GAMES or KEEP_INCUMBENT — a local tie ALWAYS favors the
incumbent (a new submission needs evidence, not noise). The engine is also
explicitly allowed to choose nobody: a selector forced to always pick a
challenger overfits by construction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_agents import FACTORIES, INCUMBENT, LB_ANCHORS, external_path  # noqa: E402
from scripts.league_selector_calibration import extract_features  # noqa: E402
from scripts.league_selector_stats import (  # noqa: E402
    P_ABOVE_SECOND_MIN,
    P_BEATS_INCUMBENT_MIN,
    TIEBREAK_P_MIN,
    beta_p_greater,
    p_beats_incumbent,
    selector_score,
)
from scripts.league_submit_ruler import CHOICE_GATES_4P, LATENCY_WARN_MS  # noqa: E402

SELECTOR_VERSION = "selector_v1_2026_06"
CHOICE_WORST_BUCKET_MIN = 0.15
MIN_DECISIVE_VS_INCUMBENT = 100

DECISION_SUBMIT = "SUBMIT_CANDIDATE"
DECISION_KEEP = "KEEP_INCUMBENT"
DECISION_MORE = "RUN_MORE_GAMES"
DECISION_CAL_FAILED = "CALIBRATION_FAILED"
DECISION_NO_VALID = "NO_TECHNICALLY_VALID_CANDIDATE"
DECISION_BAD_POOL = "INVALID_REFERENCE_POOL"


def preflight_references(names: list[str], *, smoke: bool = False) -> dict[str, Any]:
    """Fail EARLY on a broken reference pool: unknown names, missing external
    files, factories that do not instantiate, (optionally) agents that cannot
    answer one smoke observation with a list."""
    problems: dict[str, str] = {}
    for name in dict.fromkeys(names):
        if name not in FACTORIES:
            problems[name] = "not in FACTORIES"
            continue
        path = external_path(name)
        if path is not None and not path.exists():
            problems[name] = f"missing file: {path}"
            continue
        if smoke:
            try:
                agent = FACTORIES[name]()
                moves = agent({
                    "step": 0, "player": 0, "angular_velocity": 0.03,
                    "planets": [
                        [0, 0, -3.0, 0.0, 1.0, 20.0],
                        [1, 1, 3.0, 0.0, 1.0, 20.0],
                        [2, -1, 0.0, 3.0, 1.0, 10.0],
                    ],
                    "fleets": [],
                })
                if not isinstance(moves, list):
                    problems[name] = f"smoke call returned {type(moves).__name__}, not list"
            except Exception as exc:  # noqa: BLE001 — report, never crash preflight
                problems[name] = f"{type(exc).__name__}: {exc}"
    return {"valid": not problems, "problems": problems, "smoke": smoke}


def _check(rules: list[dict[str, Any]], name: str, passed: bool, details: Any) -> bool:
    rules.append({"rule": name, "passed": bool(passed), "details": details})
    return bool(passed)


def candidate_eligibility(summary: dict[str, Any], *, incumbent: str) -> tuple[bool, list[dict]]:
    """Rules 3-7 and 10-14 of etapa 19 for one candidate (choice-level — the
    ruler's own hard gates already ran at pass-to-selector levels)."""
    rules: list[dict[str, Any]] = []
    checks = {c["name"]: c for c in summary.get("checks", [])}

    faults = summary["overall"]["faults"]
    _check(rules, "zero_crashes", faults.get("crashes", 0) == 0, faults)
    _check(rules, "zero_timeouts", faults.get("timeouts", 0) == 0, faults)
    _check(rules, "zero_invalid_moves", faults.get("invalid_moves", 0) == 0, faults)
    _check(rules, "all_status_done", summary["overall"]["bad_status"] == 0,
           {"bad_status": summary["overall"]["bad_status"]})
    _check(rules, "verdict_pass_local", summary["verdict"] == "PASS_LOCAL",
           {"verdict": summary["verdict"]})

    latency = summary.get("latency_p95_max")
    _check(rules, "latency_p95_le_500", latency is not None and latency <= LATENCY_WARN_MS,
           {"latency_p95_max": latency, "max_ms": LATENCY_WARN_MS})

    inc = summary["pairwise_fixed"].get(incumbent)
    decisive = inc["decisive"] if inc else 0
    _check(rules, "decisive_vs_incumbent_coverage", decisive >= MIN_DECISIVE_VS_INCUMBENT,
           {"decisive": decisive, "required": MIN_DECISIVE_VS_INCUMBENT})

    fp = summary["four_player"]
    _check(rules, "choice_4p_winrate", fp["win_rate"] >= CHOICE_GATES_4P["min_4p_winrate"],
           {"win_rate": fp["win_rate"], "required": CHOICE_GATES_4P["min_4p_winrate"]})
    _check(rules, "choice_4p_annihilation",
           fp["annihilation_rate"] <= CHOICE_GATES_4P["max_4p_annihilation"],
           {"annihilation_rate": fp["annihilation_rate"],
            "required_max": CHOICE_GATES_4P["max_4p_annihilation"]})
    templates = {
        label: t["win_rate"]
        for label, t in (summary.get("four_player_templates") or {}).items()
        if t.get("appearances")
    }
    _check(rules, "choice_worst_template_4p",
           bool(templates)
           and min(templates.values()) >= CHOICE_GATES_4P["min_worst_template_4p_winrate"],
           {"per_template": templates,
            "required": CHOICE_GATES_4P["min_worst_template_4p_winrate"]})
    _check(rules, "no_single_template_dependence",
           "template_dependence" not in (summary.get("risk_components") or {}),
           {"risk_components": summary.get("risk_components")})

    worst_bucket = summary.get("worst_bucket_score")
    _check(rules, "worst_bucket_above_floor",
           worst_bucket is None or worst_bucket >= CHOICE_WORST_BUCKET_MIN,
           {"worst_bucket_score": worst_bucket, "required": CHOICE_WORST_BUCKET_MIN})

    # rule 13: must not lose to producer/oep by a statistically relevant margin
    for ref in ("producer", "oep"):
        entry = summary["pairwise_fixed"].get(ref)
        if entry is None or summary["candidate"] == ref:
            continue
        p_above = p_beats_incumbent(entry["wins"], entry["losses"])
        _check(rules, f"not_significantly_below_{ref}", p_above >= 0.05,
               {"wins": entry["wins"], "losses": entry["losses"], "p_above_05": p_above})

    _check(rules, "ruler_audit_checks_present",
           "no_faults" in checks and "all_status_done" in checks,
           {"present": sorted(checks)})
    return all(r["passed"] for r in rules), rules


def decide(
    report: dict[str, Any],
    calibration: dict[str, Any] | None,
    *,
    preflight: dict[str, Any],
    current_code_hash: str | None = None,
    tiebreak_p_a_over_b: float | None = None,
    allow_any_split: bool = False,
) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    incumbent = report.get("incumbent", INCUMBENT)

    def result(decision: str, *, chosen: str | None = None, reason: str,
               confidence: dict | None = None, scores: dict | None = None,
               eligible: list | None = None, rejected: dict | None = None) -> dict[str, Any]:
        return {
            "selector_version": SELECTOR_VERSION,
            "selector_valid": preflight["valid"],
            "calibration_valid": bool(calibration and calibration.get("calibration_valid")),
            "chosen_candidate": chosen,
            "fallback_choice": incumbent,
            "confidence": confidence or {},
            "decision": decision,
            "reason": reason,
            "eligible_candidates": eligible or [],
            "rejected_candidates": rejected or {},
            "score_components": scores or {},
            "risk": {
                name: {
                    "risk_penalty": s.get("risk_penalty"),
                    "risk_components": s.get("risk_components"),
                }
                for name, s in (report.get("candidates") or {}).items()
            },
            "calibration": {
                "version": calibration.get("calibration_version") if calibration else None,
                "spearman": calibration.get("spearman_selector_lb") if calibration else None,
                "seed_split": calibration.get("seed_split") if calibration else None,
                "code_hash": calibration.get("code_hash") if calibration else None,
            },
            "seed_split": (report.get("settings") or {}).get("seed_split"),
            "preflight": preflight,
            "rules": rules,
        }

    if not _check(rules, "reference_pool_valid", preflight["valid"], preflight["problems"]):
        return result(DECISION_BAD_POOL, reason="reference pool failed preflight")

    cal_ok = bool(calibration and calibration.get("calibration_valid"))
    _check(rules, "calibration_valid", cal_ok,
           {"present": calibration is not None,
            "valid": calibration.get("calibration_valid") if calibration else None})
    if cal_ok and current_code_hash is not None:
        cal_ok = _check(rules, "calibration_code_hash_matches",
                        calibration.get("code_hash") == current_code_hash,
                        {"calibration": calibration.get("code_hash"),
                         "current": current_code_hash})
    if cal_ok:
        cal_ok = _check(rules, "calibration_anchor_set_current",
                        calibration.get("lb_anchors") == dict(LB_ANCHORS),
                        {"stale": calibration.get("lb_anchors") != dict(LB_ANCHORS)})
    if not cal_ok:
        return result(DECISION_CAL_FAILED,
                      reason="no valid calibration: the league remains veto-only")

    settings = report.get("settings") or {}
    split_ok = allow_any_split or settings.get("seed_split") == "selector"
    profile_ok = allow_any_split or settings.get("profile") == "selector"
    if not _check(rules, "report_is_selector_holdout", split_ok and profile_ok,
                  {"seed_split": settings.get("seed_split"),
                   "profile": settings.get("profile"), "allow_any_split": allow_any_split}):
        return result(
            DECISION_NO_VALID,
            reason="decision requires a selector-profile report on the selector holdout split",
        )

    weights = calibration.get("weights")
    summaries: dict[str, dict[str, Any]] = report.get("candidates") or {}
    eligible: list[str] = []
    rejected: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    for name, summary in summaries.items():
        features = extract_features(summary, incumbent=incumbent)
        score = selector_score(features, weights, float(summary.get("risk_penalty", 0.0)))
        scores[name] = {"features": features, "selector_score": score,
                        "risk_penalty": summary.get("risk_penalty", 0.0)}
        if name == incumbent:
            continue  # the incumbent is the fallback, never a challenger
        ok, candidate_rules = candidate_eligibility(summary, incumbent=incumbent)
        scores[name]["eligibility_rules"] = candidate_rules
        if ok:
            eligible.append(name)
        else:
            rejected[name] = [r["rule"] for r in candidate_rules if not r["passed"]]

    if not _check(rules, "any_eligible_candidate", bool(eligible), {"rejected": rejected}):
        return result(DECISION_NO_VALID, reason="no candidate passed the choice-level gates",
                      scores=scores, rejected=rejected)

    ranked = sorted(eligible, key=lambda n: scores[n]["selector_score"], reverse=True)
    top = ranked[0]
    top_summary = summaries[top]
    inc_entry = top_summary["pairwise_fixed"].get(incumbent) or {"wins": 0, "losses": 0,
                                                                 "decisive": 0}
    p_inc = p_beats_incumbent(inc_entry["wins"], inc_entry["losses"])
    confidence: dict[str, Any] = {"p_beats_incumbent": p_inc}
    if not _check(rules, "p_beats_incumbent", p_inc >= P_BEATS_INCUMBENT_MIN,
                  {"p": p_inc, "required": P_BEATS_INCUMBENT_MIN,
                   "record": [inc_entry["wins"], inc_entry["losses"]]}):
        # enough evidence already and still inseparable -> the tie favors the
        # incumbent; thin evidence -> ask for more games instead of guessing
        if inc_entry["decisive"] >= 2 * MIN_DECISIVE_VS_INCUMBENT:
            return result(DECISION_KEEP, reason=(
                f"{top} does not separate from the incumbent "
                f"(p={p_inc:.3f} over {inc_entry['decisive']} decisive games); "
                "ties favor the incumbent"), confidence=confidence,
                scores=scores, eligible=ranked, rejected=rejected)
        return result(DECISION_MORE, reason=(
            f"{top} vs incumbent is undecided (p={p_inc:.3f}, "
            f"{inc_entry['decisive']} decisive games)"), confidence=confidence,
            scores=scores, eligible=ranked, rejected=rejected)

    if len(ranked) > 1:
        second = ranked[1]
        a, b = scores[top], scores[second]
        rec_a = summaries[top]
        rec_b = summaries[second]
        p_second = beta_p_greater(
            sum(p["wins"] for p in rec_a["pairwise_fixed"].values()),
            sum(p["losses"] for p in rec_a["pairwise_fixed"].values()),
            sum(p["wins"] for p in rec_b["pairwise_fixed"].values()),
            sum(p["losses"] for p in rec_b["pairwise_fixed"].values()),
        )
        confidence["p_above_second_best"] = p_second
        if not _check(rules, "p_above_second_best", p_second >= P_ABOVE_SECOND_MIN,
                      {"p": p_second, "required": P_ABOVE_SECOND_MIN,
                       "second": second, "score_gap":
                       a["selector_score"] - b["selector_score"]}):
            if tiebreak_p_a_over_b is not None:
                if _check(rules, "tiebreak_protocol", tiebreak_p_a_over_b >= TIEBREAK_P_MIN,
                          {"p_a_over_b": tiebreak_p_a_over_b, "required": TIEBREAK_P_MIN}):
                    confidence["tiebreak_p"] = tiebreak_p_a_over_b
                else:
                    return result(DECISION_KEEP, reason=(
                        f"tiebreak did not separate {top} from {second}; "
                        "ties favor the incumbent"), confidence=confidence,
                        scores=scores, eligible=ranked, rejected=rejected)
            else:
                return result(DECISION_MORE, reason=(
                    f"{top} and {second} are within the margin of error; "
                    "run the selector_tiebreak protocol"), confidence=confidence,
                    scores=scores, eligible=ranked, rejected=rejected)

    return result(DECISION_SUBMIT, chosen=top, reason=(
        f"{top} passed every hard gate, beats the incumbent with "
        f"p={p_inc:.3f} and is separated from the field"), confidence=confidence,
        scores=scores, eligible=ranked, rejected=rejected)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/league/submission_decision.json"))
    parser.add_argument("--smoke-preflight", action="store_true")
    parser.add_argument("--tiebreak-p", type=float, default=None,
                        help="P(top1 > top2) measured by the selector_tiebreak protocol run")
    parser.add_argument("--dry-run", action="store_true",
                        help="backtest mode: accept non-selector splits, skip hash check")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    calibration = json.loads(args.calibration.read_text())
    pool = list(dict.fromkeys(
        [report.get("incumbent", INCUMBENT), *(report.get("references") or [])]
    ))
    preflight = preflight_references(pool, smoke=args.smoke_preflight)
    decision = decide(
        report,
        calibration,
        preflight=preflight,
        current_code_hash=None if args.dry_run else _git_head(),
        tiebreak_p_a_over_b=args.tiebreak_p,
        allow_any_split=args.dry_run,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2, sort_keys=True))
    print(json.dumps({
        "out": str(args.out),
        "decision": decision["decision"],
        "chosen_candidate": decision["chosen_candidate"],
        "reason": decision["reason"],
        "confidence": decision["confidence"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
