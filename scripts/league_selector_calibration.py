"""Calibrate the submission selector against real LB anchors (etapas 8 + 16).

The league was falsified as a promotion gate (2026-06-10: local Spearman vs LB
= 0.0). The selector therefore has NO right to choose a submission until this
script proves, on bots with KNOWN leaderboard scores, that the selector_score
orders the field. Each anchor is evaluated exactly like a candidate (fixed
panel, no peer H2H — panel independence makes anchors blind to each other),
features are extracted, and the calibration passes only if ALL checks hold:

1. spearman(selector_score, LB) >= 0.60
2. zero grave inversions (> 75 LB points between anchors)
3. pgs_allscripts (the known floor) fails the hard gates
4. pgs_holdwave (the known top) is first or statistically tied with the top
5. backtest: pgs_hold never ranks above producer (the historical false
   positive); scoring is candidate-isolated (panel invariance)

calibration_valid=false is a SUCCESS of the mechanism, not a failure of this
script: it means the league stays veto-only and no candidate may be chosen.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_agents import INCUMBENT, LB_ANCHORS  # noqa: E402
from scripts.league_report import lb_inversions, spearman  # noqa: E402
from scripts.league_selector_stats import (  # noqa: E402
    DEFAULT_SELECTOR_WEIGHTS,
    beta_p_greater,
    selector_score,
)
from scripts.league_submit_ruler import (  # noqa: E402
    DEFAULT_4P_TEMPLATES,
    DEFAULT_REFERENCES,
    FIELD_2P_WEIGHT,
    PROFILE_DEFAULTS,
    SEED_SPLITS,
    _load_games,
    build_tasks,
    run_tasks,
    summarize_candidate,
)

CALIBRATION_SPEARMAN_MIN = 0.60
GRAVE_INVERSION_LB_GAP = 75.0
STAT_TIE_P_MIN = 0.30


def _task_signature(entry: dict[str, Any] | Any) -> tuple:
    get = entry.get if isinstance(entry, dict) else lambda k: getattr(entry, k)
    return (
        str(get("mode")),
        tuple(str(n) for n in get("names")),
        int(get("seeds")),
        int(get("seed_base")),
        int(get("steps")),
    )


def match_reusable(tasks: list, prior_results: list[dict[str, Any]]) -> tuple[list[dict], list]:
    """Split tasks into (reused_results, remaining_tasks). A prior result is
    reusable only when its full signature matches AND its game JSON exists and
    passes the strict payload validation — half-matching artifacts rerun."""
    by_sig: dict[tuple, dict[str, Any]] = {}
    for result in prior_results:
        if result.get("returncode") != 0:
            continue
        try:
            by_sig[_task_signature(result)] = result
        except (KeyError, TypeError, ValueError):
            continue
    reused: list[dict[str, Any]] = []
    remaining = []
    for task in tasks:
        prior = by_sig.get(_task_signature(task))
        if prior is not None:
            out = Path(prior["out"])
            if not out.is_absolute() or not out.exists():
                # repo-root-relative path from another checkout
                for base in (ROOT, Path(prior["out"]).parent):
                    candidate_path = base / prior["out"] if not out.is_absolute() else out
                    if candidate_path.exists():
                        out = candidate_path
                        break
            if out.exists():
                try:
                    _load_games(out, {**prior, "out": str(out)})
                except (ValueError, KeyError, json.JSONDecodeError):
                    remaining.append(task)
                    continue
                adopted = {**prior, "out": str(out),
                           "candidate": task.candidate, "label": task.label,
                           "role": task.role, "reused": True}
                reused.append(adopted)
                continue
        remaining.append(task)
    return reused, remaining


def resume_completed(tasks: list) -> tuple[list[dict], list]:
    """Adopt a task's own out-file when a previous interrupted run already
    produced and validated it."""
    done: list[dict[str, Any]] = []
    remaining = []
    for task in tasks:
        meta = {
            "label": task.label, "mode": task.mode, "candidate": task.candidate,
            "names": list(task.names), "role": task.role, "seeds": task.seeds,
            "seed_base": task.seed_base, "steps": task.steps,
            "chunk_size": task.chunk_size, "out": str(task.out),
            "returncode": 0, "seconds": 0.0, "stdout": "", "stderr": "",
            "resumed": True,
        }
        if task.out.exists():
            try:
                _load_games(task.out, meta)
            except (ValueError, KeyError, json.JSONDecodeError):
                remaining.append(task)
                continue
            done.append(meta)
        else:
            remaining.append(task)
    return done, remaining


def extract_features(summary: dict[str, Any], *, incumbent: str) -> dict[str, float]:
    """Feature vector restricted to fixed-panel-derivable values (etapa 8)."""
    pairwise = summary["pairwise_fixed"]

    def h2h(name: str) -> float:
        if summary["candidate"] == name:
            return 0.5  # self: neutral
        entry = pairwise.get(name)
        wr = entry["decisive_win_rate"] if entry else None
        return float(wr) if wr is not None else 0.0

    worst_bucket = summary.get("worst_bucket_score")
    fixed_rates = [s["win_rate"] for s in pairwise.values()]
    # stability: 1 - dispersion of fixed-panel win rates (a bot that is 0.9 on
    # half the panel and 0.1 on the other half is less trustworthy than a flat
    # 0.5, even at the same mean)
    dispersion = statistics.pstdev(fixed_rates) if len(fixed_rates) >= 2 else 0.0
    return {
        "cal_2p_adv": float(summary["adv_2p_fixed"]),
        "cal_4p_adv": float(summary["adv_4p_fixed"]),
        "h2h_vs_incumbent": h2h(incumbent),
        "h2h_vs_producer": h2h("producer"),
        "worst_bucket": float(worst_bucket) if worst_bucket is not None else 0.5,
        "stability": max(0.0, 1.0 - 2.0 * dispersion),
        "annihilation_4p": float(summary["four_player"]["annihilation_rate"]),
        "latency_p95_max": float(summary["latency_p95_max"] or 0.0),
        "faults": float(sum(summary["overall"]["faults"].values())),
    }


def anchor_rows(
    summaries: dict[str, dict[str, Any]],
    lb_anchors: dict[str, float],
    *,
    incumbent: str,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for name, summary in summaries.items():
        if name not in lb_anchors:
            continue
        features = extract_features(summary, incumbent=incumbent)
        rows.append({
            "name": name,
            "lb": float(lb_anchors[name]),
            "features": features,
            "selector_score": selector_score(
                features, weights, float(summary.get("risk_penalty", 0.0))
            ),
            "verdict": summary["verdict"],
            "hard_gates_passed": summary["verdict"] != "REJECT_LOCAL",
            "fixed_record": {
                "wins": sum(p["wins"] for p in summary["pairwise_fixed"].values()),
                "losses": sum(p["losses"] for p in summary["pairwise_fixed"].values()),
            },
        })
    return rows


def calibration_checks(
    rows: list[dict[str, Any]],
    *,
    spearman_min: float = CALIBRATION_SPEARMAN_MIN,
    inversion_noise: float = GRAVE_INVERSION_LB_GAP,
) -> tuple[list[dict[str, Any]], float, list]:
    by_name = {row["name"]: row for row in rows}
    scores = {row["name"]: row["selector_score"] for row in rows}
    lbs = {row["name"]: row["lb"] for row in rows}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    names = sorted(scores)
    rho = spearman([scores[n] for n in names], [lbs[n] for n in names])
    add("spearman_ge_min", rho == rho and rho >= spearman_min,
        {"spearman": rho, "required": spearman_min, "anchors": len(names)})

    inversions = lb_inversions(scores, lbs, noise=inversion_noise)
    add("no_grave_inversion", not inversions,
        {"inversions": [list(i) for i in inversions], "noise_lb": inversion_noise})

    floor = by_name.get("pgs_allscripts")
    add("allscripts_rejected", floor is not None and not floor["hard_gates_passed"],
        {"verdict": floor["verdict"] if floor else "MISSING"})

    top_row = max(rows, key=lambda r: r["selector_score"]) if rows else None
    holdwave = by_name.get("pgs_holdwave")
    if holdwave is None or top_row is None:
        add("holdwave_top_or_tied", False, {"missing": "pgs_holdwave"})
    elif top_row["name"] == "pgs_holdwave":
        add("holdwave_top_or_tied", True, {"top": "pgs_holdwave"})
    else:
        p_tie = beta_p_greater(
            holdwave["fixed_record"]["wins"], holdwave["fixed_record"]["losses"],
            top_row["fixed_record"]["wins"], top_row["fixed_record"]["losses"],
        )
        add("holdwave_top_or_tied", p_tie >= STAT_TIE_P_MIN,
            {"top": top_row["name"], "p_holdwave_above_top": p_tie,
             "required": STAT_TIE_P_MIN})

    hold = by_name.get("pgs_hold")
    producer = by_name.get("producer")
    add(
        "pgs_hold_not_above_producer",
        hold is None or producer is None
        or hold["selector_score"] <= producer["selector_score"],
        {
            "pgs_hold": hold["selector_score"] if hold else None,
            "producer": producer["selector_score"] if producer else None,
        },
    )
    return checks, rho, inversions


def check_scoring_isolation(
    candidate: str, task_results: list[dict[str, Any]], summarize_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Backtest item (e): a candidate's fixed scores must be identical whether
    it is scored from the full panel's task_results or only from its own rows
    — any difference means cross-candidate contamination."""
    full = summarize_candidate(candidate, task_results, **summarize_kwargs)
    own_rows = [r for r in task_results if r.get("candidate") == candidate]
    alone = summarize_candidate(candidate, own_rows, **summarize_kwargs)
    keys = ("score_2p_fixed", "score_4p_fixed", "field_advantage")
    return {
        "name": "panel_invariant_scoring",
        "passed": all(full[k] == alone[k] for k in keys),
        "details": {k: {"full": full[k], "alone": alone[k]} for k in keys},
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", default=",".join(sorted(LB_ANCHORS)))
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="selector")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed-split", choices=sorted(SEED_SPLITS), default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--label", default="calibration")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/league/selector_calibration"))
    parser.add_argument("--reuse-task-results", action="append", type=Path, default=[],
                        help="prior task_results.json files whose matching games are adopted")
    parser.add_argument("--resume", action="store_true",
                        help="adopt this run's own validated out-files from a prior interrupt")
    parser.add_argument("--min-decisive-2p", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    profile = PROFILE_DEFAULTS[args.profile]
    seeds = args.seeds or int(profile["seeds"])
    steps = args.steps or int(profile["steps"])
    min_decisive_2p = args.min_decisive_2p or int(profile["min_decisive_2p"])
    if args.seed_base is not None and args.seed_split is not None:
        raise SystemExit("--seed-base and --seed-split are mutually exclusive")
    seed_split = args.seed_split
    seed_base = args.seed_base
    if seed_base is None:
        seed_split = seed_split or "validation"
        seed_base = SEED_SPLITS[seed_split]

    anchors = [a for a in args.anchors.split(",") if a.strip()]
    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        anchors,
        incumbent=INCUMBENT,
        references=DEFAULT_REFERENCES,
        four_player_templates=DEFAULT_4P_TEMPLATES,
        seeds=seeds,
        seed_base=seed_base,
        steps=steps,
        out_dir=out_dir,
        peer_h2h=False,  # anchors must stay blind to each other
    )

    prior: list[dict[str, Any]] = []
    for path in args.reuse_task_results:
        prior.extend(json.loads(Path(path).read_text()))
    reused, tasks_left = match_reusable(tasks, prior)
    resumed: list[dict[str, Any]] = []
    if args.resume:
        resumed, tasks_left = resume_completed(tasks_left)
    if not args.quiet:
        print(f"[calibration] tasks={len(tasks)} reused={len(reused)} "
              f"resumed={len(resumed)} to_run={len(tasks_left)}", file=sys.stderr, flush=True)

    ran = run_tasks(
        tasks_left, args.jobs, progress=not args.quiet,
        task_results_out=out_dir / "task_results_partial.json",
    ) if tasks_left else []
    task_results = reused + resumed + ran
    (out_dir / "task_results.json").write_text(json.dumps(task_results, indent=2, sort_keys=True))

    summarize_kwargs = dict(
        incumbent=INCUMBENT,
        min_decisive_2p=min_decisive_2p,
        min_producer_winrate=0.50,
        min_incumbent_winrate=0.50,
        min_floor_winrate=0.60,
        max_annihilation_rate_4p=0.30,
        weight_2p=FIELD_2P_WEIGHT,
    )
    summaries = {a: summarize_candidate(a, task_results, **summarize_kwargs) for a in anchors}
    rows = anchor_rows(summaries, LB_ANCHORS, incumbent=INCUMBENT)
    checks, rho, inversions = calibration_checks(rows)
    top = max(rows, key=lambda r: r["selector_score"], default=None)
    if top is not None:
        checks.append(check_scoring_isolation(top["name"], task_results, summarize_kwargs))

    calibration = {
        "calibration_version": f"cal_{args.label}",
        "calibration_valid": all(c["passed"] for c in checks),
        "code_hash": _git_head(),
        "seed_split": seed_split,
        "seed_base": seed_base,
        "profile": args.profile,
        "seeds": seeds,
        "steps": steps,
        "weights": dict(DEFAULT_SELECTOR_WEIGHTS),
        "lb_anchors": dict(LB_ANCHORS),
        "anchors": {r["name"]: r for r in rows},
        "spearman_selector_lb": rho,
        "inversions": [list(i) for i in inversions],
        "checks": checks,
        "task_results": str(out_dir / "task_results.json"),
        "reused_tasks": len(reused),
        "resumed_tasks": len(resumed),
        "ran_tasks": len(ran),
    }
    out = out_dir / "calibration.json"
    out.write_text(json.dumps(calibration, indent=2, sort_keys=True))
    print(json.dumps({
        "out": str(out),
        "calibration_valid": calibration["calibration_valid"],
        "spearman": rho,
        "checks": [{c["name"]: c["passed"]} for c in checks],
        "scores": {r["name"]: round(r["selector_score"], 4) for r in rows},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
