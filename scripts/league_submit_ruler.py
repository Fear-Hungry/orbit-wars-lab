"""Strict local ruler for deciding whether a bot deserves a Kaggle submission.

This is deliberately different from the continuous league. The continuous league
is exploratory and population-biased by construction; this script runs a paired,
balanced schedule against fixed references and emits an explicit local verdict.
It is a strong local veto/selection tool, not a replacement for a stabilized LB
score.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_agents import FACTORIES, INCUMBENT  # noqa: E402
from scripts.league_report import decisive_winner  # noqa: E402

DEFAULT_REFERENCES = [
    "producer",
    "oep",
    "pgs_holdwave",
    "pgs_wave_s100",
    "pgs_bigwave",
    "pgs_allscripts",
    "ext_lb1050",
    "ext_hellburner",
]

DEFAULT_4P_TEMPLATES = [
    ("producer", "oep", "pgs_holdwave"),
    ("producer", "pgs_bigwave", "pgs_allscripts"),
    ("oep", "pgs_wave_s100", "pgs_bigwave"),
    ("pgs_holdwave", "pgs_allscripts", "ext_hellburner"),
]

PROFILE_DEFAULTS = {
    "quick": {"seeds": 4, "steps": 250, "min_decisive_2p": 4},
    "standard": {"seeds": 8, "steps": 500, "min_decisive_2p": 12},
    "strong": {"seeds": 24, "steps": 500, "min_decisive_2p": 40},
}


@dataclass(frozen=True)
class MatchTask:
    label: str
    mode: str
    candidate: str
    names: tuple[str, ...]
    seeds: int
    seed_base: int
    steps: int
    out: Path


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _known(names: Iterable[str]) -> list[str]:
    return [name for name in names if name in FACTORIES]


def _safe_label(parts: Iterable[str]) -> str:
    return "__".join(part.replace("/", "_") for part in parts)


def _complete_4p_lineup(candidate: str, template: tuple[str, ...], references: list[str]) -> tuple[str, ...] | None:
    names = [candidate]
    for name in template:
        if name != candidate and name not in names and name in FACTORIES:
            names.append(name)
    for name in references:
        if len(names) >= 4:
            break
        if name != candidate and name not in names and name in FACTORIES:
            names.append(name)
    return tuple(names[:4]) if len(names) == 4 else None


def build_tasks(
    candidates: list[str],
    *,
    incumbent: str,
    references: list[str],
    four_player_templates: list[tuple[str, ...]],
    seeds: int,
    seed_base: int,
    steps: int,
    out_dir: Path,
) -> list[MatchTask]:
    tasks: list[MatchTask] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in candidates:
        if candidate not in FACTORIES:
            raise ValueError(f"unknown candidate: {candidate}")
        ref_order = []
        for name in [incumbent, *references]:
            if name != candidate and name in FACTORIES and name not in ref_order:
                ref_order.append(name)
        for ref_idx, ref in enumerate(ref_order):
            names = (candidate, ref)
            key = ("2p", names)
            if key in seen:
                continue
            seen.add(key)
            # Shared seeds per reference across candidates: candidate A vs
            # producer and candidate B vs producer must see the same map slice,
            # otherwise the ruler reintroduces scheduler luck into selection.
            base = seed_base + 100 * ref_idx
            label = _safe_label([candidate, "2p", ref])
            tasks.append(MatchTask(
                label=label,
                mode="2p",
                candidate=candidate,
                names=names,
                seeds=seeds,
                seed_base=base,
                steps=steps,
                out=out_dir / f"{label}.json",
            ))
        for tpl_idx, template in enumerate(four_player_templates):
            names = _complete_4p_lineup(candidate, template, ref_order)
            if names is None:
                continue
            key = ("4p", names)
            if key in seen:
                continue
            seen.add(key)
            # Shared 4p seeds per template across candidates for the same reason
            # as the 2p tasks above. league_match rotates seats by seed index, so
            # requiring seeds % 4 == 0 gives balanced seats per 4p template.
            base = seed_base + 5_000 + 100 * tpl_idx
            label = _safe_label([candidate, "4p", f"line{tpl_idx}"])
            tasks.append(MatchTask(
                label=label,
                mode="4p",
                candidate=candidate,
                names=names,
                seeds=seeds,
                seed_base=base,
                steps=steps,
                out=out_dir / f"{label}.json",
            ))
    return tasks


def _run_task(task: MatchTask) -> dict[str, Any]:
    task.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/league_match.py",
        "--agents",
        ",".join(task.names),
        "--seeds",
        str(task.seeds),
        "--seed-base",
        str(task.seed_base),
        "--steps",
        str(task.steps),
        "--out",
        str(task.out),
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    return {
        "label": task.label,
        "mode": task.mode,
        "candidate": task.candidate,
        "names": list(task.names),
        "out": str(task.out),
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - start,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


def _write_task_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def run_tasks(
    tasks: list[MatchTask],
    jobs: int,
    *,
    progress: bool = False,
    task_results_out: Path | None = None,
) -> list[dict[str, Any]]:
    def _note(done: int, total: int, result: dict[str, Any]) -> None:
        if progress:
            print(
                f"[{done:02d}/{total:02d}] {result['label']} rc={result['returncode']} "
                f"{result['seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    if jobs <= 1:
        out = []
        total = len(tasks)
        for task in tasks:
            result = _run_task(task)
            out.append(result)
            _note(len(out), total, result)
            if task_results_out is not None:
                _write_task_results(task_results_out, out)
        return out
    out = []
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            out.append(result)
            _note(len(out), total, result)
            if task_results_out is not None:
                _write_task_results(task_results_out, out)
    return out


def _load_games(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    games = payload["games"]
    for game in games:
        game["mode"] = payload["mode"]
    return games


def _fault_totals(games: list[dict[str, Any]], name: str) -> dict[str, int]:
    totals = {"crashes": 0, "timeouts": 0, "invalid_moves": 0}
    for game in games:
        faults = game.get("faults") or {}
        entry = faults.get(name) or {}
        for key in totals:
            totals[key] += int(entry.get(key, 0))
    return totals


def _score_games(games: list[dict[str, Any]], name: str) -> dict[str, Any]:
    appearances = wins = losses = ties = annihilations = audited = bad_status = 0
    for game in games:
        if name not in game["seats"]:
            continue
        appearances += 1
        if "faults" in game:
            audited += 1
        idx = game["seats"].index(name)
        status = (game.get("agent_status") or ["DONE"] * len(game["seats"]))[idx]
        if status != "DONE":
            bad_status += 1
        if game.get("died_at", [None] * len(game["seats"]))[idx] is not None:
            annihilations += 1
        winner = decisive_winner(game)
        if winner == name:
            wins += 1
        elif winner is None:
            ties += 1
        else:
            losses += 1
    decisive = wins + losses
    return {
        "appearances": appearances,
        "audited": audited,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "decisive": decisive,
        "win_rate": wins / appearances if appearances else 0.0,
        "decisive_win_rate": wins / decisive if decisive else None,
        "nonloss_rate": (wins + ties) / appearances if appearances else 0.0,
        "annihilation_rate": annihilations / appearances if appearances else 0.0,
        "bad_status": bad_status,
        "faults": _fault_totals(games, name),
    }


def _empty_pair_summary() -> dict[str, Any]:
    return {
        "appearances": 0,
        "audited": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "decisive": 0,
        "win_rate": 0.0,
        "decisive_win_rate": None,
        "nonloss_rate": 0.0,
        "annihilation_rate": 0.0,
        "bad_status": 0,
        "faults": {"crashes": 0, "timeouts": 0, "invalid_moves": 0},
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_candidate(
    candidate: str,
    task_results: list[dict[str, Any]],
    *,
    incumbent: str,
    min_decisive_2p: int,
    min_producer_winrate: float,
    min_incumbent_winrate: float,
    min_floor_winrate: float,
    max_annihilation_rate_4p: float,
    weight_2p: float,
) -> dict[str, Any]:
    pairwise: dict[str, dict[str, Any]] = {}
    four_player_games: list[dict[str, Any]] = []
    all_games: list[dict[str, Any]] = []
    failed_runs = [r for r in task_results if r["candidate"] == candidate and r["returncode"] != 0]
    for result in task_results:
        if result["candidate"] != candidate or result["returncode"] != 0:
            continue
        games = _load_games(Path(result["out"]))
        all_games.extend(games)
        if result["mode"] == "2p":
            opponent = [name for name in result["names"] if name != candidate][0]
            pairwise[opponent] = _score_games(games, candidate)
        else:
            four_player_games.extend(games)

    overall = _score_games(all_games, candidate)
    four_player = _score_games(four_player_games, candidate) if four_player_games else _empty_pair_summary()
    decisive_rates = [
        s["decisive_win_rate"] for s in pairwise.values()
        if s["decisive_win_rate"] is not None
    ]
    score_2p = _mean(decisive_rates)
    score_4p = four_player["win_rate"]
    overall_score = weight_2p * score_2p + (1.0 - weight_2p) * score_4p

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: dict[str, Any] | None = None,
                  *, severity: str = "fail") -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "details": details or {},
        })

    add_check("subprocesses_completed", not failed_runs, {"failed": failed_runs})
    add_check("all_games_audited", overall["audited"] == overall["appearances"], {
        "audited": overall["audited"],
        "appearances": overall["appearances"],
    })
    add_check("no_faults", not any(overall["faults"].values()), {"faults": overall["faults"]})
    add_check("all_status_done", overall["bad_status"] == 0, {"bad_status": overall["bad_status"]})

    for opponent, summary in sorted(pairwise.items()):
        add_check(
            f"coverage_2p_vs_{opponent}",
            summary["decisive"] >= min_decisive_2p,
            {"decisive": summary["decisive"], "required": min_decisive_2p},
            severity="inconclusive",
        )

    producer = pairwise.get("producer")
    if producer is not None and candidate != "producer":
        wr = producer["decisive_win_rate"]
        add_check(
            "beats_or_ties_producer_floor",
            wr is not None and wr >= min_producer_winrate,
            {"decisive_win_rate": wr, "required": min_producer_winrate},
        )

    inc = pairwise.get(incumbent)
    if candidate == incumbent:
        add_check("incumbent_h2h", True, {"candidate_is_incumbent": True}, severity="info")
    elif inc is not None:
        wr = inc["decisive_win_rate"]
        add_check(
            "beats_or_ties_incumbent_h2h",
            wr is not None and wr >= min_incumbent_winrate,
            {"decisive_win_rate": wr, "required": min_incumbent_winrate},
        )

    floor = pairwise.get("pgs_allscripts")
    if floor is not None and candidate != "pgs_allscripts":
        wr = floor["decisive_win_rate"]
        add_check(
            "clears_rejected_floor",
            wr is not None and wr >= min_floor_winrate,
            {"decisive_win_rate": wr, "required": min_floor_winrate},
        )

    add_check(
        "survives_4p",
        four_player["appearances"] == 0 or four_player["annihilation_rate"] <= max_annihilation_rate_4p,
        {
            "annihilation_rate": four_player["annihilation_rate"],
            "required_max": max_annihilation_rate_4p,
            "appearances": four_player["appearances"],
        },
    )

    hard_failures = [c for c in checks if not c["passed"] and c["severity"] == "fail"]
    inconclusive = [c for c in checks if not c["passed"] and c["severity"] == "inconclusive"]
    if hard_failures:
        verdict = "REJECT_LOCAL"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS_LOCAL"

    return {
        "candidate": candidate,
        "verdict": verdict,
        "overall_score": overall_score,
        "score_2p": score_2p,
        "score_4p": score_4p,
        "overall": overall,
        "four_player": four_player,
        "pairwise": pairwise,
        "checks": checks,
    }


def build_report(
    candidates: list[str],
    task_results: list[dict[str, Any]],
    *,
    incumbent: str,
    min_decisive_2p: int,
    min_producer_winrate: float,
    min_incumbent_winrate: float,
    min_floor_winrate: float,
    max_annihilation_rate_4p: float,
    weight_2p: float,
) -> dict[str, Any]:
    summaries = [
        summarize_candidate(
            candidate,
            task_results,
            incumbent=incumbent,
            min_decisive_2p=min_decisive_2p,
            min_producer_winrate=min_producer_winrate,
            min_incumbent_winrate=min_incumbent_winrate,
            min_floor_winrate=min_floor_winrate,
            max_annihilation_rate_4p=max_annihilation_rate_4p,
            weight_2p=weight_2p,
        )
        for candidate in candidates
    ]
    ranking = sorted(summaries, key=lambda s: (s["verdict"] == "PASS_LOCAL", s["overall_score"]), reverse=True)
    recommended = next((s["candidate"] for s in ranking if s["verdict"] == "PASS_LOCAL"), None)
    return {
        "recommended_candidate": recommended,
        "ranking": [
            {
                "candidate": s["candidate"],
                "verdict": s["verdict"],
                "overall_score": s["overall_score"],
                "score_2p": s["score_2p"],
                "score_4p": s["score_4p"],
            }
            for s in ranking
        ],
        "candidates": {s["candidate"]: s for s in summaries},
    }


def _parse_4p_templates(values: list[str] | None) -> list[tuple[str, ...]]:
    if not values:
        return list(DEFAULT_4P_TEMPLATES)
    out = []
    for value in values:
        names = tuple(_split_csv(value))
        if len(names) != 3:
            raise ValueError(f"4p lineup template needs exactly 3 opponents: {value!r}")
        out.append(names)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="strong")
    parser.add_argument("--incumbent", default=INCUMBENT)
    parser.add_argument("--references", default=",".join(DEFAULT_REFERENCES))
    parser.add_argument("--four-player-lineup", action="append",
                        help="three comma-separated opponents; candidate is inserted as seat 0")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=70_000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/league/submit_ruler"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--task-results-out",
        type=Path,
        default=None,
        help="write completed match metadata incrementally while the ruler is still running",
    )
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-decisive-2p", type=int, default=None)
    parser.add_argument("--min-producer-winrate", type=float, default=0.50)
    parser.add_argument("--min-incumbent-winrate", type=float, default=0.50)
    parser.add_argument("--min-floor-winrate", type=float, default=0.60)
    parser.add_argument("--max-annihilation-rate-4p", type=float, default=0.35)
    parser.add_argument("--weight-2p", type=float, default=0.50)
    args = parser.parse_args(argv)
    profile = PROFILE_DEFAULTS[args.profile]
    if args.seeds is None:
        args.seeds = int(profile["seeds"])
    if args.steps is None:
        args.steps = int(profile["steps"])
    if args.min_decisive_2p is None:
        args.min_decisive_2p = int(profile["min_decisive_2p"])
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.seeds % 4 != 0:
        raise SystemExit("--seeds must be a multiple of 4 so 4p seat rotation is balanced")

    candidates = list(dict.fromkeys(args.candidates))
    references = _known(_split_csv(args.references))
    if args.incumbent not in FACTORIES:
        raise SystemExit(f"unknown incumbent: {args.incumbent}")
    templates = _parse_4p_templates(args.four_player_lineup)
    out_dir = args.out_dir
    tasks = build_tasks(
        candidates,
        incumbent=args.incumbent,
        references=references,
        four_player_templates=templates,
        seeds=args.seeds,
        seed_base=args.seed_base,
        steps=args.steps,
        out_dir=out_dir,
    )
    if not tasks:
        raise SystemExit("no runnable tasks")
    if args.skip_run:
        task_results = [
            {
                "label": task.label,
                "mode": task.mode,
                "candidate": task.candidate,
                "names": list(task.names),
                "out": str(task.out),
                "returncode": 0,
                "seconds": 0.0,
                "stdout": "",
                "stderr": "",
            }
            for task in tasks
        ]
    else:
        task_results_out = args.task_results_out
        if task_results_out is None:
            task_results_out = out_dir / "task_results.json"
        task_results = run_tasks(
            tasks,
            args.jobs,
            progress=not args.quiet,
            task_results_out=task_results_out,
        )
    report = build_report(
        candidates,
        task_results,
        incumbent=args.incumbent,
        min_decisive_2p=args.min_decisive_2p,
        min_producer_winrate=args.min_producer_winrate,
        min_incumbent_winrate=args.min_incumbent_winrate,
        min_floor_winrate=args.min_floor_winrate,
        max_annihilation_rate_4p=args.max_annihilation_rate_4p,
        weight_2p=args.weight_2p,
    )
    report.update({
        "incumbent": args.incumbent,
        "references": references,
        "tasks": task_results,
        "settings": {
            "seeds": args.seeds,
            "seed_base": args.seed_base,
            "steps": args.steps,
            "profile": args.profile,
            "min_decisive_2p": args.min_decisive_2p,
            "min_producer_winrate": args.min_producer_winrate,
            "min_incumbent_winrate": args.min_incumbent_winrate,
            "min_floor_winrate": args.min_floor_winrate,
            "max_annihilation_rate_4p": args.max_annihilation_rate_4p,
            "weight_2p": args.weight_2p,
        },
    })
    out = args.out or (out_dir / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "out": str(out),
        "recommended_candidate": report["recommended_candidate"],
        "ranking": report["ranking"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
