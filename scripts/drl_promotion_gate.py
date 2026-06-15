"""Strict PPO/BReP promotion gate for Orbit Wars.

This is the executable version of the DRL objective: PPO checkpoints are first
exported to Kaggle-format ``agent(obs)`` files, then judged in the same strict
local ruler as BReP/PGS/OEP. A candidate is promotable only if it clears the
technical gates and the frozen 2p/4p reference pool.
"""

from __future__ import annotations
# ruff: noqa: E402,I001

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_submission import render_submission
from scripts.league_agents import FACTORIES, INCUMBENT, register_submission_file
from scripts.league_submit_ruler import (
    FIELD_2P_WEIGHT,
    PROFILE_DEFAULTS,
    build_report,
    build_tasks,
    run_tasks,
)
from scripts.ppo_export_parity import check_checkpoint_export_parity
from scripts.select_ppo_checkpoint import _expand_checkpoints, checkpoint_candidate_id

DRL_REFERENCES: tuple[str, ...] = (
    "producer",
    "oep",
    "brep",
    "pgs_bigwave",
    "greedy",
    "rush",
    "pgs_allscripts",
)

DRL_REQUIRED_2P: tuple[str, ...] = (
    "producer",
    "oep",
    "brep",
    "pgs_bigwave",
    "greedy",
    "rush",
    INCUMBENT,
)

DRL_4P_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("producer", "oep", INCUMBENT),
    ("producer", "brep", "pgs_bigwave"),
    ("oep", "greedy", "rush"),
    (INCUMBENT, "brep", "pgs_bigwave"),
)


@dataclass(frozen=True)
class PreparedCandidate:
    name: str
    kind: str
    checkpoint: str | None = None
    submission: str | None = None
    export_parity: dict[str, Any] | None = None


def required_2p_winrates(threshold: float) -> dict[str, float]:
    return {name: float(threshold) for name in DRL_REQUIRED_2P}


def _export_checkpoint(
    checkpoint: Path,
    submissions_dir: Path,
    *,
    parity_dir: Path,
    skip_parity: bool,
    parity_seeds: int,
    parity_steps: int,
) -> PreparedCandidate:
    candidate_id = f"ppo_{checkpoint_candidate_id(checkpoint)}"
    submission_path = submissions_dir / f"{candidate_id}.py"
    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    rendered = render_submission(template, checkpoint=str(checkpoint))
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(rendered, encoding="utf-8")
    league_submission_path = ROOT / "artifacts" / "league" / "submissions" / f"{candidate_id}.py"
    league_submission_path.parent.mkdir(parents=True, exist_ok=True)
    league_submission_path.write_text(rendered, encoding="utf-8")
    parity_report: dict[str, Any] | None = None
    if not skip_parity:
        parity_report = check_checkpoint_export_parity(
            checkpoint,
            submission_path=submission_path,
            seeds=list(range(max(1, int(parity_seeds)))),
            steps=int(parity_steps),
            player_counts=(2, 4),
        )
        parity_dir.mkdir(parents=True, exist_ok=True)
        (parity_dir / f"{candidate_id}.json").write_text(
            json.dumps(parity_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not parity_report["passed"]:
            raise ValueError(f"PPO export parity failed for {checkpoint}: {parity_report['mismatches'][:1]}")
    register_submission_file(candidate_id, league_submission_path)
    return PreparedCandidate(
        name=candidate_id,
        kind="ppo_checkpoint",
        checkpoint=str(checkpoint),
        submission=str(league_submission_path),
        export_parity=parity_report,
    )


def prepare_candidates(
    *,
    checkpoint_patterns: list[str],
    league_candidates: list[str],
    submissions_dir: Path,
    parity_dir: Path,
    skip_parity: bool,
    parity_seeds: int,
    parity_steps: int,
) -> list[PreparedCandidate]:
    prepared: list[PreparedCandidate] = []
    seen: set[str] = set()
    for name in league_candidates:
        if name not in FACTORIES:
            raise ValueError(f"unknown league candidate: {name}")
        if name not in seen:
            prepared.append(PreparedCandidate(name=name, kind="league_agent"))
            seen.add(name)

    if checkpoint_patterns:
        for checkpoint in _expand_checkpoints(checkpoint_patterns):
            item = _export_checkpoint(
                checkpoint,
                submissions_dir,
                parity_dir=parity_dir,
                skip_parity=skip_parity,
                parity_seeds=parity_seeds,
                parity_steps=parity_steps,
            )
            if item.name in seen:
                raise ValueError(f"duplicate prepared candidate name: {item.name}")
            prepared.append(item)
            seen.add(item.name)

    if not prepared:
        raise ValueError("provide at least one --candidate or --checkpoint")
    return prepared


def _completed_task_results(tasks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": task.label,
            "mode": task.mode,
            "candidate": task.candidate,
            "names": list(task.names),
            "seeds": task.seeds,
            "seed_base": task.seed_base,
            "steps": task.steps,
            "chunk_size": task.chunk_size,
            "out": str(task.out),
            "returncode": 0,
            "seconds": 0.0,
            "stdout": "",
            "stderr": "",
        }
        for task in tasks
    ]


def run_drl_promotion_gate(
    *,
    checkpoint_patterns: list[str],
    league_candidates: list[str],
    out_dir: Path,
    profile: str,
    seeds: int | None,
    seed_base: int,
    steps: int | None,
    jobs: int,
    match_chunk_size: int,
    skip_run: bool,
    required_2p_threshold: float,
    min_decisive_2p: int | None,
    min_producer_winrate: float,
    min_incumbent_winrate: float,
    min_floor_winrate: float,
    max_annihilation_rate_4p: float,
    weight_2p: float,
    skip_parity: bool = False,
    parity_seeds: int = 2,
    parity_steps: int = 16,
) -> dict[str, Any]:
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"unknown profile: {profile}")
    profile_defaults = PROFILE_DEFAULTS[profile]
    seeds = int(profile_defaults["seeds"] if seeds is None else seeds)
    steps = int(profile_defaults["steps"] if steps is None else steps)
    min_decisive_2p = int(
        profile_defaults["min_decisive_2p"] if min_decisive_2p is None else min_decisive_2p
    )
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seeds % 4 != 0:
        raise ValueError("seeds must be a multiple of 4 so 4p seat rotation is balanced")

    prepared = prepare_candidates(
        checkpoint_patterns=checkpoint_patterns,
        league_candidates=league_candidates,
        submissions_dir=out_dir / "submissions",
        parity_dir=out_dir / "export_parity",
        skip_parity=skip_parity,
        parity_seeds=parity_seeds,
        parity_steps=parity_steps,
    )
    candidates = [item.name for item in prepared]
    tasks = build_tasks(
        candidates,
        incumbent=INCUMBENT,
        references=list(DRL_REFERENCES),
        four_player_templates=[tuple(item) for item in DRL_4P_TEMPLATES],
        seeds=seeds,
        seed_base=seed_base,
        steps=steps,
        out_dir=out_dir / "matches",
        match_chunk_size=max(0, int(match_chunk_size)),
    )
    task_results = _completed_task_results(tasks) if skip_run else run_tasks(
        tasks,
        max(1, int(jobs)),
        progress=True,
        task_results_out=out_dir / "task_results.json",
    )
    required = required_2p_winrates(required_2p_threshold)
    report = build_report(
        candidates,
        task_results,
        incumbent=INCUMBENT,
        min_decisive_2p=min_decisive_2p,
        min_producer_winrate=min_producer_winrate,
        min_incumbent_winrate=min_incumbent_winrate,
        min_floor_winrate=min_floor_winrate,
        max_annihilation_rate_4p=max_annihilation_rate_4p,
        required_2p_winrates=required,
        weight_2p=weight_2p,
    )
    prepared_by_name = {item.name: asdict(item) for item in prepared}
    promotable = [
        {
            **prepared_by_name[row["candidate"]],
            "field_advantage": row["field_advantage"],
            "score_2p_fixed": row["score_2p_fixed"],
            "score_4p_fixed": row["score_4p_fixed"],
        }
        for row in report["ranking"]
        if row["verdict"] == "PASS_LOCAL"
    ]
    report.update(
        {
            "prepared_candidates": prepared_by_name,
            "promotable": promotable,
            "drl_gate": {
                "incumbent": INCUMBENT,
                "references": list(DRL_REFERENCES),
                "required_2p_winrates": required,
                "four_player_templates": [list(item) for item in DRL_4P_TEMPLATES],
                "profile": profile,
                "seeds": seeds,
                "seed_base": seed_base,
                "steps": steps,
                "min_decisive_2p": min_decisive_2p,
                "min_producer_winrate": min_producer_winrate,
                "min_incumbent_winrate": min_incumbent_winrate,
                "min_floor_winrate": min_floor_winrate,
                "max_annihilation_rate_4p": max_annihilation_rate_4p,
                "weight_2p": weight_2p,
                "skip_parity": skip_parity,
                "parity_seeds": parity_seeds,
                "parity_steps": parity_steps,
            },
            "tasks": task_results,
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="PPO checkpoint path or glob")
    parser.add_argument("--candidate", action="append", default=[], help="existing league candidate, e.g. brep")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/drl_promotion_gate"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="quick")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=90_000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--match-chunk-size", type=int, default=0)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--parity-seeds", type=int, default=2)
    parser.add_argument("--parity-steps", type=int, default=16)
    parser.add_argument("--required-2p-threshold", type=float, default=0.50)
    parser.add_argument("--min-decisive-2p", type=int, default=None)
    parser.add_argument("--min-producer-winrate", type=float, default=0.50)
    parser.add_argument("--min-incumbent-winrate", type=float, default=0.50)
    parser.add_argument("--min-floor-winrate", type=float, default=0.60)
    parser.add_argument("--max-annihilation-rate-4p", type=float, default=0.35)
    parser.add_argument("--weight-2p", type=float, default=FIELD_2P_WEIGHT)
    args = parser.parse_args(argv)

    report = run_drl_promotion_gate(
        checkpoint_patterns=list(args.checkpoint),
        league_candidates=list(args.candidate),
        out_dir=args.out_dir,
        profile=str(args.profile),
        seeds=args.seeds,
        seed_base=int(args.seed_base),
        steps=args.steps,
        jobs=int(args.jobs),
        match_chunk_size=int(args.match_chunk_size),
        skip_run=bool(args.skip_run),
        required_2p_threshold=float(args.required_2p_threshold),
        min_decisive_2p=args.min_decisive_2p,
        min_producer_winrate=float(args.min_producer_winrate),
        min_incumbent_winrate=float(args.min_incumbent_winrate),
        min_floor_winrate=float(args.min_floor_winrate),
        max_annihilation_rate_4p=float(args.max_annihilation_rate_4p),
        weight_2p=float(args.weight_2p),
        skip_parity=bool(args.skip_parity),
        parity_seeds=int(args.parity_seeds),
        parity_steps=int(args.parity_steps),
    )
    out = args.out or (args.out_dir / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "local_veto_passes": report["local_veto_passes"],
                "selection_status": report["selection_status"],
                "promotable": report["promotable"],
                "ranking": report["ranking"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
