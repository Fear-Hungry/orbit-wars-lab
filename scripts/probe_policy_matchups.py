from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_policy_pool import (
    _load_policies,
    _normalized_margin,
    _run_match,
    _win_points,
)


def _parse_seeds(raw: str) -> list[int]:
    if "," in raw:
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        seeds = list(range(max(1, int(raw))))
    if not seeds:
        raise ValueError("probe requires at least one seed")
    return seeds


def _summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    margins = [float(record["margin"]) for record in records]
    wins = [float(record["win_points"]) for record in records]
    return {
        "games": float(len(records)),
        "win_rate": fmean(wins) if wins else 0.0,
        "mean_margin": fmean(margins) if margins else 0.0,
        "min_margin": min(margins) if margins else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--seeds", default="4")
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--margin-below", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--disable-comets", action="store_true")
    args = parser.parse_args()

    policies = _load_policies()
    names = [args.candidate, *args.opponents]
    missing = [name for name in names if name not in policies]
    if missing:
        raise SystemExit(f"unknown policies: {', '.join(sorted(set(missing)))}")

    seeds = _parse_seeds(args.seeds)
    records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for opponent_name in args.opponents:
        for seed in seeds:
            for candidate_first in (True, False):
                players = (
                    [policies[args.candidate], policies[opponent_name]]
                    if candidate_first
                    else [policies[opponent_name], policies[args.candidate]]
                )
                candidate_idx = 0 if candidate_first else 1
                scores = _run_match(
                    players,
                    seed=seed,
                    episode_steps=args.episode_steps,
                    enable_comets=not args.disable_comets,
                    act_timeout=args.act_timeout,
                )
                margin = _normalized_margin(scores, candidate_idx)
                record = {
                    "opponent": opponent_name,
                    "seed": seed,
                    "candidate_position": candidate_idx,
                    "scores": scores,
                    "margin": margin,
                    "win_points": _win_points(scores, candidate_idx),
                }
                all_records.append(record)
                if margin < args.margin_below:
                    records.append(record)
                    if args.max_records > 0 and len(records) >= args.max_records:
                        print(
                            json.dumps(
                                {
                                    "candidate": args.candidate,
                                    "summary": _summarize(all_records),
                                    "records": records,
                                    "truncated": True,
                                },
                                indent=2,
                                sort_keys=True,
                            )
                        )
                        return

    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "summary": _summarize(all_records),
                "records": records,
                "truncated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
