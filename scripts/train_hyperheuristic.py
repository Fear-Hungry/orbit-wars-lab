"""H7a — out-of-sample contextual selector (anti-max-bias).

The oracle's ``oracle_new_family_margin = +0.08`` is a *contaminated upper bound*:
``mean(max_per_snapshot)`` over noisy estimates is optimistic (Cawley & Talbot,
2010 — selection on a noisy criterion inflates the reported score). A realizable
agent cannot pick by the posterior outcome; it must pick by features available at
decision time.

This script separates the two:

* ``oracle_choice_known_outcome`` — picks the best family by the *realized*
  margin of each snapshot. Theoretical (contaminated) ceiling.
* ``selector_choice_predicted``  — picks by the runtime context ``bucket`` only,
  using a per-bucket rule LEARNED ON A TRAIN SPLIT and applied to a disjoint
  HOLDOUT split. This is the realizable-agent metric.

The verdict is keyed on the HOLDOUT selector margin (never the oracle):

    selected_holdout <= -0.045  -> SELECTOR_DOES_NOT_CAPTURE  (fix features/fitness)
    -0.045 < selected_holdout < 0 -> SELECTOR_HAS_SIGNAL       (run 32 seeds)
    selected_holdout >= 0        -> SELECTOR_CROSSES_FLOOR     (run 96 vs Producer)

Per-family regret diagnoses sparse specialization: where each family wins, and
whether the selector's error is aggressive false-positive or conservative
false-negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from bots.oep.family_h import (
    CAPTURE_MARGIN,
    RESERVE,
    _aim,
    _projected_defense,
    _split_planets,
)
from bots.oep.hyperheuristic import context_bucket
from scripts.oracle_candidates import DEFAULT_BASELINE, collect_candidate_action_records

Record = dict[str, Any]
PRIMITIVES = (
    "production_projected_attack",
    "timeline_risk",
    "hammer_multiprong",
    "regroup_dominance",
)


def _hammer_regime(obs: dict[str, Any]) -> bool:
    """True when >=2 own sources can synchronize to overwhelm one enemy target.

    This is the hammer family's OWN precondition (domain-motivated, not read off
    the holdout regret table), added because regret showed hammer_multiprong is
    worth ~+0.17 yet never selected by the coarse bucket selector.
    """

    _me, own, others = _split_planets(obs)
    enemies = [p for p in others if p.owner >= 0]
    angvel = float(obs.get("angular_velocity", 0.03))
    sources = [p for p in own if p.ships - RESERVE >= 1]
    if len(sources) < 2 or not enemies:
        return False
    total = sum(s.ships - RESERVE for s in sources)
    for target in enemies:
        nearest_eta = min(_aim(s, target, angvel)[1] for s in sources)
        if total > _projected_defense(target, nearest_eta) + CAPTURE_MARGIN:
            return True
    return False


def snapshot_features(obs: dict[str, Any]) -> dict[str, Any]:
    """Runtime-available context features (no post-action leak)."""

    return {"bucket": context_bucket(obs), "hammer_regime": _hammer_regime(obs)}


def coarse_context(features: dict[str, Any]) -> str:
    return str(features["bucket"])


def rich_context(features: dict[str, Any]) -> str:
    """Bucket refined by the hammer regime so the selector can route hammer states."""

    if features.get("hammer_regime"):
        return "hammer|" + str(features["bucket"])
    return str(features["bucket"])


def split_seeds(seeds: list[int]) -> tuple[list[int], list[int]]:
    """Disjoint train/holdout seed split (distinct games — true out-of-sample)."""

    half = len(seeds) // 2
    return seeds[:half], seeds[half:]


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def learn_per_bucket(
    train: list[Record], primitives: tuple[str, ...]
) -> dict[str, Any]:
    """Learn ``bucket -> family`` by best mean train margin; global best as fallback."""

    per_bucket: dict[str, str] = {}
    buckets = {r["bucket"] for r in train}
    for bucket in buckets:
        rows = [r for r in train if r["bucket"] == bucket]
        per_bucket[bucket] = max(
            primitives, key=lambda f: _mean([r["margins"][f] for r in rows])
        )
    fallback = max(primitives, key=lambda f: _mean([r["margins"][f] for r in train]))
    return {"per_bucket": per_bucket, "fallback": fallback}


def selector_pick(bucket: str, learned: dict[str, Any]) -> str:
    return learned["per_bucket"].get(bucket, learned["fallback"])


def evaluate(
    holdout: list[Record], learned: dict[str, Any], primitives: tuple[str, ...]
) -> dict[str, float]:
    selected = _mean([r["margins"][selector_pick(r["bucket"], learned)] for r in holdout])
    oracle = _mean([max(r["margins"][f] for f in primitives) for r in holdout])
    oep = _mean([r["margins"].get("oep", 0.0) for r in holdout])
    return {
        "selector_choice_predicted": selected,
        "oracle_choice_known_outcome": oracle,
        "oep_margin": oep,
        "regret": oracle - selected,
        "selected_vs_oep_delta": selected - oep,
    }


def regret_table(
    holdout: list[Record], learned: dict[str, Any], primitives: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    n = len(holdout) or 1
    table: dict[str, dict[str, float]] = {}
    for f in primitives:
        oracle_rows = [r for r in holdout if max(primitives, key=lambda g: r["margins"][g]) == f]
        sel_rows = [r for r in holdout if selector_pick(r["bucket"], learned) == f]
        table[f] = {
            "oracle_pick_rate": len(oracle_rows) / n,
            "selected_pick_rate": len(sel_rows) / n,
            "mean_delta_when_oracle": _mean([r["margins"][f] for r in oracle_rows]),
            "mean_delta_when_selected": _mean([r["margins"][f] for r in sel_rows]),
        }
    return table


def holdout_verdict(selected: float, baseline: float = DEFAULT_BASELINE) -> str:
    if selected <= baseline:
        return "SELECTOR_DOES_NOT_CAPTURE"
    if selected < 0.0:
        return "SELECTOR_HAS_SIGNAL"  # run 32 seeds
    return "SELECTOR_CROSSES_FLOOR"  # run 96 vs Producer + OEP sanity


def _evaluate_context(
    records: list[Record], train_seeds: list[int], holdout_seeds: list[int], context_fn
) -> dict[str, Any]:
    """Learn + holdout-evaluate keyed on ``context_fn(features)`` (no leak)."""

    tagged = [{**r, "bucket": context_fn(r.get("features", {"bucket": r["bucket"]}))} for r in records]
    train = [r for r in tagged if r["seed"] in set(train_seeds)]
    holdout = [r for r in tagged if r["seed"] in set(holdout_seeds)]
    learned = learn_per_bucket(train, PRIMITIVES)
    metrics = evaluate(holdout, learned, PRIMITIVES)
    return {
        "learned_per_bucket": learned["per_bucket"],
        "learned_fallback": learned["fallback"],
        "oracle_new_family_margin_train": _mean(
            [max(r["margins"][f] for f in PRIMITIVES) for r in train]
        ),
        "oracle_new_family_margin_holdout": metrics["oracle_choice_known_outcome"],
        "selected_new_family_margin_holdout": metrics["selector_choice_predicted"],
        "regret_holdout": metrics["regret"],
        "selected_vs_oep_delta": metrics["selected_vs_oep_delta"],
        "regret_table": regret_table(holdout, learned, PRIMITIVES),
        "verdict": holdout_verdict(metrics["selector_choice_predicted"]),
    }


def run(
    *,
    seeds: list[int],
    episode_steps: int,
    snapshot_stride: int,
    max_snapshots: int,
    enable_comets: bool,
) -> dict[str, Any]:
    families = [*PRIMITIVES, "oep"]
    records, legality = collect_candidate_action_records(
        seeds=seeds,
        families=families,
        episode_steps=episode_steps,
        snapshot_stride=snapshot_stride,
        max_snapshots=max_snapshots,
        enable_comets=enable_comets,
        feature_fn=snapshot_features,
    )
    train_seeds, holdout_seeds = split_seeds(seeds)
    # Same data, two feature schemes — clean coarse-vs-rich comparison (H7b).
    coarse = _evaluate_context(records, train_seeds, holdout_seeds, coarse_context)
    rich = _evaluate_context(records, train_seeds, holdout_seeds, rich_context)
    hammer_rate = _mean([1.0 if r["features"].get("hammer_regime") else 0.0 for r in records])

    return {
        "seeds": list(seeds),
        "train_seeds": train_seeds,
        "holdout_seeds": holdout_seeds,
        "episode_steps": episode_steps,
        "snapshot_stride": snapshot_stride,
        "max_snapshots": max_snapshots,
        "enable_comets": enable_comets,
        "primitives": list(PRIMITIVES),
        "n_train_snapshots": len([r for r in records if r["seed"] in set(train_seeds)]),
        "n_holdout_snapshots": len([r for r in records if r["seed"] in set(holdout_seeds)]),
        "hammer_regime_rate": hammer_rate,
        "coarse_bucket": coarse,
        "rich_context": rich,
        # Primary verdict = rich context (H7b); coarse kept for contrast.
        "selected_new_family_margin_holdout": rich["selected_new_family_margin_holdout"],
        "verdict": rich["verdict"],
        "crashes": legality["crashes"],
        "invalid_actions": legality["invalid"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H7a out-of-sample contextual selector")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--episode-steps", type=int, default=96)
    p.add_argument("--snapshot-stride", type=int, default=32)
    p.add_argument("--max-snapshots", type=int, default=3)
    p.add_argument("--no-comets", action="store_true")
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed (use fresh seeds, e.g. 32, to avoid reusing a diagnosed holdout)",
    )
    p.add_argument("--out", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = int(args.seed_start)
    summary = run(
        seeds=list(range(start, start + int(args.seeds))),
        episode_steps=int(args.episode_steps),
        snapshot_stride=int(args.snapshot_stride),
        max_snapshots=int(args.max_snapshots),
        enable_comets=not bool(args.no_comets),
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
