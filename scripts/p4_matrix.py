"""F0 — Producer x OEP matrix per seed x format (2p and 4p). EVAL ONLY (no training).

For each seed and format this runs Producer against OEP and records, PER PLAYER,
the normalized margin, win, contested production/ship share AUC, survival and the
first-collapse step. The per-seed granularity is the point: it exposes the REGIMES
where one archetype beats the other, which is

  - the decision table for the conservative selector (Front A), and
  - the per-candidate value calibration for the PPO candidate-selector (Front B).

Opponents come from the registry as ISOLATED instances (each resets its own memory
on ``step == 0``), so a 4p lineup with two Producers / two OEPs does not share a
module-singleton runtime across seats (cross-contamination would corrupt the matrix).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from scripts.dense_eval import COLLAPSE_PROD_SHARE, _shares


def _run_match(
    policies: list[Any],
    *,
    seed: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> tuple[list[float], list[dict[str, Any]], int, float]:
    """Run one game with ``len(policies)`` seats; return scores + per-seat stats."""
    n = len(policies)
    backend = RustBatchBackend(
        num_envs=1,
        num_players=n,
        seed=int(seed),
        config=RustConfig(
            episode_steps=int(episode_steps),
            enable_comets=bool(enable_comets),
            act_timeout=float(act_timeout),
        ),
    )
    state = backend.reset(seed)[0]
    per: list[dict[str, Any]] = [
        {"prod": [], "ship": [], "planets": 0, "survival": 0, "collapse": None}
        for _ in range(n)
    ]
    outcome: dict[str, Any] = {"scores": [0.0] * n, "done": False}
    invalid = 0.0
    steps = 0
    while True:
        actions: list[list[list[float]]] = [[] for _ in range(n)]
        for idx, policy in enumerate(policies):
            try:
                moves = policy(state, idx)
                if not isinstance(moves, list):
                    moves = []
            except Exception:  # noqa: BLE001 - a crashing policy just passes (counted)
                invalid += 1.0
                moves = []
            if not moves_are_legal(state, idx, moves):
                invalid += 1.0
                moves = []
            actions[idx] = moves
        outcomes, states = backend.step_with_states([actions])
        outcome = outcomes[0]
        state = states[0]
        steps += 1
        for idx in range(n):
            own_planets, prod_share, ship_share = _shares(state, idx)
            per[idx]["prod"].append(prod_share)
            per[idx]["ship"].append(ship_share)
            per[idx]["planets"] = own_planets
            if own_planets > 0 or ship_share > 0.0:
                per[idx]["survival"] = steps
            if per[idx]["collapse"] is None and (
                prod_share < COLLAPSE_PROD_SHARE or own_planets == 0
            ):
                per[idx]["collapse"] = steps
        if outcome["done"]:
            break
    scores = [float(s) for s in outcome["scores"]]
    return scores, per, steps, invalid


def _seat_record(scores: list[float], per: list[dict[str, Any]], seat: int, steps: int) -> dict[str, float]:
    others = max(s for i, s in enumerate(scores) if i != seat)
    p = per[seat]
    return {
        "margin": normalized_margin(scores, seat),
        "win": 1.0 if scores[seat] > others else 0.0,
        "prod_share_auc": fmean(p["prod"]) if p["prod"] else 0.0,
        "ship_share_auc": fmean(p["ship"]) if p["ship"] else 0.0,
        "survival_step": float(p["survival"]),
        "first_collapse_step": float(p["collapse"] if p["collapse"] is not None else steps),
    }


def _agg(records: list[dict[str, float]], key: str) -> float:
    return fmean(r[key] for r in records) if records else 0.0


def _matrix_2p(seeds: int, episode_steps: int, enable_comets: bool, act_timeout: float) -> dict[str, Any]:
    prod = get_isolated_opponents("producer", 1)[0]
    oep = get_isolated_opponents("oep", 1)[0]
    per_seed: list[dict[str, Any]] = []
    invalid_total = 0.0
    for seed in range(max(1, seeds)):
        prod_seats: list[dict[str, float]] = []
        oep_seats: list[dict[str, float]] = []
        # both seatings to cancel any seat/first-move bias
        for lineup, prod_idx, oep_idx in (([prod, oep], 0, 1), ([oep, prod], 1, 0)):
            scores, per, steps, invalid = _run_match(
                lineup, seed=seed, episode_steps=episode_steps,
                enable_comets=enable_comets, act_timeout=act_timeout,
            )
            invalid_total += invalid
            prod_seats.append(_seat_record(scores, per, prod_idx, steps))
            oep_seats.append(_seat_record(scores, per, oep_idx, steps))
        pm = _agg(prod_seats, "margin")
        om = _agg(oep_seats, "margin")
        per_seed.append({
            "seed": seed,
            "producer": {k: _agg(prod_seats, k) for k in prod_seats[0]},
            "oep": {k: _agg(oep_seats, k) for k in oep_seats[0]},
            "oep_minus_producer_margin": om - pm,
        })
    return {"per_seed": per_seed, "invalid_actions": invalid_total}


def _matrix_4p(seeds: int, episode_steps: int, enable_comets: bool, act_timeout: float) -> dict[str, Any]:
    # FFA lineup [P, O, P, O] with isolated instances (two of each, no shared runtime).
    prods = get_isolated_opponents("producer", 2)
    oeps = get_isolated_opponents("oep", 2)
    per_seed: list[dict[str, Any]] = []
    invalid_total = 0.0
    for seed in range(max(1, seeds)):
        lineup = [prods[0], oeps[0], prods[1], oeps[1]]
        scores, per, steps, invalid = _run_match(
            lineup, seed=seed, episode_steps=episode_steps,
            enable_comets=enable_comets, act_timeout=act_timeout,
        )
        invalid_total += invalid
        prod_seats = [_seat_record(scores, per, s, steps) for s in (0, 2)]
        oep_seats = [_seat_record(scores, per, s, steps) for s in (1, 3)]
        pm = _agg(prod_seats, "margin")
        om = _agg(oep_seats, "margin")
        per_seed.append({
            "seed": seed,
            "lineup": ["producer", "oep", "producer", "oep"],
            "producer": {k: _agg(prod_seats, k) for k in prod_seats[0]},
            "oep": {k: _agg(oep_seats, k) for k in oep_seats[0]},
            "oep_minus_producer_margin": om - pm,
        })
    return {"per_seed": per_seed, "invalid_actions": invalid_total}


def _regimes(per_seed: list[dict[str, Any]], edge: float) -> dict[str, Any]:
    oep_wins = [r["seed"] for r in per_seed if r["oep_minus_producer_margin"] > edge]
    prod_wins = [r["seed"] for r in per_seed if r["oep_minus_producer_margin"] < -edge]
    ties = [r["seed"] for r in per_seed if abs(r["oep_minus_producer_margin"]) <= edge]
    deltas = [r["oep_minus_producer_margin"] for r in per_seed]
    return {
        "edge": edge,
        "oep_beats_producer_seeds": oep_wins,
        "producer_beats_oep_seeds": prod_wins,
        "tie_seeds": ties,
        "oep_advantage_mean": fmean(deltas) if deltas else 0.0,
        "oep_advantage_max": max(deltas) if deltas else 0.0,
        "producer_margin_mean": fmean(r["producer"]["margin"] for r in per_seed) if per_seed else 0.0,
        "oep_margin_mean": fmean(r["oep"]["margin"] for r in per_seed) if per_seed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--enable-comets", action="store_true")
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--edge", type=float, default=0.10,
                        help="|oep-producer| margin gap above which a seed is a clear regime")
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/p4/matrix_producer_oep.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "config": {
            "seeds": args.seeds, "episode_steps": args.episode_steps,
            "enable_comets": args.enable_comets, "act_timeout": args.act_timeout,
            "edge": args.edge,
        },
    }
    m2 = _matrix_2p(args.seeds, args.episode_steps, args.enable_comets, args.act_timeout)
    report["format_2p"] = {**m2, "regimes": _regimes(m2["per_seed"], args.edge)}
    if not args.skip_4p:
        m4 = _matrix_4p(args.seeds, args.episode_steps, args.enable_comets, args.act_timeout)
        report["format_4p"] = {**m4, "regimes": _regimes(m4["per_seed"], args.edge)}

    invalid = m2["invalid_actions"] + (0.0 if args.skip_4p else report["format_4p"]["invalid_actions"])
    report["health"] = {"invalid_actions": invalid}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # compact stdout summary
    print(json.dumps({
        "out": str(args.out),
        "invalid_actions": invalid,
        "2p": report["format_2p"]["regimes"],
        **({"4p": report["format_4p"]["regimes"]} if not args.skip_4p else {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
