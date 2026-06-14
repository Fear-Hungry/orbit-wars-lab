"""LB calibration — does the robustness gate's fitness PREDICT the leaderboard?

This is the decisive validity test for the whole auto-research loop. The loop
optimises ``fitness = mean_margin - 2*death_rate`` from the multi-opponent gate
(scripts/research_loop/evaluator.py). That signal is only worth optimising if it
ORDERS configs the way the real Kaggle leaderboard does — the project already
learned (the hard way) that the local LEAGUE's BT rating does NOT (Spearman ~0 /
-0.6 vs LB; memory: local_league_is_submission_gate). The open question is
whether the *robustness gate* (a different instrument) does better.

Method: re-score the configs we have REAL LB scores for (scripts/league_agents.py
``LB_ANCHORS``) through the exact same gate the loop uses, then Spearman the gate
fitness against the LB. No new Kaggle submissions are spent — we reuse the
anchors we already paid for.

ANCHORS use the EXACT kwargs the submitted config used (from league_agents.py
FACTORIES), NOT the loop's BASELINE: the anchors are PLAIN configs (e.g. holdwave
is scripts="hold" with no threat_value_4p), and we must reproduce what actually
scored that LB number.

Verdict bands (n is small — treat as a directional health check, not proof):
    rho >= 0.6  -> gate PREDICTS (matches the league's formal bar); loop trustworthy
    0.3 <= rho  -> WEAK signal; loop usable but calibrate more before trusting promotions
    rho < 0.3   -> gate FALSIFIED like the league; redesign fitness BEFORE trusting the loop
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.research_loop.evaluator import DEFAULT_POOL, evaluate
from scripts.research_loop.genome import fitness

ROOT = Path(__file__).resolve().parent.parent.parent

# (name, exact PGSConfig kwargs as submitted, real LB score). Source of truth:
# scripts/league_agents.py FACTORIES + LB_ANCHORS (refreshed 2026-06-11 via CLI).
# Only PGS-expressible anchors: producer/oep/brep are non-PGS and the evaluator
# only materialises PGSConfig subjects (extending it to raw agents would add the
# 5th+ anchor — see docs/auto_research_pipeline.md "upgrade path").
ANCHORS: list[tuple[str, dict, float]] = [
    ("pgs_holdwave",   {"scripts": "hold", "wave_min_ships": 60.0, "wave_start_step": 150}, 1228.8),
    ("pgs_wave_s100",  {"scripts": "hold", "wave_min_ships": 60.0, "wave_start_step": 100}, 1146.1),
    ("pgs_hold",       {"scripts": "hold"}, 1057.6),
    ("pgs_allscripts", {}, 1021.5),
]

LB_NOISE = 60.0  # Kaggle resubmit noise (memory: ~±60; identical config 1228.8 vs 1156.7)


def _rankdata(xs: list[float]) -> list[float]:
    """Average-rank (ties share the mean rank), ascending."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via Pearson on ranks (handles ties; matches league_report)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def inversions(rows: list[dict]) -> list[str]:
    """Pairs the gate orders BACKWARDS vs LB, beyond the ±noise band — these are
    the concrete configs the gate gets wrong (a global rho ~0 hides WHO)."""
    bad = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            d_lb = a["lb"] - b["lb"]
            if abs(d_lb) < LB_NOISE:
                continue  # LB can't distinguish them — not a real inversion
            d_fit = a["fitness"] - b["fitness"]
            if d_lb * d_fit < 0:
                hi, lo = (a, b) if d_lb > 0 else (b, a)
                bad.append(f"{hi['name']} (LB {hi['lb']:.0f}) ranked BELOW "
                           f"{lo['name']} (LB {lo['lb']:.0f}) by the gate")
    return bad


def competitive_tied(rows: list[dict], eps: float = 1e-6) -> bool:
    """True if the gate gives IDENTICAL fitness to every anchor except the single
    lowest-LB floor. When this holds, a high rho is an ARTIFACT of correctly
    ranking only the floor outlier — the gate is blind to the competitive range
    (the exact 'topo plano / floor-veto-only' failure that falsified the league).
    A passing rho here is a FALSE PASS and must not be trusted."""
    if len(rows) < 3:
        return False
    by_lb = sorted(rows, key=lambda r: r["lb"])
    competitive = by_lb[1:]  # drop the floor (lowest LB)
    fits = [r["fitness"] for r in competitive]
    return (max(fits) - min(fits)) < eps


def verdict(rho: float, n: int, tied: bool = False) -> str:
    if rho != rho:  # nan
        return "INDETERMINATE (degenerate ranks)"
    if tied:
        return ("FALSE PASS — FLOOR-VETO ONLY: the competitive anchors got IDENTICAL gate "
                f"fitness; rho={rho:+.2f} is driven solely by ranking the floor outlier. The "
                "gate canNOT order the top (same 'topo plano' that falsified the league). Do "
                "NOT trust for promotion; the search space/fitness must discriminate the top first.")
    if rho >= 0.6:
        return "GATE PREDICTS (>=0.6, league bar) -> loop fitness TRUSTWORTHY"
    if rho >= 0.3:
        return "WEAK SIGNAL (0.3-0.6) -> usable, but add anchors before trusting promotions"
    return "GATE FALSIFIED (<0.3, like the old league) -> REDESIGN fitness before trusting the loop"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6, help="eval seeds per opponent")
    ap.add_argument("--steps", type=int, default=500, help="eval horizon (500 = honest)")
    ap.add_argument("--seats", default="4", choices=("2", "4", "mix"),
                    help="2 = 1v1 mirror; 4 = the (falsified) 4p gate; mix = field 0.54·4p+0.46·2p")
    ap.add_argument("--pool", default=",".join(DEFAULT_POOL), help="comma list of opponents")
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: artifacts/research_loop/calibration[_2p|_mix].json")
    args = ap.parse_args()
    pool = tuple(p.strip() for p in args.pool.split(",") if p.strip())
    seats = "mix" if args.seats == "mix" else int(args.seats)
    if args.out is None:
        suffix = {"2": "_2p", "mix": "_mix"}.get(args.seats, "")
        args.out = ROOT / "artifacts" / "research_loop" / f"calibration{suffix}.json"

    print(f"=== LB calibration (seats={args.seats}): does the gate predict the leaderboard? ===")
    print(f"anchors={len(ANCHORS)} seats={args.seats} seeds={args.seeds} "
          f"steps={args.steps} pool={','.join(pool)}\n")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for name, cfg, lb in ANCHORS:
        print(f"[{name}] LB={lb:.1f}  cfg={cfg or '{} (default)'}")
        metrics = evaluate(cfg, seeds=args.seeds, steps=args.steps, pool=pool,
                           seed_base=args.seed_base, enable_comets=not args.no_comets,
                           verbose=True, seats=seats)
        fit = fitness(metrics)
        rows.append({"name": name, "lb": lb, "fitness": fit,
                     "mean_margin": metrics["mean_margin"], "death_rate": metrics["death_rate"],
                     "per_opponent": {o: v["mean_margin"] for o, v in metrics["per_opponent"].items()}})
        print(f"    -> gate fitness = {fit:+.4f}\n", flush=True)

    rho = spearman([r["fitness"] for r in rows], [r["lb"] for r in rows])
    invs = inversions(rows)

    # ranked table (by LB desc) so eyeballing the gate-rank column is easy
    by_lb = sorted(rows, key=lambda r: -r["lb"])
    gate_order = sorted(rows, key=lambda r: -r["fitness"])
    gate_rank = {r["name"]: i + 1 for i, r in enumerate(gate_order)}
    print("=== RESULT ===")
    print(f"{'config':16s} {'LB':>8s} {'LB#':>4s} {'fitness':>9s} {'gate#':>6s}")
    for i, r in enumerate(by_lb):
        print(f"{r['name']:16s} {r['lb']:8.1f} {i+1:>4d} {r['fitness']:+9.4f} {gate_rank[r['name']]:>6d}")
    # Per-opponent discrimination: spread of an opponent's margin across anchors.
    # ~0 means that opponent ranks every anchor the same → dead weight in the pool
    # (e.g. rush/greedy saturate at +1.0). Surfaces which opponents carry the signal.
    discrimination = {}
    if rows and rows[0].get("per_opponent"):
        for opp in rows[0]["per_opponent"]:
            vals = [r["per_opponent"][opp] for r in rows if opp in r.get("per_opponent", {})]
            discrimination[opp] = round(max(vals) - min(vals), 4) if vals else 0.0

    tied = competitive_tied(rows)
    n_distinct = len({round(r["fitness"], 6) for r in rows})
    print(f"\nSpearman(gate fitness, LB) = {rho:+.3f}   (n={len(rows)}, "
          f"{n_distinct} distinct fitness values)")
    if tied:
        print("⚠ COMPETITIVE ANCHORS TIED: the gate gave identical fitness to every anchor "
              "except the floor — rho is a FALSE PASS (see verdict).")
    if invs:
        print("inversions (gate disagrees with LB beyond ±%.0f noise):" % LB_NOISE)
        for s in invs:
            print(f"  - {s}")
    else:
        print("no inversions beyond the LB noise band.")
    if discrimination:
        ranked = sorted(discrimination.items(), key=lambda kv: -kv[1])
        print("\nper-opponent discrimination (margin spread across anchors; ~0 = dead weight):")
        for opp, d in ranked:
            flag = "  ← SATURATED (drop/down-weight)" if d < 0.05 else ""
            print(f"  {opp:9s}: {d:.4f}{flag}")

    print(f"\nVERDICT: {verdict(rho, len(rows), tied)}")
    print(f"(elapsed {time.perf_counter() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rho": rho, "n": len(rows), "n_distinct_fitness": n_distinct,
        "competitive_tied": tied, "seats": args.seats, "seeds": args.seeds,
        "steps": args.steps, "pool": list(pool), "discrimination": discrimination,
        "rows": rows, "inversions": invs,
        "verdict": verdict(rho, len(rows), tied),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
