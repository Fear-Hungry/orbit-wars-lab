"""Nash-vs-LB calibration — does Nash averaging rank our LB-anchored agents
BETTER than the naive mean-vs-pool win-rate?

This is the decisive measurability test for the "learning as a search component"
pivot (six-hats 2026-06-18). The whole bet rests on a TRUSTWORTHY veto, and the
project already proved the local league's mean-vs-pool rating does NOT predict the
LB (Spearman ~0 / -0.6; memory: local_league_is_submission_gate). The open claim
(Balduzzi et al. 2018, "Re-evaluating Evaluation", Zotero HJJAI6Q6, scripts/nash_eval.py)
is that Nash averaging — invariant to redundant/weak agents — orders a pool better
when the pool contains a redundant cluster.

Our 7 LB-anchored agents are EXACTLY that adversarial case: a redundant 4-member
hold-family cluster (pgs_holdwave/hold/wave_s100/allscripts, all `scripts="hold"`
variants) plus 3 structurally distinct agents (producer/oep/brep). Mean-vs-pool
should be biased by the cluster; Nash should not.

Method (no new Kaggle submissions — reuse the LB we already paid for):
  1. Play every unordered pair of the LB-anchored agents seat-rotated 2p through
     the SAME ruler primitive the promotion gate uses (league_match.py via
     league_submit_ruler), giving an antisymmetric win-rate matrix W.
  2. Rank by nash_rating(W) (redundancy-invariant) AND by naive mean-vs-pool
     win-rate (the instrument that was falsified).
  3. Spearman each ranking against the real LB.
  4. Verdict: Nash is validated as a better veto iff rho_nash >= 0.6 (league bar)
     AND rho_nash > rho_mean by a clear margin.

Run:
  .venv/bin/python -m scripts.nash_lb_calibrate --seeds 24 --jobs 6
  .venv/bin/python -m scripts.nash_lb_calibrate --seeds 4 --jobs 6   # wiring smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from scripts.league_agents import FACTORIES, LB_ANCHORS
from scripts.league_submit_ruler import (
    MatchTask,
    _load_games,
    _safe_label,
    _score_games,
    _seed_slice_base,
    run_tasks,
)
from scripts.research_loop.calibrate import spearman
from scripts.nash_eval import maxent_nash, nash_rating, winrate_to_advantage

ROOT = Path(__file__).resolve().parent.parent


def lb_anchored_pool() -> list[str]:
    """Named agents that are BOTH in FACTORIES (the ruler can play them) and have
    a measured LB score in LB_ANCHORS, ordered by LB descending."""
    names = [n for n in LB_ANCHORS if n in FACTORIES]
    return sorted(names, key=lambda n: -LB_ANCHORS[n])


def build_winrate_matrix(
    pool: list[str], *, seeds: int, seed_base: int, steps: int, jobs: int,
    out_dir: Path, progress: bool,
) -> tuple[np.ndarray, dict]:
    """W[i,j] = P(pool[i] beats pool[j]) from seat-rotated 2p ruler matches.

    One match per UNORDERED pair: league_match rotates seats by seed index, so a
    single (a,b) task already balances both seats — running (b,a) too would only
    duplicate games. Both win-rates are read off the one game set via _score_games.
    """
    n = len(pool)
    W = np.zeros((n, n), dtype=float)
    idx = {name: i for i, name in enumerate(pool)}
    tasks = []
    for a, b in itertools.combinations(pool, 2):
        label = _safe_label([a, "vs", b])
        tasks.append(MatchTask(
            label=label, mode="2p", candidate=a, names=(a, b), seeds=seeds,
            seed_base=_seed_slice_base(seed_base, "2p", f"{a}__{b}"),
            steps=steps, out=out_dir / f"{label}.json",
        ))
    results = run_tasks(tasks, jobs, progress=progress)
    by_label = {r["label"]: r for r in results}
    diag = {}
    for a, b in itertools.combinations(pool, 2):
        label = _safe_label([a, "vs", b])
        result = by_label[label]
        if result["returncode"] != 0:
            raise RuntimeError(
                f"match {a} vs {b} failed rc={result['returncode']}:\n"
                f"{result['stderr'][-1500:]}"
            )
        games = _load_games(Path(result["out"]))
        sa, sb = _score_games(games, a), _score_games(games, b)
        # win_rate = wins/appearances; ties cancel in the antisymmetric A=W-W.T,
        # so this is the correct advantage signal even when sa+sb < 1 (ties).
        W[idx[a], idx[b]] = sa["win_rate"]
        W[idx[b], idx[a]] = sb["win_rate"]
        diag[label] = {
            "a": a, "b": b, "a_win_rate": round(sa["win_rate"], 4),
            "b_win_rate": round(sb["win_rate"], 4),
            "a_decisive_wr": sa["decisive_win_rate"],
            "appearances": sa["appearances"], "seconds": round(result["seconds"], 1),
        }
    return W, diag


def mean_vs_pool(W: np.ndarray) -> np.ndarray:
    """Naive rating: each agent's mean win-rate vs the rest (off-diagonal row mean).
    This is the redundancy-BIASED instrument the local league was falsified on."""
    n = W.shape[0]
    out = np.zeros(n)
    for i in range(n):
        others = [W[i, j] for j in range(n) if j != i]
        out[i] = float(np.mean(others)) if others else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24, help="2p seeds per pair (rotated seats)")
    ap.add_argument("--steps", type=int, default=500, help="episode horizon (500 = official)")
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=6, help="<= physical cores (threads pinned to 1)")
    ap.add_argument("--pool", default=None,
                    help="comma list overriding the default LB-anchored pool")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "research_loop" / "nash_calibration.json")
    ap.add_argument("--work-dir", type=Path,
                    default=ROOT / "artifacts" / "research_loop" / "nash_calibration_matches")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    pool = ([p.strip() for p in args.pool.split(",") if p.strip()]
            if args.pool else lb_anchored_pool())
    missing = [n for n in pool if n not in FACTORIES or n not in LB_ANCHORS]
    if missing:
        raise SystemExit(f"pool entries not in FACTORIES∩LB_ANCHORS: {missing}")
    lb = np.array([LB_ANCHORS[n] for n in pool], dtype=float)

    print("=== Nash-vs-LB calibration: does Nash averaging beat mean-vs-pool? ===")
    print(f"pool (n={len(pool)}, by LB desc): "
          + ", ".join(f"{n}={LB_ANCHORS[n]:.0f}" for n in pool))
    print(f"seeds={args.seeds} steps={args.steps} jobs={args.jobs}\n")

    t0 = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    W, diag = build_winrate_matrix(
        pool, seeds=args.seeds, seed_base=args.seed_base, steps=args.steps,
        jobs=args.jobs, out_dir=args.work_dir, progress=not args.quiet)

    nash = nash_rating(W)                 # redundancy-invariant
    mean = mean_vs_pool(W)                # redundancy-biased baseline
    nash_mix = maxent_nash(winrate_to_advantage(W))

    rho_nash = spearman(list(nash), list(lb))
    rho_mean = spearman(list(mean), list(lb))

    print("=== win-rate matrix W[i,j]=P(row beats col) ===")
    hdr = " " * 16 + "".join(f"{n[:8]:>9s}" for n in pool)
    print(hdr)
    for i, n in enumerate(pool):
        row = "".join(f"{W[i, j]:9.2f}" for j in range(len(pool)))
        print(f"{n[:15]:16s}{row}")

    print("\n=== ratings (sorted by LB desc) ===")
    print(f"{'agent':16s} {'LB':>8s} {'nash':>8s} {'nash_mix':>9s} {'mean_wr':>8s}")
    for i, n in enumerate(pool):
        print(f"{n:16s} {lb[i]:8.1f} {nash[i]:+8.3f} {nash_mix[i]:9.3f} {mean[i]:8.3f}")

    margin = rho_nash - rho_mean
    if rho_nash >= 0.6 and margin > 0.05:
        verdict = (f"NASH VALIDATED: rho_nash={rho_nash:+.2f} clears the 0.6 league bar "
                   f"and beats mean-vs-pool ({rho_mean:+.2f}) by {margin:+.2f} -> "
                   f"use nash_rating as the pool veto.")
    elif rho_nash >= 0.6:
        verdict = (f"BOTH PREDICT: rho_nash={rho_nash:+.2f} clears 0.6 but mean "
                   f"({rho_mean:+.2f}) is close (margin {margin:+.2f}); pool not "
                   f"redundant enough to separate them here. Nash still safer.")
    elif margin > 0.05:
        verdict = (f"NASH BETTER BUT WEAK: rho_nash={rho_nash:+.2f} > mean ({rho_mean:+.2f}) "
                   f"yet below the 0.6 bar — add LB anchors / seeds before trusting promotions.")
    else:
        verdict = (f"INCONCLUSIVE: rho_nash={rho_nash:+.2f}, rho_mean={rho_mean:+.2f} "
                   f"(n={len(pool)} is small). Nash does not yet beat the falsified instrument; "
                   f"add anchors/seeds or the gate stays unproven.")

    print(f"\nSpearman(nash, LB) = {rho_nash:+.3f}   Spearman(mean_wr, LB) = {rho_mean:+.3f}"
          f"   (n={len(pool)})")
    print(f"\nVERDICT: {verdict}")
    print(f"(elapsed {time.perf_counter() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "pool": pool, "lb": list(lb), "seeds": args.seeds, "steps": args.steps,
        "rho_nash": rho_nash, "rho_mean": rho_mean, "margin": margin,
        "nash_rating": list(nash), "nash_mixture": list(nash_mix),
        "mean_vs_pool": list(mean), "winrate_matrix": W.tolist(),
        "pairs": diag, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
