"""Runner — ties the 5 components into a generational research loop.

  parent selection (DuckDB top-N | baseline at gen 0)
    -> mutate K candidates
    -> evaluate each vs the opponent pool (h9 gate)
    -> record each in experiments.duckdb (applied if beats parent, else rejected)
    -> best becomes the next generation's parent
    -> repeat for --generations.

MVP success criterion: >=3 candidates evaluated in a run, every one written to
the DB, and DIFFERENT genomes producing DIFFERENT fitness (real discrimination,
not a trivial tie). NO Kaggle submission anywhere in this loop.

Usage:
    python -m scripts.research_loop.runner --generations 2 --candidates-per-gen 2 --seeds 4 \
        [--steps 500] [--pool producer,oep,rush,greedy] [--rng-seed 0] [--db experiments.duckdb] \
        [--no-record] [--no-comets]

Run with the project's .venv python directly (NOT `uv run`, which reverts the
fresh Rust .so — memory: build_uv_reverts_fresh_so):
    .venv/bin/python -m scripts.research_loop.runner ...
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import date as _date
from pathlib import Path

import torch

from scripts.research_loop.evaluator import DEFAULT_POOL, evaluate
from scripts.research_loop.genome import (
    baseline_genome,
    fitness,
    mutate,
    serialize,
)
from scripts.research_loop import registry


def _fmt_genome(g: dict) -> str:
    """Short readable view of the search-space knobs only (skip fixed regime)."""
    from scripts.research_loop.search_space import SEARCH_SPACE

    parts = []
    for k in SEARCH_SPACE:
        v = g.get(k)
        parts.append(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-research loop over PGS strategy knobs (MVP).")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--candidates-per-gen", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=4, help="eval seeds per opponent (keep low for speed)")
    ap.add_argument("--steps", type=int, default=500, help="eval horizon (500 = honest; lower for smoke)")
    ap.add_argument("--pool", default=",".join(DEFAULT_POOL), help="comma list of opponents")
    ap.add_argument("--rng-seed", type=int, default=0, help="seed for the MUTATION rng (determinism)")
    ap.add_argument("--seed-base", type=int, default=2000, help="base of the eval seed range")
    ap.add_argument("--db", type=Path, default=registry.DEFAULT_DB)
    ap.add_argument("--no-record", action="store_true", help="dry run: do not write to DuckDB")
    ap.add_argument("--no-comets", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    rng = random.Random(args.rng_seed)
    pool = tuple(o.strip() for o in args.pool.split(",") if o.strip())
    today = _date.today().isoformat()
    enable_comets = not args.no_comets

    print(f"=== Auto-research loop ===", flush=True)
    print(f"generations={args.generations} candidates/gen={args.candidates_per_gen} "
          f"seeds={args.seeds} steps={args.steps} pool={','.join(pool)} "
          f"rng_seed={args.rng_seed} record={not args.no_record}", flush=True)

    # Gen-0 parent = baseline. Keep a list of (genome, fitness) seen this run for
    # next-gen selection (independent of DB so --no-record still loops).
    parent_genome = baseline_genome()
    parent_fit: float | None = None
    all_results: list[tuple[int, int, dict, float]] = []  # (gen, cand, genome, fitness)
    t0 = time.perf_counter()

    for gen in range(args.generations):
        print(f"\n----- generation {gen} (parent fitness="
              f"{'baseline' if parent_fit is None else f'{parent_fit:.4f}'}) -----", flush=True)
        gen_results: list[tuple[dict, float, dict]] = []
        for cand in range(args.candidates_per_gen):
            genome = mutate(parent_genome, rng)
            print(f"  [g{gen}c{cand}] genome: {_fmt_genome(genome)}", flush=True)
            metrics = evaluate(
                genome, seeds=args.seeds, steps=args.steps, pool=pool,
                seed_base=args.seed_base, enable_comets=enable_comets,
            )
            fit = fitness(metrics)
            beats = parent_fit is None or fit > parent_fit
            print(f"  [g{gen}c{cand}] -> fitness={fit:+.4f} "
                  f"(death={metrics['death_rate']:.3f} margin={metrics['mean_margin']:+.3f}) "
                  f"{'BEATS parent' if beats else 'below parent'}", flush=True)
            if not args.no_record:
                new_id = registry.record(
                    genome, metrics, fit, parent_fit,
                    date=today, db_path=args.db, generation=gen, cand=cand,
                )
                print(f"  [g{gen}c{cand}] recorded experiment id={new_id} "
                      f"(status={'applied' if beats else 'rejected'})", flush=True)
            gen_results.append((genome, fit, metrics))
            all_results.append((gen, cand, genome, fit))

        # Select best of this generation as next parent (elitist; >= keeps parent
        # if no child beats it — guards against drift on noisy fitness).
        best_genome, best_fit, _ = max(gen_results, key=lambda t: t[1])
        if parent_fit is None or best_fit > parent_fit:
            parent_genome, parent_fit = best_genome, best_fit
            print(f"  new parent: fitness={parent_fit:+.4f} ({_fmt_genome(parent_genome)})", flush=True)
        else:
            print(f"  no child beat parent (best={best_fit:+.4f} <= {parent_fit:+.4f}); parent unchanged", flush=True)

    # --- summary + discrimination check ---
    fits = [f for *_, f in all_results]
    n = len(fits)
    distinct = len({round(f, 6) for f in fits})
    spread = (max(fits) - min(fits)) if fits else 0.0
    print(f"\n=== SUMMARY ({time.perf_counter() - t0:.0f}s) ===", flush=True)
    print(f"candidates evaluated: {n}", flush=True)
    for gen, cand, genome, f in all_results:
        print(f"  g{gen}c{cand}: fitness={f:+.4f}  {_fmt_genome(genome)}", flush=True)
    print(f"distinct fitness values: {distinct}/{n}  (spread={spread:.4f})", flush=True)
    discriminates = distinct >= 2 and spread > 1e-6
    print(f"FITNESS DISCRIMINATES: {discriminates}  "
          f"(>=2 distinct values across genomes)", flush=True)
    print(f"best fitness this run: {max(fits):+.4f}" if fits else "no candidates", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
