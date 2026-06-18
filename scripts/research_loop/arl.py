"""arl — the Auto-Research Loop MVP runner (goal.md).

One governed loop, adapted from the "recursive self-improvement" video pattern to
this competition:

    select parent -> propose candidate -> (short) eval -> parse metrics
      -> keep/discard policy -> log to DuckDB -> JSON/MD report

It deliberately reuses the pieces that already exist (``genome``/``evaluator``/
``registry``) and adds the missing MVP contract: explicit **modes**
(``--dry-run`` / ``--smoke`` / ``--research``), the five-decision vocabulary, a
per-iteration **contract** (``run_id``/``parent``/``hypothesis``/``candidate``/
``patch``/``commands``/``seeds``/``metrics``/``faults``/``decision``), parseable
JSON+Markdown reports, and DuckDB logging.

Guardrails (goal.md invariants — the loop must NOT do these):
- it never touches gates/seeds/thresholds/pool used as off-limits validation;
- a *technical* failure is logged ``technical_fail``, never competitive
  ``rejected`` (see ``policy.keep_or_discard``);
- if the calibration says the fitness does NOT predict the LB, a ``promoted``
  verdict is downgraded to ``inconclusive`` (the search is on an unverified
  signal — fix the ruler before trusting it);
- NO Kaggle submission anywhere.

Run with the project's .venv python directly (NOT ``uv run`` — memory:
build_uv_reverts_fresh_so):

    .venv/bin/python -m scripts.research_loop.arl --dry-run  --iterations 1
    .venv/bin/python -m scripts.research_loop.arl --smoke    --iterations 1
    .venv/bin/python -m scripts.research_loop.arl --research --iterations 6 --seeds 24
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from python.lab.experiments import add_experiment, connect
from scripts.research_loop import registry
from scripts.research_loop.evaluator import DEFAULT_POOL
from scripts.research_loop.genome import baseline_genome, fitness, mutate, serialize
from scripts.research_loop.policy import (
    build_promotion_command,
    candidate_name,
    keep_or_discard,
    parse_metrics,
    select_survivors,
    status_for,
)
from scripts.research_loop.search_space import SEARCH_SPACE

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "research_loop"
ACT_TIMEOUT_MS = 1000.0  # Kaggle act budget; p95 over this is a fault.
TAG = "ARL"

# Smoke = tiny budget to validate WIRING only (never a competitive promotion;
# the seed floor in the policy guarantees smoke can only reach needs_more_seeds).
SMOKE_DEFAULTS = {"seeds": 2, "steps": 60, "pool": ("producer",)}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _knob_diff(parent: dict, child: dict) -> dict:
    """{knob: [old, new]} for every search-space knob the mutation changed."""
    diff = {}
    for k in SEARCH_SPACE:
        a, b = parent.get(k), child.get(k)
        if a != b:
            diff[k] = [a, b]
    return diff


def _hypothesis(diff: dict) -> str:
    if not diff:
        return "no-op mutation (candidate identical to parent)"
    parts = [f"{k}:{a:g}->{b:g}" if isinstance(b, (int, float)) else f"{k}:{a}->{b}"
             for k, (a, b) in diff.items()]
    return "perturb " + ", ".join(parts)


def _load_calibration():
    try:
        return json.loads((ART / "calibration.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _trust_line(calib) -> tuple[str, bool]:
    """LB-predictiveness verdict of the fitness signal (None/degenerate => untrusted)."""
    if not calib:
        return ("calibration NOT run (no artifacts/research_loop/calibration.json) -> "
                "fitness LB-predictiveness UNKNOWN; treat promotions as exploratory.", False)
    rho = calib.get("rho")
    if rho is None or rho != rho:
        return (f"calibration rho is degenerate ({rho}); treat promotions as exploratory.", False)
    if calib.get("competitive_tied"):
        return (f"calibration rho={rho:+.3f} is a FALSE PASS (gate tied every competitive anchor, "
                "floor-veto only); promotions are EXPLORATORY.", False)
    trusted = rho >= 0.3
    band = ">=0.6 PREDICTS" if rho >= 0.6 else "0.3-0.6 WEAK" if trusted else "<0.3 FALSIFIED"
    return (f"calibration Spearman(gate, LB) = {rho:+.3f} [{band}]", trusted)


def _evaluate(genome: dict, *, seeds: int, steps: int, pool, seats, seed_base: int,
              enable_comets: bool, verbose: bool) -> dict:
    """Run the fitness eval; ANY exception is captured as a fault payload.

    Returning ``{"error": ...}`` (rather than raising) is deliberate: it routes a
    crashed eval through ``parse_metrics`` -> ``technical_fail``, so a harness
    problem never masquerades as a weak candidate (goal.md).
    """
    from scripts.research_loop.evaluator import evaluate  # lazy: pulls the Rust .so

    try:
        return evaluate(genome, seeds=seeds, steps=steps, pool=pool, seed_base=seed_base,
                        enable_comets=enable_comets, verbose=verbose, seats=seats)
    except Exception as exc:  # noqa: BLE001 — we WANT to capture everything as a fault
        return {"error": f"{type(exc).__name__}: {exc}"}


def _record(contract: dict, *, db_path: Path, dry: bool) -> int | None:
    """Persist one iteration into ``experiments.duckdb`` (or validate-only if dry).

    Keeps the ``fitness=<x>`` prefix in ``result`` and the ``genome={...}`` suffix
    in ``idea`` so ``registry.select_parents`` can still recover this candidate as
    a future parent. ``status`` is set EXPLICITLY from the decision so
    ``technical_fail`` lands in ``logged``, never ``rejected``.
    """
    dec = contract["decision"]
    fit = contract.get("fitness")
    genome = contract["candidate"]["genome"]
    result = (f"fitness={fit:.4f} | " if isinstance(fit, (int, float)) else "fitness=na | ") + (
        f"decision={dec} | mode={contract['mode']} | "
        f"{json.dumps(contract['metrics_summary'], separators=(',', ':'))} | "
        f"faults={json.dumps(contract['faults'], separators=(',', ':'))}"
    )
    idea = f"ARL[{contract['mode']}] {contract['hypothesis']} | genome={serialize(genome)}"
    command = contract["commands"][0] if contract["commands"] else ""
    kwargs = dict(
        db_path=db_path, date=contract["date"], idea=idea, command=command,
        result=result, decision=f"{dec} — {contract['reason']}",
        status=status_for(dec), tags=TAG,
    )
    if dry:
        # Validate the row would insert: required fields present + DB opens.
        if not kwargs["date"] or not kwargs["idea"]:
            raise ValueError("dry-run record validation failed: missing date/idea")
        connect(db_path).close()
        return None
    return add_experiment(**kwargs)


def _metrics_summary(parsed) -> dict:
    return {
        "death_rate": parsed.death_rate, "mean_margin": parsed.mean_margin,
        "mean_final_planets": parsed.mean_final_planets, "n_seeds": parsed.n_seeds,
        "steps": parsed.steps, "seats": parsed.seats, "pool": list(parsed.pool),
        "valid": parsed.valid,
    }


# --------------------------------------------------------------------------- #
# one iteration
# --------------------------------------------------------------------------- #
def run_iteration(*, idx: int, mode: str, parent_genome: dict, parent_fitness: float | None,
                  parent_src: str, rng: random.Random, budget: dict, seats, seed_base: int,
                  enable_comets: bool, min_promotion_seeds: int, noise_band: float,
                  trusted: bool, run_cmd: str, today: str, verbose: bool) -> dict:
    """Propose -> (maybe) evaluate -> parse -> decide. Returns the iteration contract."""
    child = mutate(parent_genome, rng)
    diff = _knob_diff(parent_genome, child)
    seeds_n = budget["seeds"]
    steps = budget["steps"]
    pool = budget["pool"]
    seed_list = list(range(seed_base, seed_base + seeds_n))

    if mode == "dry-run":
        raw = None  # Mode 0: no eval, no edits.
    else:
        raw = _evaluate(child, seeds=seeds_n, steps=steps, pool=pool, seats=seats,
                        seed_base=seed_base, enable_comets=enable_comets, verbose=verbose)

    parsed = parse_metrics(raw, act_timeout_ms=ACT_TIMEOUT_MS, n_seeds=seeds_n)
    fit = fitness(parsed.raw) if parsed.valid else None
    decision = keep_or_discard(parsed, fitness=fit, parent_fitness=parent_fitness,
                               min_promotion_seeds=min_promotion_seeds, noise_band=noise_band)

    # Honesty downgrade: a promotion on an unverified ruler is only exploratory.
    dec_token, reason = decision.decision, decision.reason
    if dec_token == "promoted" and not trusted:
        dec_token = "inconclusive"
        reason = ("downgraded from promoted: fitness signal is NOT a verified LB predictor "
                  f"(calibration untrusted). Original: {decision.reason}")

    return {
        "run_id": f"{today.replace('-', '')}-{mode}-i{idx}",
        "mode": mode,
        "date": today,
        "parent": {"source": parent_src, "fitness": parent_fitness, "genome": parent_genome},
        "hypothesis": _hypothesis(diff),
        "candidate": {"genome": child, "knobs_changed": diff},
        "patch": {"kind": "config-mutation", "diff": diff},  # no code patch in the MVP
        "commands": [run_cmd,
                     f"(eval) evaluator.evaluate(seats={seats}, pool={','.join(pool)}, "
                     f"seeds={seeds_n}, steps={steps})" if mode != "dry-run" else "(no eval — dry-run)"],
        "seeds": seed_list,
        "metrics_summary": _metrics_summary(parsed),
        "per_opponent": parsed.per_opponent,
        "faults": parsed.faults,
        "fitness": fit,
        "parent_fitness": parent_fitness,
        "delta": decision.delta,
        "decision": dec_token,
        "competitive": dec_token in ("promoted", "rejected"),
        "reason": reason,
        "note": parsed.note,
    }


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def _render_md(summary: dict, iters: list[dict], trust_line: str, trusted: bool) -> str:
    L = ["# Auto-Research Loop — run report\n"]
    L.append(f"> **TRUST FIRST:** {trust_line}\n")
    if not trusted:
        L.append("> ⚠ Fitness is NOT a verified LB predictor — any promotion is downgraded to "
                 "`inconclusive`. Fix the ruler before trusting the search "
                 "(`docs/auto_research_pipeline.md`).\n")
    L.append("\n## Summary\n")
    for k in ("mode", "iterations", "parent_source", "parent_fitness", "min_promotion_seeds",
              "noise_band", "elapsed_s"):
        L.append(f"- {k}: `{summary.get(k)}`")
    L.append("- decisions: " + ", ".join(f"`{d}`={n}" for d, n in summary["decision_counts"].items()))
    handoff = summary.get("handoff")
    if handoff:
        L.append("\n## Promotion handoff (Mode 3)\n")
        L.append(f"> {handoff['note']}\n")
        if handoff["survivors"]:
            L.append(f"- survivors: {', '.join(f'`{s}`' for s in handoff['survivors'])}")
            L.append(f"- run the **seat-rotated** ruler (real verdict):\n\n  ```bash\n  {handoff['command']}\n  ```")
        else:
            L.append("- no survivor beat the parent on local fitness; nothing to hand off.")
    L.append("\n## Iterations\n")
    L.append("| run_id | hypothesis | seeds | fitness | Δparent | decision | faults |")
    L.append("|---|---|---|---|---|---|---|")
    for it in iters:
        f = it["fitness"]
        d = it["delta"]
        fr = f"{f:+.4f}" if isinstance(f, (int, float)) else "na"
        dr = f"{d:+.4f}" if isinstance(d, (int, float)) else "na"
        fault_txt = ", ".join(f"{k}={it['faults'].get(k)}" for k in
                              ("timeouts", "invalid_moves", "bad_status", "fallbacks", "exceptions")
                              if it["faults"].get(k)) or "—"
        L.append(f"| {it['run_id']} | {it['hypothesis'][:48]} | {len(it['seeds'])} | {fr} | {dr} "
                 f"| **{it['decision']}** | {fault_txt} |")
    L.append("\n## Reasons\n")
    for it in iters:
        L.append(f"- `{it['run_id']}` → **{it['decision']}**: {it['reason']}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-Research Loop MVP (dry-run/smoke/research).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_const", dest="mode", const="dry-run",
                      help="Mode 0: select parent + build contract, NO eval, NO edits, validate DB record.")
    mode.add_argument("--smoke", action="store_const", dest="mode", const="smoke",
                      help="Mode 1: tiny eval to validate wiring; never promotes competitively.")
    mode.add_argument("--research", action="store_const", dest="mode", const="research",
                      help="Mode 2: full-budget local research iterations.")
    ap.set_defaults(mode="dry-run")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=None, help="eval seeds (smoke default 2; research 6)")
    ap.add_argument("--steps", type=int, default=None, help="eval horizon (smoke default 60; else 500)")
    ap.add_argument("--pool", default=None, help="comma list of opponents (smoke default producer)")
    ap.add_argument("--seats", default="4", choices=("2", "4", "mix"))
    ap.add_argument("--rng-seed", type=int, default=0, help="mutation rng seed (determinism)")
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--min-promotion-seeds", type=int, default=16,
                    help="seed floor below which a candidate can only reach needs_more_seeds "
                         "(memory: never decide by 12-16 seeds)")
    ap.add_argument("--noise-band", type=float, default=0.10,
                    help="|fitness-parent| within this band => inconclusive (flat-top guard)")
    ap.add_argument("--db", type=Path, default=registry.DEFAULT_DB)
    ap.add_argument("--no-record", action="store_true", help="do not write to DuckDB")
    ap.add_argument("--ruler-profile", default="strong", choices=("quick", "standard", "strong"),
                    help="profile of the seat-rotated ruler command emitted for survivors (Mode 3)")
    ap.add_argument("--no-handoff", action="store_true",
                    help="research mode: do not emit the promotion-ruler handoff for survivors")
    ap.add_argument("--run-ruler", action="store_true",
                    help="Mode 3: ACTUALLY execute the emitted seat-rotated ruler command "
                         "(explicit, governed opt-in). Default OFF = emit only, never auto-run.")
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report-json", type=Path, default=ART / "arl_report.json")
    ap.add_argument("--report-md", type=Path, default=ART / "arl_report.md")
    ap.add_argument("--survivors-json", type=Path, default=ART / "survivors.json",
                    help="research mode: consolidated survivors manifest (list + ruler command)")
    args = ap.parse_args(argv)

    m = args.mode
    seats = "mix" if args.seats == "mix" else int(args.seats)
    enable_comets = not args.no_comets
    verbose = not args.quiet
    today = datetime.now(timezone.utc).date().isoformat()
    ART.mkdir(parents=True, exist_ok=True)

    # Budget per mode (smoke is tiny; research/dry honest unless overridden).
    if m == "smoke":
        seeds = args.seeds or SMOKE_DEFAULTS["seeds"]
        steps = args.steps or SMOKE_DEFAULTS["steps"]
        pool = tuple(args.pool.split(",")) if args.pool else SMOKE_DEFAULTS["pool"]
    else:
        seeds = args.seeds if args.seeds is not None else 6
        steps = args.steps if args.steps is not None else 500
        pool = tuple(o.strip() for o in args.pool.split(",")) if args.pool else DEFAULT_POOL
    budget = {"seeds": seeds, "steps": steps, "pool": pool}

    if m != "dry-run":
        import torch
        torch.set_num_threads(1)

    calib = _load_calibration()
    trust_line, trusted = _trust_line(calib)

    # Parent = best ARL candidate in the DB, else the shipped baseline.
    parents = registry.select_parents(1, db_path=args.db)
    if parents:
        parent_genome, parent_fitness = parents[0]
        parent_src = "db-frontier"
    else:
        parent_genome, parent_fitness, parent_src = baseline_genome(), None, "baseline (empty frontier)"

    run_cmd = (f".venv/bin/python -m scripts.research_loop.arl --{m} --iterations {args.iterations} "
               f"--seeds {seeds} --steps {steps} --pool {','.join(pool)} --seats {args.seats} "
               f"--rng-seed {args.rng_seed} --seed-base {args.seed_base} "
               f"--min-promotion-seeds {args.min_promotion_seeds} --noise-band {args.noise_band}")

    print(f"=== Auto-Research Loop ({m}) ===", flush=True)
    print(f"TRUST: {trust_line}", flush=True)
    print(f"parent ({parent_src}): fitness="
          f"{'unknown' if parent_fitness is None else f'{parent_fitness:+.4f}'}", flush=True)
    print(f"budget: seeds={seeds} steps={steps} pool={','.join(pool)} seats={args.seats} | "
          f"iterations={args.iterations} min_promotion_seeds={args.min_promotion_seeds} "
          f"noise_band={args.noise_band} record={not args.no_record}", flush=True)

    t0 = time.perf_counter()

    # Establish a comparison bar when the parent's fitness is unknown (skip in
    # dry-run, which never evaluates). Recorded as a non-candidate baseline row.
    if parent_fitness is None and m != "dry-run":
        print("measuring baseline parent fitness (no prior bar)…", flush=True)
        base_raw = _evaluate(parent_genome, seeds=seeds, steps=steps, pool=pool, seats=seats,
                             seed_base=args.seed_base, enable_comets=enable_comets, verbose=verbose)
        base_parsed = parse_metrics(base_raw, act_timeout_ms=ACT_TIMEOUT_MS, n_seeds=seeds)
        if base_parsed.valid:
            parent_fitness = fitness(base_parsed.raw)
            print(f"baseline parent fitness = {parent_fitness:+.4f}", flush=True)
            if not args.no_record:
                add_experiment(db_path=args.db, date=today,
                               idea=f"ARL[{m}] baseline bar measurement",
                               command=run_cmd,
                               result=f"fitness={parent_fitness:.4f} | role=baseline | "
                                      f"seeds={seeds} steps={steps} pool={','.join(pool)}",
                               decision="inconclusive — baseline bar (not a candidate)",
                               status="logged", tags=TAG)
        else:
            print(f"baseline eval did not produce a valid sample ({base_parsed.note}); "
                  "candidates will compare against no bar.", flush=True)

    iters: list[dict] = []
    for i in range(args.iterations):
        rng = random.Random(args.rng_seed + i)
        it = run_iteration(idx=i, mode=m, parent_genome=parent_genome, parent_fitness=parent_fitness,
                           parent_src=parent_src, rng=rng, budget=budget, seats=seats,
                           seed_base=args.seed_base, enable_comets=enable_comets,
                           min_promotion_seeds=args.min_promotion_seeds, noise_band=args.noise_band,
                           trusted=trusted, run_cmd=run_cmd, today=today, verbose=verbose)
        new_id = _record(it, db_path=args.db, dry=args.no_record or m == "dry-run")
        it["db_id"] = new_id
        fr = f"{it['fitness']:+.4f}" if isinstance(it["fitness"], (int, float)) else "na"
        print(f"  [{it['run_id']}] {it['hypothesis'][:60]}", flush=True)
        print(f"  [{it['run_id']}] fitness={fr} -> {it['decision'].upper()} "
              f"({'validated-only' if (args.no_record or m == 'dry-run') else f'db id={new_id}'})", flush=True)
        if it["reason"]:
            print(f"  [{it['run_id']}] reason: {it['reason']}", flush=True)
        # Elitist: promote the parent within the run only on a real promotion.
        if it["decision"] == "promoted":
            parent_genome, parent_fitness, parent_src = it["candidate"]["genome"], it["fitness"], it["run_id"]
        iters.append(it)

    elapsed = time.perf_counter() - t0
    counts: dict[str, int] = {}
    for it in iters:
        counts[it["decision"]] = counts.get(it["decision"], 0) + 1
    summary = {
        "mode": m, "iterations": len(iters), "parent_source": parent_src,
        "parent_fitness": parent_fitness, "min_promotion_seeds": args.min_promotion_seeds,
        "noise_band": args.noise_band, "elapsed_s": round(elapsed, 2),
        "trusted": trusted, "trust_line": trust_line, "recorded": not (args.no_record or m == "dry-run"),
        "decision_counts": counts,
    }

    # Mode 3 — promotion handoff (research only): write each local-veto survivor as
    # a PGS genome JSON so league_agents auto-registers it, then EMIT (not run) the
    # seat-rotated ruler command. The real promotion verdict stays human + governed.
    handoff = None
    if m == "research" and not args.no_handoff:
        survivors = select_survivors(iters, noise_band=args.noise_band)
        cand_dir = ART / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for it in survivors:
            name = candidate_name(it["run_id"])
            (cand_dir / f"{name}.json").write_text(
                json.dumps(it["candidate"]["genome"], sort_keys=True, indent=2), encoding="utf-8")
            names.append(name)
        cmd = build_promotion_command(names, profile=args.ruler_profile)
        handoff = {
            "survivors": names, "noise_band": args.noise_band, "ruler_profile": args.ruler_profile,
            "command": cmd, "candidate_dir": str(cand_dir),
            "note": ("local-veto survivors (beat parent on UNVERIFIED local fitness). The "
                     "seat-rotated league_submit_ruler is the REAL promotion verdict — run it "
                     "manually; ARL never auto-promotes/submits. Staging dir is not auto-pruned."),
        }
        # Consolidated manifest (list of survivors + the exact ruler command), always
        # written in research mode so the handoff has a single stable artifact path.
        args.survivors_json.write_text(json.dumps(handoff, indent=2, default=str), encoding="utf-8")
        print("\n=== PROMOTION HANDOFF (Mode 3) ===", flush=True)
        if names:
            sh = ART / "promote_survivors.sh"
            sh.write_text("#!/usr/bin/env bash\n# ARL -> seat-rotated promotion ruler (governed; "
                          "run manually, human-in-the-loop)\nset -euo pipefail\n" + cmd + "\n",
                          encoding="utf-8")
            print(f"{len(names)} survivor(s) cleared the local veto: {', '.join(names)}", flush=True)
            print("Local 'promoted' is NOT competitive — run the seat-rotated ruler:", flush=True)
            print(f"  {cmd}", flush=True)
            print(f"(genomes in {cand_dir}; command also written to {sh})", flush=True)
            if args.run_ruler:
                import shlex
                import subprocess
                print("\n--run-ruler set: executing the seat-rotated ruler now (explicit opt-in)…",
                      flush=True)
                rc = subprocess.run(shlex.split(cmd), cwd=str(REPO)).returncode
                handoff["ruler_executed"] = True
                handoff["ruler_returncode"] = rc
                print(f"seat-rotated ruler exited with code {rc}", flush=True)
            else:
                handoff["ruler_executed"] = False
                print("(--run-ruler NOT set → ruler NOT executed; run the command above manually)",
                      flush=True)
        else:
            handoff["ruler_executed"] = False
            print("no survivor beat the parent on local fitness; nothing to hand off.", flush=True)
    summary["handoff"] = handoff

    args.report_json.write_text(json.dumps({"summary": summary, "iterations": iters}, indent=2,
                                           default=str), encoding="utf-8")
    args.report_md.write_text(_render_md(summary, iters, trust_line, trusted), encoding="utf-8")

    print(f"\n=== DONE ({elapsed:.1f}s) ===", flush=True)
    print("decisions: " + ", ".join(f"{d}={n}" for d, n in counts.items()), flush=True)
    print(f"wrote {args.report_json}", flush=True)
    print(f"wrote {args.report_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
