"""self_research — the AUTONOMOUS driver: keep hunting for better bots on its own.

Where ``runner.py`` runs one bounded generational batch, this is the continuous,
STATEFUL, budget-bounded daemon meant to run unattended (overnight / under cron).
Progress COMPOUNDS across invocations because the incumbent parent is always read
back from the DuckDB frontier (``registry.select_parents``) — every ARL candidate
ever recorded, including those from prior runs and from ``runner.py``.

Loop (until --max-hours or --max-candidates):
    incumbent = best ARL candidate in DuckDB  (baseline genome if the DB is empty)
    repeat:
        mutate incumbent -> K children
        evaluate each honest (vs the diverse pool) ; record each in DuckDB (ARL)
        incumbent = best(incumbent, children)        # elitist, strict-> keeps champ on noise
        log frontier ; check budget

Then a CONFIRMATION re-eval of the champion at 2x seeds (memory:
local_league_is_submission_gate / "never decide by 12-16 seeds" — single-pass
fitness inflates ~3-4x; a lucky-high child must hold up before we believe it).

HONESTY GUARD: the run reads ``artifacts/research_loop/calibration.json`` and the
final report LEADS with the gate's LB-predictiveness verdict. If the gate is
falsified (rho < 0.3), the search is optimising a signal that does NOT predict the
leaderboard — the report flags every result EXPLORATORY and tells you to fix the
fitness before trusting/submitting anything. NO Kaggle submission happens here
(governance: submit is a separate, budgeted, human-in-the-loop step).

Run with the project's .venv python (NOT `uv run` — memory: build_uv_reverts_fresh_so):
    .venv/bin/python -m scripts.research_loop.self_research --max-hours 6 --seeds 6
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import date as _date
from pathlib import Path

import torch

from python.lab.experiments import add_experiment
from scripts.research_loop import registry
from scripts.research_loop.evaluator import DEFAULT_POOL, evaluate
from scripts.research_loop.genome import baseline_genome, fitness, mutate, serialize
from scripts.research_loop.search_space import SEARCH_SPACE

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "research_loop"
CALIB_JSON = ART / "calibration.json"


def _fmt_genome(g: dict) -> str:
    parts = []
    for k in SEARCH_SPACE:
        v = g.get(k)
        parts.append(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}")
    return " ".join(parts)


def _load_calibration() -> dict | None:
    try:
        return json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _incumbent(db_path: Path) -> tuple[dict, float | None, str]:
    """Best ARL candidate in the DB, else the shipped baseline (fitness unknown)."""
    parents = registry.select_parents(1, db_path=db_path)
    if parents:
        g, f = parents[0]
        return g, f, "db-frontier"
    return baseline_genome(), None, "baseline (empty frontier)"


def _trust_line(calib: dict | None) -> tuple[str, bool]:
    if not calib:
        return ("calibration NOT run yet (no artifacts/research_loop/calibration.json) "
                "-> fitness LB-predictiveness UNKNOWN; treat results as exploratory.", False)
    rho = calib.get("rho")
    if rho is None or rho != rho:
        return (f"calibration ran but rho is degenerate ({rho}); treat as exploratory.", False)
    if calib.get("competitive_tied"):
        return (f"calibration rho={rho:+.3f} is a FALSE PASS — the gate tied every competitive "
                "anchor (floor-veto only). The search cannot find a real winner on this signal; "
                "results are EXPLORATORY. Fix the discriminating power first.", False)
    trusted = rho >= 0.3
    band = (">=0.6 PREDICTS" if rho >= 0.6 else "0.3-0.6 WEAK" if trusted else "<0.3 FALSIFIED")
    return (f"calibration Spearman(gate, LB) = {rho:+.3f} [{band}]  ->  "
            + ("fitness is LB-predictive enough to trust the search."
               if trusted else
               "fitness does NOT predict the LB; results are EXPLORATORY, fix fitness first "
               "(docs/auto_research_pipeline.md)."), trusted)


def _holdwave_anchor_fit(calib: dict | None) -> float | None:
    if not calib:
        return None
    for r in calib.get("rows", []):
        if r.get("name") == "pgs_holdwave":
            return r.get("fitness")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous bot-search daemon (self-research).")
    ap.add_argument("--max-hours", type=float, default=6.0, help="wall-clock budget; stop after this")
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = unlimited (until --max-hours)")
    ap.add_argument("--candidates-per-batch", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=6, help="eval seeds per opponent")
    ap.add_argument("--steps", type=int, default=500, help="eval horizon (500 = honest)")
    ap.add_argument("--confirm-seeds", type=int, default=12, help="re-eval champion at this many seeds")
    ap.add_argument("--seats", default="4", choices=("2", "4", "mix"),
                    help="2 = 1v1 mirror; 4 = 4p gate; mix = field 0.54·4p+0.46·2p")
    ap.add_argument("--pool", default=",".join(DEFAULT_POOL))
    ap.add_argument("--seed-base", type=int, default=2000, help="base of the eval seed range")
    ap.add_argument("--rng-base", type=int, default=1000, help="mutation rng base (per-batch offset added)")
    ap.add_argument("--db", type=Path, default=registry.DEFAULT_DB)
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--report", type=Path, default=ART / "self_research_report.md")
    ap.add_argument("--state", type=Path, default=ART / "self_research_state.json")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    seats = "mix" if args.seats == "mix" else int(args.seats)
    pool = tuple(o.strip() for o in args.pool.split(",") if o.strip())
    today = _date.today().isoformat()
    enable_comets = not args.no_comets
    deadline_s = args.max_hours * 3600.0
    ART.mkdir(parents=True, exist_ok=True)

    calib = _load_calibration()
    trust_line, trusted = _trust_line(calib)
    hw_fit = _holdwave_anchor_fit(calib)

    inc_genome, inc_fit, inc_src = _incumbent(args.db)

    print("=== self_research (autonomous bot search) ===", flush=True)
    print(f"max_hours={args.max_hours} batch={args.candidates_per_batch} seeds={args.seeds} "
          f"steps={args.steps} pool={','.join(pool)} confirm_seeds={args.confirm_seeds}", flush=True)
    print(f"TRUST: {trust_line}", flush=True)
    print(f"incumbent ({inc_src}): fitness="
          f"{'unknown' if inc_fit is None else f'{inc_fit:+.4f}'}  {_fmt_genome(inc_genome)}", flush=True)
    if hw_fit is not None:
        print(f"holdwave anchor (LB 1228.8) gate fitness = {hw_fit:+.4f}  <- the bar to beat", flush=True)

    t0 = time.perf_counter()
    start_fit = inc_fit
    champ_genome, champ_fit = dict(inc_genome), inc_fit
    n_eval = 0
    batch = 0
    history: list[dict] = []

    while True:
        if time.perf_counter() - t0 >= deadline_s:
            print(f"\n[budget] reached {args.max_hours}h wall-clock; stopping.", flush=True)
            break
        if args.max_candidates and n_eval >= args.max_candidates:
            print(f"\n[budget] reached {args.max_candidates} candidates; stopping.", flush=True)
            break

        rng = random.Random(args.rng_base + batch)
        print(f"\n----- batch {batch} (champion="
              f"{'?' if champ_fit is None else f'{champ_fit:+.4f}'}) "
              f"[{(time.perf_counter()-t0)/3600:.2f}h / {args.max_hours}h] -----", flush=True)
        for cand in range(args.candidates_per_batch):
            if time.perf_counter() - t0 >= deadline_s:
                break
            genome = mutate(champ_genome, rng)
            print(f"  [b{batch}c{cand}] {_fmt_genome(genome)}", flush=True)
            metrics = evaluate(genome, seeds=args.seeds, steps=args.steps, pool=pool,
                               seed_base=args.seed_base, enable_comets=enable_comets,
                               seats=seats)
            fit = fitness(metrics)
            beats = champ_fit is None or fit > champ_fit
            new_id = registry.record(genome, metrics, fit, champ_fit, date=today,
                                     db_path=args.db, generation=batch, cand=cand)
            print(f"  [b{batch}c{cand}] -> fitness={fit:+.4f} "
                  f"(death={metrics['death_rate']:.3f} margin={metrics['mean_margin']:+.3f}) "
                  f"{'NEW CHAMPION' if beats else 'below champ'}  id={new_id}", flush=True)
            n_eval += 1
            history.append({"batch": batch, "cand": cand, "id": new_id, "fitness": fit,
                            "death_rate": metrics["death_rate"], "mean_margin": metrics["mean_margin"]})
            if beats:
                champ_genome, champ_fit = dict(genome), fit
        batch += 1

    # --- confirmation re-eval of the champion (anti triage-inflation) ---
    confirm = None
    if champ_fit is not None and n_eval > 0 and args.confirm_seeds > args.seeds:
        print(f"\n=== confirming champion at {args.confirm_seeds} seeds (was {args.seeds}) ===", flush=True)
        cm = evaluate(champ_genome, seeds=args.confirm_seeds, steps=args.steps, pool=pool,
                      seed_base=args.seed_base, enable_comets=enable_comets, seats=seats)
        confirm_fit = fitness(cm)
        held = (start_fit is None) or (confirm_fit >= start_fit)
        confirm = {"seeds": args.confirm_seeds, "fitness": confirm_fit,
                   "death_rate": cm["death_rate"], "mean_margin": cm["mean_margin"],
                   "held_vs_start": held}
        print(f"champion confirmed fitness={confirm_fit:+.4f} "
              f"(quick was {champ_fit:+.4f}); holds vs start "
              f"({'n/a' if start_fit is None else f'{start_fit:+.4f}'}): {held}", flush=True)

    # --- write report + state + one summary DB row ---
    elapsed_h = (time.perf_counter() - t0) / 3600.0
    improved = (start_fit is not None and champ_fit is not None and champ_fit > start_fit)
    beats_hw = (hw_fit is not None and champ_fit is not None and champ_fit > hw_fit)
    report = _render_report(trust_line, trusted, inc_src, start_fit, champ_genome, champ_fit,
                            confirm, hw_fit, beats_hw, improved, n_eval, elapsed_h, history)
    args.report.write_text(report, encoding="utf-8")
    args.state.write_text(json.dumps({
        "champion_genome": champ_genome, "champion_fitness": champ_fit,
        "start_fitness": start_fit, "confirm": confirm, "candidates_evaluated": n_eval,
        "elapsed_hours": elapsed_h, "trusted": trusted, "beats_holdwave_anchor": beats_hw,
        "history": history,
    }, indent=2), encoding="utf-8")

    if n_eval > 0:
        verdict = ("EXPLORATORY (gate not LB-predictive)" if not trusted
                   else "improved over start" if improved else "no improvement over start")
        add_experiment(
            db_path=args.db, date=today,
            idea=f"ARL self_research summary champion={serialize(champ_genome)}",
            command=f"python -m scripts.research_loop.self_research --max-hours {args.max_hours} --seeds {args.seeds}",
            result=(f"fitness={champ_fit:.4f} | candidates={n_eval} elapsed={elapsed_h:.2f}h | "
                    f"start={'na' if start_fit is None else f'{start_fit:.4f}'} "
                    f"holdwave_anchor={'na' if hw_fit is None else f'{hw_fit:.4f}'} "
                    f"beats_holdwave={beats_hw} | trusted={trusted}"),
            decision=verdict, status="logged", tags="ARL_RUN",
        )

    print(f"\n=== DONE: {n_eval} candidates in {elapsed_h:.2f}h ===", flush=True)
    print(f"champion fitness={'?' if champ_fit is None else f'{champ_fit:+.4f}'} | "
          f"improved_over_start={improved} | beats_holdwave_anchor={beats_hw} | trusted={trusted}", flush=True)
    print(f"wrote {args.report}", flush=True)
    print(f"wrote {args.state}", flush=True)
    return 0


def _render_report(trust_line, trusted, inc_src, start_fit, champ_genome, champ_fit,
                   confirm, hw_fit, beats_hw, improved, n_eval, elapsed_h, history) -> str:
    L = []
    L.append("# self_research — autonomous bot search report\n")
    L.append(f"> **TRUST FIRST:** {trust_line}\n")
    if not trusted:
        L.append("> ⚠ The gate fitness is NOT a verified LB predictor, so the \"champion\" below is the "
                 "best on an UNVERIFIED signal. Do **not** submit based on it. Priority is to fix the "
                 "fitness (add LB anchors / redesign the gate) — see `docs/auto_research_pipeline.md`.\n")
    L.append("\n## Result\n")
    L.append(f"- candidates evaluated: **{n_eval}**  (in {elapsed_h:.2f} h)")
    L.append(f"- incumbent source: {inc_src}")
    L.append(f"- start fitness: {'unknown' if start_fit is None else f'{start_fit:+.4f}'}")
    L.append(f"- champion fitness (quick): {'?' if champ_fit is None else f'{champ_fit:+.4f}'}")
    if confirm:
        L.append(f"- champion CONFIRMED @ {confirm['seeds']} seeds: **{confirm['fitness']:+.4f}** "
                 f"(death={confirm['death_rate']:.3f} margin={confirm['mean_margin']:+.3f}) — "
                 f"holds vs start: **{confirm['held_vs_start']}**")
    L.append(f"- improved over start: **{improved}**")
    if hw_fit is not None:
        L.append(f"- holdwave anchor (LB 1228.8) gate fitness: {hw_fit:+.4f}  ->  "
                 f"champion beats it: **{beats_hw}**")
    L.append("\n## Champion genome\n")
    L.append("```json")
    L.append(serialize(champ_genome))
    L.append("```")
    L.append("\nSearch-space knobs:\n")
    for k in SEARCH_SPACE:
        L.append(f"- `{k}` = {champ_genome.get(k)}")
    L.append("\n## What to do next\n")
    if not trusted:
        L.append("1. **Do not submit.** Fix the fitness signal first (the search is blind without it).")
        L.append("2. Add LB anchors (extend `calibrate.py` to non-PGS subjects) to get n>=5 and a real rho.")
        L.append("3. If rho stays <0.3, the robustness gate is the wrong instrument — rethink (e.g. "
                 "field-style exploiters in the pool, or a learned LB-proxy).")
    elif beats_hw and confirm and confirm["held_vs_start"]:
        L.append("1. The champion beats the holdwave LB-champion on the (LB-predictive) gate AND held the "
                 "confirmation re-eval. This is the strongest candidate yet.")
        L.append("2. **Human review + 1/day submit budget:** package it and submit ONE probe to convert the "
                 "gate prediction into a real LB anchor (governance in `docs/auto_research_pipeline.md`).")
    else:
        L.append("1. Champion did not clearly beat the holdwave bar (or did not hold confirmation). Keep the "
                 "loop running for more batches; the frontier compounds across runs.")
        L.append("2. Consider widening the search space or adding a style-exploiter to the pool.")
    L.append("\n## Top candidates this run\n")
    top = sorted(history, key=lambda h: -h["fitness"])[:8]
    L.append("| id | batch.cand | fitness | death | margin |")
    L.append("|---|---|---|---|---|")
    for h in top:
        L.append(f"| {h['id']} | b{h['batch']}c{h['cand']} | {h['fitness']:+.4f} "
                 f"| {h['death_rate']:.3f} | {h['mean_margin']:+.3f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
