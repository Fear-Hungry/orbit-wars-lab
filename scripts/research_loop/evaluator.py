"""Multi-opponent fitness evaluator — reuses scripts/h9_4p_gate.run_config.

Runs the genome-configured PGS (seat 0) against each opponent in a diverse pool
(producer, oep, rush, greedy), 4p, ``steps`` steps x ``seeds`` seeds, then
averages death_rate / mean_margin / mean_final_planets across the pool. The gate
already instantiates opponents via ``get_isolated_opponents`` per env (memory:
make_isolated_opponent — the gate batches one instance per (env, seat), never
reusing a name within a game), so we inherit that correctness for free.

We import ``run_config`` directly rather than shelling out: it is a plain
function, deterministic for fixed seeds, and the in-process path lets us hold the
fresh Rust ``.so`` binding the project's .venv already loaded (memory:
build_uv_reverts_fresh_so — never invoke via `uv run`).
"""
from __future__ import annotations

import time

from scripts.h9_4p_gate import run_config
from scripts.research_loop.eval2p import run_config_2p
from scripts.research_loop.genome import to_pgs_config

DEFAULT_POOL = ("producer", "oep", "rush", "greedy")

# Field regime split (measured via EpisodeService, memory: field_is_majority_4p).
# The LB is a MIXTURE — neither pure-4p (collapses) nor pure-2p (inverts) can match
# it. The "mix" gate weights both so the fitness measures what the LB measures.
FIELD_4P, FIELD_2P = 0.54, 0.46


def evaluate(genome: dict | None, *, seeds: int, steps: int, pool=DEFAULT_POOL,
             seed_base: int = 2000, enable_comets: bool = True, verbose: bool = True,
             seats=4, subject_factory=None) -> dict:
    """Evaluate one candidate over the opponent pool. Returns pool-averaged metrics.

    The subject (seat 0) is either a PGS ``genome`` (mutated knobs) or a
    ``subject_factory`` callable returning a fresh official-obs agent (the
    materialiser path — any FACTORIES entry: pgs_*, producer, exported PPO, a new
    planner, exploiters). Exactly one of ``genome`` / ``subject_factory`` is used.

    ``seats`` selects the harness:
      4     → the 4p gate (run_config) — collapses the hold-family to a tied floor.
      2     → the 2p mirror (run_config_2p) — separates, but inverts vs LB.
      "mix" → field-weighted 0.54·4p + 0.46·2p, the only gate that can match a
              mixed-regime LB (see the six-hats validation analysis 2026-06-14).
    Metrics are comparable WITHIN a seat mode, not across modes.
    """
    if seats == "mix":
        r4 = evaluate(genome, seeds=seeds, steps=steps, pool=pool, seed_base=seed_base,
                      enable_comets=enable_comets, verbose=verbose, seats=4,
                      subject_factory=subject_factory)
        r2 = evaluate(genome, seeds=seeds, steps=steps, pool=pool, seed_base=seed_base,
                      enable_comets=enable_comets, verbose=verbose, seats=2,
                      subject_factory=subject_factory)
        keys = ("death_rate", "mean_margin", "mean_final_planets")
        mixed = {k: FIELD_4P * r4[k] + FIELD_2P * r2[k] for k in keys}
        mixed["per_opponent"] = {
            opp: {k: FIELD_4P * r4["per_opponent"][opp][k] + FIELD_2P * r2["per_opponent"][opp][k]
                  for k in keys}
            for opp in r4["per_opponent"]
        }
        mixed.update(timeouts=r4["timeouts"] + r2["timeouts"],
                     eval_elapsed_s=r4["eval_elapsed_s"] + r2["eval_elapsed_s"],
                     seeds=seeds, steps=steps, seats="mix", pool=list(pool),
                     components={"4p": {k: r4[k] for k in keys},
                                 "2p": {k: r2[k] for k in keys}})
        if verbose:
            print(f"    [mix] 4p×{FIELD_4P}+2p×{FIELD_2P}: "
                  f"death={mixed['death_rate']:.3f} margin={mixed['mean_margin']:+.3f}", flush=True)
        return mixed

    _run = run_config if seats == 4 else run_config_2p
    pgs_config = to_pgs_config(genome) if genome is not None else {}
    seed_list = list(range(seed_base, seed_base + seeds))
    # Pass subject_factory ONLY when set: keeps the genome path byte-identical to
    # the pre-materialiser signature (any run_config-shaped callable still works).
    sf_kw = {"subject_factory": subject_factory} if subject_factory is not None else {}
    per_opp: dict[str, dict] = {}
    t0 = time.perf_counter()
    for opp in pool:
        r = _run(opp, pgs_config, seed_list, steps, enable_comets, opponent=opp, **sf_kw)
        per_opp[opp] = {
            "death_rate": r["death_rate"],
            "mean_margin": r["mean_margin"],
            "mean_final_planets": r["mean_final_planets"],
            "timeouts": r["timeouts"],
        }
        if verbose:
            print(f"    vs {opp:9s}: death={r['death_rate']:.3f} "
                  f"margin={r['mean_margin']:+.3f} planets={r['mean_final_planets']:.2f} "
                  f"timeouts={r['timeouts']} ({r['elapsed_s']:.0f}s)", flush=True)

    n = len(per_opp)
    agg = {
        "death_rate": sum(v["death_rate"] for v in per_opp.values()) / n,
        "mean_margin": sum(v["mean_margin"] for v in per_opp.values()) / n,
        "mean_final_planets": sum(v["mean_final_planets"] for v in per_opp.values()) / n,
        "timeouts": sum(v["timeouts"] for v in per_opp.values()),
        "per_opponent": per_opp,
        "eval_elapsed_s": time.perf_counter() - t0,
        "seeds": seeds,
        "steps": steps,
        "seats": seats,
        "pool": list(pool),
    }
    return agg
