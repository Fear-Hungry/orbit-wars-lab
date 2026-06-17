"""H9 4p death-rate gate — the real test of the threat-value 4p model.

DEPRECATED AS A PROMOTION GATE (2026-06-16). This harness pins the candidate to
SEAT 0 in every 4p game (seats 1..3 = isolated opponents) and never rotates it.
A strong result here can therefore be a seat-0 advantage artifact, not real
strength. Do NOT use it to decide promotion/submission. The seat-rotated,
fault-checked promotion ruler is ``scripts/league_submit_ruler.py`` (rotates
seats by seed index and validates the seat counter per task). The CLI now
refuses to run unless you pass ``--ack-seat-biased`` (or set
``OWL_ALLOW_SEAT_BIASED_GATE=1``), which marks the output as a non-promotion
diagnostic. The library functions (``run_config``/``decide``) stay importable so
research-loop fitness keeps working — but that fitness inherits the same seat
bias and must not gate submission either.

The probe (h9_threat_probe) only proved the threat features SEPARATE actions.
This proves the bot built on them actually SURVIVES 4p better than the holdwave
floor. Seat 0 = candidate PGS; seats 1..3 = isolated Producer. 500 steps (the
only honest horizon, [kaggle_500_step_eval_required]). Metric = annihilation rate
(seat 0 ends with 0 planets) + mean normalised ship margin + mean final planets.

PASS iff death_4p(threat_value_4p) < death_4p(holdwave) with no margin regression.
This is the gate that decides whether we export/submit (DB 234 plan).

Decision rule (six-hats validation fix, 2026-06-14): the verdict is the FIELD
AGGREGATE over the opponent pool, not a per-opponent tally — a per-opponent
"N/len PASS" line is gameable (win vs producer, lose vs rush) and a death-only
verdict green-lights configs that buy survival by shedding margin (exactly the
LB ~1048 stagnation, memory: h9_threat_value_4p_cracks_floor). So we require all
three, aggregated: death strictly lower, margin not regressed beyond --margin-tol,
and zero timeouts. The pure ``decide()`` carries the rule so it is unit-testable
without the simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from bots.pgs.planner import PGSConfig, make_runtime
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import planet_owner, planet_ships
from python.orbit_wars_gym.observation import to_official_observation

SEAT = 0
ACT_TIMEOUT = 1.0


def _own_enemy_ships(state, seat):
    own = sum(planet_ships(p) for p in state["planets"] if planet_owner(p) == seat)
    enemy = sum(planet_ships(p) for p in state["planets"] if planet_owner(p) not in (-1, seat))
    return float(own), float(enemy)


def _own_planets(state, seat):
    return sum(1 for p in state["planets"] if planet_owner(p) == seat)


def run_config(name, pgs_config, seeds, steps, enable_comets, opponent="producer"):
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=4, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=enable_comets, episode_steps=steps,
                                                 act_timeout=ACT_TIMEOUT))
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [make_runtime(PGSConfig(**pgs_config)) for _ in range(n)]
    others = [1, 2, 3]
    opp = get_isolated_opponents(opponent, n * len(others))
    timeouts = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        rows = []
        for i in range(n):
            obs = to_official_observation(states[i], SEAT)
            ts = time.perf_counter()
            moves = agents[i].act(obs)
            if time.perf_counter() - ts > ACT_TIMEOUT:
                timeouts += 1
                moves = []
            for m in moves:
                if len(m) >= 3:
                    rows.append([float(i), float(SEAT), float(m[0]), float(m[1]), float(m[2])])
            for j, seat in enumerate(others):
                for m in opp[i * len(others) + j](states[i], seat):
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, SEAT)
        states = backend.states()

    deaths, margins, planets = 0, [], []
    for i in range(n):
        own, enemy = _own_enemy_ships(states[i], SEAT)
        npl = _own_planets(states[i], SEAT)
        if npl == 0:
            deaths += 1
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else -1.0)
        planets.append(npl)
    return {
        "name": name,
        "death_rate": deaths / n,
        "mean_margin": float(np.mean(margins)),
        "mean_final_planets": float(np.mean(planets)),
        "timeouts": timeouts,
        "elapsed_s": time.perf_counter() - t0,
    }


def _aggregate(results: list[dict]) -> dict:
    """Field aggregate over the opponent pool (equal-weighted, lower death better)."""
    n = max(1, len(results))
    return {
        "death_rate": sum(r["death_rate"] for r in results) / n,
        "mean_margin": sum(r["mean_margin"] for r in results) / n,
        "mean_final_planets": sum(r["mean_final_planets"] for r in results) / n,
        "timeouts": sum(r["timeouts"] for r in results),
    }


def decide(base: dict, h9: dict, *, margin_tol: float = 0.02) -> dict:
    """Pure verdict: does h9 beat the holdwave floor over the pool aggregate?

    PASS iff ALL hold (the documented spec, enforced — not death-only):
      - death_rate(h9) < death_rate(holdwave)          [strictly safer]
      - mean_margin(h9) >= mean_margin(holdwave) - tol  [no margin regression]
      - timeouts(h9) == 0                               [fits the 1s act budget]

    ``base``/``h9`` are aggregate summaries ({death_rate, mean_margin, timeouts}).
    Returns the verdict plus the deltas, so a caller can log WHY it failed.
    """
    d_death = h9["death_rate"] - base["death_rate"]      # want < 0
    d_margin = h9["mean_margin"] - base["mean_margin"]    # want >= -tol
    safer = d_death < 0.0
    no_margin_regression = d_margin >= -float(margin_tol)
    no_timeouts = int(h9.get("timeouts", 0)) == 0
    reasons = []
    if not safer:
        reasons.append(f"death not reduced (Δ={d_death:+.3f})")
    if not no_margin_regression:
        reasons.append(f"margin regressed (Δ={d_margin:+.3f} < -{margin_tol:.3f})")
    if not no_timeouts:
        reasons.append(f"timeouts={h9.get('timeouts', 0)}")
    return {
        "passed": safer and no_margin_regression and no_timeouts,
        "delta_death": d_death,
        "delta_margin": d_margin,
        "margin_tol": float(margin_tol),
        "reasons": reasons,
    }


def _refuse_as_promotion_gate(ack: bool, *, tool: str, bias: str) -> None:
    """Block CLI use as a promotion gate unless the seat bias is acknowledged.

    Reversible and non-destructive: library imports are unaffected; only the
    ``__main__`` path is guarded. Pass ``--ack-seat-biased`` or set
    ``OWL_ALLOW_SEAT_BIASED_GATE=1`` to run it as a labelled diagnostic.
    """
    import os
    if ack or os.environ.get("OWL_ALLOW_SEAT_BIASED_GATE") == "1":
        print(f"[{tool}] WARNING: seat-biased diagnostic ({bias}); "
              "NOT a promotion gate. Use scripts/league_submit_ruler.py to promote.",
              file=sys.stderr, flush=True)
        return
    sys.stderr.write(
        f"\n[{tool}] REFUSING TO RUN AS A PROMOTION GATE.\n"
        f"  Seat bias: {bias}. Results can be a seat-0 advantage artifact.\n"
        "  The seat-rotated, fault-checked promotion ruler is:\n"
        "    python -m scripts.league_submit_ruler --candidates <name> "
        "--incumbent <name> --seeds 96 --steps 500\n"
        "  To run this anyway as a NON-promotion diagnostic, pass --ack-seat-biased\n"
        "  (or set OWL_ALLOW_SEAT_BIASED_GATE=1).\n\n")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ack-seat-biased", action="store_true",
                    help="acknowledge this is a seat-0-pinned diagnostic, NOT a promotion gate")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--opponents", default="producer", help="comma list: producer,rush,greedy,oep")
    ap.add_argument("--margin-tol", type=float, default=0.02,
                    help="max allowed aggregate margin drop vs holdwave before FAIL")
    ap.add_argument("--out", type=Path, default=None, help="write verdict JSON here")
    ap.add_argument("--no-comets", action="store_true")
    args = ap.parse_args()
    _refuse_as_promotion_gate(args.ack_seat_biased, tool="h9_4p_gate",
                              bias="candidate pinned to seat 0 in all 4p games, no rotation")
    torch.set_num_threads(1)
    seeds = list(range(2000, 2000 + args.seeds))
    enable_comets = not args.no_comets

    # Shipped-fidelity configs: baseline = ACTUAL submitted PGS config; H9 only adds
    # threat_value_4p=True (auto-enables 4p survival portfolio; 2p stays frozen).
    shipped = {"scripts": "hold", "wave_min_ships": 60.0, "wave_start_step": 150}
    configs = {"holdwave": dict(shipped), "h9_threat": dict(shipped, threat_value_4p=True)}
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    per_opp = []          # (opp, base_summary, h9_summary)
    base_results, h9_results = [], []
    for opp in opponents:
        print(f"\n########## OPPONENT = {opp} ##########", flush=True)
        res = {}
        for name, cfg in configs.items():
            print(f"[gate] {name} vs {opp} ({args.seeds} seeds x {args.steps} steps 4p)...", flush=True)
            res[name] = run_config(name, cfg, seeds, args.steps, enable_comets, opponent=opp)
            r = res[name]
            print(f"  {name}: death={r['death_rate']:.3f} margin={r['mean_margin']:+.3f} "
                  f"planets={r['mean_final_planets']:.2f} timeouts={r['timeouts']} ({r['elapsed_s']:.0f}s)", flush=True)
        base, h9 = res["holdwave"], res["h9_threat"]
        base_results.append(base)
        h9_results.append(h9)
        per_opp.append((opp, base, h9))
        print(f"  vs {opp}: death {base['death_rate']:.3f}->{h9['death_rate']:.3f}  "
              f"margin {base['mean_margin']:+.3f}->{h9['mean_margin']:+.3f}", flush=True)

    base_agg, h9_agg = _aggregate(base_results), _aggregate(h9_results)
    verdict = decide(base_agg, h9_agg, margin_tol=args.margin_tol)

    print("\n=== H9 ROBUSTNESS SUMMARY (per-opponent; verdict is the AGGREGATE) ===")
    for opp, b, h in per_opp:
        print(f"  {opp:10s}: death {b['death_rate']:.3f}->{h['death_rate']:.3f}  "
              f"margin {b['mean_margin']:+.3f}->{h['mean_margin']:+.3f}")
    print(f"\n  AGGREGATE: death {base_agg['death_rate']:.3f}->{h9_agg['death_rate']:.3f} "
          f"(Δ={verdict['delta_death']:+.3f})  "
          f"margin {base_agg['mean_margin']:+.3f}->{h9_agg['mean_margin']:+.3f} "
          f"(Δ={verdict['delta_margin']:+.3f}, tol={args.margin_tol:.3f})")
    print(f"  VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}"
          + (f"  [{'; '.join(verdict['reasons'])}]" if verdict["reasons"] else ""))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "passed": verdict["passed"],
            "seeds": args.seeds, "steps": args.steps, "margin_tol": args.margin_tol,
            "opponents": opponents,
            "aggregate": {"holdwave": base_agg, "h9_threat": h9_agg},
            "delta_death": verdict["delta_death"], "delta_margin": verdict["delta_margin"],
            "reasons": verdict["reasons"],
            "per_opponent": [
                {"opponent": opp, "holdwave": b, "h9_threat": h} for opp, b, h in per_opp
            ],
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)

    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
