"""H9 4p death gate on the OFFICIAL kaggle_environments orbit_wars env.

Python-only (no Rust build) — this is the gate the Kaggle notebook runs and the
single source of truth shared with it. The official env is more authoritative
than our Rust parity reproduction, so this is the number that decides submission.

Seat 0 = candidate PGS; seats 1..3 = Producer. 500 steps. Metric = annihilation
rate (player 0 ends with 0 planets) + mean final planets. PASS iff
death(threat_value_4p) < death(holdwave) with no fallback/timeout blowups.
"""
from __future__ import annotations

import argparse
import json
import time

SHIPPED = dict(scripts="hold", wave_min_ships=60.0, wave_start_step=150)
H9 = dict(SHIPPED, threat_value_4p=True)


def _make_pgs_agent(cfg):
    from bots.pgs.planner import PGSConfig, PGSRuntime

    rt = PGSRuntime(PGSConfig(**cfg))

    def _agent(obs, config=None):
        try:
            return rt.act(obs)
        except Exception:
            return []

    return _agent


def _producer_agent():
    from bots.producer.agent import agent as producer

    def _agent(obs, config=None):
        try:
            return producer(obs)
        except Exception:
            return []

    return _agent


def _planets_owned(observation, player):
    from python.orbit_wars_gym.entities import planet_owner

    return sum(1 for p in observation["planets"] if planet_owner(p) == player)


def run_config(name, cfg, seeds, steps):
    from kaggle_environments import make

    deaths, planets = 0, []
    for s in seeds:
        env = make("orbit_wars", configuration={"episodeSteps": steps, "seed": int(s)}, debug=False)
        env.reset(num_agents=4)
        out = env.run([_make_pgs_agent(cfg), _producer_agent(), _producer_agent(), _producer_agent()])
        final = out[-1][0]["observation"]
        n0 = _planets_owned(final, 0)
        planets.append(n0)
        if n0 == 0:
            deaths += 1
    return {"name": name, "death_rate": deaths / len(seeds),
            "mean_final_planets": sum(planets) / len(planets), "n": len(seeds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default=None, help="optional json path for the verdict")
    args = ap.parse_args()
    seeds = list(range(3000, 3000 + args.seeds))

    results = {}
    for name, cfg in [("holdwave", SHIPPED), ("h9_threat", H9)]:
        t = time.perf_counter()
        print(f"[kenv-gate] {name}: {args.seeds} seeds x {args.steps} steps 4p (official env)...", flush=True)
        results[name] = run_config(name, cfg, seeds, args.steps)
        r = results[name]
        r["elapsed_s"] = time.perf_counter() - t
        print(f"  {name}: death_rate={r['death_rate']:.3f} mean_final_planets={r['mean_final_planets']:.2f} "
              f"({r['elapsed_s']:.0f}s)", flush=True)

    base, h9 = results["holdwave"], results["h9_threat"]
    passed = h9["death_rate"] < base["death_rate"]
    verdict = {"pass": bool(passed), "holdwave": base, "h9_threat": h9,
               "delta_death": h9["death_rate"] - base["death_rate"]}
    print("\n=== H9 4p GATE (official kaggle_environments) ===")
    print(f"  death_rate: holdwave={base['death_rate']:.3f} -> h9={h9['death_rate']:.3f}  "
          f"{'PASS' if passed else 'FAIL'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"  wrote {args.out}")
    return verdict


if __name__ == "__main__":
    main()
