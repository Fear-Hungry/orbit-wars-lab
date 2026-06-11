"""H4 — MAP-Elites / Quality-Diversity over the PGS strategy space.

Illuminates (does not optimize) the behavioural space of PGS configs: each cell
of a (wave-size x launch-cadence) grid keeps the highest-fitness genome that
*behaves* that way. Fitness = mean win-margin vs a DIVERSE reference set (never
vs Producer alone — that overfit trap is documented). Behaviour descriptors are
MEASURED from the candidate's actual launches in-game, not read off the genome
(the QD principle: illuminate what the agent DOES).

Lit: Mouret & Clune 2015 (arXiv:1504.04909); Cully et al. 2015 (Nature).

Output: artifacts/mapelites/archive.json — {cell_key: {genome, fitness, desc,
beats_producer, n_games}}. Resumable: re-run appends evals into the same archive.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/mapelites_pgs.py --evals 60 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import fleet_owner, fleet_ships, planet_owner, planet_ships
from python.orbit_wars_gym.observation import to_official_observation
from scripts.league_agents import make as make_named

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "artifacts/mapelites/archive.json"

# ---- genome space (strategy knobs of PGSConfig) -------------------------------
SCRIPTS_OPTS = [
    "hold",
    "hold,snipe",
    "hold,capture,reinforce",
    "hold,snipe,capture,reinforce,evac",
]
# name -> (lo, hi) for numeric genes; scripts handled separately
GENES = {
    "wave_min_ships": (0.0, 140.0),
    "wave_start_step": (0, 250),
    "wave_max_delay": (1, 35),
    "prod_weight": (3.0, 40.0),
    "arbiter_margin": (0.0, 80.0),
    "value_horizon": (25, 55),
}
INT_GENES = {"wave_start_step", "wave_max_delay", "value_horizon"}

# ---- behaviour descriptor grid ------------------------------------------------
WAVE_BINS = [0, 15, 30, 50, 80, 120, 1e9]      # mean ships per launch -> 6 cells
CADENCE_BINS = [0, 0.05, 0.10, 0.18, 0.30, 1.01]  # frac of steps that launched -> 5 cells

# ---- evaluation reference set (DIVERSE — not Producer alone). Two opponents
# keep the per-eval cost down for the QD search; finalists get re-checked in the
# league against the full pool. producer = floor/spray; holdwave = wave-discipline.
REFERENCE = ["producer", "pgs_holdwave"]


def _totals(state, np_):
    tot = [0.0] * np_
    for p in state.get("planets", []):
        o = planet_owner(p)
        if 0 <= o < np_:
            tot[o] += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if 0 <= o < np_:
            tot[o] += fleet_ships(f)
    return tot


def _candidate_act(genome):
    from bots.pgs.planner import PGSConfig, PGSRuntime

    kw = {k: (int(round(v)) if k in INT_GENES else float(v))
          for k, v in genome.items() if k != "scripts"}
    kw["scripts"] = genome["scripts"]
    return PGSRuntime(PGSConfig(**kw)).act


def _play_vs(genome, opp_name, seeds, steps, cand_seat):
    """One batched run: candidate at cand_seat, opponent at the other seat.
    Returns (margins list per seed, launch_stats dict for the candidate)."""
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=2, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=True))
    backend.reset(int(seeds[0]))
    states = backend.states()
    cand = [_candidate_act(genome) for _ in range(n)]
    opp = [make_named(opp_name) for _ in range(n)]
    died = [[None, None] for _ in range(n)]
    launches, ships_sent, active_steps = 0, 0.0, 0  # candidate behaviour accumulators
    for t in range(steps):
        rows = []
        for i in range(n):
            for seat in range(2):
                if died[i][seat] is not None:
                    continue
                agent = cand[i] if seat == cand_seat else opp[i]
                obs = to_official_observation(states[i], seat)
                try:
                    moves = agent(obs) or []
                except Exception:
                    moves = []
                if seat == cand_seat:
                    launched = [m for m in moves if len(m) >= 3 and float(m[2]) >= 1.0]
                    if launched:
                        active_steps += 1
                        launches += len(launched)
                        ships_sent += sum(float(m[2]) for m in launched)
                for m in moves:
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
        for i in range(n):
            tot = _totals(states[i], 2)
            for seat in range(2):
                if died[i][seat] is None and tot[seat] <= 0.0:
                    died[i][seat] = t + 1
    margins = []
    for i in range(n):
        tot = _totals(states[i], 2)
        c, o = tot[cand_seat], tot[1 - cand_seat]
        margins.append(1.0 if c > o else (-1.0 if o > c else 0.0))
    stats = {"launches": launches, "ships_sent": ships_sent,
             "active_steps": active_steps, "steps": steps * n}
    return margins, stats


def evaluate(genome, seeds, steps=500):
    """Margin vs the reference set (both seats) + measured behaviour descriptors."""
    all_margins, prod_margins = [], []
    L, S, A, T = 0, 0.0, 0, 0
    for opp in REFERENCE:
        for cand_seat in (0, 1):
            m, st = _play_vs(genome, opp, seeds, steps, cand_seat)
            all_margins += m
            if opp == "producer":
                prod_margins += m
            L += st["launches"]
            S += st["ships_sent"]
            A += st["active_steps"]
            T += st["steps"]
    wave_size = (S / L) if L else 0.0          # mean ships per launch
    cadence = (A / T) if T else 0.0            # frac of (env-step) with a launch
    fitness = sum(all_margins) / len(all_margins)
    beats_prod = (sum(prod_margins) / len(prod_margins)) > 0 if prod_margins else False
    return {"fitness": fitness, "wave_size": wave_size, "cadence": cadence,
            "beats_producer": beats_prod, "n_games": len(all_margins)}


def cell_of(desc):
    w = max(b for b in range(len(WAVE_BINS) - 1) if WAVE_BINS[b] <= desc["wave_size"])
    c = max(b for b in range(len(CADENCE_BINS) - 1) if CADENCE_BINS[b] <= desc["cadence"])
    return f"w{w}_c{c}"


def random_genome(rng):
    g = {k: (rng.randint(int(lo), int(hi)) if k in INT_GENES else rng.uniform(lo, hi))
         for k, (lo, hi) in GENES.items()}
    g["scripts"] = rng.choice(SCRIPTS_OPTS)
    return g


def mutate(genome, rng):
    g = dict(genome)
    for k, (lo, hi) in GENES.items():
        if rng.random() < 0.5:
            span = (hi - lo) * 0.25
            v = g[k] + rng.gauss(0, span)
            v = min(max(v, lo), hi)
            g[k] = int(round(v)) if k in INT_GENES else v
    if rng.random() < 0.25:
        g["scripts"] = rng.choice(SCRIPTS_OPTS)
    return g


# seed genomes = the configs we already know (so the archive starts grounded)
SEED_GENOMES = [
    {"scripts": "hold", "wave_min_ships": 0.0, "wave_start_step": 50, "wave_max_delay": 8,
     "prod_weight": 15.0, "arbiter_margin": 25.0, "value_horizon": 40},  # pgs_hold
    {"scripts": "hold", "wave_min_ships": 60.0, "wave_start_step": 150, "wave_max_delay": 8,
     "prod_weight": 15.0, "arbiter_margin": 25.0, "value_horizon": 40},  # pgs_holdwave
    {"scripts": "hold", "wave_min_ships": 60.0, "wave_start_step": 100, "wave_max_delay": 8,
     "prod_weight": 15.0, "arbiter_margin": 25.0, "value_horizon": 40},  # pgs_wave_s100
    {"scripts": "hold", "wave_min_ships": 100.0, "wave_start_step": 50, "wave_max_delay": 25,
     "prod_weight": 15.0, "arbiter_margin": 25.0, "value_horizon": 40},  # pgs_bigwave
]


def load_archive():
    if ARCHIVE.exists():
        return json.loads(ARCHIVE.read_text())
    return {}


def save_archive(arch):
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(json.dumps(arch, indent=1))


def place(arch, genome, desc):
    key = cell_of(desc)
    cur = arch.get(key)
    if cur is None or desc["fitness"] > cur["fitness"]:
        arch[key] = {"genome": genome, "fitness": desc["fitness"],
                     "wave_size": round(desc["wave_size"], 2), "cadence": round(desc["cadence"], 3),
                     "beats_producer": desc["beats_producer"], "n_games": desc["n_games"]}
        return True
    return False


def eval_one():
    """Subprocess worker: evaluate one genome (JSON) and write the descriptor."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-one", action="store_true")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(1)
    g = json.loads(args.genome)
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    desc = evaluate(g, seeds, args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"genome": g, "desc": desc}))


def _spawn(genome, out, seeds, seed_base, steps):
    import subprocess
    import sys
    cmd = [sys.executable, "scripts/mapelites_pgs.py", "--eval-one",
           "--genome", json.dumps(genome), "--out", str(out),
           "--seeds", str(seeds), "--seed-base", str(seed_base), "--steps", str(steps)]
    return subprocess.Popen(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": "."},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=2, help="seeds per opponent per seat")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.rng)
    arch = load_archive()
    sb = args.seed_base
    tmp = ROOT / "artifacts/mapelites/_w"
    tmp.mkdir(parents=True, exist_ok=True)

    # work queue: seed genomes first (fresh archive), then mutations of elites
    pending = list(SEED_GENOMES) if not arch else []
    done = 0
    running = {}  # proc -> (out_path, genome)
    t_start = time.perf_counter()

    def next_genome():
        if pending:
            return pending.pop(0)
        if arch:
            return mutate(rng.choice(list(arch.values()))["genome"], rng)
        return random_genome(rng)

    while done < args.evals:
        while len(running) < args.workers and done + len(running) < args.evals:
            g = next_genome()
            out = tmp / f"w{sb}.json"
            proc = _spawn(g, out, args.seeds, sb, args.steps)
            sb += args.seeds
            running[proc] = (out, g)
        # poll
        finished = [p for p in running if p.poll() is not None]
        if not finished:
            time.sleep(1.0)
            continue
        for p in finished:
            out, g = running.pop(p)
            done += 1
            try:
                res = json.loads(out.read_text())
                desc = res["desc"]
                g = res["genome"]
            except Exception as e:
                print(f"[{done:3d}] worker failed ({e})", flush=True)
                continue
            new = place(arch, g, desc)
            save_archive(arch)
            beat = sum(1 for v in arch.values() if v["beats_producer"])
            print(f"[{done:3d}] cell={cell_of(desc):7s} fit={desc['fitness']:+.2f} "
                  f"wave={desc['wave_size']:6.1f} cad={desc['cadence']:.3f} "
                  f"beatP={desc['beats_producer']!s:5s} {'NEW' if new else '   '} | "
                  f"cells={len(arch)} beatsP={beat} ({time.perf_counter()-t_start:.0f}s)", flush=True)

    print("\n=== ARCHIVE ===")
    for k in sorted(arch):
        v = arch[k]
        print(f"{k:8s} fit={v['fitness']:+.2f} wave={v['wave_size']:6.1f} cad={v['cadence']:.3f} "
              f"beatP={v['beats_producer']!s:5s} scripts={v['genome']['scripts']}")
    beat = sum(1 for v in arch.values() if v["beats_producer"])
    print(f"\ncells filled: {len(arch)} | cells beating Producer: {beat}")


if __name__ == "__main__":
    import sys
    if "--eval-one" in sys.argv:
        eval_one()
    else:
        main()
