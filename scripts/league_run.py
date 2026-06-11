"""Continuous local league: keep scheduling games, accumulate rating forever.

Each round schedules K matchups (2p/4p ~50/50 like the LB), every matchup on
FRESH seeds (state.json tracks the next seed — no seed is ever reused), runs
them in parallel, then refreshes the standings (league_report) into
artifacts/league/v1/standings.log.

Matchmaking: 70% uniform random (coverage for BT), 30% among the current top-5
(precision at the top — the league is a VETO instrument, it does not promote).
4p tables force style mix: at most 3 producer-lineage bots per table when
possible.

Stop/resume freely: every result is an append-only JSON; the report aggregates
whatever exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# allow running as a standalone script (`python scripts/league_run.py`) — put the
# repo root on sys.path so the sibling-module import below resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.league_report import DEFAULT_GLOBS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "artifacts/league/v1"
STATE = DIR / "state.json"
POOL = ["producer", "oep", "brep", "pgs_hold", "pgs_holdwave", "pgs_wave_s100", "brep_league3",
        "pgs_allscripts", "ext_lb1050", "ext_hellburner",
        # style exploiter (2026-06-10 falsification: pool needs the field's loss axes).
        # NOTE: "rusher" (v1 spread/s50 AND v2 focus/f120) went 0-win vs the whole pool
        # — a free win adds population bias (Balduzzi 2018), so it stays OUT until a
        # variant actually wins games.
        "pgs_bigwave"]
PRODUCER_LINEAGE = {"producer", "oep", "brep", "pgs_hold", "pgs_holdwave", "pgs_wave_s100",
                    "brep_league3", "pgs_allscripts", "pgs_bigwave"}


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"next_seed": 1016, "round": 0}


def top5():
    rep = DIR / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        return [n for n in r["ranking"] if n in POOL][:5]
    return POOL[:5]


def pick_matchups(rng, k):
    tops = top5()
    picks = []
    for j in range(k):
        if j % 2 == 0:  # 2p
            pool = tops if rng.random() < 0.3 else POOL
            picks.append(rng.sample(pool, 2))
        else:  # 4p with style mix: try to cap producer-lineage at 3
            for _ in range(10):
                c = rng.sample(POOL, 4)
                if sum(1 for x in c if x in PRODUCER_LINEAGE) <= 3:
                    break
            picks.append(c)
    return picks


def match_filename(rnd, names):
    """Deterministic, collision-free per (round, names). Truncated names alone
    collide (pgs_hold/pgs_holdwave -> "pgs_ho": round 92 lost a match to a silent
    overwrite), so append a short hash of the FULL names tuple."""
    tag = hashlib.sha1(",".join(names).encode()).hexdigest()[:8]
    return f"r{rnd:04d}_{len(names)}p_{'_'.join(n[:6] for n in names)}_{tag}.json"


def run_match(names, seed_base, seeds, rnd):
    out = DIR / "cont" / match_filename(rnd, names)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/league_match.py", "--agents", ",".join(names),
           "--seeds", str(seeds), "--seed-base", str(seed_base), "--steps", "500",
           "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."})
    return out.name, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--matchups-per-round", type=int, default=4)
    ap.add_argument("--seeds-per-matchup", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random()
    state = load_state()
    log = open(DIR / "standings.log", "a")
    for _ in range(args.rounds):
        state["round"] += 1
        picks = pick_matchups(rng, args.matchups_per_round)
        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for names in picks:
                base = state["next_seed"]
                state["next_seed"] += args.seeds_per_matchup
                jobs.append(ex.submit(run_match, names, base, args.seeds_per_matchup, state["round"]))
            results = [j.result() for j in jobs]
        STATE.write_text(json.dumps(state))
        fails = [n for n, rc in results if rc != 0]
        rep = subprocess.run(
            [sys.executable, "scripts/league_report.py", DEFAULT_GLOBS, "50"],
            cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
        standings = next((ln for ln in rep.stdout.splitlines() if ln.startswith("calibration")), "")
        top_lines = "\n".join(rep.stdout.splitlines()[2:11])
        line = (f"=== round {state['round']:4d} | new={len(picks)} fails={len(fails)} "
                f"| next_seed={state['next_seed']} | {standings}\n{top_lines}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()


if __name__ == "__main__":
    main()
