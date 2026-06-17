"""Does the top5_proxy ruler SEPARATE known-relative-strength styles, or is it
only a robustness VETO?

The goal (2026-06-16) requires: "Se a régua não separa producer, pgs_holdwave,
pgs_bigwave e floor de forma coerente, ela continua sendo veto, não seleção."

This probe runs each REFERENCE candidate (whose relative strength we already
know) in 4p AGAINST the strong public proxy trio (ext_lb1224 / ext_lb1110 /
ext_lb1100), SEAT-ROTATED via scripts/league_match.py (all four seat rotations
per seed — NOT the seat-0-pinned benchmark_submission 4p path). We then read the
candidate's win_rate / annihilation_rate off the ruler's own scoring and check
whether the pool orders them COHERENTLY:

  producer  ~  pgs_holdwave  >=  pgs_bigwave   >   pgs_allscripts (rejected floor)

If the spread between the competitive bots and the rejected floor is within the
sampling band, the proxy pool can rank nothing here -> it is a VETO, not a
selector, and PPO/PBT fitness built on it would chase noise (goal item 6).

Run (AFTER the drain gate frees cores; thread-pinned so jobs==cores):
    PYTHONPATH=. .venv/bin/python scripts/top5_proxy_separation_probe.py
    SEP_SEEDS=16 SEP_JOBS=5 PYTHONPATH=. .venv/bin/python scripts/top5_proxy_separation_probe.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.league_submit_ruler import _score_games  # noqa: E402

# Reference candidates: known RELATIVE strength (producer/holdwave competitive;
# allscripts is the REJECTED floor). The probe asks whether the proxy pool
# recovers this ordering.
CANDIDATES = ["producer", "pgs_holdwave", "pgs_bigwave", "pgs_allscripts"]
# Strong public LB proxies = the top range of the ruler pool.
TRIO = ["ext_lb1224", "ext_lb1110", "ext_lb1100"]
STEPS = 500
SEEDS = int(os.environ.get("SEP_SEEDS", "12"))   # seeds per candidate (x4 rotations each)
SEED_BASE = 80_000
CHUNK = int(os.environ.get("SEP_CHUNK", "3"))     # seeds per parallel league_match call
JOBS = int(os.environ.get("SEP_JOBS", "5"))

OUT = ROOT / "artifacts" / "top5_proxy" / "separation_probe"
OUT.mkdir(parents=True, exist_ok=True)


def _jobs():
    jobs = []
    for ci, cand in enumerate(CANDIDATES):
        base = SEED_BASE + ci * 1000
        for off in range(0, SEEDS, CHUNK):
            n = min(CHUNK, SEEDS - off)
            jobs.append((cand, n, base + off))
    return jobs


def _run(job):
    cand, n, base = job
    names = [cand, *TRIO]
    label = f"{cand}__b{base}"
    out = OUT / f"{label}.json"
    cmd = [
        sys.executable, "scripts/league_match.py",
        "--agents", ",".join(names),
        "--seeds", str(n), "--seed-base", str(base),
        "--steps", str(STEPS), "--out", str(out),
    ]
    env = {**os.environ, "PYTHONPATH": ".",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    return cand, label, p.returncode, p.stderr[-1500:], str(out)


def main() -> int:
    jobs = _jobs()
    print(f"[sep] {len(jobs)} chunks across {JOBS} workers; "
          f"candidates={len(CANDIDATES)} trio={TRIO} seeds={SEEDS} steps={STEPS}",
          flush=True)
    per_cand_files: dict[str, list[str]] = {c: [] for c in CANDIDATES}
    bad = []
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(_run, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            cand, label, rc, stderr, out = fut.result()
            done += 1
            crash = "[crash]" in stderr
            print(f"  [{done}/{len(jobs)}] {label} rc={rc}"
                  + ("  CRASH" if crash else ""), flush=True)
            if rc != 0 or crash:
                bad.append((label, rc, stderr.strip().splitlines()[-1:] or ["?"]))
            per_cand_files[cand].append(out)

    rows = []
    for cand in CANDIDATES:
        games = []
        for f in per_cand_files[cand]:
            payload = json.loads(Path(f).read_text())
            for g in payload["games"]:
                g["mode"] = "4p"
            games.extend(payload["games"])
        sc = _score_games(games, cand)
        seat_counter = Counter(g["seats"].index(cand) for g in games if cand in g["seats"])
        rows.append((cand, sc, dict(seat_counter), len(games)))

    # Coherence: competitive bots must clear the rejected floor by > noise.
    wr = {c: sc["win_rate"] for c, sc, _, _ in rows}
    floor_wr = wr.get("pgs_allscripts", 0.0)
    comp = max(wr.get("producer", 0.0), wr.get("pgs_holdwave", 0.0))
    spread = comp - floor_wr
    # ~1/sqrt(games) band on a win-rate; require the spread to clear it clearly.
    n_floor = next((g for c, _, _, g in rows if c == "pgs_allscripts"), 0) or 1
    band = 1.0 / (n_floor ** 0.5)
    separates = spread > 2.0 * band

    report = {
        "candidates": CANDIDATES, "trio": TRIO, "steps": STEPS, "seeds": SEEDS,
        "rows": [{"candidate": c, "win_rate": sc["win_rate"],
                  "annihilation_rate": sc["annihilation_rate"],
                  "decisive_win_rate": sc["decisive_win_rate"],
                  "faults": sc["faults"], "seat_counts": seats, "games": g}
                 for c, sc, seats, g in rows],
        "spread_competitive_minus_floor": spread, "noise_band": band,
        "separates": bool(separates), "bad_runs": bad,
    }
    rp = OUT / "separation_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== TOP5_PROXY SEPARATION PROBE (4p, seat-rotated vs strong trio) ===")
    print(f"  trio = {TRIO}")
    for c, sc, seats, g in sorted(rows, key=lambda r: -r[1]["win_rate"]):
        faults = sc["faults"]
        nf = sum(faults.values())
        print(f"  {c:18s} win_rate={sc['win_rate']:.3f} "
              f"death={sc['annihilation_rate']:.3f} games={g:3d} "
              f"seats={seats} faults={nf}")
    print(f"\n  spread(competitive - floor) = {spread:+.3f}  (2x noise band = {2*band:.3f})")
    print(f"  SEPARATES = {separates}  -> "
          + ("usable as a 4p SELECTOR" if separates
             else "VETO ONLY (cannot rank these styles; do not gate PPO/PBT on it)"))
    if bad:
        print(f"  WARNING: {len(bad)} bad/crashed runs: {bad[:3]}")
    print(f"\n  wrote {rp}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
