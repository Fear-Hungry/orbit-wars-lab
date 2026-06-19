"""Does the EXTERNAL field-proxy pool order the FINE hold-family by real LB?

The internal LB-anchored pool (producer/oep/pgs-family) demonstrably CANNOT order
the top (the league + nash calibrations show competitive_tied / inversions: the
LB champion pgs_holdwave gets ranked BELOW weaker hold variants). Root-cause
hypothesis: the discriminative signal lives in the DIVERSE PRIVATE FIELD, not the
redundant internal pool. The strong public LB-derived agents (ext_lb1224/1110/
1100) are the closest local proxy for that field.

This probe asks the decisive question for the only lever with a real LB signal
(wave timing): run the hold-family — IDENTICAL config except wave_start_step —
  pgs_holdwave (start=150, LB 1228.8)
  pgs_wave_s100 (start=100, LB 1146.1)
  pgs_hold      (no wave,   LB 1057.6)
  pgs_allscripts(floor,     LB 1021.5)
in 4p, SEAT-ROTATED, against the strong external trio, and check whether the
pool orders them by their REAL LB (Spearman). If it does (>= 0.8 with enough
seeds), the external pool is a field-predictive SELECTOR for the wave lever and a
config search can find a bot that beats holdwave LOCALLY. If not, no local gate
orders the wave lever and holdwave is the verified ceiling.

Run:
    SEP_SEEDS=24 SEP_JOBS=6 PYTHONPATH=. .venv/bin/python scripts/wave_family_field_probe.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.league_submit_ruler import _score_games  # noqa: E402

# Hold-family LB anchors (mirror scripts/league_agents.py LB_ANCHORS).
CAND_LB = {
    "pgs_holdwave": 1228.8,
    "pgs_wave_s100": 1146.1,
    "pgs_hold": 1057.6,
    "pgs_allscripts": 1021.5,
}
CANDIDATES = list(CAND_LB)
TRIO = ["ext_lb1224", "ext_lb1110", "ext_lb1100"]
STEPS = 500
SEEDS = int(os.environ.get("SEP_SEEDS", "24"))
SEED_BASE = 90_000
CHUNK = int(os.environ.get("SEP_CHUNK", "3"))
JOBS = int(os.environ.get("SEP_JOBS", "6"))

OUT = ROOT / "artifacts" / "top5_proxy" / "wave_family_field_probe"
OUT.mkdir(parents=True, exist_ok=True)


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy). Ties -> average rank."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


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
    print(f"[wave-field] {len(jobs)} chunks across {JOBS} workers; "
          f"candidates={CANDIDATES} trio={TRIO} seeds={SEEDS} steps={STEPS}",
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
                bad.append((label, rc, (stderr.strip().splitlines()[-1:] or ["?"])[0]))
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

    lb = [CAND_LB[c] for c, *_ in rows]
    win = [sc["win_rate"] for _, sc, _, _ in rows]
    dec = [sc["decisive_win_rate"] if sc["decisive_win_rate"] is not None else 0.0
           for _, sc, _, _ in rows]
    surv = [-sc["annihilation_rate"] for _, sc, _, _ in rows]  # higher survival ~ higher LB
    rho_win = _spearman(win, lb)
    rho_dec = _spearman(dec, lb)
    rho_surv = _spearman(surv, lb)
    rho_best = max(rho_win, rho_dec, rho_surv)

    # Verdict: a field-predictive SELECTOR for the wave lever needs to order the
    # fine hold-family by LB. >= 0.8 over 4 points = at most one adjacent swap.
    selector = rho_best >= 0.8

    report = {
        "candidates": CANDIDATES, "trio": TRIO, "steps": STEPS, "seeds": SEEDS,
        "lb": {c: CAND_LB[c] for c in CANDIDATES},
        "rows": [{"candidate": c, "lb": CAND_LB[c], "win_rate": sc["win_rate"],
                  "decisive_win_rate": sc["decisive_win_rate"],
                  "annihilation_rate": sc["annihilation_rate"],
                  "faults": sc["faults"], "seat_counts": seats, "games": g}
                 for c, sc, seats, g in rows],
        "rho_win_vs_lb": rho_win, "rho_decisive_vs_lb": rho_dec,
        "rho_survival_vs_lb": rho_surv, "rho_best": rho_best,
        "is_field_predictive_selector": bool(selector), "bad_runs": bad,
    }
    rp = OUT / "wave_family_field_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== WAVE-FAMILY FIELD PROBE (4p seat-rotated vs strong external trio) ===")
    print(f"  trio = {TRIO}")
    print(f"  {'candidate':16s} {'LB':>8s} {'win':>7s} {'decW':>7s} {'death':>7s} {'games':>6s} faults")
    for c, sc, seats, g in sorted(rows, key=lambda r: -CAND_LB[r[0]]):
        nf = sum(sc["faults"].values())
        dw = sc["decisive_win_rate"]
        print(f"  {c:16s} {CAND_LB[c]:8.1f} {sc['win_rate']:7.3f} "
              f"{(dw if dw is not None else 0.0):7.3f} {sc['annihilation_rate']:7.3f} {g:6d} {nf}")
    print(f"\n  Spearman vs LB:  win={rho_win:+.3f}  decisive={rho_dec:+.3f}  survival={rho_surv:+.3f}")
    print(f"  rho_best = {rho_best:+.3f}  ->  "
          + ("FIELD-PREDICTIVE SELECTOR (can rank wave lever; search is trustable)"
             if selector else
             "NOT A SELECTOR (external pool cannot order the wave family by LB either)"))
    if bad:
        print(f"  WARNING: {len(bad)} bad/crashed runs: {bad[:3]}")
    print(f"  wrote {rp}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
