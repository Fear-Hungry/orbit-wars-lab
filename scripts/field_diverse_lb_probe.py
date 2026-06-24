"""Does a DIVERSE weak->strong opponent pool order the hold-family by LB?

The strong-trio probe (wave_family_field_probe.py) gave Spearman -0.40: vs the 3
STRONGEST externals, allscripts (worst LB) won MOST and holdwave (best LB) was mid.
Plausible flaw: a FIXED STRONG trio is a narrow slice. allscripts may beat strong
opponents yet collapse vs weak ones (overextends/dies); holdwave is robust across
the WHOLE field distribution, which is what the real LB rewards. So a field-
representative gate must span weak->strong opponents and several lineups.

This probe runs the hold-family in 4p, seat-rotated, across DIVERSE lineups that
span the strength range, then correlates aggregate win-rate with real LB.

  pgs_holdwave (s150) 1228.8 | pgs_wave_s100 (s100) 1146.1
  pgs_hold (no wave)  1057.6 | pgs_allscripts (floor) 1021.5

If aggregate win-rate orders them by LB (Spearman >= 0.8) the diverse pool is a
field-predictive selector for the wave lever; else no local eval orders the top.

Run:
    SEP_SEEDS=10 SEP_JOBS=7 PYTHONPATH=. .venv/bin/python scripts/field_diverse_lb_probe.py
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
from scripts.wave_family_field_probe import _spearman  # noqa: E402

CAND_LB = {
    "pgs_holdwave": 1228.8,
    "pgs_wave_s100": 1146.1,
    "pgs_hold": 1057.6,
    "pgs_allscripts": 1021.5,
}
# CANDS env overrides the candidate set (e.g. screen wave variants by survival,
# the +0.80-LB-correlated metric). Variants without an LB anchor are ranked but
# excluded from the Spearman-vs-LB (which needs >= 3 anchored points).
_cands_env = os.environ.get("CANDS", "").strip()
CANDIDATES = ([c.strip() for c in _cands_env.split(",") if c.strip()]
              if _cands_env else list(CAND_LB))
# Diverse judge lineups spanning weak -> strong (each is the 3 opponents in a 4p game).
LINEUPS = {
    "strong": ["ext_lb1224", "ext_lb1110", "ext_lb1100"],
    "mixed":  ["ext_rulebase_ml", "ext_hellburner", "ext_lb1100"],
    "weakmix": ["producer", "greedy", "rush"],
}
STEPS = 500
SEEDS = int(os.environ.get("SEP_SEEDS", "10"))   # seeds per (candidate, lineup)
SEED_BASE = 110_000
CHUNK = int(os.environ.get("SEP_CHUNK", "2"))
JOBS = int(os.environ.get("SEP_JOBS", "7"))

OUT = ROOT / "artifacts" / "top5_proxy" / "field_diverse_lb_probe"
OUT.mkdir(parents=True, exist_ok=True)


def _jobs():
    jobs = []
    for ci, cand in enumerate(CANDIDATES):
        for li, (lname, trio) in enumerate(LINEUPS.items()):
            base = SEED_BASE + ci * 10_000 + li * 1000
            for off in range(0, SEEDS, CHUNK):
                n = min(CHUNK, SEEDS - off)
                jobs.append((cand, lname, trio, n, base + off))
    return jobs


def _run(job):
    cand, lname, trio, n, base = job
    names = [cand, *trio]
    label = f"{cand}__{lname}__b{base}"
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
    return cand, lname, label, p.returncode, p.stderr[-1500:], str(out)


def main() -> int:
    jobs = _jobs()
    print(f"[field-diverse] {len(jobs)} chunks / {JOBS} workers; "
          f"cands={CANDIDATES} lineups={list(LINEUPS)} seeds={SEEDS} steps={STEPS}",
          flush=True)
    files: dict[str, list[str]] = {c: [] for c in CANDIDATES}
    per_lineup: dict[tuple, list[str]] = {}
    bad = []
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(_run, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            cand, lname, label, rc, stderr, out = fut.result()
            done += 1
            crash = "[crash]" in stderr
            print(f"  [{done}/{len(jobs)}] {label} rc={rc}" + ("  CRASH" if crash else ""), flush=True)
            if rc != 0 or crash:
                bad.append((label, rc, (stderr.strip().splitlines()[-1:] or ["?"])[0]))
            files[cand].append(out)
            per_lineup.setdefault((cand, lname), []).append(out)

    def score_files(flist, cand):
        games = []
        for f in flist:
            payload = json.loads(Path(f).read_text())
            for g in payload["games"]:
                g["mode"] = "4p"
            games.extend(payload["games"])
        return _score_games(games, cand), len(games)

    rows = []
    for cand in CANDIDATES:
        sc, ng = score_files(files[cand], cand)
        # per-lineup breakdown
        lw = {}
        for lname in LINEUPS:
            s2, n2 = score_files(per_lineup.get((cand, lname), []), cand)
            lw[lname] = (round(s2["win_rate"], 3), n2)
        rows.append((cand, sc, ng, lw))

    anchored = [(c, sc) for c, sc, _, _ in rows if c in CAND_LB]
    if len(anchored) >= 3:
        lb = [CAND_LB[c] for c, _ in anchored]
        rho_win = _spearman([sc["win_rate"] for _, sc in anchored], lb)
        rho_surv = _spearman([-sc["annihilation_rate"] for _, sc in anchored], lb)
        rho_best = max(rho_win, rho_surv)
    else:
        rho_win = rho_surv = rho_best = None
    selector = rho_best is not None and rho_best >= 0.8

    report = {
        "candidates": CANDIDATES, "lineups": LINEUPS, "steps": STEPS, "seeds": SEEDS,
        "lb": CAND_LB,
        "rows": [{"candidate": c, "lb": CAND_LB.get(c), "win_rate": sc["win_rate"],
                  "annihilation_rate": sc["annihilation_rate"],
                  "decisive_win_rate": sc["decisive_win_rate"],
                  "per_lineup_winrate": lw, "games": ng, "faults": sc["faults"]}
                 for c, sc, ng, lw in rows],
        "rho_win_vs_lb": rho_win, "rho_survival_vs_lb": rho_surv, "rho_best": rho_best,
        "is_field_predictive_selector": bool(selector), "bad_runs": bad,
    }
    rp = OUT / "field_diverse_lb_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== FIELD-DIVERSE LB PROBE (4p seat-rotated, weak->strong lineups) ===")
    print("  (death-rate is the +0.80-LB-correlated metric; lower = better)")
    print(f"  {'candidate':18s} {'LB':>8s} {'win':>6s} {'death':>6s}  per-lineup win (strong/mixed/weakmix)")
    for c, sc, ng, lw in sorted(rows, key=lambda r: r[1]["annihilation_rate"]):
        s = lw.get("strong", ('?',0))[0]; m = lw.get("mixed", ('?',0))[0]; w = lw.get("weakmix", ('?',0))[0]
        lbtxt = f"{CAND_LB[c]:8.1f}" if c in CAND_LB else "     n/a"
        print(f"  {c:18s} {lbtxt} {sc['win_rate']:6.3f} {sc['annihilation_rate']:6.3f}  "
              f"{s}/{m}/{w}  (n={ng})")
    if rho_best is not None:
        print(f"\n  Spearman vs LB (anchored only): win={rho_win:+.3f}  survival={rho_surv:+.3f}  best={rho_best:+.3f}")
        print(f"  -> " + ("FIELD-PREDICTIVE SELECTOR" if selector else "not a strict selector (rho<0.8)"))
    else:
        print("\n  (variant screen: no Spearman — ranking by death-rate vs holdwave reference)")
    if bad:
        print(f"  WARNING {len(bad)} bad runs: {bad[:3]}")
    print(f"  wrote {rp}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
