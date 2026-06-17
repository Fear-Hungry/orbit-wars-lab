"""Seat-rotated H2H promotion gate for the G3.2 decisive-wave (even_attrition_2p).

Candidate = pgs_decisive2p (pgs_holdwave + decisive_wave_2p concentration),
incumbent = pgs_holdwave (the live LB submission). The decisive wave only
activates in 2p (player_count==2 guard), so this gate WEIGHTS 2p heavily (the
loss class it targets) and keeps a 4p veto purely to prove no 4p regression
(the 4p path is byte-identical to pgs_holdwave by construction).

Same engine, scoring, seat rotation and zero-fault discipline as
scripts/gate_drain_h2h.py (it reuses the ruler's summarize_candidate). Threads
are pinned so CPU contention can't manufacture false per-move timeouts. Run AFTER
the drain gate frees cores:

    PYTHONPATH=. GATE_JOBS=5 .venv/bin/python scripts/gate_decisive2p_h2h.py
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

from scripts.league_submit_ruler import _score_games, summarize_candidate  # noqa: E402

CAND = "pgs_decisive2p"
INC = "pgs_holdwave"
STEPS = 500
CHUNK = 8
JOBS = int(os.environ.get("GATE_JOBS", "5"))

OUT = ROOT / "artifacts" / "league" / "submit_ruler" / "decisive2p_gate"
OUT.mkdir(parents=True, exist_ok=True)

# matchup: (mode, names, total_seeds, frozen_seed_base). 2p-heavy (target class).
MATCHUPS = [
    ("2p", (CAND, INC), 96, 60_000),               # THE H2H vs incumbent (live submission)
    ("2p", (CAND, "producer"), 96, 61_000),        # producer floor (ruler hard check)
    ("2p", (CAND, "pgs_allscripts"), 48, 62_000),  # rejected-floor (ruler hard check)
    ("2p", (CAND, "oep"), 48, 63_000),             # field style
    ("2p", (CAND, "ext_lb1110"), 48, 64_000),      # strong public proxy (hold the field)
    ("4p", (CAND, "producer", "oep", "pgs_bigwave"), 48, 65_000),  # no-4p-regression veto
]


def _expected_games(mode: str, n: int) -> int:
    return n * (2 if mode == "2p" else 4)


def _chunk_jobs():
    jobs = []
    for mode, names, seeds, base in MATCHUPS:
        for off in range(0, seeds, CHUNK):
            n = min(CHUNK, seeds - off)
            label = f"{mode}__{'_'.join(names)}__b{base + off}"
            jobs.append((mode, names, n, base + off, label))
    return jobs


def _run_chunk(job):
    mode, names, n, base, label = job
    out = OUT / f"{label}.json"
    if out.exists():
        try:
            payload = json.loads(out.read_text())
            if len(payload.get("games", [])) == _expected_games(mode, n):
                return label, 0, 0.0, "", str(out)
        except (json.JSONDecodeError, OSError):
            pass
    cmd = [
        sys.executable, "scripts/league_match.py",
        "--agents", ",".join(names),
        "--seeds", str(n), "--seed-base", str(base),
        "--steps", str(STEPS), "--out", str(out),
    ]
    env = {**os.environ, "PYTHONPATH": ".",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    return label, p.returncode, time.perf_counter() - t, p.stderr[-2000:], str(out)


def main() -> int:
    jobs = _chunk_jobs()
    print(f"[gate] decisive2p: {len(jobs)} chunks across {JOBS} workers; "
          f"matchups={len(MATCHUPS)} steps={STEPS}", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(_run_chunk, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            label, rc, secs, stderr, out = fut.result()
            done += 1
            crash = "[crash]" in stderr
            print(f"  [{done}/{len(jobs)}] {label} rc={rc} {secs:.0f}s"
                  + ("  CRASH-STDERR" if crash else ""), flush=True)
            if crash:
                print("      " + stderr.strip().splitlines()[-1], flush=True)
            results[label] = (rc, out)

    task_results = []
    per_matchup = []
    for mode, names, seeds, base in MATCHUPS:
        merged_games = []
        crashes = Counter()
        bad_rc = False
        for off in range(0, seeds, CHUNK):
            label = f"{mode}__{'_'.join(names)}__b{base + off}"
            rc, out = results[label]
            if rc != 0:
                bad_rc = True
            payload = json.loads(Path(out).read_text())
            for g in payload["games"]:
                g["mode"] = mode
            merged_games.extend(payload["games"])
            for k, v in (payload.get("crashes") or {}).items():
                crashes[k] += v
        merged_path = OUT / f"MERGED__{mode}__{'_'.join(names)}.json"
        merged_path.write_text(json.dumps(
            {"agents": list(names), "mode": mode, "games": merged_games,
             "crashes": dict(crashes)}, indent=1))
        task_results.append({"candidate": CAND, "mode": mode, "names": list(names),
                             "out": str(merged_path), "returncode": 0 if not bad_rc else 1})
        seat_counter = Counter(" | ".join(g["seats"]) for g in merged_games)
        sc = _score_games(merged_games, CAND)
        per_matchup.append((mode, names, seeds, sc, dict(seat_counter), dict(crashes), bad_rc))

    # 2p-heavy weight (the change is 2p-only; 4p is a no-regression veto).
    # decisive_wave_2p is INERT in 4p (player_count==2 guard), so pgs_decisive2p's
    # 4p death must equal holdwave's ~0.625 inherent rate (drain gate baseline). An
    # absolute 0.35 cap would reject the incumbent too; 0.70 only catches a REAL 4p
    # regression (i.e. the 2p-guard leaking into 4p). The 2p H2H vs holdwave is what
    # actually gates this change.
    verdict = summarize_candidate(
        CAND, task_results, incumbent=INC,
        min_decisive_2p=40, min_producer_winrate=0.50,
        min_incumbent_winrate=0.50, min_floor_winrate=0.60,
        max_annihilation_rate_4p=0.70, weight_2p=0.80,
    )

    report = {"candidate": CAND, "incumbent": INC, "steps": STEPS,
              "matchups": [
                  {"mode": m, "names": list(nm), "seeds": s,
                   "score": sc, "seat_counts": seats, "crashes": cr, "bad_rc": brc}
                  for (m, nm, s, sc, seats, cr, brc) in per_matchup],
              "verdict": verdict}
    rp = OUT / "gate_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== DECISIVE-WAVE (even_attrition_2p) H2H GATE (seat-rotated, thread-pinned) ===")
    for m, nm, s, sc, seats, cr, brc in per_matchup:
        opp = "+".join(x for x in nm if x != CAND)
        dec = sc["decisive_win_rate"]
        print(f"  {m} vs {opp:24s} seeds={s:3d}  "
              f"win_rate={sc['win_rate']:.3f} dec_wr={dec if dec is not None else float('nan'):.3f} "
              f"annih={sc['annihilation_rate']:.3f} faults={sc['faults']} "
              + ("  BAD_RC" if brc else ""))
    print(f"\n  VERDICT = {verdict['verdict']}  "
          f"(score_2p={verdict['score_2p']:.3f} score_4p={verdict['score_4p']:.3f})")
    for c in verdict["checks"]:
        if c["severity"] != "info":
            mark = "PASS" if c["passed"] else ("WARN" if c["severity"] == "inconclusive" else "FAIL")
            print(f"    [{mark}] {c['name']}: {c['details']}")
    print(f"\n  wrote {rp}")
    return 0 if verdict["verdict"] == "PASS_LOCAL" else 1


if __name__ == "__main__":
    sys.exit(main())
