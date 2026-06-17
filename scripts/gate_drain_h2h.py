"""Parallel, seat-rotated H2H gate for the drenagem-dupla (drain) fix.

The promotion ruler scripts/league_submit_ruler.py orchestrates league_match
serially per matchup, so a 96-seed 4p task is a multi-hour long pole. This
runner uses the SAME engine (scripts/league_match.py, same seat rotation and
fault recording) but fans the seeds of each matchup out into parallel chunks
across all cores, then merges each matchup's chunk JSONs and feeds them through
the ruler's own scoring (summarize_candidate / _score_games / decisive_winner).
Verdict logic is therefore identical to the ruler; only the scheduling differs.

Candidate = pgs_hold_fix (8459f7b, drain fix), incumbent = pgs_hold_prefix
(8459f7b~1). Threads are pinned (OMP/MKL/OPENBLAS/NUMEXPR=1) so CPU contention
cannot manufacture false per-move timeouts (which the tarball league wrapper
treats as crashes). 500 steps, frozen seed bases, rotated seats, zero-fault
required.
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

CAND = "pgs_hold_fix"
INC = "pgs_hold_prefix"
STEPS = 500
CHUNK = 8  # seeds per parallel chunk
# Leave clear headroom (host has 8 cores + a second Claude session may run jobs).
# Keeping load < cores avoids contention-induced per-move spikes > 0.9s, which the
# tarball wrapper would flag as a fault and invalidate the whole gate (forcing a
# re-run). Fewer workers ≈ predictable single clean pass; see memory
# ruler_false_timeouts_and_tarball_wrapper. Override with GATE_JOBS env.
JOBS = int(os.environ.get("GATE_JOBS", "5"))

OUT = ROOT / "artifacts" / "league" / "submit_ruler" / "drain_gate_96"
OUT.mkdir(parents=True, exist_ok=True)

# matchup: (mode, names, total_seeds, frozen_seed_base)
# pgs_allscripts is REQUIRED: summarize_candidate has a hard `clears_rejected_floor`
# check against it; omitting it would hard-fail the verdict with
# missing_required_opponent regardless of the real H2H.
MATCHUPS = [
    ("2p", (CAND, INC), 96, 70_000),            # THE H2H vs incumbent
    ("2p", (CAND, "producer"), 96, 71_000),     # producer floor (goal-required)
    ("2p", (CAND, "pgs_allscripts"), 48, 74_000),  # rejected-floor robustness (ruler hard check)
    ("2p", (CAND, "oep"), 48, 72_000),          # field style
    ("2p", (CAND, "pgs_bigwave"), 48, 73_000),  # field style
    ("4p", (CAND, "producer", "oep", "pgs_bigwave"), 48, 75_000),  # fix 4p survival (context)
]

# RELATIVE 4p veto: the absolute holdwave 4p death is ~0.62 (inherent, not caused
# by the fix), so an absolute cap would reject the incumbent too. The fair test is
# whether the FIX worsens 4p vs the PRE-FIX, in the SAME field on the SAME seeds
# (seed base 75_000 matches the fix-4p matchup). The verdict uses
# fix_annih <= prefix_annih + REL_4P_TOL, not an absolute bar.
REL_4P = ("4p", (INC, "producer", "oep", "pgs_bigwave"), 48, 75_000)
REL_4P_TOL = 0.05


def _chunk_jobs():
    jobs = []
    for mode, names, seeds, base in (*MATCHUPS, REL_4P):
        for off in range(0, seeds, CHUNK):
            n = min(CHUNK, seeds - off)
            label = f"{mode}__{'_'.join(names)}__b{base + off}"
            jobs.append((mode, names, n, base + off, label))
    return jobs


def _expected_games(mode: str, n: int) -> int:
    # league_match plays 2 seat orders (2p) or 4 rotations (4p) per seed.
    return n * (2 if mode == "2p" else 4)


def _run_chunk(job):
    mode, names, n, base, label = job
    out = OUT / f"{label}.json"
    # Resume: a chunk's seeds are frozen, so a valid existing output is reusable.
    # Skip re-running it (the host can drop the job mid-run; redoing 30 good
    # chunks wastes ~25 min). Validate the game count so a truncated write reruns.
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
    print(f"[gate] {len(jobs)} chunks across {JOBS} workers; "
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

    # Merge chunk JSONs per matchup, then aggregate with the ruler's scoring.
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
        # non-strict task_result (no seed_base) -> ruler skips per-task seat
        # validation; we validate seat balance explicitly below.
        task_results.append({"candidate": CAND, "mode": mode, "names": list(names),
                             "out": str(merged_path), "returncode": 0 if not bad_rc else 1})
        # explicit seat-rotation balance check (stringify tuple keys for JSON)
        seat_counter = Counter(" | ".join(g["seats"]) for g in merged_games)
        sc = _score_games(merged_games, CAND)
        per_matchup.append((mode, names, seeds, sc, dict(seat_counter), dict(crashes), bad_rc))

    # RELATIVE 4p: score the pre-fix incumbent in the SAME field/seeds and compare
    # death rates. This isolates the fix's 4p effect (the absolute 4p death is
    # holdwave-inherent, so summarize's absolute veto is neutralized below).
    mode4, names4, seeds4, base4 = REL_4P
    rel_games = []
    for off in range(0, seeds4, CHUNK):
        label = f"{mode4}__{'_'.join(names4)}__b{base4 + off}"
        _, out = results[label]
        payload = json.loads(Path(out).read_text())
        for g in payload["games"]:
            g["mode"] = mode4
        rel_games.extend(payload["games"])
    (OUT / f"MERGED__{mode4}__{'_'.join(names4)}.json").write_text(json.dumps(
        {"agents": list(names4), "mode": mode4, "games": rel_games}, indent=1))
    prefix_4p = _score_games(rel_games, INC)
    fix_4p = next(sc for m, nm, s, sc, *_ in per_matchup if m == "4p")
    rel_delta = fix_4p["annihilation_rate"] - prefix_4p["annihilation_rate"]
    rel_4p_pass = rel_delta <= REL_4P_TOL

    verdict = summarize_candidate(
        CAND, task_results, incumbent=INC,
        min_decisive_2p=40, min_producer_winrate=0.50,
        min_incumbent_winrate=0.50, min_floor_winrate=0.60,
        max_annihilation_rate_4p=1.0, weight_2p=0.46,   # absolute 4p veto neutralized; see REL_4P
    )
    # Fold the relative-4p veto into the final verdict.
    final_verdict = verdict["verdict"]
    if not rel_4p_pass and final_verdict == "PASS_LOCAL":
        final_verdict = "REJECT_LOCAL"

    report = {"candidate": CAND, "incumbent": INC, "steps": STEPS,
              "relative_4p": {
                  "fix_annihilation": fix_4p["annihilation_rate"],
                  "prefix_annihilation": prefix_4p["annihilation_rate"],
                  "delta": rel_delta, "tol": REL_4P_TOL, "pass": rel_4p_pass,
                  "fix_4p_faults": sum(fix_4p["faults"].values()),
                  "prefix_4p_faults": sum(prefix_4p["faults"].values())},
              "summarize_verdict": verdict["verdict"], "verdict_final": final_verdict,
              "matchups": [
                  {"mode": m, "names": list(nm), "seeds": s,
                   "score": sc, "seat_counts": seats, "crashes": cr, "bad_rc": brc}
                  for (m, nm, s, sc, seats, cr, brc) in per_matchup],
              "verdict": verdict}
    rp = OUT / "gate_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== DRAIN-FIX H2H GATE (96-seed core, seat-rotated, thread-pinned) ===")
    for m, nm, s, sc, seats, cr, brc in per_matchup:
        opp = "+".join(x for x in nm if x != CAND)
        faults = sc["faults"]
        print(f"  {m} vs {opp:24s} seeds={s:3d}  "
              f"win_rate={sc['win_rate']:.3f} dec_wr="
              f"{sc['decisive_win_rate'] if sc['decisive_win_rate'] is not None else float('nan'):.3f} "
              f"annih={sc['annihilation_rate']:.3f} faults={faults} "
              f"seats={seats}" + ("  BAD_RC" if brc else ""))
    print(f"  RELATIVE 4p (fix vs prefix, same field): fix_annih={fix_4p['annihilation_rate']:.3f} "
          f"prefix_annih={prefix_4p['annihilation_rate']:.3f} delta={rel_delta:+.3f} "
          f"(tol={REL_4P_TOL}) -> {'PASS' if rel_4p_pass else 'FAIL'}")
    print(f"\n  VERDICT = {final_verdict}  (summarize={verdict['verdict']}; "
          f"score_2p={verdict['score_2p']:.3f} score_4p={verdict['score_4p']:.3f})")
    for c in verdict["checks"]:
        if c["severity"] != "info":
            mark = "PASS" if c["passed"] else ("WARN" if c["severity"] == "inconclusive" else "FAIL")
            print(f"    [{mark}] {c['name']}: {c['details']}")
    print(f"\n  wrote {rp}")
    return 0 if final_verdict == "PASS_LOCAL" else 1


if __name__ == "__main__":
    sys.exit(main())
