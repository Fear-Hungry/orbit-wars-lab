"""Seat-rotated 4p-HEAVY gate for the H9 threat-value 4p survival lever (item 5:
kingmaker/overextension — reduce exposure when we become prey).

Candidate = pgs_h9threat (pgs_holdwave + threat_value_4p; auto-adds `reinforce`
in 4p so the forward per-enemy threat value can SELECT survival plans). Incumbent
= pgs_holdwave. The change is 4p-ONLY (2p frozen, scripts="hold"), so the 4p
matchups are the primary axis and the 2p matchups are frozen/floor sanity checks.

Re-validates H9 with the CORRECTED verdict the old scripts/h9_4p_gate.py lacked
(it was death-only and seat-0-pinned, so it green-lit configs that cut death but
BLED MARGIN -> LB ~1048; see memory validation_gates_hardened_2026-06-14 +
h9_threat_value_4p_cracks_floor). Here the 4p verdict is RELATIVE and AGGREGATE:
candidate must cut 4p death (annihilation) vs the SAME field/seeds AND not lose
4p win-rate (margin), with zero faults and seat rotation.

Run clean (no concurrent compute -> contention manufactures false timeouts):
    PYTHONPATH=. GATE_JOBS=5 .venv/bin/python scripts/gate_h9_threat_h2h.py
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

CAND = "pgs_h9threat"
INC = "pgs_holdwave"
FIELD = ("producer", "oep", "pgs_bigwave")
STEPS = 500
CHUNK = 8
JOBS = int(os.environ.get("GATE_JOBS", "5"))

OUT = ROOT / "artifacts" / "league" / "submit_ruler" / "h9_threat_gate"
OUT.mkdir(parents=True, exist_ok=True)

# 2p matchups = FROZEN/floor sanity (threat is 4p-only; 2p must equal holdwave).
MATCHUPS = [
    ("2p", (CAND, INC), 48, 50_000),               # 2p H2H (must tie: 2p frozen)
    ("2p", (CAND, "producer"), 48, 51_000),        # producer floor (ruler hard check)
    ("2p", (CAND, "pgs_allscripts"), 48, 52_000),  # rejected floor (ruler hard check)
    ("4p", (CAND, *FIELD), 48, 67_000),            # candidate 4p survival (PRIMARY axis)
]
# RELATIVE 4p baseline: incumbent in the SAME field on the SAME seeds (67_000).
REL_4P = ("4p", (INC, *FIELD), 48, 67_000)
# 4p verdict: candidate must CUT death and NOT bleed margin (the H9 question).
DEATH_IMPROVE_MIN = 0.02   # candidate annih must be at least this much BELOW incumbent
MARGIN_REGRESS_TOL = 0.03  # candidate 4p win-rate may dip at most this below incumbent


def _expected_games(mode: str, n: int) -> int:
    return n * (2 if mode == "2p" else 4)


def _chunk_jobs():
    jobs = []
    for mode, names, seeds, base in (*MATCHUPS, REL_4P):
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


def _merge(mode, names, seeds, base, results):
    games = []
    bad_rc = False
    crashes = Counter()
    for off in range(0, seeds, CHUNK):
        label = f"{mode}__{'_'.join(names)}__b{base + off}"
        rc, out = results[label]
        if rc != 0:
            bad_rc = True
        payload = json.loads(Path(out).read_text())
        for g in payload["games"]:
            g["mode"] = mode
        games.extend(payload["games"])
        for k, v in (payload.get("crashes") or {}).items():
            crashes[k] += v
    merged_path = OUT / f"MERGED__{mode}__{'_'.join(names)}.json"
    merged_path.write_text(json.dumps(
        {"agents": list(names), "mode": mode, "games": games, "crashes": dict(crashes)}, indent=1))
    return games, str(merged_path), bad_rc, dict(crashes)


def main() -> int:
    jobs = _chunk_jobs()
    print(f"[gate] h9_threat: {len(jobs)} chunks across {JOBS} workers; "
          f"4p-heavy; steps={STEPS}", flush=True)
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
        games, merged_path, bad_rc, crashes = _merge(mode, names, seeds, base, results)
        task_results.append({"candidate": CAND, "mode": mode, "names": list(names),
                             "out": merged_path, "returncode": 0 if not bad_rc else 1})
        sc = _score_games(games, CAND)
        seats = Counter(" | ".join(g["seats"]) for g in games)
        per_matchup.append((mode, names, seeds, sc, dict(seats), crashes, bad_rc))

    # RELATIVE 4p: incumbent in the same field/seeds.
    rmode, rnames, rseeds, rbase = REL_4P
    rel_games, _, _, _ = _merge(rmode, rnames, rseeds, rbase, results)
    inc_4p = _score_games(rel_games, INC)
    cand_4p = next(sc for m, nm, s, sc, *_ in per_matchup if m == "4p")
    death_delta = cand_4p["annihilation_rate"] - inc_4p["annihilation_rate"]   # want < 0
    win_delta = cand_4p["win_rate"] - inc_4p["win_rate"]                       # want >= -tol
    death_pass = death_delta <= -DEATH_IMPROVE_MIN
    margin_pass = win_delta >= -MARGIN_REGRESS_TOL
    h9_4p_pass = death_pass and margin_pass

    verdict = summarize_candidate(
        CAND, task_results, incumbent=INC,
        min_decisive_2p=40, min_producer_winrate=0.50,
        min_incumbent_winrate=0.50, min_floor_winrate=0.60,
        max_annihilation_rate_4p=1.0, weight_2p=0.30,   # 4p-heavy; absolute 4p veto -> relative below
    )
    # The H9 lever must IMPROVE 4p survival without bleeding margin, on top of the
    # ruler's 2p/floor/fault checks. Both required for PASS.
    final = verdict["verdict"]
    if final == "PASS_LOCAL" and not h9_4p_pass:
        final = "REJECT_LOCAL"

    report = {"candidate": CAND, "incumbent": INC, "steps": STEPS,
              "relative_4p": {
                  "cand_annihilation": cand_4p["annihilation_rate"],
                  "inc_annihilation": inc_4p["annihilation_rate"],
                  "death_delta": death_delta, "death_improve_min": DEATH_IMPROVE_MIN,
                  "death_pass": death_pass,
                  "cand_win_rate": cand_4p["win_rate"], "inc_win_rate": inc_4p["win_rate"],
                  "win_delta": win_delta, "margin_regress_tol": MARGIN_REGRESS_TOL,
                  "margin_pass": margin_pass, "h9_4p_pass": h9_4p_pass,
                  "cand_4p_faults": sum(cand_4p["faults"].values()),
                  "inc_4p_faults": sum(inc_4p["faults"].values())},
              "summarize_verdict": verdict["verdict"], "verdict_final": final,
              "matchups": [{"mode": m, "names": list(nm), "seeds": s, "score": sc,
                            "seat_counts": seats, "crashes": cr, "bad_rc": brc}
                           for (m, nm, s, sc, seats, cr, brc) in per_matchup],
              "verdict": verdict}
    rp = OUT / "gate_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n=== H9 THREAT-VALUE 4p SURVIVAL GATE (seat-rotated, 4p-heavy) ===")
    for m, nm, s, sc, seats, cr, brc in per_matchup:
        opp = "+".join(x for x in nm if x != CAND)
        dec = sc["decisive_win_rate"]
        print(f"  {m} vs {opp:24s} seeds={s:3d}  win_rate={sc['win_rate']:.3f} "
              f"dec_wr={dec if dec is not None else float('nan'):.3f} "
              f"annih={sc['annihilation_rate']:.3f} faults={sum(sc['faults'].values())}"
              + ("  BAD_RC" if brc else ""))
    print(f"\n  RELATIVE 4p (cand vs incumbent holdwave, same field/seeds):")
    print(f"    death: cand={cand_4p['annihilation_rate']:.3f} inc={inc_4p['annihilation_rate']:.3f} "
          f"delta={death_delta:+.3f} (need <=-{DEATH_IMPROVE_MIN}) -> {'PASS' if death_pass else 'FAIL'}")
    print(f"    win  : cand={cand_4p['win_rate']:.3f} inc={inc_4p['win_rate']:.3f} "
          f"delta={win_delta:+.3f} (need >=-{MARGIN_REGRESS_TOL}) -> {'PASS' if margin_pass else 'FAIL'}")
    print(f"\n  VERDICT = {final}  (summarize={verdict['verdict']}; h9_4p_pass={h9_4p_pass})")
    for c in verdict["checks"]:
        if c["severity"] != "info" and not c["passed"]:
            print(f"    [FAIL] {c['name']}: {c['details']}")
    print(f"\n  wrote {rp}")
    return 0 if final == "PASS_LOCAL" else 1


if __name__ == "__main__":
    sys.exit(main())
