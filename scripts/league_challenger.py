"""All-vs-all CHALLENGER validator — "does this bot beat every previous bot?".

The bias-free way to validate a new bot (e.g. a PPO checkpoint): its pairwise
head-to-head win rate vs EACH pool member, not the population-biased BT/Elo
aggregate (the bias that made the league call s100 #1 and the field disagree —
Balduzzi et al. 2018). This is what the Kaggle field does: winner-take-all,
all-vs-all. A bot "domina o pool" iff it wins >= 0.50 vs every member.

For each opponent it runs league_match (2p, BOTH seat orders) over N seeds at
500 steps, in parallel across opponents, then prints the H2H win rate + Wilson
95% CI + a PASS/FAIL verdict.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/league_challenger.py --candidate pgs_holdwave --seeds 12
  PYTHONPATH=. .venv/bin/python scripts/league_challenger.py --candidate <ppo_name> --seeds 20 --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TMP = ROOT / "artifacts/league/challenger"

# the "previous bots" the candidate must beat (the all-vs-all pool minus junk).
# ext_* are weak field proxies kept as floor calibration; the verdict separates
# "beats our real bots" from "beats the floor".
DEFAULT_POOL = ["producer", "oep", "brep", "pgs_hold", "pgs_holdwave",
                "pgs_wave_s100", "brep_league3", "pgs_allscripts", "pgs_bigwave",
                "ext_lb1050", "ext_hellburner"]
REAL_BOTS = {"producer", "oep", "brep", "pgs_hold", "pgs_holdwave",
             "pgs_wave_s100", "brep_league3", "pgs_allscripts", "pgs_bigwave"}


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _spawn(cand, opp, out, seeds, seed_base):
    cmd = [sys.executable, "scripts/league_match.py", "--agents", f"{cand},{opp}",
           "--seeds", str(seeds), "--seed-base", str(seed_base), "--steps", "500",
           "--out", str(out)]
    return subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def h2h_from_file(path, cand):
    """win rate of `cand` over the opponent in a 2p match file (both seats)."""
    from scripts.league_report import decisive_winner

    d = json.loads(Path(path).read_text())
    w = dec = 0
    for g in d["games"]:
        # shared rule (ties, wipes, Kaggle agent_status eligibility) — a local
        # argmax would crown an ERROR/TIMEOUT seat the field never rewards
        winner = decisive_winner(g)
        if winner is None:
            continue  # draw/wipe/all-errored — not decisive
        dec += 1
        if winner == cand:
            w += 1
    return w, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="agent name in FACTORIES (or a tarball auto-registered as one)")
    ap.add_argument("--pool", default=",".join(DEFAULT_POOL))
    ap.add_argument("--seeds", type=int, default=12, help="seeds per opponent (x2 seat orders = 2x games)")
    ap.add_argument("--seed-base", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-winrate", type=float, default=0.50)
    args = ap.parse_args()

    cand = args.candidate
    pool = [p for p in args.pool.split(",") if p and p != cand]
    TMP.mkdir(parents=True, exist_ok=True)
    sb = args.seed_base
    running = {}
    t0 = time.perf_counter()

    # launch one match job per opponent, capped at --workers
    queue = list(pool)
    results = {}
    while queue or running:
        while queue and len(running) < args.workers:
            opp = queue.pop(0)
            out = TMP / f"{cand}__{opp}.json"
            p = _spawn(cand, opp, out, args.seeds, sb)
            sb += args.seeds
            running[p] = (opp, out)
        done = [p for p in running if p.poll() is not None]
        if not done:
            time.sleep(1.0)
            continue
        for p in done:
            opp, out = running.pop(p)
            try:
                w, dec = h2h_from_file(out, cand)
            except Exception as e:
                print(f"  {opp:16s} FALHOU ({e})", flush=True)
                results[opp] = None
                continue
            results[opp] = (w, dec)

    # report, ordered by win rate
    print(f"\n=== CHALLENGER: {cand} vs pool (2p, ambos assentos, 500 steps) ===")
    print(f"{'oponente':16s} {'win':>5s} {'CI95':>13s} {'jogos':>6s} {'LB':>6s}")
    try:
        from scripts.league_agents import LB_ANCHORS
    except Exception:
        LB_ANCHORS = {}
    beats_real = beats_all = total_real = 0
    rows = []
    for opp, r in results.items():
        if r is None:
            continue
        w, dec = r
        wr = w / dec if dec else 0.0
        lo, hi = wilson(w, dec)
        rows.append((opp, wr, lo, hi, dec))
    for opp, wr, lo, hi, dec in sorted(rows, key=lambda x: x[1]):
        lb = f"{LB_ANCHORS[opp]:6.0f}" if opp in LB_ANCHORS else "     -"
        flag = "✓" if wr >= args.min_winrate else "✗"
        print(f"{opp:16s} {wr:5.2f} [{lo:.2f},{hi:.2f}] {dec:6d} {lb} {flag}")
        if opp in REAL_BOTS:
            total_real += 1
            if wr >= args.min_winrate:
                beats_real += 1
        if wr >= args.min_winrate:
            beats_all += 1
    n = len([r for r in results.values() if r])
    print(f"\nbate {beats_all}/{n} do pool | bate {beats_real}/{total_real} dos NOSSOS bots reais "
          f"(exclui ext_*) | {time.perf_counter()-t0:.0f}s")
    verdict = "DOMINA o pool (bate todos os bots reais)" if beats_real == total_real else \
              f"NÃO domina — perde/empata p/ {total_real-beats_real} bot(s) real(is)"
    print(f"VEREDITO: {cand} {verdict}")


if __name__ == "__main__":
    main()
