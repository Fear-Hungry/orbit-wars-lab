"""Aggregate league match JSONs -> payoff matrix + Bradley-Terry rating
+ bootstrap CIs + LB estimate + submission-gate verdict.

Pairwise outcomes: 2p games directly; 4p games decompose winner-beats-each-loser
(same winner-take-all signal the LB emits). Ties in final ships = draw (no
pairwise outcome for the tied players).

Calibration: Spearman between BT rating and LB_ANCHORS + hard gate
"pgs_allscripts below producer/oep/brep". Gate stat: P(bot >= producer) from
bootstrap resampling of games.
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys
from collections import defaultdict

from scripts.league_agents import GATE_REFERENCE, LB_ANCHORS


def load_games(pattern):
    """pattern: one glob or several comma-separated globs."""
    import os

    games = []
    for pat in pattern.split(","):
        for f in sorted(glob.glob(pat.strip()), key=lambda p: (os.path.getmtime(p), p)):
            d = json.load(open(f))
            for g in d["games"]:
                g["mode"] = d["mode"]
                games.append(g)
    return games


# Kaggle-like online rating, reverse-engineered from 429 real episode updates of
# our own submissions (DB id=145): logistic expectation at scale 500 vs the MEAN
# opponent rating; K decays exponentially with games played (faster in 4p).
def kaggle_mu(games, mu0=600.0, D=500.0):
    mu = defaultdict(lambda: mu0)
    n = defaultdict(int)
    for g in games:
        seats, ships = g["seats"], g["final_ships"]
        top = max(ships)
        winners = [s for s, sh in zip(seats, ships) if sh == top]
        if len(winners) != 1:
            continue
        w = winners[0]
        deltas = {}
        for nm in seats:
            opp = [mu[o] for o in seats if o != nm]
            E = 1.0 / (1.0 + 10 ** ((sum(opp) / len(opp) - mu[nm]) / D))
            if len(seats) == 2:
                K = max(230.0 * math.exp(-n[nm] / 15.0), 12.0)
            else:
                K = max(270.0 * math.exp(-n[nm] / 10.0), 4.0)
            S = 1.0 if nm == w else 0.0
            deltas[nm] = K * (S - E)
        for nm in seats:
            mu[nm] += deltas[nm]
            n[nm] += 1
    return dict(mu), dict(n)


def pairwise_outcomes(games):
    """[(winner, loser), ...] with draws skipped."""
    out = []
    for g in games:
        seats, ships = g["seats"], g["final_ships"]
        top = max(ships)
        winners = [s for s, sh in zip(seats, ships) if sh == top]
        if len(winners) != 1:
            continue  # draw (or total wipe) — no signal
        w = winners[0]
        out.append((g, w, [s for s in seats if s != w]))
    return out


def bt_from_outcomes(outcomes, names, iters=300):
    wins = defaultdict(lambda: defaultdict(float))
    for _, w, losers in outcomes:
        for l in losers:
            wins[w][l] += 1.0
    s = {n: 1.0 for n in names}
    for _ in range(iters):
        new = {}
        for i in names:
            num = sum(wins[i][j] for j in names if j != i)
            den = 0.0
            for j in names:
                if j == i:
                    continue
                nij = wins[i][j] + wins[j][i]
                if nij:
                    den += nij / (s[i] + s[j])
            new[i] = num / den if den > 0 else s[i]
        z = sum(new.values()) / len(new)
        s = {k: max(v / z, 1e-9) for k, v in new.items()}
    return {k: 400.0 * math.log10(v) + 1000.0 for k, v in s.items()}, wins


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "artifacts/league/v1/p*.json"
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    games = load_games(pattern)
    outcomes = pairwise_outcomes(games)
    names = sorted({s for g in games for s in g["seats"]})
    bt, wins = bt_from_outcomes(outcomes, names)

    # bootstrap over games -> rating CI + P(bot >= producer)
    rng = random.Random(7)
    boots = []
    for _ in range(n_boot):
        sample = [outcomes[rng.randrange(len(outcomes))] for _ in range(len(outcomes))]
        b, _ = bt_from_outcomes(sample, names, iters=150)
        boots.append(b)
    ci = {}
    p_ge_prod = {}
    for n in names:
        vals = sorted(b[n] for b in boots)
        ci[n] = (vals[int(0.05 * n_boot)], vals[int(0.95 * n_boot)])
        p_ge_prod[n] = sum(1 for b in boots if b[n] >= b[GATE_REFERENCE]) / n_boot

    # LB estimate: least-squares BT -> LB on anchors
    ax = [bt[n] for n in LB_ANCHORS if n in bt]
    ay = [LB_ANCHORS[n] for n in LB_ANCHORS if n in bt]
    mx, my = sum(ax) / len(ax), sum(ay) / len(ay)
    sxx = sum((x - mx) ** 2 for x in ax)
    slope = sum((x - mx) * (y - my) for x, y in zip(ax, ay)) / sxx if sxx else 0.0
    icpt = my - slope * mx
    lb_est = {n: slope * bt[n] + icpt for n in names}

    # per-mode breakdown
    stats = {n: {"2p": [0, 0], "4p": [0, 0], "annih": [0, 0]} for n in names}
    for g in games:
        ships = g["final_ships"]
        top = max(ships)
        winners = [s for s, sh in zip(g["seats"], ships) if sh == top]
        for seat, nm in enumerate(g["seats"]):
            st = stats[nm]
            st[g["mode"]][1] += 1
            if len(winners) == 1 and nm == winners[0]:
                st[g["mode"]][0] += 1
            st["annih"][1] += 1
            if g["died_at"][seat] is not None:
                st["annih"][0] += 1

    kmu, kn = kaggle_mu(games)
    ranking = sorted(names, key=lambda n: -bt[n])
    anchored = [n for n in ranking if n in LB_ANCHORS]
    rho = spearman([bt[n] for n in anchored], [LB_ANCHORS[n] for n in anchored])
    rho_k = spearman([kmu[n] for n in anchored], [LB_ANCHORS[n] for n in anchored])
    hard_gate = ("pgs_allscripts" in bt and
                 all(bt["pgs_allscripts"] < bt[n] for n in ("producer", "oep", "brep") if n in bt))

    print(f"games: { {m: sum(1 for g in games if g['mode']==m) for m in ('2p','4p')} } "
          f"(decisive pairwise: {len(outcomes)}; bootstrap n={n_boot})")
    print(f"\n{'bot':16s} {'BT':>5s} {'CI90':>12s} {'µ-kgl':>6s} {'LB_est':>6s} {'LB':>5s} {'P>=ref':>7s} "
          f"{'win2p':>6s} {'win4p':>6s} {'annih':>6s}")
    for n in ranking:
        lo, hi = ci[n]
        lb = f"{LB_ANCHORS[n]:5.0f}" if n in LB_ANCHORS else "    -"
        s = stats[n]
        w2 = f"{s['2p'][0]/s['2p'][1]:.2f}" if s["2p"][1] else "  -"
        w4 = f"{s['4p'][0]/s['4p'][1]:.2f}" if s["4p"][1] else "  -"
        an = f"{s['annih'][0]/s['annih'][1]:.2f}"
        print(f"{n:16s} {bt[n]:5.0f} [{lo:4.0f},{hi:4.0f}] {kmu[n]:6.0f} {lb_est[n]:6.0f} {lb} "
              f"{p_ge_prod[n]:7.2f} {w2:>6s} {w4:>6s} {an:>6s}")
    print(f"\ncalibration: spearman BT = {rho:+.3f} | spearman µ-kaggle = {rho_k:+.3f} | "
          f"hard gate (allscripts < cluster): {'PASS' if hard_gate else 'FAIL'}")
    print(f"gate de submissão (ref={GATE_REFERENCE}): aprovado se P(bot >= ref) >= 0.6")

    out = {
        "bt": bt, "ci90": ci, "p_ge_producer": p_ge_prod, "lb_estimate": lb_est,
        "kaggle_mu": kmu, "kaggle_games": kn, "spearman_kaggle_mu": rho_k,
        "ranking": ranking, "spearman_lb": rho, "hard_gate": hard_gate,
        "per_mode": stats,
        "wins": {r: dict(wins[r]) for r in names},
    }
    open("artifacts/league/v1/report.json", "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
