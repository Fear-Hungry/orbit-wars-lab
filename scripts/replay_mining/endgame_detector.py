"""End-game detector for G3.2 (attacks even_attrition_collapse, 2p).

The 2p attrition loss: start even (~50% share), trade planets all game
(recaptures ~= losses), collapse in the closing phase. Phase-1 diagnostic: on the
mid-late game (default steps >= 60), measure what separates these losses from 2p
WINS, with a strong prior from memory[lb_elite_style_taxonomy]: elite (1500+) play
FEW BIG waves (>=50 ships) and HOARD 2-5x; Producer/PGS "spray" small waves (the
losers' profile). So we measure wave discipline + hoarding + the ship/production
divergence (winning ships but losing the economy?).

No target resolution needed — wave SIZE is read straight from our actions
([from, angle, ships]). Compare loss vs win to isolate the tell.

    .venv/bin/python scripts/replay_mining/endgame_detector.py --set attrition_2p --out artifacts/replay_mining/endgame_loss.csv
    .venv/bin/python scripts/replay_mining/endgame_detector.py --set win_2p --out artifacts/replay_mining/endgame_win.csv
    .venv/bin/python scripts/replay_mining/endgame_detector.py --compare A.csv B.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay_mining.parse import _full_obs, _player_state, parse_replay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "replay_mining"
REPLAYS = OUT / "replays"
CONTROL_2P = OUT / "control_replays_2p"
TAXO = OUT / "loss_taxonomy.csv"

MID = 60          # start of the "mid-late" window
BIG_WAVE = 50     # elite threshold: a "big" wave


def detect(replay: dict) -> dict:
    f = parse_replay(replay)
    oi = f["our_index"]
    n = f["n_players"]
    steps = replay.get("steps") or []
    n_steps = len(steps)
    lo = min(MID, max(0, n_steps - 1))

    # Our wave sizes over the window (each action move = [from, angle, ships]).
    wave_sizes: list[float] = []
    for t in range(lo, n_steps):
        if oi < len(steps[t]):
            for mv in steps[t][oi].get("action") or []:
                try:
                    s = float(mv[2])
                    if s > 0:
                        wave_sizes.append(s)
                except (IndexError, TypeError, ValueError):
                    pass
    total_wave_ships = sum(wave_sizes)
    n_waves = len(wave_sizes)
    big = [w for w in wave_sizes if w >= BIG_WAVE]

    # Hoard + share/production trajectory over the window.
    hoard_fracs = []
    ship_share = {}
    prod_share = {}
    for t in range(lo, n_steps):
        st = _player_state(_full_obs(steps[t]), n)
        tot = st["total"]
        ssum = sum(tot) or 1
        if tot[oi] > 0:
            hoard_fracs.append(st["max_stack"][oi] / tot[oi])
        psum = sum(st["prod"]) or 1
        for mark in (100, 150):
            if t == min(mark, n_steps - 1):
                ship_share[mark] = tot[oi] / ssum
                prod_share[mark] = st["prod"][oi] / psum

    return {
        "episode_id": f["episode_id"],
        "format": f["format"],
        "n_steps": n_steps,
        "final_reward": f["final_reward"],
        "elim_step": f["elim_step"],
        # wave discipline
        "n_waves": n_waves,
        "mean_wave": round(statistics.mean(wave_sizes), 2) if wave_sizes else 0.0,
        "median_wave": round(statistics.median(wave_sizes), 2) if wave_sizes else 0.0,
        "max_wave": round(max(wave_sizes), 1) if wave_sizes else 0.0,
        "n_big_waves": len(big),
        "frac_waves_ge50": round(len(big) / n_waves, 4) if n_waves else 0.0,
        "frac_ships_in_big": round(sum(big) / total_wave_ships, 4) if total_wave_ships else 0.0,
        "waves_per_100steps": round(n_waves / max(1, n_steps - lo) * 100, 2),
        # hoarding
        "mean_hoard_frac": round(statistics.mean(hoard_fracs), 4) if hoard_fracs else 0.0,
        "max_hoard_frac": round(max(hoard_fracs), 4) if hoard_fracs else 0.0,
        # economy vs ships
        "ship_share_100": round(ship_share.get(100, float("nan")), 4),
        "prod_share_100": round(prod_share.get(100, float("nan")), 4),
        "ship_share_150": round(ship_share.get(150, float("nan")), 4),
        "prod_share_150": round(prod_share.get(150, float("nan")), 4),
    }


def _files(which: str, limit: int):
    if which == "attrition_2p":
        rows = list(csv.DictReader(open(TAXO)))
        ids = [r["episode_id"] for r in rows
               if r["primary_class"] == "even_attrition_collapse" and r["format"] == "2p"]
        out = []
        for eid in ids:
            hits = list(REPLAYS.glob(f"*/loss/*/{eid}.json"))
            if hits:
                out.append(hits[0])
        return out[: limit or None]
    if which == "win_2p":
        return sorted(CONTROL_2P.glob("*.json"))[: limit or None]
    raise SystemExit(f"unknown set {which}")


def cmd_run(args):
    files = _files(args.set, args.limit)
    print(f"[detect] {args.set}: {len(files)} replays")
    rows = [detect(json.loads(p.read_text())) for p in files]
    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {out} ({len(rows)} rows)")
    _summary(args.set, rows)


def _summary(label, rows):
    keys = ["n_waves", "mean_wave", "median_wave", "max_wave", "frac_waves_ge50",
            "frac_ships_in_big", "waves_per_100steps", "mean_hoard_frac",
            "max_hoard_frac", "ship_share_100", "prod_share_100",
            "ship_share_150", "prod_share_150"]
    print(f"\n=== {label} (n={len(rows)}) ===")
    for k in keys:
        v = [r[k] for r in rows if isinstance(r[k], (int, float)) and r[k] == r[k]]
        if v:
            print(f"  {k:20s}: mean={statistics.mean(v):.3f} med={statistics.median(v):.3f}")


def cmd_compare(a_path, b_path):
    a = list(csv.DictReader(open(a_path)))
    b = list(csv.DictReader(open(b_path)))
    print(f"A={Path(a_path).stem} (n={len(a)})  vs  B={Path(b_path).stem} (n={len(b)})\n")
    print(f"{'metric':20s} {'A(med)':>10s} {'B(med)':>10s}  delta")
    keys = ["n_waves", "mean_wave", "median_wave", "max_wave", "frac_waves_ge50",
            "frac_ships_in_big", "waves_per_100steps", "mean_hoard_frac",
            "max_hoard_frac", "ship_share_100", "prod_share_100",
            "ship_share_150", "prod_share_150"]
    for k in keys:
        va = [float(r[k]) for r in a if r.get(k) not in (None, "", "nan", "None")]
        vb = [float(r[k]) for r in b if r.get(k) not in (None, "", "nan", "None")]
        if not va or not vb:
            continue
        ma, mb = statistics.median(va), statistics.median(vb)
        print(f"{k:20s} {ma:10.3f} {mb:10.3f}  {ma-mb:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["attrition_2p", "win_2p"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "endgame_metrics.csv"))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        cmd_compare(*args.compare)
    elif args.set:
        cmd_run(args)
    else:
        ap.error("pass --set or --compare")


if __name__ == "__main__":
    main()
