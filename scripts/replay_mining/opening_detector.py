"""Opening-phase detector for G3.1a (attacks bad_opening_4p).

Phase-1 diagnostic: on a set of replays, measure what our agent DID in the first
~50 steps (the expansion phase) and isolate the "tell" of a bad 4p opening.

Per fleet WE launch in the window, the target is resolved from ground truth: the
planet nearest to the fleet's last observed position (a fleet vanishes when it
reaches a planet's radius, so last-pos ~ target edge). This sidesteps the
rotating-planet aim math entirely. Intent = owner of that planet AT LAUNCH:
  neutral  -> expansion
  us       -> reinforce
  other    -> PvP (the thing we want to suppress early in 4p)

The 🐛 check is built in: `arrival_dist` (how close the fleet got to ANY planet)
and `expansion_success_rate` (did launches at neutrals actually capture them).
If openings fail because fleets miss / arrive short, that's an aim/ETA bug, not a
strategy problem — and the headline becomes "fix targeting", not "suppress PvP".

Usage:
    .venv/bin/python scripts/replay_mining/opening_detector.py \
        --set bad_opening_4p --out artifacts/replay_mining/opening_bad.csv
    .venv/bin/python scripts/replay_mining/opening_detector.py \
        --set win_4p --limit 124 --out artifacts/replay_mining/opening_win.csv
    # then compare:
    .venv/bin/python scripts/replay_mining/opening_detector.py --compare \
        artifacts/replay_mining/opening_bad.csv artifacts/replay_mining/opening_win.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay_mining import episode_service as es
from replay_mining.parse import _full_obs, parse_replay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "replay_mining"
REPLAYS = OUT / "replays"
CONTROL = OUT / "control_replays"   # downloaded wins for comparison
TAXO = OUT / "loss_taxonomy.csv"
INDEX = OUT / "episode_index.json"

WINDOW = 50          # opening / expansion phase length in steps
CAPTURE_GRACE = 6    # steps after arrival to check whether we captured the target


def _obs_with(step, key):
    for a in step:
        ob = a.get("observation") or {}
        if ob.get(key) is not None:
            return ob
    return {}


def _planets_at(step):
    return _full_obs(step).get("planets") or []


def _nearest_planet(planets, x, y):
    best, bd = None, 1e18
    for p in planets:
        d = (p[2] - x) ** 2 + (p[3] - y) ** 2
        if d < bd:
            bd, best = d, p
    return best, math.sqrt(bd)


def detect(replay: dict) -> dict:
    f = parse_replay(replay)
    oi = f["our_index"]
    n = f["n_players"]
    steps = replay.get("steps") or []
    n_steps = len(steps)
    win = min(WINDOW, n_steps - 1)

    # Track our fleets: id -> list of (step, x, y, from_planet, ships).
    tracks: dict[int, list[tuple]] = {}
    for t, st in enumerate(steps):
        obs = _obs_with(st, "fleets")
        for fl in obs.get("fleets") or []:
            if fl[1] == oi:
                tracks.setdefault(fl[0], []).append((t, fl[2], fl[3], fl[5], fl[6]))

    # Planet owner timeline (by id) for intent-at-launch and capture checks.
    def owner_at(t, pid):
        if t < 0 or t >= n_steps:
            t = max(0, min(t, n_steps - 1))
        for p in _planets_at(steps[t]):
            if p[0] == pid:
                return p[1]
        return None

    expansion_ships = pvp_ships = reinforce_ships = 0
    n_launch = exp_attempts = exp_success = 0
    arrival_dists = []
    first_pvp_launch = None

    for fid, tr in tracks.items():
        launch_step = tr[0][0]
        if launch_step >= win:
            continue
        n_launch += 1
        ships = tr[0][4]
        last_t, lx, ly = tr[-1][0], tr[-1][1], tr[-1][2]
        # resolve target from arrival position (planet positions rotate -> use the
        # step after the fleet was last seen, i.e. arrival)
        arr_step = min(last_t + 1, n_steps - 1)
        planets = _planets_at(steps[arr_step])
        tgt, dist = _nearest_planet(planets, lx, ly)
        arrival_dists.append(dist)
        if tgt is None:
            continue
        tgt_id = tgt[0]
        owner = owner_at(launch_step, tgt_id)
        if owner == -1:
            expansion_ships += ships
            exp_attempts += 1
            # captured if we own it within the grace window after arrival
            if any(owner_at(arr_step + k, tgt_id) == oi for k in range(CAPTURE_GRACE + 1)):
                exp_success += 1
        elif owner == oi:
            reinforce_ships += ships
        else:
            pvp_ships += ships
            if first_pvp_launch is None or launch_step < first_pvp_launch:
                first_pvp_launch = launch_step

    # Ownership-based opening metrics.
    init_planets = _planets_at(steps[0])
    initial_neutrals = sum(1 for p in init_planets if p[1] == -1)
    win_planets = _planets_at(steps[win])
    free_neutrals_50 = sum(1 for p in win_planets if p[1] == -1)
    owned_50 = {i: 0 for i in range(n)}
    for p in win_planets:
        if 0 <= p[1] < n:
            owned_50[p[1]] += 1

    # neutral captures in [1, win]: transitions -1 -> player, and OUR capture fate.
    prev = {p[0]: p[1] for p in init_planets}
    captures = {i: 0 for i in range(n)}
    our_captured_at: dict[int, int] = {}   # pid -> step we took it from neutral
    eta_first_loss = None
    for t in range(1, win + 1):
        for p in _planets_at(steps[t]):
            pid, ow = p[0], p[1]
            was = prev.get(pid)
            if was == -1 and 0 <= ow < n:
                captures[ow] += 1
                if ow == oi and pid not in our_captured_at:
                    our_captured_at[pid] = t
            if was == oi and ow != oi and 0 <= ow < n and ow != -1 and eta_first_loss is None:
                eta_first_loss = t
            prev[pid] = ow
    total_caps = sum(captures.values()) or 1
    strongest_opp_caps = max((captures[i] for i in range(n) if i != oi), default=0)
    sent = expansion_ships + pvp_ships + reinforce_ships

    # Fate of OUR opening captures: did we hold them to step `win`? When we lost
    # one, what was our garrison just before it flipped? (under-garrison signal)
    def owner_ships(t, pid):
        for p in _planets_at(steps[min(max(t, 0), n_steps - 1)]):
            if p[0] == pid:
                return p[1], p[5]
        return None, None

    held = lost = 0
    garr_at_loss = []
    for pid, cs in our_captured_at.items():
        o50, _ = owner_ships(win, pid)
        if o50 == oi:
            held += 1
            continue
        lost += 1
        for t in range(cs + 1, win + 1):
            o, _ = owner_ships(t, pid)
            if o != oi:
                _, g = owner_ships(t - 1, pid)
                if g is not None:
                    garr_at_loss.append(g)
                break
    n_cap = len(our_captured_at)

    return {
        "episode_id": f["episode_id"],
        "our_index": oi,
        "n_players": n,
        "n_steps": n_steps,
        "n_planets": len(init_planets),
        "initial_neutrals": initial_neutrals,
        # expansion outcome
        "our_captures_50": captures[oi],
        "opp_captures_50": total_caps - captures[oi],
        "strongest_opp_caps_50": strongest_opp_caps,
        "neutral_share_50": round(captures[oi] / total_caps, 4),
        "owned_50": owned_50[oi],
        "free_neutrals_50": free_neutrals_50,
        # capture-and-HOLD: do opening neutral captures survive to step `win`?
        "opening_captures": n_cap,
        "captures_held_50": held,
        "captures_lost_50": lost,
        "capture_hold_rate": round(held / n_cap, 4) if n_cap else None,
        "mean_garrison_at_loss": round(statistics.mean(garr_at_loss), 2) if garr_at_loss else None,
        # launch intent mix
        "n_launch_50": n_launch,
        "expansion_ships_50": expansion_ships,
        "pvp_ships_50": pvp_ships,
        "reinforce_ships_50": reinforce_ships,
        "pvp_share_50": round(pvp_ships / sent, 4) if sent else 0.0,
        # timing
        "eta_first_pvp_launch": first_pvp_launch,
        "eta_first_loss_to_player": eta_first_loss,
        # 🐛 aim/ETA diagnostics
        "exp_attempts_50": exp_attempts,
        "exp_success_50": exp_success,
        "exp_success_rate": round(exp_success / exp_attempts, 4) if exp_attempts else None,
        "mean_arrival_dist": round(statistics.mean(arrival_dists), 3) if arrival_dists else None,
        "max_arrival_dist": round(max(arrival_dists), 3) if arrival_dists else None,
    }


def _episode_files(which: str, limit: int) -> list[tuple[int, Path]]:
    if which == "bad_opening_4p":
        rows = list(csv.DictReader(open(TAXO)))
        ids = [r["episode_id"] for r in rows
               if r["primary_class"] == "bad_opening" and r["format"] == "4p"]
        out = []
        for eid in ids:
            hits = list(REPLAYS.glob(f"*/loss/*/{eid}.json"))
            if hits:
                out.append((int(eid), hits[0]))
        return out[: limit or None]
    if which == "win_4p":
        idx = json.loads(INDEX.read_text())
        wins = [e for e in idx["episodes"]
                if e["outcome"] == "win" and e["n_players"] == 4]
        wins = wins[: limit or None]
        out = []
        CONTROL.mkdir(parents=True, exist_ok=True)
        for e in wins:
            dest = CONTROL / f"{e['episode_id']}.json"
            es.download_replay(e["episode_id"], dest)
            out.append((e["episode_id"], dest))
        return out
    raise SystemExit(f"unknown set {which}")


def cmd_run(args):
    files = _episode_files(args.set, args.limit)
    print(f"[detect] {args.set}: {len(files)} replays")
    rows = []
    for eid, path in files:
        try:
            rows.append(detect(json.loads(path.read_text())))
        except Exception as exc:  # pragma: no cover
            print(f"   FAIL {eid}: {exc}")
    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {out} ({len(rows)} rows)")
    _summarize(args.set, rows)


def _summarize(label, rows):
    def col(name, cast=float):
        return [cast(r[name]) for r in rows if r.get(name) not in (None, "", "None")]

    def stat(name):
        v = col(name)
        if not v:
            return "n/a"
        return f"mean={statistics.mean(v):.3f} med={statistics.median(v):.3f}"

    print(f"\n=== {label} (n={len(rows)}) ===")
    for k in ["neutral_share_50", "our_captures_50", "opp_captures_50",
              "strongest_opp_caps_50", "owned_50", "free_neutrals_50",
              "opening_captures", "captures_held_50", "captures_lost_50",
              "capture_hold_rate", "mean_garrison_at_loss",
              "pvp_share_50", "pvp_ships_50", "expansion_ships_50",
              "eta_first_pvp_launch", "eta_first_loss_to_player",
              "exp_success_rate", "mean_arrival_dist"]:
        print(f"  {k:26s}: {stat(k)}")
    pvp_early = sum(1 for r in rows
                    if r.get("eta_first_pvp_launch") not in (None, "", "None")
                    and float(r["eta_first_pvp_launch"]) < 25)
    print(f"  PvP-launch-before-step25  : {pvp_early}/{len(rows)} = {pvp_early/len(rows):.1%}")


def cmd_compare(args):
    a = list(csv.DictReader(open(args.files[0])))
    b = list(csv.DictReader(open(args.files[1])))
    la = Path(args.files[0]).stem
    lb = Path(args.files[1]).stem
    print(f"Comparing A={la} (n={len(a)})  vs  B={lb} (n={len(b)})\n")
    keys = ["neutral_share_50", "our_captures_50", "opp_captures_50",
            "strongest_opp_caps_50", "owned_50", "free_neutrals_50",
            "opening_captures", "captures_held_50", "captures_lost_50",
            "capture_hold_rate", "mean_garrison_at_loss",
            "pvp_share_50", "pvp_ships_50", "expansion_ships_50",
            "eta_first_pvp_launch", "eta_first_loss_to_player",
            "exp_success_rate", "mean_arrival_dist"]
    print(f"{'metric':26s} {'A(med)':>10s} {'B(med)':>10s}  delta")
    for k in keys:
        va = [float(r[k]) for r in a if r.get(k) not in (None, "", "None")]
        vb = [float(r[k]) for r in b if r.get(k) not in (None, "", "None")]
        if not va or not vb:
            continue
        ma, mb = statistics.median(va), statistics.median(vb)
        print(f"{k:26s} {ma:10.3f} {mb:10.3f}  {ma-mb:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--set", choices=["bad_opening_4p", "win_4p"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "opening_metrics.csv"))
    ap.add_argument("--compare", nargs=2, dest="files", metavar=("A", "B"))
    args = ap.parse_args()
    if args.files:
        cmd_compare(args)
    elif args.set:
        cmd_run(args)
    else:
        ap.error("pass --set <name> or --compare A B")


if __name__ == "__main__":
    main()
