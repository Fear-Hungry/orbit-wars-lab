"""Play one league matchup (2p pair or 4p composition) over batched seeds.

2p: plays BOTH seat orderings per seed. 4p: rotates the composition across
seats per seed (seed i uses rotation i % 4). Winner = argmax final total ships
(official reward semantics: one +1, rest -1). Also records annihilation
(first step a player's ships+planets hit 0) — the LB failure mode our old
margin-only evals never measured.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig  # noqa: E402
from python.orbit_wars_gym.entities import (  # noqa: E402
    fleet_owner,
    fleet_ships,
    planet_owner,
    planet_ships,
)
from python.orbit_wars_gym.observation import to_official_observation  # noqa: E402
from python.orbit_wars_gym.rules import moves_are_legal  # noqa: E402
from scripts.league_agents import make  # noqa: E402


def _totals(state, num_players):
    tot = [0.0] * num_players
    for p in state.get("planets", []):
        o = planet_owner(p)
        if 0 <= o < num_players:
            tot[o] += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if 0 <= o < num_players:
            tot[o] += fleet_ships(f)
    return tot


def _bump_fault(game_faults, name, key):
    f = game_faults.setdefault(name, {"crashes": 0, "timeouts": 0, "invalid_moves": 0})
    f[key] += 1


# Kaggle act semantics (kaggle_environments agent.py/core.py/schemas.json):
# each act may exceed actTimeout=1s by drawing on a banked overage budget
# (observation.remainingOverageTime, default 12s); the act whose overrun
# EXCEEDS the remaining bank becomes DeadlineExceeded -> status TIMEOUT. Any
# exception -> status ERROR. Either way the agent never acts again and core
# overrides its reward to None (it cannot win, regardless of final ships).
ACT_TIMEOUT_S = 1.0
OVERAGE_BANK_S = 12.0


def play_batch(names_by_seat, seeds, steps, decision_ms, crashes):
    """names_by_seat: list of agent names, one per seat. Returns per-seed games."""
    n, np_ = len(seeds), len(names_by_seat)
    backend = RustBatchBackend(num_envs=n, num_players=np_, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=True))
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [[make(name) for name in names_by_seat] for _ in range(n)]
    died_at = [[None] * np_ for _ in range(n)]
    # per-game fault counters {agent_name: {crashes, timeouts, invalid_moves}}
    faults = [{} for _ in range(n)]
    # Kaggle semantics state: remaining overage bank + terminal error status
    # ("ERROR"/"TIMEOUT") per seat — an errored seat stops acting for the rest
    # of the game (its planets keep producing, fleets in flight continue)
    overage = [[OVERAGE_BANK_S] * np_ for _ in range(n)]
    errored = [[None] * np_ for _ in range(n)]
    for t in range(steps):
        rows = []
        for i in range(n):
            for seat, name in enumerate(names_by_seat):
                if died_at[i][seat] is not None or errored[i][seat] is not None:
                    continue
                obs = to_official_observation(states[i], seat)
                t0 = time.perf_counter()
                try:
                    raw_moves = agents[i][seat](obs)
                except Exception as e:
                    # Kaggle: any raise -> status ERROR, agent dead for the rest
                    # of the episode (never silently: corrupts H2H/BT otherwise)
                    if crashes.get(name, 0) == 0:
                        print(f"[crash] {name} seat={seat} seed={seeds[i]} step={t}: {e!r}",
                              file=sys.stderr, flush=True)
                    crashes[name] = crashes.get(name, 0) + 1
                    _bump_fault(faults[i], name, "crashes")
                    errored[i][seat] = "ERROR"
                    raw_moves = []
                dt_ms = (time.perf_counter() - t0) * 1000.0
                decision_ms.setdefault(name, []).append(dt_ms)
                if dt_ms > ACT_TIMEOUT_S * 1000.0:
                    _bump_fault(faults[i], name, "timeouts")
                    over_s = dt_ms / 1000.0 - ACT_TIMEOUT_S
                    # Kaggle agent.py checks the overrun against the bank BEFORE
                    # decrementing it; the killing act is replaced by
                    # DeadlineExceeded (this turn's moves are lost)
                    if over_s > overage[i][seat]:
                        errored[i][seat] = errored[i][seat] or "TIMEOUT"
                        raw_moves = []
                    else:
                        overage[i][seat] -= over_s
                if raw_moves is None:
                    moves = []
                elif isinstance(raw_moves, list):
                    moves = raw_moves
                else:
                    _bump_fault(faults[i], name, "invalid_moves")
                    moves = []
                valid_moves = []
                for m in moves:
                    try:
                        # official process_moves drops len != 3 entries exactly
                        ok = len(m) == 3 and all(math.isfinite(float(m[k])) for k in range(3))
                    except (TypeError, ValueError):
                        ok = False
                    if not ok:
                        # invalid entry was previously dropped UNCOUNTED — count it
                        _bump_fault(faults[i], name, "invalid_moves")
                        continue
                    mv = [float(m[0]), float(m[1]), float(m[2])]
                    valid_moves.append(mv)
                if valid_moves and not moves_are_legal(states[i], seat, valid_moves):
                    # Well-formed but semantically illegal as a TURN: wrong owner,
                    # non-positive ships, or aggregate overbudget from one source.
                    # The engine may ignore only the impossible launches, so keep
                    # forwarding to mirror dynamics, but never let the game look clean.
                    _bump_fault(faults[i], name, "invalid_moves")
                for mv in valid_moves:
                    rows.append([float(i), float(seat), mv[0], mv[1], mv[2]])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
        for i in range(n):
            tot = _totals(states[i], np_)
            for seat in range(np_):
                if died_at[i][seat] is None and tot[seat] <= 0.0:
                    died_at[i][seat] = t + 1
    games = []
    for i in range(n):
        tot = _totals(states[i], np_)
        # Tie detection on the ROUNDED totals we emit as final_ships: keeps the
        # JSON self-consistent (a consumer recomputing the winner from
        # final_ships agrees) and avoids invisible 1e-12 float gaps deciding a
        # "winner" the record itself can't distinguish.
        final = [round(x, 1) for x in tot]
        # Kaggle core overrides reward to None for ERROR/TIMEOUT agents — an
        # errored seat cannot win even holding the max ships; winner is the
        # unique argmax among NON-errored seats only.
        eligible = [s for s in range(np_) if errored[i][s] is None]
        mx = max((final[s] for s in eligible), default=0.0)
        top = [s for s in eligible if final[s] == mx]
        tie = len(top) > 1
        winner = top[0] if (not tie and top and mx > 0) else -1
        game = {
            "seed": int(seeds[i]),
            "seats": list(names_by_seat),
            "final_ships": final,
            "winner_seat": winner,
            "winner": names_by_seat[winner] if winner >= 0 else None,
            "tie": tie,
            "died_at": died_at[i],
            # per-seat terminal status, Kaggle vocabulary; always present —
            # a missing key marks pre-instrumentation games (audit rule)
            "agent_status": [errored[i][s] or "DONE" for s in range(np_)],
        }
        # always present, even when clean ({}): a MISSING "faults" key means the
        # game predates the fault instrumentation (UNAUDITED — crash/timeout/
        # invalid invisible), while present-but-empty means audited clean. The
        # old omit-when-clean contract made the two cases indistinguishable
        # (2026-06-11: all 5k pre-fix games read as "clean" in the report).
        game["faults"] = faults[i]
        games.append(game)
    return games


def _seed_chunks(seeds: list[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [seeds[i: i + size] for i in range(0, len(seeds), size)]


def _write_report(out_path: str, names: list[str], games: list[dict], decision_ms: dict[str, list[float]],
                  crashes: dict[str, int]) -> None:
    out = {
        "agents": names,
        "mode": f"{len(names)}p",
        "games": games,
        "decision_ms_p95": {k: float(np.percentile(v, 95)) for k, v in decision_ms.items() if v},
        "crashes": crashes,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True, help="comma list: 2 names (pair) or 4 (composition)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="write partial output after this many seeds per seat-order batch (0 = all seeds)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    names = [s.strip() for s in args.agents.split(",")]
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else len(seeds)
    decision_ms: dict[str, list[float]] = {}
    crashes: dict[str, int] = {}
    games = []
    if len(names) == 2:
        # Interleave seat orders per seed chunk so partial checkpoint JSONs are
        # already seat-balanced enough for monitoring. The final report was
        # balanced before this; the fix is for honest long-run progress reads.
        for batch in _seed_chunks(seeds, chunk_size):
            for seat_names in (names, names[::-1]):
                games += play_batch(seat_names, batch, args.steps, decision_ms, crashes)
                _write_report(args.out, names, games, decision_ms, crashes)
    elif len(names) == 4:
        rotation_chunks: list[tuple[list[str], list[list[int]]]] = []
        for r in range(4):
            rot = names[r:] + names[:r]
            batch = [s for j, s in enumerate(seeds) if j % 4 == r]
            rotation_chunks.append((rot, _seed_chunks(batch, chunk_size)))
        max_chunks = max((len(chunks) for _, chunks in rotation_chunks), default=0)
        for idx in range(max_chunks):
            for rot, chunks in rotation_chunks:
                if idx >= len(chunks):
                    continue
                chunk = chunks[idx]
                if chunk:
                    games += play_batch(rot, chunk, args.steps, decision_ms, crashes)
                    _write_report(args.out, names, games, decision_ms, crashes)
    else:
        raise SystemExit("need 2 or 4 agents")
    _write_report(args.out, names, games, decision_ms, crashes)
    print(json.dumps({"out": args.out, "games": len(games), "crashes": crashes}))


if __name__ == "__main__":
    main()
