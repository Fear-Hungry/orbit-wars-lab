"""Eval our agents vs EXTERNAL kaggle-style agents (T0 top-5 proxy ruler).

External agent = a single .py file exposing `agent(obs[, config])` in official
kaggle obs format (stdlib-only, e.g. pulled from public notebooks into
artifacts/opponents/top5_proxy/<name>/agent.py).

One fresh module instance per (env, seat) — these agents keep state in module
globals, so each game needs its own copy (same isolation rule as the registry).

Reports the same metrics as eval_pgs_direct: seat-averaged normalized ship-score
margin, per-seat margins, win rate, per-seed margins, decision-time stats.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

_EXT_COUNTER = [0]
ACT_TIMEOUT_SECONDS = 1.0


def load_external_agent(path: str):
    """Load a fresh, isolated instance of an external agent module."""
    _EXT_COUNTER[0] += 1
    name = f"_external_agent_{_EXT_COUNTER[0]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def make_our_agent(kind: str, pgs_config: dict):
    if kind == "pgs":
        from bots.pgs.planner import PGSConfig, make_runtime

        rt = make_runtime(PGSConfig(**pgs_config))
        return rt.act
    if kind == "producer":
        from bots.producer.agent import make_agent

        return make_agent()
    raise ValueError(f"unknown agent kind: {kind}")


def _ships(state, player):
    own = enemy = 0.0
    for p in state.get("planets", []):
        o = planet_owner(p)
        if o == player:
            own += planet_ships(p)
        elif o >= 0:
            enemy += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if o == player:
            own += fleet_ships(f)
        elif o >= 0:
            enemy += fleet_ships(f)
    return own, enemy


def _empty_faults():
    return {"crashes": 0, "timeouts": 0, "invalid_actions": 0}


def _sanitize_moves(state, player, moves, faults):
    if not isinstance(moves, list):
        faults["invalid_actions"] += 1
        return []
    if not moves_are_legal(state, player, moves):
        faults["invalid_actions"] += 1
        return []
    return moves


def _play_seat(agent_seat, seeds, episode_steps, enable_comets, external_path,
               agent_kind, pgs_config, decision_ms, opp_ms, faults, opp_faults):
    n = len(seeds)
    backend = RustBatchBackend(
        num_envs=n, num_players=2, seed=int(seeds[0]),
        config=RustConfig(
            enable_comets=enable_comets,
            episode_steps=episode_steps,
            act_timeout=ACT_TIMEOUT_SECONDS,
        ),
    )
    backend.reset(int(seeds[0]))
    states = backend.states()
    agents = [make_our_agent(agent_kind, pgs_config) for _ in range(n)]
    opp_seat = 1 - agent_seat
    opps = [load_external_agent(external_path) for _ in range(n)]
    for _ in range(episode_steps):
        rows = []
        for i in range(n):
            obs = to_official_observation(states[i], agent_seat)
            t0 = time.perf_counter()
            try:
                moves = agents[i](obs)
            except Exception:
                faults["crashes"] += 1
                decision_ms.append((time.perf_counter() - t0) * 1000.0)
                moves = []
            else:
                elapsed = time.perf_counter() - t0
                decision_ms.append(elapsed * 1000.0)
                if elapsed > ACT_TIMEOUT_SECONDS:
                    faults["timeouts"] += 1
                    moves = []
                else:
                    moves = _sanitize_moves(states[i], agent_seat, moves, faults)
            for m in moves:
                if len(m) >= 3:
                    rows.append([float(i), float(agent_seat), float(m[0]), float(m[1]), float(m[2])])
            opp_obs = to_official_observation(states[i], opp_seat)
            t0 = time.perf_counter()
            try:
                opp_moves = opps[i](opp_obs)
            except Exception:
                opp_faults["crashes"] += 1
                opp_ms.append((time.perf_counter() - t0) * 1000.0)
                opp_moves = []
            else:
                elapsed = time.perf_counter() - t0
                opp_ms.append(elapsed * 1000.0)
                if elapsed > ACT_TIMEOUT_SECONDS:
                    opp_faults["timeouts"] += 1
                    opp_moves = []
                else:
                    opp_moves = _sanitize_moves(states[i], opp_seat, opp_moves, opp_faults)
            for m in opp_moves:
                if len(m) >= 3:
                    rows.append([float(i), float(opp_seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, agent_seat)
        states = backend.states()
    margins = []
    for i in range(n):
        own, enemy = _ships(states[i], agent_seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True, help="path to external agent.py")
    ap.add_argument("--agent", default="pgs", choices=["pgs", "producer"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--max-deviations", type=int, default=None)
    ap.add_argument("--scripts", type=str, default=None)
    ap.add_argument("--arbiter-margin", type=float, default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.set_num_threads(1)
    pgs_config = {}
    for key in ("max_deviations", "scripts", "arbiter_margin"):
        v = getattr(args, key)
        if v is not None:
            pgs_config[key] = v

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    decision_ms: list[float] = []
    opp_ms: list[float] = []
    faults = _empty_faults()
    opp_faults = _empty_faults()
    by_seat = {
        seat: _play_seat(seat, seeds, args.episode_steps, not args.no_comets,
                         args.external, args.agent, pgs_config, decision_ms, opp_ms,
                         faults, opp_faults)
        for seat in (0, 1)
    }
    all_margins = [m for ms in by_seat.values() for m in ms]
    arr = np.asarray(decision_ms)
    oarr = np.asarray(opp_ms)
    result = {
        "agent": args.agent,
        "pgs_config": pgs_config,
        "external": args.external,
        "mean_score_margin": float(np.mean(all_margins)),
        **{f"seat{seat}_margin": float(np.mean(ms)) for seat, ms in by_seat.items()},
        "win_rate": float(np.mean([m > 0 for m in all_margins])),
        "annihilated_rate": float(np.mean([m <= -0.99 for m in all_margins])),
        "annihilation_wins": float(np.mean([m >= 0.99 for m in all_margins])),
        "seeds": args.seeds,
        "episode_steps": args.episode_steps,
        **{f"per_seed_seat{seat}": {str(s): round(m, 4) for s, m in zip(seeds, ms, strict=False)}
           for seat, ms in by_seat.items()},
        "decision_ms": {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())},
        "opp_decision_ms": {"mean": float(oarr.mean()), "p95": float(np.percentile(oarr, 95))},
        "runtime_faults": faults,
        "opp_runtime_faults": opp_faults,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
