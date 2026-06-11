"""H-P0 — simulation budget per turn for the mission-portfolio search (heuristic front).

Measures, on REAL game states (Producer vs Producer via the Rust backend), the cost of
the primitives a per-turn assignment search (PGS, Churchill & Buro 2013) would pay:

  1. per-turn SETUP: official obs -> tensors -> parse_obs -> ensure_planet_movement
     -> build_distance_cache -> baseline garrison_status(H);
  2. per-CANDIDATE eval: _clone_movement + cache invalidation (what _debit_entry_sources
     does) + garrison_status(H) recompute + production-territory reduce — the lower
     bound a candidate plan evaluation costs;
  3. full OEP agent step, for reference (it already runs a candidate search per turn).

Output: evals/turn that fit in the ~700ms usable slice of Kaggle's actTimeout=1s,
per horizon. Decision gate (todo H-P0): go if >= ~20 useful evals/turn.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from bots.oep.planner import _clone_movement, _invalidate_garrison_cache
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import ensure_planet_movement
from orbit_lite.obs import parse_obs
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation


def _flat_rows(player: int, moves) -> list[list[float]]:
    return [[0.0, float(player), float(m[0]), float(m[1]), float(m[2])] for m in moves if len(m) >= 3]


def collect_states(seed: int, capture_steps: list[int]) -> dict[int, dict]:
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(enable_comets=True))
    backend.reset(seed)
    # one call with count=2 -> two distinct isolated instances (two calls with the
    # same name would hand back the same pool slots and cross-contaminate)
    policies = get_isolated_opponents("producer", 2)
    captured: dict[int, dict] = {}
    last = max(capture_steps)
    for step in range(last + 1):
        state = backend.states()[0]
        if step in capture_steps:
            captured[step] = state
        rows = _flat_rows(0, policies[0](state, 0)) + _flat_rows(1, policies[1](state, 1))
        import numpy as np

        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
    return captured


def _ms(fn, reps: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) * 1000.0 / reps


def bench_state(state: dict, horizons: list[int], cand_reps: int) -> dict:
    obs = to_official_observation(state, 0)
    out: dict = {"step": int(state.get("step", -1))}

    t0 = time.perf_counter()
    obs_tensors = single_obs_to_tensor(obs, player_id=0)
    parsed = parse_obs(obs_tensors)
    out["obs_to_tensor_ms"] = (time.perf_counter() - t0) * 1000.0
    out["P"] = int(parsed.P)

    per_h: dict[str, dict] = {}
    for H in horizons:
        cfg = MovementConfig(movement_horizon=H, drift_epsilon=1e-3, track_fleets=True,
                             player_count=2, max_tracked_fleets=128)
        t0 = time.perf_counter()
        movement = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=cfg, cached_movement=None)
        build_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        cache = build_distance_cache(movement, max_k=H)
        cache_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        status = movement.garrison_status(max_horizon=H)
        garrison_ms = (time.perf_counter() - t0) * 1000.0
        prod = movement.planet_prod

        def one_candidate_eval(movement=movement, H=H, prod=prod) -> float:
            clone = _clone_movement(movement)
            # what _debit_entry_sources does: mutate garrisons + drop the projection cache,
            # forcing garrison_status to recompute exactly like a real candidate eval
            clone.planet_ships = (clone.planet_ships - 1.0).clamp(min=0.0)
            _invalidate_garrison_cache(clone)
            st = clone.garrison_status(max_horizon=H)
            owner_h = st.owner[:, H]
            me = owner_h == 0
            enemy = (owner_h >= 0) & (~me)
            return float((prod[me].sum() - prod[enemy].sum()).item())

        samples = [_ms(one_candidate_eval) for _ in range(cand_reps)]
        cand_ms = statistics.median(samples)
        setup_ms = out["obs_to_tensor_ms"] + build_ms + cache_ms + garrison_ms
        per_h[str(H)] = {
            "movement_build_ms": build_ms,
            "distance_cache_ms": cache_ms,
            "garrison_status_ms": garrison_ms,
            "candidate_eval_ms_median": cand_ms,
            "candidate_eval_ms_p95": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
            "setup_total_ms": setup_ms,
            "evals_in_700ms": int(max(0.0, 700.0 - setup_ms) / cand_ms) if cand_ms > 0 else None,
        }
        _ = status, cache
    out["per_horizon"] = per_h
    return out


def bench_full_oep(state: dict, reps: int = 3) -> float:
    oep = get_isolated_opponents("oep", 1)[0]
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        oep(state, 0)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--capture-steps", default="30,100,200,350")
    ap.add_argument("--horizons", default="13,25,50")
    ap.add_argument("--cand-reps", type=int, default=30)
    ap.add_argument("--out", default="artifacts/hp0_sim_budget.json")
    args = ap.parse_args()

    torch.set_num_threads(1)  # Kaggle CPU is effectively single-threaded for the agent
    capture_steps = [int(s) for s in args.capture_steps.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]

    states = collect_states(args.seed, capture_steps)
    results = []
    for step, state in sorted(states.items()):
        r = bench_state(state, horizons, args.cand_reps)
        r["full_oep_step_ms"] = bench_full_oep(state)
        results.append(r)
        print(f"step {step}: P={r['P']} oep_full={r['full_oep_step_ms']:.1f}ms "
              + " ".join(
                  f"H{h}[setup={v['setup_total_ms']:.1f}ms cand={v['candidate_eval_ms_median']:.2f}ms "
                  f"evals/700ms={v['evals_in_700ms']}]"
                  for h, v in r["per_horizon"].items()
              ))

    summary = {
        "seed": args.seed,
        "torch_threads": 1,
        "states": results,
        "worst_case_evals_in_700ms": {
            h: min(r["per_horizon"][h]["evals_in_700ms"] for r in results) for h in map(str, horizons)
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("worst-case evals/700ms by horizon:", summary["worst_case_evals_in_700ms"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
