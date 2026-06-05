from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.movement import MovementConfig, PlanetMovement
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation


def _obs_tensors(state: dict[str, Any], *, player: int) -> dict[str, Any]:
    obs = to_official_observation(state, player=int(player))
    obs["planets"] = [_planet_row(planet) for planet in obs.get("planets", [])]
    obs["initial_planets"] = [
        _planet_row(planet) for planet in obs.get("initial_planets", [])
    ]
    obs["fleets"] = [_fleet_row(fleet) for fleet in obs.get("fleets", [])]
    return single_obs_to_tensor(obs, player_id=int(player))


def _planet_row(planet: Any) -> Any:
    if not isinstance(planet, dict):
        return planet
    return [
        planet["id"],
        planet["owner"],
        planet["x"],
        planet["y"],
        planet["radius"],
        planet["ships"],
        planet["production"],
    ]


def _fleet_row(fleet: Any) -> Any:
    if not isinstance(fleet, dict):
        return fleet
    return [
        fleet["id"],
        fleet["owner"],
        fleet["x"],
        fleet["y"],
        fleet["angle"],
        fleet["from_planet_id"],
        fleet["ships"],
    ]


def _measure_fresh(states: list[dict[str, Any]], *, horizon: int, player: int) -> list[float]:
    elapsed_ms: list[float] = []
    cfg = MovementConfig(
        movement_horizon=int(horizon),
        drift_epsilon=1.0e-3,
        track_fleets=True,
        player_count=2,
        max_tracked_fleets=128,
    )
    for state in states:
        obs = _obs_tensors(state, player=player)
        start = perf_counter()
        movement = PlanetMovement.from_obs_tensors(obs, config=cfg)
        movement.garrison_status(max_horizon=int(horizon))
        elapsed_ms.append(1000.0 * (perf_counter() - start))
    return elapsed_ms


def _measure_cached(states: list[dict[str, Any]], *, horizon: int, player: int) -> dict[str, Any]:
    cfg = MovementConfig(
        movement_horizon=int(horizon),
        drift_epsilon=1.0e-3,
        track_fleets=True,
        player_count=2,
        max_tracked_fleets=128,
    )
    movement: PlanetMovement | None = None
    elapsed_ms: list[float] = []
    update_ms: list[float] = []
    status_ms: list[float] = []
    dirty_before: list[float] = []
    dirty_after: list[float] = []
    dirty_from_before: list[float] = []
    rebuilt = 0
    for state in states:
        obs = _obs_tensors(state, player=player)
        start = perf_counter()
        if movement is None:
            update_start = perf_counter()
            movement = PlanetMovement.from_obs_tensors(obs, config=cfg)
            update_ms.append(1000.0 * (perf_counter() - update_start))
            rebuilt += 1
        else:
            previous_obj = id(movement)
            update_start = perf_counter()
            movement.update(obs)
            update_ms.append(1000.0 * (perf_counter() - update_start))
            rebuilt += int(id(movement) != previous_obj)
        if movement.garrison_dirty_from is not None:
            dirty_before.append(float((movement.garrison_dirty_from <= int(horizon)).sum().item()))
            dirty_from_before.append(float(movement.garrison_dirty_from.float().mean().item()))
        status_start = perf_counter()
        movement.garrison_status(max_horizon=int(horizon))
        status_ms.append(1000.0 * (perf_counter() - status_start))
        if movement.garrison_dirty_from is not None:
            dirty_after.append(float((movement.garrison_dirty_from <= int(horizon)).sum().item()))
        elapsed_ms.append(1000.0 * (perf_counter() - start))
    return {
        "elapsed_ms": elapsed_ms,
        "update_ms": update_ms,
        "status_ms": status_ms,
        "dirty_before": dirty_before,
        "dirty_from_before": dirty_from_before,
        "dirty_after": dirty_after,
        "rebuilt": rebuilt,
    }


def _states(*, seed: int, steps: int, horizon: int, enable_comets: bool) -> list[dict[str, Any]]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(
            episode_steps=max(int(steps) + int(horizon) + 2, 64),
            enable_comets=bool(enable_comets),
            act_timeout=1.0,
        ),
    )
    state = backend.reset(seed)[0]
    out = [state]
    for _ in range(max(0, int(steps) - 1)):
        _, states = backend.step_with_states([[[], []]])
        state = states[0]
        out.append(state)
    return out


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": fmean(values),
        "max": max(values),
        "min": min(values),
    }


def run(*, seeds: int, steps: int, horizon: int, enable_comets: bool) -> dict[str, Any]:
    fresh_ms: list[float] = []
    cached_ms: list[float] = []
    dirty_before: list[float] = []
    dirty_after: list[float] = []
    dirty_from_before: list[float] = []
    cached_update_ms: list[float] = []
    cached_status_ms: list[float] = []
    rebuilt = 0
    for seed in range(max(1, int(seeds))):
        states = _states(
            seed=seed,
            steps=int(steps),
            horizon=int(horizon),
            enable_comets=bool(enable_comets),
        )
        fresh_ms.extend(_measure_fresh(states, horizon=int(horizon), player=0))
        cached = _measure_cached(states, horizon=int(horizon), player=0)
        cached_ms.extend(cached["elapsed_ms"])
        cached_update_ms.extend(cached["update_ms"])
        cached_status_ms.extend(cached["status_ms"])
        dirty_before.extend(cached["dirty_before"])
        dirty_from_before.extend(cached["dirty_from_before"])
        dirty_after.extend(cached["dirty_after"])
        rebuilt += int(cached["rebuilt"])
    fresh_mean = fmean(fresh_ms) if fresh_ms else 0.0
    cached_mean = fmean(cached_ms) if cached_ms else 0.0
    return {
        "seeds": int(seeds),
        "steps": int(steps),
        "horizon": int(horizon),
        "enable_comets": bool(enable_comets),
        "fresh_ms": _summary(fresh_ms),
        "cached_ms": _summary(cached_ms),
        "cached_update_ms": _summary(cached_update_ms),
        "cached_status_ms": _summary(cached_status_ms),
        "speedup": fresh_mean / cached_mean if cached_mean > 0.0 else 0.0,
        "dirty_slots_before": _summary(dirty_before),
        "dirty_from_before": _summary(dirty_from_before),
        "dirty_slots_after": _summary(dirty_after),
        "rebuilt": rebuilt,
        "samples": len(cached_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run(
        seeds=max(1, int(args.seeds)),
        steps=max(1, int(args.steps)),
        horizon=max(1, int(args.horizon)),
        enable_comets=not bool(args.disable_comets),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
