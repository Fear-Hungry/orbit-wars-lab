from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from kaggle_environments import make
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.snapshots import snapshot_to_state

PLANET_FIELDS = ("id", "owner", "x", "y", "radius", "ships", "production")
FLEET_FIELDS = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")
FLOAT_FIELDS = {"x", "y", "radius", "angle"}
XY_ATOL = 1.0e-6


def _planet_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "radius": float(row[4]),
        "ships": int(row[5]),
        "production": int(row[6]),
    }


def _fleet_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "angle": float(row[4]),
        "from_planet_id": int(row[5]),
        "ships": int(row[6]),
    }


def _comet_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    return row


def _snapshot_from_observation(observation: Any) -> dict[str, Any]:
    return {
        "planets": deepcopy(observation.planets),
        "fleets": deepcopy(observation.fleets),
        "angular_velocity": float(observation.angular_velocity),
        "initial_planets": deepcopy(observation.initial_planets),
        "next_fleet_id": int(observation.next_fleet_id),
        "comets": deepcopy(observation.comets),
        "comet_planet_ids": list(observation.comet_planet_ids),
    }


def _state_from_official_observation(observation: Any, *, num_players: int) -> dict[str, Any]:
    state = snapshot_to_state(
        _snapshot_from_observation(observation),
        num_players=int(num_players),
        step=int(observation.step),
        done=False,
    )
    state["planets"] = [_planet_dict(planet) for planet in state["planets"]]
    state["initial_planets"] = [_planet_dict(planet) for planet in state["initial_planets"]]
    state["fleets"] = [_fleet_dict(fleet) for fleet in state["fleets"]]
    state["comets"] = [_comet_dict(comet) for comet in state["comets"]]
    return state


def _ordered_by_id(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row["id"]))


def _field_equal(field: str, expected: Any, actual: Any, *, atol: float) -> bool:
    if field in FLOAT_FIELDS:
        return math.isclose(float(actual), float(expected), rel_tol=1.0e-6, abs_tol=atol)
    return int(actual) == int(expected)


def _compare_rows(
    *,
    kind: str,
    official_rows: Sequence[dict[str, Any]],
    rust_rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    seed: int,
    step: int,
    atol: float,
) -> str | None:
    official_ordered = _ordered_by_id(official_rows)
    rust_ordered = _ordered_by_id(rust_rows)
    if len(official_ordered) != len(rust_ordered):
        return (
            f"[PARITY-{kind.upper()}] seed={seed} step={step} field=count "
            f"official={len(official_ordered)} rust={len(rust_ordered)}"
        )
    for official, rust in zip(official_ordered, rust_ordered, strict=True):
        official_id = int(official["id"])
        rust_id = int(rust["id"])
        if official_id != rust_id:
            return (
                f"[PARITY-{kind.upper()}] seed={seed} step={step} field=id "
                f"official={official_id} rust={rust_id}"
            )
        for field in fields:
            if not _field_equal(field, official[field], rust[field], atol=atol):
                return (
                    f"[PARITY-{kind.upper()}] seed={seed} step={step} id={official_id} "
                    f"field={field} official={official[field]!r} rust={rust[field]!r}"
                )
    return None


def _compare_comets(
    *,
    official: Any,
    rust: Any,
    seed: int,
    step: int,
    atol: float,
) -> str | None:
    official_ids = sorted(int(pid) for pid in official.comet_planet_ids)
    rust_ids = sorted(int(pid) for pid in rust.get("comet_planet_ids", []))
    if official_ids != rust_ids:
        return (
            f"[PARITY-COMETS] seed={seed} step={step} field=comet_planet_ids "
            f"official={official_ids} rust={rust_ids}"
        )
    official_groups = list(official.comets)
    rust_groups = list(rust.get("comets", []))
    if len(official_groups) != len(rust_groups):
        return (
            f"[PARITY-COMETS] seed={seed} step={step} field=group_count "
            f"official={len(official_groups)} rust={len(rust_groups)}"
        )
    for group_idx, (official_group, rust_group) in enumerate(
        zip(official_groups, rust_groups, strict=True)
    ):
        official_group = _comet_dict(official_group)
        rust_group = _comet_dict(rust_group)
        if int(official_group["path_index"]) != int(rust_group["path_index"]):
            return (
                f"[PARITY-COMETS] seed={seed} step={step} group={group_idx} "
                f"field=path_index official={official_group['path_index']} "
                f"rust={rust_group['path_index']}"
            )
        if [int(pid) for pid in official_group["planet_ids"]] != [
            int(pid) for pid in rust_group["planet_ids"]
        ]:
            return (
                f"[PARITY-COMETS] seed={seed} step={step} group={group_idx} "
                f"field=planet_ids official={official_group['planet_ids']} "
                f"rust={rust_group['planet_ids']}"
            )
        official_paths = official_group["paths"]
        rust_paths = rust_group["paths"]
        if len(official_paths) != len(rust_paths):
            return (
                f"[PARITY-COMETS] seed={seed} step={step} group={group_idx} "
                f"field=paths_count official={len(official_paths)} rust={len(rust_paths)}"
            )
        for path_idx, (official_path, rust_path) in enumerate(
            zip(official_paths, rust_paths, strict=True)
        ):
            if len(official_path) != len(rust_path):
                return (
                    f"[PARITY-COMETS] seed={seed} step={step} group={group_idx} "
                    f"path={path_idx} field=path_len official={len(official_path)} "
                    f"rust={len(rust_path)}"
                )
            sample_points = {0, max(0, len(official_path) - 1)}
            if official_group["path_index"] < len(official_path):
                sample_points.add(int(official_group["path_index"]))
            for point_idx in sorted(sample_points):
                official_xy = official_path[point_idx]
                rust_xy = rust_path[point_idx]
                for axis, official_value, rust_value in zip(
                    ("x", "y"), official_xy, rust_xy, strict=True
                ):
                    if not math.isclose(
                        float(rust_value),
                        float(official_value),
                        rel_tol=1.0e-6,
                        abs_tol=atol,
                    ):
                        return (
                            f"[PARITY-COMETS] seed={seed} step={step} group={group_idx} "
                            f"path={path_idx} point={point_idx} field={axis} "
                            f"official={official_value!r} rust={rust_value!r}"
                        )
    return None


def _compare_state(
    *,
    official_observation: Any,
    rust_state: dict[str, Any],
    seed: int,
    atol: float,
) -> str | None:
    step = int(official_observation.step)
    if int(rust_state["step"]) != step:
        return (
            f"[PARITY-STATE] seed={seed} field=step "
            f"official={step} rust={rust_state['step']}"
        )
    if int(official_observation.next_fleet_id) != int(rust_state["next_fleet_id"]):
        return (
            f"[PARITY-STATE] seed={seed} step={step} field=next_fleet_id "
            f"official={official_observation.next_fleet_id} rust={rust_state['next_fleet_id']}"
        )
    if not math.isclose(
        float(rust_state["angular_velocity"]),
        float(official_observation.angular_velocity),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        return (
            f"[PARITY-STATE] seed={seed} step={step} field=angular_velocity "
            f"official={official_observation.angular_velocity!r} "
            f"rust={rust_state['angular_velocity']!r}"
        )
    diff = _compare_rows(
        kind="planets",
        official_rows=[_planet_dict(row) for row in official_observation.planets],
        rust_rows=[_planet_dict(row) for row in rust_state["planets"]],
        fields=PLANET_FIELDS,
        seed=seed,
        step=step,
        atol=atol,
    )
    if diff is not None:
        return diff
    diff = _compare_rows(
        kind="fleets",
        official_rows=[_fleet_dict(row) for row in official_observation.fleets],
        rust_rows=[_fleet_dict(row) for row in rust_state["fleets"]],
        fields=FLEET_FIELDS,
        seed=seed,
        step=step,
        atol=atol,
    )
    if diff is not None:
        return diff
    return _compare_comets(
        official=official_observation,
        rust=rust_state,
        seed=seed,
        step=step,
        atol=atol,
    )


def run_probe(
    *,
    episodes: int,
    steps: int,
    num_players: int,
    enable_comets: bool,
    atol: float,
) -> dict[str, Any]:
    checked_steps = 0
    for seed in range(int(episodes)):
        env = make(
            "orbit_wars",
            configuration={"seed": seed, "episodeSteps": int(steps)},
            debug=True,
        )
        env.reset(int(num_players))
        rust = RustBatchBackend(
            num_envs=1,
            num_players=int(num_players),
            seed=seed,
            config=RustConfig(episode_steps=int(steps), enable_comets=bool(enable_comets)),
        )
        official_obs = env.state[0].observation
        rust_state = _state_from_official_observation(
            official_obs,
            num_players=int(num_players),
        )
        rust.reset_from_states([rust_state])
        diff = _compare_state(
            official_observation=official_obs,
            rust_state=rust.states()[0],
            seed=seed,
            atol=float(atol),
        )
        if diff is not None:
            return {"passed": False, "error": diff, "seed": seed, "checked_steps": checked_steps}

        while not env.done and int(official_obs.step) < int(steps):
            actions = [[] for _ in range(int(num_players))]
            env.step(actions)
            _, rust_states = rust.step_with_states([actions])
            official_obs = env.state[0].observation
            checked_steps += 1
            diff = _compare_state(
                official_observation=official_obs,
                rust_state=rust_states[0],
                seed=seed,
                atol=float(atol),
            )
            if diff is not None:
                return {
                    "passed": False,
                    "error": diff,
                    "seed": seed,
                    "checked_steps": checked_steps,
                }
    return {
        "passed": True,
        "episodes": int(episodes),
        "steps": int(steps),
        "num_players": int(num_players),
        "enable_comets": bool(enable_comets),
        "checked_steps": checked_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--atol", type=float, default=XY_ATOL)
    args = parser.parse_args()

    report = run_probe(
        episodes=max(1, int(args.episodes)),
        steps=max(1, int(args.steps)),
        num_players=int(args.num_players),
        enable_comets=not bool(args.disable_comets),
        atol=float(args.atol),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
