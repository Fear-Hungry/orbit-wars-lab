from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from bots.oep.planner import (
    OEPLiteRuntime,
    _cheap_opponent_entries,
    _config_for,
    _effective_config,
    _entries_from_moves,
    _movement_config,
    _opponent_id,
    _to_list_observation,
)
from bots.producer._upstream import ProducerLiteRuntime
from bots.producer.agent import agent as producer_agent
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.movement_step import ensure_planet_movement
from orbit_lite.obs import parse_obs
from orbit_lite.planner_core import largest_initial_player_count
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation


def _entry_lanes(entries, planet_ids: torch.Tensor) -> dict[tuple[int, int], float]:
    valid = torch.where(entries.valid & (entries.ships >= 1.0))[0]
    lanes: dict[tuple[int, int], float] = {}
    for idx in valid.tolist():
        source_slot = int(entries.source_slots[idx].item())
        target_slot = int(entries.target_slots[idx].item())
        if source_slot < 0 or target_slot < 0:
            continue
        if source_slot >= int(planet_ids.shape[0]) or target_slot >= int(planet_ids.shape[0]):
            continue
        key = (int(planet_ids[source_slot].item()), int(planet_ids[target_slot].item()))
        lanes[key] = lanes.get(key, 0.0) + float(entries.ships[idx].item())
    return lanes


def _compare_lanes(
    *,
    real: dict[tuple[int, int], float],
    model: dict[tuple[int, int], float],
) -> dict[str, float]:
    real_keys = set(real)
    model_keys = set(model)
    overlap = real_keys & model_keys
    ship_abs_error = 0.0
    if overlap:
        ship_abs_error = sum(abs(float(model[key]) - float(real[key])) for key in overlap) / len(overlap)
    return {
        "samples": 1.0,
        "real_lane_count": float(len(real_keys)),
        "model_lane_count": float(len(model_keys)),
        "overlap_lane_count": float(len(overlap)),
        "lane_recall": float(len(overlap) / len(real_keys)) if real_keys else 1.0,
        "lane_precision": float(len(overlap) / len(model_keys)) if model_keys else (1.0 if not real_keys else 0.0),
        "empty_real": 1.0 if not real_keys else 0.0,
        "empty_model": 1.0 if not model_keys else 0.0,
        "matched_ship_abs_error": float(ship_abs_error),
    }


def _disagreement_score(row: dict[str, float]) -> float:
    return (
        (1.0 - float(row["lane_recall"]))
        + (1.0 - float(row["lane_precision"]))
        + 0.25 * abs(float(row["model_lane_count"]) - float(row["real_lane_count"]))
        + 0.02 * float(row["matched_ship_abs_error"])
    )


def _lane_rows(lanes: dict[tuple[int, int], float]) -> list[dict[str, float]]:
    return [
        {"source": float(source), "target": float(target), "ships": float(ships)}
        for (source, target), ships in sorted(lanes.items())
    ]


def _example_row(
    *,
    seed: int,
    step: int,
    oep_player: int,
    model_name: str,
    row: dict[str, float],
    real: dict[tuple[int, int], float],
    model: dict[tuple[int, int], float],
) -> dict[str, Any]:
    real_keys = set(real)
    model_keys = set(model)
    missing = {key: real[key] for key in sorted(real_keys - model_keys)}
    extra = {key: model[key] for key in sorted(model_keys - real_keys)}
    return {
        "score": float(_disagreement_score(row)),
        "seed": int(seed),
        "step": int(step),
        "oep_player": int(oep_player),
        "model": str(model_name),
        "metrics": {key: float(value) for key, value in sorted(row.items()) if key != "samples"},
        "real_lanes": _lane_rows(real),
        "model_lanes": _lane_rows(model),
        "missing_lanes": _lane_rows(missing),
        "extra_lanes": _lane_rows(extra),
    }


def _record_example(
    examples: dict[str, list[dict[str, Any]]],
    *,
    max_examples: int,
    example: dict[str, Any],
) -> None:
    if int(max_examples) <= 0:
        return
    bucket = examples.setdefault(str(example["model"]), [])
    bucket.append(example)
    bucket.sort(key=lambda item: float(item["score"]), reverse=True)
    del bucket[int(max_examples) :]


def _top_lanes(lanes: dict[tuple[int, int], float], max_lanes: int) -> dict[tuple[int, int], float]:
    if int(max_lanes) <= 0 or len(lanes) <= int(max_lanes):
        return dict(lanes)
    return dict(
        sorted(lanes.items(), key=lambda item: float(item[1]), reverse=True)[: int(max_lanes)]
    )


def _merge(target: dict[str, float], row: dict[str, float]) -> None:
    for key, value in row.items():
        target[key] = target.get(key, 0.0) + float(value)


def _mean_rows(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, totals in sorted(rows.items()):
        samples = max(1.0, float(totals.get("samples", 0.0)))
        out[name] = {key: float(value) / samples for key, value in sorted(totals.items())}
        out[name]["samples"] = samples
    return out


def _producer_runtime_moves(
    runtime: ProducerLiteRuntime,
    state: dict[str, Any],
    player: int,
) -> list[list[float]]:
    obs = _to_list_observation(to_official_observation(state, player=int(player)))
    obs_tensors = single_obs_to_tensor(obs, player_id=int(player))
    with torch.no_grad():
        row = runtime.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(row, obs, player_id=int(player))


def _producer_policy_moves(state: dict[str, Any], player: int) -> list[list[float]]:
    obs = _to_list_observation(to_official_observation(state, player=int(player)))
    moves = producer_agent(obs)
    return list(moves) if isinstance(moves, list) else []


def _evaluate_state(
    *,
    state: dict[str, Any],
    oep_player: int,
    real_moves_by_player: list[list[list[float]]],
    runtime: OEPLiteRuntime,
    shadow_seed_for_shared: bool,
) -> dict[str, tuple[dict[str, float], dict[tuple[int, int], float], dict[tuple[int, int], float]]]:
    raw_obs = _to_list_observation(to_official_observation(state, player=int(oep_player)))
    obs_tensors = single_obs_to_tensor(raw_obs, player_id=int(oep_player))
    player_count = largest_initial_player_count(obs_tensors)
    opp_id = _opponent_id(int(oep_player), player_count)
    if opp_id is None:
        return {}

    base_config = _effective_config(_config_for(player_count), step=int(obs_tensors["step"].item()))
    obs = parse_obs(obs_tensors)
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(base_config, player_count=player_count),
        cached_movement=None,
    )
    cache = build_distance_cache(movement, max_k=int(base_config.horizon))
    status = movement.garrison_status(max_horizon=int(base_config.horizon))
    alive_by_step = movement.alive_by_step[: int(base_config.horizon) + 1]

    real_entries = _entries_from_moves(
        moves=real_moves_by_player[int(opp_id)],
        movement=movement,
        obs_tensors=obs_tensors,
        player_id=int(opp_id),
    )
    real_lanes = _entry_lanes(real_entries, movement.planet_ids)
    rows: dict[
        str,
        tuple[dict[str, float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    ] = {}

    cheap_entries = _cheap_opponent_entries(
        movement=movement,
        obs=obs,
        cache=cache,
        status=status,
        config=base_config,
        opponent_id=int(opp_id),
    )
    cheap_lanes = _entry_lanes(cheap_entries, movement.planet_ids)
    rows["cheap"] = (_compare_lanes(real=real_lanes, model=cheap_lanes), cheap_lanes, real_lanes)

    inline_entries = runtime._producer_entries_inline(
        owner_id=int(opp_id),
        obs_tensors=obs_tensors,
        movement=movement,
        cache=cache,
        status=status,
        alive_by_step=alive_by_step,
        base_config=base_config,
        player_count=player_count,
    )
    inline_lanes = _entry_lanes(inline_entries, movement.planet_ids)
    rows["producer_inline"] = (
        _compare_lanes(real=real_lanes, model=inline_lanes),
        inline_lanes,
        real_lanes,
    )
    for max_lanes in (1, 2, 3):
        model_lanes = _top_lanes(inline_lanes, max_lanes)
        rows[f"producer_inline_top{max_lanes}"] = _compare_lanes(
            real=real_lanes,
            model=model_lanes,
        ), model_lanes, real_lanes

    tensor_entries = runtime._producer_entries_tensor(
        owner_id=int(opp_id),
        obs_tensors=obs_tensors,
        movement=movement,
    )
    tensor_lanes = _entry_lanes(tensor_entries, movement.planet_ids)
    rows["producer_tensor"] = (
        _compare_lanes(real=real_lanes, model=tensor_lanes),
        tensor_lanes,
        real_lanes,
    )
    for max_lanes in (1, 2, 3):
        model_lanes = _top_lanes(tensor_lanes, max_lanes)
        rows[f"producer_tensor_top{max_lanes}"] = _compare_lanes(
            real=real_lanes,
            model=model_lanes,
        ), model_lanes, real_lanes

    if bool(shadow_seed_for_shared):
        runtime._producer_entries_shared_tensor(
            owner_id=int(oep_player),
            obs_tensors=obs_tensors,
            movement=movement,
        )
    shared_entries = runtime._producer_entries_shared_tensor(
        owner_id=int(opp_id),
        obs_tensors=obs_tensors,
        movement=movement,
    )
    shared_lanes = _entry_lanes(shared_entries, movement.planet_ids)
    rows["producer_shared_tensor"] = (
        _compare_lanes(real=real_lanes, model=shared_lanes),
        shared_lanes,
        real_lanes,
    )
    return rows


def compare_models(
    *,
    seeds: int,
    steps: int,
    enable_comets: bool,
    examples: int,
    reference: str,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(dict)
    example_rows: dict[str, list[dict[str, Any]]] = {}
    runtime = OEPLiteRuntime(seed_policy=producer_agent, opponent_policy=producer_agent)
    producer_runtimes = [ProducerLiteRuntime(), ProducerLiteRuntime()]
    for seed in range(max(1, int(seeds))):
        runtime.reset()
        backend = RustBatchBackend(
            num_envs=1,
            num_players=2,
            seed=int(seed),
            config=RustConfig(
                episode_steps=int(steps),
                enable_comets=bool(enable_comets),
                act_timeout=1.0,
            ),
        )
        state = backend.reset(seed)[0]
        for producer_runtime in producer_runtimes:
            producer_runtime.reset()
        for step in range(max(1, int(steps))):
            if str(reference) == "isolated":
                real_moves_by_player = [
                    _producer_runtime_moves(producer_runtimes[player], state, player)
                    for player in (0, 1)
                ]
            elif str(reference) == "policy_after_seed":
                real_moves_by_player = []
            else:
                raise ValueError(f"unknown reference: {reference!r}")
            for oep_player in (0, 1):
                if str(reference) == "policy_after_seed":
                    opp_id = 1 - int(oep_player)
                    _producer_policy_moves(state, int(oep_player))
                    opponent_moves = _producer_policy_moves(state, int(opp_id))
                    real_moves_by_player = [[], []]
                    real_moves_by_player[int(opp_id)] = opponent_moves
                rows = _evaluate_state(
                    state=state,
                    oep_player=oep_player,
                    real_moves_by_player=real_moves_by_player,
                    runtime=runtime,
                    shadow_seed_for_shared=str(reference) == "policy_after_seed",
                )
                for name, (row, model_lanes, real_lanes) in rows.items():
                    _merge(totals[name], row)
                    _record_example(
                        example_rows,
                        max_examples=int(examples),
                        example=_example_row(
                            seed=int(seed),
                            step=int(step),
                            oep_player=int(oep_player),
                            model_name=name,
                            row=row,
                            real=real_lanes,
                            model=model_lanes,
                        ),
                    )
            if str(reference) == "policy_after_seed":
                actions = [
                    [
                        _producer_runtime_moves(producer_runtimes[player], state, player)
                        for player in (0, 1)
                    ]
                ]
            else:
                actions = [real_moves_by_player]
            outcomes, states = backend.step_with_states(actions)
            state = states[0]
            if bool(outcomes[0].get("done", False)):
                break
    report = {
        "seeds": int(seeds),
        "steps": int(steps),
        "enable_comets": bool(enable_comets),
        "reference": str(reference),
        "models": _mean_rows(totals),
    }
    if int(examples) > 0:
        report["examples"] = example_rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cheap OEP opponent models to Producer lanes.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--reference", choices=("isolated", "policy_after_seed"), default="isolated")
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = compare_models(
        seeds=int(args.seeds),
        steps=int(args.steps),
        enable_comets=not bool(args.disable_comets),
        examples=int(args.examples),
        reference=str(args.reference),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
