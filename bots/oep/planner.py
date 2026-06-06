from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.garrison_launch import LaunchSet, sparse_launch_flow_delta
from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    LaunchEntries,
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    safe_drain,
)
from torch import Tensor

from bots.producer._upstream import (
    ProducerLiteConfig,
)
from bots.producer._upstream import (
    ProducerLiteRuntime as _ProducerLiteRuntime,
)
from bots.producer._upstream import (
    plan_lite_waves as _producer_plan_lite_waves,
)

COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
Policy = Callable[[dict[str, Any]], list[list[float]]]


@dataclass(frozen=True)
class LaneIntent:
    source_planet_id: int
    target_planet_id: int
    fraction: float


@dataclass(frozen=True)
class OEPPlannerConfig:
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


CONFIG_4P = dataclasses.replace(
    OEPPlannerConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


def _config_for(player_count: int) -> OEPPlannerConfig:
    return CONFIG_4P if int(player_count) >= 4 else OEPPlannerConfig()


def _movement_config(config: OEPPlannerConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


@dataclass(frozen=True)
class OEPLiteConfig:
    """Experimental one-turn OEP overlay for the tracked Producer fixture."""

    base: OEPPlannerConfig = dataclasses.field(default_factory=OEPPlannerConfig)
    fractions: tuple[float, ...] = (0.5, 1.0)
    opponent_response: bool = True
    opponent_response_mode: str = "producer"
    producer_plan_mode: str = "policy"
    min_advantage: float = 0.0
    max_advantage: float = float("inf")
    max_sources_per_lane: int = 6
    max_offensive_targets: int = 6
    max_defensive_targets: int = 2
    max_waves_per_turn: int = 4
    enable_regroup: bool = True
    late_config_step: int = -1
    late_max_sources_per_lane: int = 0
    profile_stages: bool = False


def _effective_config(config: OEPPlannerConfig, *, step: int) -> OEPPlannerConfig:
    """Cap horizon at the next seeded-comet spawn boundary.

    The forward model intentionally does not predict future comet RNG. Existing
    comets remain modeled by ``PlanetMovement``; only scoring past a future
    spawn boundary is disabled.
    """
    horizon = int(config.horizon)
    future = [spawn - int(step) for spawn in COMET_SPAWN_STEPS if spawn > int(step)]
    if future:
        horizon = max(1, min(horizon, min(future)))
    return dataclasses.replace(config, horizon=horizon)


def _opponent_id(player_id: int, player_count: int) -> int | None:
    if int(player_count) != 2:
        return None
    return 1 - int(player_id)


def _with_player(obs: Any, player_id: int) -> Any:
    if not isinstance(obs, dict):
        return obs
    copied = dict(obs)
    copied["player"] = int(player_id)
    return copied


def _with_tensor_player(obs_tensors: dict, player_id: int) -> dict:
    copied = dict(obs_tensors)
    player = obs_tensors["player"]
    copied["player"] = torch.full_like(player, int(player_id))
    return copied


def _producer_config_from_oep(config: OEPPlannerConfig) -> ProducerLiteConfig:
    return ProducerLiteConfig(
        horizon=int(config.horizon),
        max_sources_per_lane=int(config.max_sources_per_lane),
        max_offensive_targets=int(config.max_offensive_targets),
        max_defensive_targets=int(config.max_defensive_targets),
        max_waves_per_turn=int(config.max_waves_per_turn),
        roi_threshold=float(config.roi_threshold),
        min_ships_to_launch=float(config.min_ships_to_launch),
        enable_regroup=bool(config.enable_regroup),
        max_regroup_time=float(config.max_regroup_time),
        regroup_pressure_delta_min=float(config.regroup_pressure_delta_min),
        max_regroup_sources_per_lane=int(config.max_regroup_sources_per_lane),
        max_regroup_targets_per_source=int(config.max_regroup_targets_per_source),
        regroup_pressure_norm=str(config.regroup_pressure_norm),
        regroup_time_penalty_weight=float(config.regroup_time_penalty_weight),
    )


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


def _to_list_observation(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return obs
    converted = dict(obs)
    converted["planets"] = [_planet_row(planet) for planet in obs.get("planets", [])]
    converted["initial_planets"] = [
        _planet_row(planet) for planet in obs.get("initial_planets", [])
    ]
    converted["fleets"] = [_fleet_row(fleet) for fleet in obs.get("fleets", [])]
    return converted


def _entries_from_moves(
    *,
    moves: list[list[float]],
    movement: PlanetMovement,
    obs_tensors: dict,
    player_id: int,
) -> LaunchEntries:
    count = len(moves)
    if count <= 0:
        return _empty_entries(movement.device, movement.dtype)
    from_ids = torch.tensor([int(move[0]) for move in moves], dtype=torch.long, device=movement.device)
    source_slots = torch.full((count,), -1, dtype=torch.long, device=movement.device)
    for idx in range(count):
        matches = torch.where(movement.planet_ids == from_ids[idx])[0]
        if int(matches.numel()) > 0:
            source_slots[idx] = matches[0]
        else:
            raise ValueError(f"policy emitted move from unknown planet id: {int(from_ids[idx].item())}")
    angle = torch.tensor([float(move[1]) for move in moves], dtype=movement.dtype, device=movement.device)
    ships = torch.tensor([float(move[2]) for move in moves], dtype=movement.dtype, device=movement.device)
    provisional = LaunchEntries(
        source_slots=source_slots,
        target_slots=torch.zeros(count, dtype=torch.long, device=movement.device),
        ships=ships,
        angle=angle,
        eta=torch.ones(count, dtype=movement.dtype, device=movement.device),
        valid=(source_slots >= 0) & (ships >= 1.0),
    )
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors,
        movement=movement,
        entries=provisional,
        player_id=int(player_id),
    )
    return LaunchEntries(
        source_slots=source_slots,
        target_slots=launches.target_slots,
        ships=ships,
        angle=angle,
        eta=launches.eta_turns.to(device=movement.device, dtype=movement.dtype),
        valid=provisional.valid & launches.valid,
    )


def _entries_from_sparse_row(
    *,
    row: dict[str, Tensor],
    movement: PlanetMovement,
    obs_tensors: dict,
    player_id: int,
) -> LaunchEntries:
    counts = int(row["counts"].item())
    if counts <= 0:
        return _empty_entries(movement.device, movement.dtype)
    from_ids = row["from_planet_id"][:counts].to(device=movement.device, dtype=torch.long)
    source_slots = torch.full((counts,), -1, dtype=torch.long, device=movement.device)
    for idx in range(counts):
        matches = torch.where(movement.planet_ids == from_ids[idx])[0]
        if int(matches.numel()) > 0:
            source_slots[idx] = matches[0]
        else:
            raise ValueError(f"policy emitted move from unknown planet id: {int(from_ids[idx].item())}")
    angle = row["angle"][:counts].to(device=movement.device, dtype=movement.dtype)
    ships = row["num_ships"][:counts].to(device=movement.device, dtype=movement.dtype)
    provisional = LaunchEntries(
        source_slots=source_slots,
        target_slots=torch.zeros(counts, dtype=torch.long, device=movement.device),
        ships=ships,
        angle=angle,
        eta=torch.ones(counts, dtype=movement.dtype, device=movement.device),
        valid=(source_slots >= 0) & (ships >= 1.0),
    )
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors,
        movement=movement,
        entries=provisional,
        player_id=int(player_id),
    )
    return LaunchEntries(
        source_slots=source_slots,
        target_slots=launches.target_slots,
        ships=ships,
        angle=angle,
        eta=launches.eta_turns.to(device=movement.device, dtype=movement.dtype),
        valid=provisional.valid & launches.valid,
    )


def _launch_set_from_entries(
    *,
    entries: LaunchEntries,
    owner_id: int,
    candidates: int,
) -> LaunchSet | None:
    valid_slots = torch.where(entries.valid & (entries.ships >= 1.0))[0]
    if int(valid_slots.numel()) == 0:
        return None
    src = entries.source_slots[valid_slots]
    tgt = entries.target_slots[valid_slots]
    ships = entries.ships[valid_slots]
    eta = entries.eta[valid_slots]
    valid = entries.valid[valid_slots]
    owner = torch.full_like(src, int(owner_id), dtype=torch.long)
    return LaunchSet(
        source_slots=src.view(1, -1).expand(candidates, -1),
        target_slots=tgt.view(1, -1).expand(candidates, -1),
        ships=ships.view(1, -1).expand(candidates, -1),
        eta=eta.view(1, -1).expand(candidates, -1),
        owner=owner.view(1, -1).expand(candidates, -1),
        valid=valid.view(1, -1).expand(candidates, -1),
    )


def _top_entries_by_ships(entries: LaunchEntries, *, max_entries: int) -> LaunchEntries:
    valid = entries.valid & (entries.ships >= 1.0)
    valid_slots = torch.where(valid)[0]
    if int(max_entries) <= 0 or int(valid_slots.numel()) <= int(max_entries):
        return entries
    keep_valid_slots = valid_slots[
        torch.argsort(entries.ships[valid_slots], descending=True)[: int(max_entries)]
    ]
    keep = torch.zeros_like(entries.valid)
    keep[keep_valid_slots] = True
    return LaunchEntries(
        source_slots=entries.source_slots,
        target_slots=entries.target_slots,
        ships=torch.where(keep, entries.ships, torch.zeros_like(entries.ships)),
        angle=entries.angle,
        eta=entries.eta,
        valid=entries.valid & keep,
    )


def _combine_launch_sets(primary: LaunchSet, extra: LaunchSet | None) -> LaunchSet:
    if extra is None:
        return primary
    return LaunchSet(
        source_slots=torch.cat([primary.source_slots, extra.source_slots], dim=-1),
        target_slots=torch.cat([primary.target_slots, extra.target_slots], dim=-1),
        ships=torch.cat([primary.ships, extra.ships], dim=-1),
        eta=torch.cat([primary.eta, extra.eta], dim=-1),
        owner=torch.cat([primary.owner, extra.owner], dim=-1),
        valid=torch.cat([primary.valid, extra.valid], dim=-1),
    )


def _score_launch_set(
    launch_set: LaunchSet,
    *,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    player_id: int,
) -> Tensor:
    diff = sparse_launch_flow_delta(
        status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launch_set,
        player_id=int(player_id),
    )
    net = diff.net_ship_delta
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp


def _plan_fitness(
    entries: LaunchEntries,
    *,
    opponent_launch_set: LaunchSet | None,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    player_id: int,
) -> float:
    own_launch_set = _launch_set_from_entries(
        entries=entries,
        owner_id=int(player_id),
        candidates=1,
    )
    if own_launch_set is None:
        return 0.0
    score = _score_launch_set(
        _combine_launch_sets(own_launch_set, opponent_launch_set),
        status=status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        player_id=int(player_id),
    )
    if opponent_launch_set is not None:
        opponent_baseline = _score_launch_set(
            opponent_launch_set,
            status=status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=int(player_count),
            player_id=int(player_id),
        )
        score = score - opponent_baseline
    return float(score.reshape(-1)[0].item())


def _planet_id_to_slot(movement: PlanetMovement) -> dict[int, int]:
    return {
        int(planet_id.item()): int(slot)
        for slot, planet_id in enumerate(movement.planet_ids)
    }


def _safe_drain_by_slot(*, status, obs, player_id: int) -> Tensor:
    if int(obs.P) == 0:
        return torch.zeros(0, dtype=obs.ships.dtype, device=obs.device)
    source_idx = torch.arange(int(obs.P), dtype=torch.long, device=obs.device)
    H = max(0, int(status.ships.shape[-1]) - 1)
    return safe_drain(
        status,
        source_idx=source_idx,
        source_ships=obs.ships.to(obs.ships.dtype),
        H_eff=torch.full((), float(H), dtype=obs.ships.dtype, device=obs.device),
        player_id=int(player_id),
    )


def _lane_intents_from_entries(
    entries: LaunchEntries,
    *,
    movement: PlanetMovement,
    status,
    obs,
    player_id: int,
) -> tuple[LaneIntent, ...]:
    valid_slots = torch.where(entries.valid & (entries.ships >= 1.0))[0]
    if int(valid_slots.numel()) == 0:
        return ()
    drain = _safe_drain_by_slot(status=status, obs=obs, player_id=int(player_id))
    lanes: list[LaneIntent] = []
    seen: set[tuple[int, int, float]] = set()
    for entry_idx in valid_slots.tolist():
        source_slot = int(entries.source_slots[entry_idx].item())
        target_slot = int(entries.target_slots[entry_idx].item())
        if source_slot < 0 or target_slot < 0 or source_slot >= obs.P or target_slot >= obs.P:
            continue
        denom = float(drain[source_slot].item()) if int(drain.shape[0]) > source_slot else 0.0
        if denom < 1.0:
            continue
        ships = float(entries.ships[entry_idx].item())
        fraction = max(0.05, min(1.0, ships / denom))
        source_id = int(movement.planet_ids[source_slot].item())
        target_id = int(movement.planet_ids[target_slot].item())
        key = (source_id, target_id, round(fraction, 3))
        if key in seen:
            continue
        seen.add(key)
        lanes.append(
            LaneIntent(
                source_planet_id=source_id,
                target_planet_id=target_id,
                fraction=fraction,
            )
        )
    return tuple(lanes)


def _entries_from_lane_intents(
    lanes: tuple[LaneIntent, ...],
    *,
    movement: PlanetMovement,
    obs,
    status,
    config: OEPPlannerConfig,
    player_id: int,
) -> LaunchEntries:
    if not lanes:
        return _empty_entries(obs.device, obs.ships.dtype)
    slot_by_id = _planet_id_to_slot(movement)
    source_slots: list[int] = []
    target_slots: list[int] = []
    fractions: list[float] = []
    for lane in lanes:
        source_slot = slot_by_id.get(int(lane.source_planet_id))
        target_slot = slot_by_id.get(int(lane.target_planet_id))
        if source_slot is None or target_slot is None:
            continue
        if source_slot == target_slot:
            continue
        if (
            not bool(obs.alive[source_slot])
            or int(obs.owner_abs[source_slot].item()) != int(player_id)
        ):
            continue
        if not bool(obs.alive[target_slot]):
            continue
        source_slots.append(source_slot)
        target_slots.append(target_slot)
        fractions.append(max(0.05, min(1.0, float(lane.fraction))))
    if not source_slots:
        return _empty_entries(obs.device, obs.ships.dtype)

    device = obs.device
    dtype = obs.ships.dtype
    src = torch.tensor(source_slots, dtype=torch.long, device=device)
    tgt = torch.tensor(target_slots, dtype=torch.long, device=device)
    frac = torch.tensor(fractions, dtype=dtype, device=device)
    drain = _safe_drain_by_slot(status=status, obs=obs, player_id=int(player_id))
    ships = (drain[src] * frac).floor()
    valid = ships >= max(1.0, float(config.min_ships_to_launch))
    if not bool(valid.any()):
        return _empty_entries(device, dtype)

    count = int(src.shape[0])
    eta_cap = torch.full((count,), float(config.horizon), dtype=dtype, device=device)
    pair_sizes = ships.view(count, 1, 1).expand(count, count, 1)
    active = reachable_mask(
        movement,
        source_idx=src,
        target_idx=tgt,
        fleet_sizes=pair_sizes,
        eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        src.view(count, 1),
        tgt.view(1, count),
        ships.view(count, 1).expand(count, count),
        active=active,
    )
    eta = aim["eta"].diagonal()
    angle = aim["angle"].diagonal()
    aimed_valid = aim["viable"].diagonal()
    valid = valid & aimed_valid & (eta <= eta_cap)

    K_eta = max(1, min(int(config.horizon), max(0, int(status.ships.shape[-1]) - 1)))
    floor = capture_floor(
        status,
        target_idx=tgt,
        k_max=K_eta,
        capture_overhead=1.0,
        player_id=int(player_id),
    )
    K = int(floor.shape[-1])
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        target_owned = obs.owner_abs[tgt] == float(player_id)
        valid = valid & (target_owned | (ships >= floor_at_arr))

    remaining = obs.ships.to(dtype).floor().clone()
    repaired_valid = torch.zeros_like(valid)
    for idx in range(int(src.shape[0])):
        if not bool(valid[idx]):
            continue
        source_slot = int(src[idx].item())
        send = ships[idx].floor()
        if float(send.item()) < 1.0 or send > remaining[source_slot]:
            continue
        ships[idx] = send
        remaining[source_slot] = (remaining[source_slot] - send).clamp(min=0.0)
        repaired_valid[idx] = True

    return LaunchEntries(
        source_slots=src,
        target_slots=tgt,
        ships=torch.where(repaired_valid, ships, torch.zeros_like(ships)),
        angle=torch.where(repaired_valid, angle, torch.zeros_like(angle)),
        eta=torch.where(repaired_valid, eta, torch.ones_like(eta)),
        valid=repaired_valid,
    )


def _cheap_opponent_entries(
    *,
    movement: PlanetMovement,
    obs,
    cache,
    status,
    config: OEPPlannerConfig,
    opponent_id: int,
) -> LaunchEntries:
    """Cheap 1-ply adversary model used only inside OEP fitness.

    This deliberately avoids calling the full Producer policy a second time.
    It models the opponent as sending a few high-garrison planets toward
    valuable non-owned targets, with legality and aiming still handled by the
    shared movement helpers.
    """
    P = int(obs.P)
    if P == 0:
        return _empty_entries(obs.device, obs.ships.dtype)
    source_mask = (
        obs.alive
        & (obs.owner_abs == float(opponent_id))
        & (obs.ships >= float(config.min_ships_to_launch))
    )
    target_mask = obs.alive & (obs.owner_abs != float(opponent_id))
    if not bool(source_mask.any()) or not bool(target_mask.any()):
        return _empty_entries(obs.device, obs.ships.dtype)

    source_cap = max(1, min(3, int(config.max_sources_per_lane), P))
    target_cap = max(1, min(5, int(config.max_offensive_targets), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, source_cap)
    target_value = (
        18.0 * (obs.owner_abs == float(obs.player_id)).to(obs.ships.dtype)
        + 5.0 * obs.prod.to(obs.ships.dtype)
        - 0.08 * obs.ships.to(obs.ships.dtype)
    )
    target_idx, target_exists = _candidate_indices(target_value, target_mask, target_cap)
    if not bool(source_exists.any()) or not bool(target_exists.any()):
        return _empty_entries(obs.device, obs.ships.dtype)

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    dist = cache.cross_dist[0][source_idx.clamp(0, P - 1)][:, target_idx.clamp(0, P - 1)]
    pair_score = (
        target_value[target_idx.clamp(0, P - 1)].view(1, T)
        - 0.04 * dist.to(obs.ships.dtype)
    )
    pair_score = torch.where(
        source_exists.view(S, 1) & target_exists.view(1, T),
        pair_score,
        torch.full_like(pair_score, float("-inf")),
    )
    flat_order = torch.argsort(pair_score.reshape(-1), descending=True)
    lanes: list[LaneIntent] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    max_lanes = max(1, min(2, int(config.max_waves_per_turn)))
    for flat in flat_order.tolist():
        if len(lanes) >= max_lanes:
            break
        if not torch.isfinite(pair_score.reshape(-1)[flat]):
            break
        source_short = flat // T
        target_short = flat % T
        source_slot = int(source_idx[source_short].item())
        target_slot = int(target_idx[target_short].item())
        if source_slot in used_sources or target_slot in used_targets:
            continue
        used_sources.add(source_slot)
        used_targets.add(target_slot)
        lanes.append(
            LaneIntent(
                source_planet_id=int(movement.planet_ids[source_slot].item()),
                target_planet_id=int(movement.planet_ids[target_slot].item()),
                fraction=1.0,
            )
        )
    return _entries_from_lane_intents(
        tuple(lanes),
        movement=movement,
        obs=obs,
        status=status,
        config=config,
        player_id=int(opponent_id),
    )


def _build_fraction_candidates(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: OEPPlannerConfig,
    fractions: tuple[float, ...],
    player_count: int,
    opponent_entries: LaunchEntries | None,
):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H_axis = int(status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return None

    source_idx, source_exists = _candidate_indices(
        obs.ships,
        source_mask,
        max(1, min(int(config.max_sources_per_lane), P)),
    )
    target_idx, target_exists = build_target_shortlist(
        obs,
        obs_tensors,
        status,
        cache,
        config=config,
        K_eta=K_eta,
        H=H,
        prod=prod,
        source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return None

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    G = len(fractions)
    fractions_t = torch.tensor(fractions, dtype=dtype, device=device)
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    drain = safe_drain(
        status,
        source_idx=source_idx,
        source_ships=source_ships,
        H_eff=torch.full((), float(H), dtype=dtype, device=device),
        player_id=pid,
    )
    sizes = (drain.view(S, 1, 1) * fractions_t.view(1, 1, G)).expand(S, T, G).floor()
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    floor = capture_floor(
        status,
        target_idx=target_idx,
        k_max=K_eta,
        capture_overhead=1.0,
        player_id=pid,
    )
    K = int(floor.shape[-1])
    active = reachable_mask(
        movement,
        source_idx=source_idx,
        target_idx=target_idx,
        fleet_sizes=sizes,
        eta_cap=eta_cap,
    )
    aim = intercept_angle(
        movement,
        source_idx.view(S, 1, 1),
        target_idx.view(1, T, 1),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T, 1))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.view(1, T, K).expand(S, T, K).gather(-1, k_arr)
        clears_floor = sizes >= floor_at_arr
    else:
        clears_floor = torch.ones_like(viable)
    valid = (
        viable
        & clears_floor
        & (sizes >= 1.0)
        & (source_idx.view(S, 1, 1) != target_idx.view(1, T, 1))
        & source_exists.view(S, 1, 1)
        & target_exists.view(1, T, 1)
    )

    C = S * T * G
    cand_src = source_idx.view(S, 1, 1).expand(S, T, G).reshape(C, 1)
    cand_tgt_slot = target_idx.view(1, T, 1).expand(S, T, G).reshape(C)
    cand_tgt_short = (
        torch.arange(T, device=device).view(1, T, 1).expand(S, T, G).reshape(C)
    )
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, 1)
    cand_angle = angle.reshape(C, 1)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, 1)
    cand_active = valid.reshape(C, 1)
    cand_valid = valid.reshape(C)

    own_launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    opp_launches = (
        _launch_set_from_entries(
            entries=opponent_entries,
            owner_id=_opponent_id(pid, player_count) if _opponent_id(pid, player_count) is not None else -1,
            candidates=C,
        )
        if opponent_entries is not None
        else None
    )
    score = _score_launch_set(
        _combine_launch_sets(own_launches, opp_launches),
        status=status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        player_id=pid,
    )
    if opp_launches is not None:
        baseline = _score_launch_set(
            opp_launches,
            status=status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=int(player_count),
            player_id=pid,
        )
        score = score - baseline
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    return {
        "P": P,
        "W": W,
        "device": device,
        "dtype": dtype,
        "score": score,
        "cand_src": cand_src,
        "cand_send": cand_send,
        "cand_angle": cand_angle,
        "cand_eta": cand_eta,
        "cand_active": cand_active,
        "cand_tgt_slot": cand_tgt_slot,
        "cand_tgt_short": cand_tgt_short,
        "cand_is_def": obs.owned[cand_tgt_slot.clamp(0, P - 1)],
        "source_budget": obs.ships.to(dtype).clone(),
        "target_exists": target_exists,
    }


def plan_oep_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: OEPPlannerConfig,
    fractions: tuple[float, ...],
    player_count: int,
    opponent_entries: LaunchEntries | None,
) -> LaunchEntries:
    built = _build_fraction_candidates(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        status=status,
        prod=prod,
        alive_by_step=alive_by_step,
        config=config,
        fractions=fractions,
        player_count=player_count,
        opponent_entries=opponent_entries,
    )
    if built is None:
        return _empty_entries(obs.device, obs.ships.dtype)
    wave_entries, leftover = _greedy_select(
        P=built["P"],
        W=built["W"],
        device=built["device"],
        dtype=built["dtype"],
        score=built["score"],
        cand_src=built["cand_src"],
        cand_send=built["cand_send"],
        cand_angle=built["cand_angle"],
        cand_eta=built["cand_eta"],
        cand_active=built["cand_active"],
        cand_tgt_slot=built["cand_tgt_slot"],
        cand_tgt_short=built["cand_tgt_short"],
        cand_is_def=built["cand_is_def"],
        source_budget=built["source_budget"],
        target_exists=built["target_exists"],
        roi_threshold=float(config.roi_threshold),
    )
    if not bool(config.enable_regroup):
        return wave_entries
    pressure = cheap_enemy_pressure(obs, cache, horizon=float(config.horizon), player_id=int(obs.player_id))
    from orbit_lite.planner_core import _plan_regroup

    regroup = _plan_regroup(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        garrison_status=status,
        leftover=leftover,
        original_ships=obs.ships.to(built["dtype"]),
        pressure=pressure,
        config=config,
        H=max(0, int(status.ships.shape[-1]) - 1),
    )
    return concat_launch_entries([wave_entries, regroup])


class OEPLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.last_lanes: tuple[LaneIntent, ...] = ()
        self.producer_runtimes: dict[int, _ProducerLiteRuntime] = {}
        self.producer_shared_runtime = _ProducerLiteRuntime()
        self.profile_totals: dict[str, float] = {}
        self.profile_counts: dict[str, int] = {}
        self.selection_counts: dict[str, int] = {}
        self.selection_delta_sum = 0.0
        self.selection_delta_min: float | None = None
        self.selection_delta_max: float | None = None
        self.selection_phase_counts: dict[str, int] = {}
        self.selection_phase_delta_sum: dict[str, float] = {}

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.last_lanes = ()
        self.producer_runtimes = {}
        self.producer_shared_runtime = _ProducerLiteRuntime()

    def reset_profile(self) -> None:
        self.profile_totals = {}
        self.profile_counts = {}
        self.selection_counts = {}
        self.selection_delta_sum = 0.0
        self.selection_delta_min = None
        self.selection_delta_max = None
        self.selection_phase_counts = {}
        self.selection_phase_delta_sum = {}

    def record_profile(self, name: str, elapsed: float) -> None:
        self.profile_totals[name] = self.profile_totals.get(name, 0.0) + float(elapsed)
        self.profile_counts[name] = self.profile_counts.get(name, 0) + 1

    def record_selection(
        self,
        *,
        step: int,
        chose_oep: bool,
        oep_fitness: float,
        producer_fitness: float,
        oep_entries: LaunchEntries,
        producer_entries: LaunchEntries,
    ) -> None:
        choice = "oep" if bool(chose_oep) else "producer"
        self.selection_counts[choice] = self.selection_counts.get(choice, 0) + 1
        if bool(oep_entries.valid.any().item()):
            self.selection_counts["oep_nonempty"] = self.selection_counts.get("oep_nonempty", 0) + 1
        if bool(producer_entries.valid.any().item()):
            self.selection_counts["producer_nonempty"] = self.selection_counts.get("producer_nonempty", 0) + 1
        delta = float(oep_fitness) - float(producer_fitness)
        self.selection_delta_sum += delta
        self.selection_delta_min = delta if self.selection_delta_min is None else min(self.selection_delta_min, delta)
        self.selection_delta_max = delta if self.selection_delta_max is None else max(self.selection_delta_max, delta)
        phase = self._selection_phase(int(step))
        phase_prefix = f"{phase}_{choice}"
        self.selection_phase_counts[phase] = self.selection_phase_counts.get(phase, 0) + 1
        self.selection_phase_counts[phase_prefix] = self.selection_phase_counts.get(phase_prefix, 0) + 1
        self.selection_phase_delta_sum[phase] = self.selection_phase_delta_sum.get(phase, 0.0) + delta

    @staticmethod
    def _selection_phase(step: int) -> str:
        if int(step) < 50:
            return "early"
        if int(step) < 150:
            return "mid"
        if int(step) < 350:
            return "late"
        return "endgame"

    def profile_summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "calls": float(self.profile_counts.get(name, 0)),
                "total_ms": 1000.0 * total,
                "mean_ms": 1000.0 * total / max(1, self.profile_counts.get(name, 0)),
            }
            for name, total in sorted(self.profile_totals.items())
        }

    def selection_summary(self) -> dict[str, float]:
        oep = float(self.selection_counts.get("oep", 0))
        producer = float(self.selection_counts.get("producer", 0))
        total = oep + producer
        summary = {
            "decisions": total,
            "oep_choices": oep,
            "producer_choices": producer,
            "oep_choice_rate": oep / max(1.0, total),
            "producer_choice_rate": producer / max(1.0, total),
            "oep_nonempty_rate": float(self.selection_counts.get("oep_nonempty", 0)) / max(1.0, total),
            "producer_nonempty_rate": float(self.selection_counts.get("producer_nonempty", 0)) / max(1.0, total),
            "mean_fitness_delta_oep_minus_producer": self.selection_delta_sum / max(1.0, total),
            "min_fitness_delta_oep_minus_producer": (
                0.0 if self.selection_delta_min is None else float(self.selection_delta_min)
            ),
            "max_fitness_delta_oep_minus_producer": (
                0.0 if self.selection_delta_max is None else float(self.selection_delta_max)
            ),
        }
        for phase in ("early", "mid", "late", "endgame"):
            phase_decisions = float(self.selection_phase_counts.get(phase, 0))
            phase_oep = float(self.selection_phase_counts.get(f"{phase}_oep", 0))
            phase_producer = float(self.selection_phase_counts.get(f"{phase}_producer", 0))
            summary[f"{phase}_decisions"] = phase_decisions
            summary[f"{phase}_oep_choice_rate"] = phase_oep / max(1.0, phase_decisions)
            summary[f"{phase}_producer_choice_rate"] = phase_producer / max(1.0, phase_decisions)
            summary[f"{phase}_mean_fitness_delta_oep_minus_producer"] = (
                float(self.selection_phase_delta_sum.get(phase, 0.0)) / max(1.0, phase_decisions)
            )
        return summary


class OEPLiteRuntime:
    def __init__(
        self,
        *,
        seed_policy: Policy,
        opponent_policy: Policy | None = None,
        config: OEPLiteConfig | None = None,
        memory: OEPLiteMemory | None = None,
    ) -> None:
        self.seed_policy = seed_policy
        self.opponent_policy = opponent_policy if opponent_policy is not None else seed_policy
        self.config = config if config is not None else OEPLiteConfig()
        self.memory = memory if memory is not None else OEPLiteMemory()

    def reset(self) -> None:
        self.memory.reset()
        self.memory.reset_profile()

    def _profile_start(self) -> float | None:
        return perf_counter() if bool(self.config.profile_stages) else None

    def _profile_record(self, name: str, start: float | None) -> None:
        if start is not None:
            self.memory.record_profile(name, perf_counter() - start)

    def profile_summary(self) -> dict[str, dict[str, float]]:
        return self.memory.profile_summary()

    def selection_summary(self) -> dict[str, float]:
        return self.memory.selection_summary()

    def _oep_config(self, base_config: OEPPlannerConfig, *, step: int) -> OEPPlannerConfig:
        max_sources_per_lane = int(self.config.max_sources_per_lane)
        if (
            int(self.config.late_config_step) >= 0
            and int(step) >= int(self.config.late_config_step)
            and int(self.config.late_max_sources_per_lane) > 0
        ):
            max_sources_per_lane = min(
                max_sources_per_lane,
                int(self.config.late_max_sources_per_lane),
            )
        return dataclasses.replace(
            base_config,
            max_sources_per_lane=min(
                int(base_config.max_sources_per_lane),
                max_sources_per_lane,
            ),
            max_offensive_targets=min(
                int(base_config.max_offensive_targets),
                int(self.config.max_offensive_targets),
            ),
            max_defensive_targets=min(
                int(base_config.max_defensive_targets),
                int(self.config.max_defensive_targets),
            ),
            max_waves_per_turn=min(
                int(base_config.max_waves_per_turn),
                int(self.config.max_waves_per_turn),
            ),
            enable_regroup=bool(base_config.enable_regroup) and bool(self.config.enable_regroup),
        )

    def _producer_entries_inline(
        self,
        *,
        owner_id: int,
        obs_tensors: dict,
        movement: PlanetMovement,
        cache,
        status,
        alive_by_step: Tensor,
        base_config: OEPPlannerConfig,
        player_count: int,
    ) -> LaunchEntries:
        owner_tensors = _with_tensor_player(obs_tensors, int(owner_id))
        obs = parse_obs(owner_tensors, player_id=int(owner_id))
        entries = _producer_plan_lite_waves(
            movement=movement,
            obs=obs,
            obs_tensors=owner_tensors,
            cache=cache,
            garrison_status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            config=_producer_config_from_oep(base_config),
            player_count=int(player_count),
        )
        return disambiguate_duplicate_launches(entries)

    def _producer_entries_tensor(
        self,
        *,
        owner_id: int,
        obs_tensors: dict,
        movement: PlanetMovement,
    ) -> LaunchEntries:
        runtime = self.memory.producer_runtimes.get(int(owner_id))
        if runtime is None:
            runtime = _ProducerLiteRuntime()
            self.memory.producer_runtimes[int(owner_id)] = runtime
        with torch.no_grad():
            row = runtime.tensor_action(_with_tensor_player(obs_tensors, int(owner_id)))
        return _entries_from_sparse_row(
            row=row,
            movement=movement,
            obs_tensors=obs_tensors,
            player_id=int(owner_id),
        )

    def _producer_entries_shared_tensor(
        self,
        *,
        owner_id: int,
        obs_tensors: dict,
        movement: PlanetMovement,
    ) -> LaunchEntries:
        with torch.no_grad():
            row = self.memory.producer_shared_runtime.tensor_action(
                _with_tensor_player(obs_tensors, int(owner_id))
            )
        return _entries_from_sparse_row(
            row=row,
            movement=movement,
            obs_tensors=obs_tensors,
            player_id=int(owner_id),
        )

    def _seed_shared_tensor_memory_from_entries(
        self,
        *,
        entries: LaunchEntries,
        obs_tensors: dict,
    ) -> None:
        runtime = self.memory.producer_shared_runtime
        mem = runtime.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        mem.last_sparse_action_row = entries_to_sparse_payload(
            entries,
            planet_ids=obs_tensors["planets"][..., 0].long(),
        )

    def _sync_shared_tensor_movement_from_entries(
        self,
        *,
        entries: LaunchEntries,
        obs_tensors: dict,
        base_config: OEPPlannerConfig,
        player_count: int,
        owner_id: int,
    ) -> None:
        runtime = self.memory.producer_shared_runtime
        mem = runtime.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.movement = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        movement = ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=_movement_config(base_config, player_count=int(player_count)),
            cached_movement=mem.movement,
        )
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors,
            movement=movement,
            entries=entries,
            player_id=int(owner_id),
        )
        apply_private_planned_launches(
            movement=movement,
            launches=launches,
            owner_id=int(owner_id),
            obs_tensors=obs_tensors,
        )
        mem.movement = movement
        mem.last_sparse_action_row = entries_to_sparse_payload(
            entries,
            planet_ids=obs_tensors["planets"][..., 0].long(),
        )

    def tensor_action(self, obs_tensors: dict, raw_obs: Any | None = None):
        action_start = self._profile_start()
        mem = self.memory
        if raw_obs is None:
            raise ValueError("OEPLiteRuntime.tensor_action requires raw_obs for policy injection")
        if bool((obs_tensors["step"] == 0).all()):
            mem.reset()
            mem.reset_profile()
            action_start = self._profile_start()
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        player_count = int(mem.cached_player_count)
        base_config = _effective_config(_config_for(player_count), step=int(obs_tensors["step"].item()))
        device = obs_tensors["planets"].device
        obs = parse_obs(obs_tensors)
        if obs.P == 0:
            row = empty_action_row(device)
            self._profile_record("action_total", action_start)
            return row

        stage_start = self._profile_start()
        movement = ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=_movement_config(base_config, player_count=player_count),
            cached_movement=mem.movement,
        )
        self._profile_record("ensure_planet_movement", stage_start)
        mem.movement = movement

        stage_start = self._profile_start()
        cache = build_distance_cache(movement, max_k=int(base_config.horizon))
        self._profile_record("build_distance_cache", stage_start)

        stage_start = self._profile_start()
        H = int(base_config.horizon)
        status = movement.garrison_status(max_horizon=H)
        alive_by_step = movement.alive_by_step[: H + 1]
        self._profile_record("garrison_status", stage_start)

        producer_plan_mode = str(self.config.producer_plan_mode)
        if producer_plan_mode == "inline":
            stage_start = self._profile_start()
            producer_entries = self._producer_entries_inline(
                owner_id=int(obs.player_id),
                obs_tensors=obs_tensors,
                movement=movement,
                cache=cache,
                status=status,
                alive_by_step=alive_by_step,
                base_config=base_config,
                player_count=player_count,
            )
            self._profile_record("producer_seed_inline", stage_start)
        elif producer_plan_mode == "policy":
            stage_start = self._profile_start()
            producer_moves = self.seed_policy(_with_player(raw_obs, int(obs.player_id)))
            self._profile_record("producer_seed_policy", stage_start)

            stage_start = self._profile_start()
            producer_entries = _entries_from_moves(
                moves=producer_moves,
                movement=movement,
                obs_tensors=obs_tensors,
                player_id=int(obs.player_id),
            )
            self._profile_record("producer_entries", stage_start)
            if str(self.config.opponent_response_mode) == "producer_shared_tensor":
                stage_start = self._profile_start()
                self._producer_entries_shared_tensor(
                    owner_id=int(obs.player_id),
                    obs_tensors=obs_tensors,
                    movement=movement,
                )
                self._profile_record("producer_seed_shared_tensor_shadow", stage_start)
            elif str(self.config.opponent_response_mode) == "producer_seeded_shared_tensor":
                stage_start = self._profile_start()
                self._seed_shared_tensor_memory_from_entries(
                    entries=producer_entries,
                    obs_tensors=obs_tensors,
                )
                self._profile_record("producer_seed_shared_tensor_memory", stage_start)
            elif str(self.config.opponent_response_mode) == "producer_synced_shared_tensor":
                stage_start = self._profile_start()
                self._sync_shared_tensor_movement_from_entries(
                    entries=producer_entries,
                    obs_tensors=obs_tensors,
                    base_config=base_config,
                    player_count=player_count,
                    owner_id=int(obs.player_id),
                )
                self._profile_record("producer_seed_shared_tensor_sync", stage_start)
        elif producer_plan_mode == "tensor":
            stage_start = self._profile_start()
            producer_entries = self._producer_entries_tensor(
                owner_id=int(obs.player_id),
                obs_tensors=obs_tensors,
                movement=movement,
            )
            self._profile_record("producer_seed_tensor", stage_start)
        else:
            raise ValueError(f"unknown OEP producer_plan_mode: {producer_plan_mode!r}")
        opponent_entries = None
        opp_id = _opponent_id(int(obs.player_id), player_count)
        if (
            bool(self.config.opponent_response)
            and opp_id is not None
        ):
            mode = str(self.config.opponent_response_mode)
            if mode == "producer":
                if producer_plan_mode == "inline":
                    stage_start = self._profile_start()
                    opponent_entries = self._producer_entries_inline(
                        owner_id=opp_id,
                        obs_tensors=obs_tensors,
                        movement=movement,
                        cache=cache,
                        status=status,
                        alive_by_step=alive_by_step,
                        base_config=base_config,
                        player_count=player_count,
                    )
                    self._profile_record("producer_opponent_inline", stage_start)
                elif producer_plan_mode == "tensor":
                    stage_start = self._profile_start()
                    opponent_entries = self._producer_entries_tensor(
                        owner_id=opp_id,
                        obs_tensors=obs_tensors,
                        movement=movement,
                    )
                    self._profile_record("producer_opponent_tensor", stage_start)
                else:
                    stage_start = self._profile_start()
                    opponent_moves = self.opponent_policy(_with_player(raw_obs, opp_id))
                    self._profile_record("producer_opponent_policy", stage_start)

                    stage_start = self._profile_start()
                    opponent_entries = _entries_from_moves(
                        moves=opponent_moves,
                        movement=movement,
                        obs_tensors=obs_tensors,
                        player_id=opp_id,
                    )
                    self._profile_record("opponent_entries", stage_start)
            elif mode == "producer_inline":
                stage_start = self._profile_start()
                opponent_entries = self._producer_entries_inline(
                    owner_id=opp_id,
                    obs_tensors=obs_tensors,
                    movement=movement,
                    cache=cache,
                    status=status,
                    alive_by_step=alive_by_step,
                    base_config=base_config,
                    player_count=player_count,
                )
                self._profile_record("producer_opponent_inline", stage_start)
            elif mode == "producer_inline_top3":
                stage_start = self._profile_start()
                opponent_entries = _top_entries_by_ships(
                    self._producer_entries_inline(
                        owner_id=opp_id,
                        obs_tensors=obs_tensors,
                        movement=movement,
                        cache=cache,
                        status=status,
                        alive_by_step=alive_by_step,
                        base_config=base_config,
                        player_count=player_count,
                    ),
                    max_entries=3,
                )
                self._profile_record("producer_opponent_inline_top3", stage_start)
            elif mode == "producer_tensor":
                stage_start = self._profile_start()
                opponent_entries = self._producer_entries_tensor(
                    owner_id=opp_id,
                    obs_tensors=obs_tensors,
                    movement=movement,
                )
                self._profile_record("producer_opponent_tensor", stage_start)
            elif mode == "producer_tensor_top3":
                stage_start = self._profile_start()
                opponent_entries = _top_entries_by_ships(
                    self._producer_entries_tensor(
                        owner_id=opp_id,
                        obs_tensors=obs_tensors,
                        movement=movement,
                    ),
                    max_entries=3,
                )
                self._profile_record("producer_opponent_tensor_top3", stage_start)
            elif mode == "producer_shared_tensor":
                stage_start = self._profile_start()
                opponent_entries = self._producer_entries_shared_tensor(
                    owner_id=opp_id,
                    obs_tensors=obs_tensors,
                    movement=movement,
                )
                self._profile_record("producer_opponent_shared_tensor", stage_start)
            elif mode == "producer_seeded_shared_tensor":
                stage_start = self._profile_start()
                opponent_entries = self._producer_entries_shared_tensor(
                    owner_id=opp_id,
                    obs_tensors=obs_tensors,
                    movement=movement,
                )
                self._profile_record("producer_opponent_seeded_shared_tensor", stage_start)
            elif mode == "producer_synced_shared_tensor":
                stage_start = self._profile_start()
                opponent_entries = self._producer_entries_shared_tensor(
                    owner_id=opp_id,
                    obs_tensors=obs_tensors,
                    movement=movement,
                )
                self._profile_record("producer_opponent_synced_shared_tensor", stage_start)
            elif mode == "cheap":
                stage_start = self._profile_start()
                opponent_entries = _cheap_opponent_entries(
                    movement=movement,
                    obs=obs,
                    cache=cache,
                    status=status,
                    config=base_config,
                    opponent_id=opp_id,
                )
                self._profile_record("cheap_opponent_entries", stage_start)
            elif mode == "none":
                opponent_entries = None
            else:
                raise ValueError(f"unknown OEP opponent_response_mode: {mode!r}")

        stage_start = self._profile_start()
        opponent_launch_set = (
            _launch_set_from_entries(
                entries=opponent_entries,
                owner_id=int(opp_id),
                candidates=1,
            )
            if opponent_entries is not None and opp_id is not None
            else None
        )
        self._profile_record("opponent_launch_set", stage_start)
        oep_config = self._oep_config(base_config, step=int(obs_tensors["step"].item()))

        stage_start = self._profile_start()
        oep_entries = plan_oep_waves(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            config=oep_config,
            fractions=self.config.fractions,
            player_count=player_count,
            opponent_entries=opponent_entries,
        )
        self._profile_record("plan_oep_waves", stage_start)

        stage_start = self._profile_start()
        producer_fitness = _plan_fitness(
            producer_entries,
            opponent_launch_set=opponent_launch_set,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            player_id=int(obs.player_id),
        )
        self._profile_record("fitness_producer", stage_start)

        stage_start = self._profile_start()
        oep_fitness = _plan_fitness(
            oep_entries,
            opponent_launch_set=opponent_launch_set,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            player_id=int(obs.player_id),
        )
        self._profile_record("fitness_oep", stage_start)
        _advantage = oep_fitness - producer_fitness
        chose_oep = (
            _advantage > float(self.config.min_advantage)
            and _advantage < float(self.config.max_advantage)
        )
        if bool(self.config.profile_stages):
            mem.record_selection(
                step=int(obs_tensors["step"].item()),
                chose_oep=bool(chose_oep),
                oep_fitness=float(oep_fitness),
                producer_fitness=float(producer_fitness),
                oep_entries=oep_entries,
                producer_entries=producer_entries,
            )
        chosen = oep_entries if chose_oep else producer_entries

        chosen = disambiguate_duplicate_launches(chosen)

        stage_start = self._profile_start()
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors,
            movement=movement,
            entries=chosen,
            player_id=int(obs.player_id),
        )
        self._profile_record("chosen_launch_inference", stage_start)

        stage_start = self._profile_start()
        apply_private_planned_launches(
            movement=movement,
            launches=launches,
            owner_id=int(obs.player_id),
            obs_tensors=obs_tensors,
        )
        self._profile_record("apply_private_launches", stage_start)

        stage_start = self._profile_start()
        row = entries_to_sparse_payload(chosen, planet_ids=obs_tensors["planets"][..., 0].long())
        self._profile_record("entries_to_payload", stage_start)
        mem.last_sparse_action_row = row
        self._profile_record("action_total", action_start)
        return row

    def act(self, obs):
        obs = _to_list_observation(obs)
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        with torch.no_grad():
            sparse_row = self.tensor_action(obs_tensors, raw_obs=obs)
        return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

def _env_config() -> OEPLiteConfig:
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return int(default)
        return int(raw)

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return float(default)
        return float(raw)

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return bool(default)
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean value")

    def _env_fractions(default: tuple[float, ...]) -> tuple[float, ...]:
        raw = os.getenv("OEP_FRACTIONS")
        if raw is None or raw.strip() == "":
            return default
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
        if not values:
            raise ValueError("OEP_FRACTIONS must contain at least one float")
        return values

    defaults = OEPLiteConfig()
    return OEPLiteConfig(
        opponent_response_mode=os.getenv(
            "OEP_OPPONENT_RESPONSE_MODE",
            defaults.opponent_response_mode,
        ),
        producer_plan_mode=os.getenv(
            "OEP_PRODUCER_PLAN_MODE",
            defaults.producer_plan_mode,
        ),
        fractions=_env_fractions(defaults.fractions),
        min_advantage=_env_float("OEP_MIN_ADVANTAGE", defaults.min_advantage),
        max_advantage=_env_float("OEP_MAX_ADVANTAGE", defaults.max_advantage),
        max_sources_per_lane=_env_int(
            "OEP_MAX_SOURCES_PER_LANE",
            defaults.max_sources_per_lane,
        ),
        max_offensive_targets=_env_int(
            "OEP_MAX_OFFENSIVE_TARGETS",
            defaults.max_offensive_targets,
        ),
        max_defensive_targets=_env_int(
            "OEP_MAX_DEFENSIVE_TARGETS",
            defaults.max_defensive_targets,
        ),
        max_waves_per_turn=_env_int(
            "OEP_MAX_WAVES_PER_TURN",
            defaults.max_waves_per_turn,
        ),
        enable_regroup=_env_bool("OEP_ENABLE_REGROUP", defaults.enable_regroup),
        late_config_step=_env_int("OEP_LATE_CONFIG_STEP", defaults.late_config_step),
        late_max_sources_per_lane=_env_int(
            "OEP_LATE_MAX_SOURCES_PER_LANE",
            defaults.late_max_sources_per_lane,
        ),
    )


def _load_private_producer_policy(name: str) -> Policy:
    module_path = Path(__file__).resolve().parents[1] / "producer" / "agent.py"
    spec = importlib.util.spec_from_file_location(f"_oep_private_producer_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load private Producer policy: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = getattr(module, "agent", None)
    if not callable(policy):
        raise ImportError(f"private Producer policy at {module_path} does not define agent(obs)")
    return policy


_RUNTIME = OEPLiteRuntime(
    seed_policy=_load_private_producer_policy("seed"),
    opponent_policy=_load_private_producer_policy("opponent"),
    config=_env_config(),
)


def agent(obs):
    return _RUNTIME.act(obs)
