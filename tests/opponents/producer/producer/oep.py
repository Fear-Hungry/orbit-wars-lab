from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
from main import (
    ProducerLiteConfig,
    ProducerLiteMemory,
    _config_for,
    _movement_config,
    cheap_enemy_pressure,
    plan_lite_waves,
    run_turn,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.garrison_launch import LaunchSet, sparse_launch_flow_delta
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import PlanetMovement
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

COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


@dataclass(frozen=True)
class OEPLiteConfig:
    """Experimental one-turn OEP overlay for the tracked Producer fixture."""

    base: ProducerLiteConfig = dataclasses.field(default_factory=ProducerLiteConfig)
    fractions: tuple[float, ...] = (0.5, 1.0)
    opponent_response: bool = True
    min_advantage: float = 0.0
    max_sources_per_lane: int = 6
    max_offensive_targets: int = 6
    max_defensive_targets: int = 2
    max_waves_per_turn: int = 4


def _effective_config(config: ProducerLiteConfig, *, step: int) -> ProducerLiteConfig:
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


def _entries_from_sparse_payload(
    *,
    payload: dict[str, Tensor],
    movement: PlanetMovement,
    obs_tensors: dict,
    player_id: int,
) -> LaunchEntries:
    count = int(payload["counts"].item())
    if count <= 0:
        return _empty_entries(movement.device, movement.dtype)
    from_ids = payload["from_planet_id"][:count].to(device=movement.device, dtype=torch.long)
    source_slots = torch.full((count,), -1, dtype=torch.long, device=movement.device)
    for idx in range(count):
        matches = torch.where(movement.planet_ids == from_ids[idx])[0]
        if int(matches.numel()) > 0:
            source_slots[idx] = matches[0]
    angle = payload["angle"][:count].to(device=movement.device, dtype=movement.dtype)
    ships = payload["num_ships"][:count].to(device=movement.device, dtype=movement.dtype)
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


def _build_fraction_candidates(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
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
    config: ProducerLiteConfig,
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
        self.producer_memory = ProducerLiteMemory()
        self.opponent_memory = ProducerLiteMemory()
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.producer_memory.reset()
        self.opponent_memory.reset()
        self.cached_player_count = None
        self.last_sparse_action_row = None


class OEPLiteRuntime:
    def __init__(self, config: OEPLiteConfig | None = None, memory: OEPLiteMemory | None = None) -> None:
        self.config = config if config is not None else OEPLiteConfig()
        self.memory = memory if memory is not None else OEPLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def _oep_config(self, base_config: ProducerLiteConfig) -> ProducerLiteConfig:
        return dataclasses.replace(
            base_config,
            max_sources_per_lane=min(
                int(base_config.max_sources_per_lane),
                int(self.config.max_sources_per_lane),
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
        )

    def tensor_action(self, obs_tensors: dict, raw_obs: Any | None = None):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.reset()
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        player_count = int(mem.cached_player_count)
        base_config = _effective_config(_config_for(player_count), step=int(obs_tensors["step"].item()))
        device = obs_tensors["planets"].device
        obs = parse_obs(obs_tensors)
        if obs.P == 0:
            return empty_action_row(device)

        movement = ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=_movement_config(base_config, player_count=player_count),
            cached_movement=mem.movement,
        )
        mem.movement = movement
        cache = build_distance_cache(movement, max_k=int(base_config.horizon))
        H = int(base_config.horizon)
        status = movement.garrison_status(max_horizon=H)
        alive_by_step = movement.alive_by_step[: H + 1]

        producer_entries = plan_lite_waves(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            config=base_config,
            player_count=player_count,
        )
        opponent_entries = None
        opp_id = _opponent_id(int(obs.player_id), player_count)
        if bool(self.config.opponent_response) and raw_obs is not None and opp_id is not None:
            opp_tensors = single_obs_to_tensor(_with_player(raw_obs, opp_id), player_id=opp_id)
            opp_row = run_turn(
                opp_tensors,
                config=base_config,
                player_count=player_count,
                memory=mem.opponent_memory,
            )
            opponent_entries = _entries_from_sparse_payload(
                payload=opp_row,
                movement=movement,
                obs_tensors=obs_tensors,
                player_id=opp_id,
            )
        elif bool(self.config.opponent_response) and opp_id is not None:
            raise ValueError("OEPLiteRuntime.tensor_action requires raw_obs for opponent response")

        opponent_launch_set = (
            _launch_set_from_entries(
                entries=opponent_entries,
                owner_id=int(opp_id),
                candidates=1,
            )
            if opponent_entries is not None and opp_id is not None
            else None
        )
        oep_entries = plan_oep_waves(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            config=self._oep_config(base_config),
            fractions=self.config.fractions,
            player_count=player_count,
            opponent_entries=opponent_entries,
        )
        producer_fitness = _plan_fitness(
            producer_entries,
            opponent_launch_set=opponent_launch_set,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            player_id=int(obs.player_id),
        )
        oep_fitness = _plan_fitness(
            oep_entries,
            opponent_launch_set=opponent_launch_set,
            status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            player_id=int(obs.player_id),
        )
        chosen = (
            oep_entries
            if oep_fitness > producer_fitness + float(self.config.min_advantage)
            else producer_entries
        )

        chosen = disambiguate_duplicate_launches(chosen)
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors,
            movement=movement,
            entries=chosen,
            player_id=int(obs.player_id),
        )
        apply_private_planned_launches(
            movement=movement,
            launches=launches,
            owner_id=int(obs.player_id),
            obs_tensors=obs_tensors,
        )
        row = entries_to_sparse_payload(chosen, planet_ids=obs_tensors["planets"][..., 0].long())
        mem.last_sparse_action_row = row
        return row


_RUNTIME = OEPLiteRuntime()


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors, raw_obs=obs)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
