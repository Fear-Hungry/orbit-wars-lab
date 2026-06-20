from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import torch
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
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
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from torch import Tensor


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs."""

    # the projection window, the movement build length, AND the target ETA cap
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12  # enemy/neutral proximity targets
    max_defensive_targets: int = 4
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5  # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    # --- G3.1a opening capture-and-HOLD (4p only) ------------------------
    # During the opening, require a neutral capture's wave to clear defenders AND
    # survive projected enemy counter-pressure (cheap_enemy_pressure * margin),
    # via capture_floor's reinforcement hook. If unaffordable, the contestable
    # capture is gated out (we don't grab neutrals we'd immediately lose).
    # Data: bad 4p openings hold only 71% of opening captures vs 100% in wins
    # (docs/LOSS_TAXONOMY.md + opening_detector). 0.0 == today's behaviour.
    opening_hold_margin: float = 0.0
    opening_hold_until_step: int = 50
    # Rank-19-style THREAT-AWARE targeting (general, all-game): penalize a capture
    # whose target the enemy can reinforce after our launch, by inflating the capture
    # floor by cheap_enemy_pressure * margin for ALL enemy/neutral targets every step
    # (the public top agent's `β·ρ(eta)·enemy_mass` ROI term, generalized from the
    # opening-only G3.1a hook). 0.0 == today's behaviour (byte-identical).
    reactive_reinforce_margin: float = 0.0
    # ρ(eta) reaction ramp for the reactive margin: enemy needs >= eta_free turns of
    # our fleet's flight to react, then reaction likelihood ramps to 1 over eta_scale.
    # So fast/near captures aren't over-penalized (faithful to the rank-19 ρ(eta) term).
    reactive_reinforce_eta_free: float = 3.0
    reactive_reinforce_eta_scale: float = 12.0
    # G3.1a literal Phase-2 rule: during the opening, suppress attacks on enemy
    # PLAYERS while the source still has a viable neutral capture in range
    # ("deixar os outros se desgastarem"). Off by default. Phase-1 predicts this
    # does not help (within losses, more early PvP correlates with a better
    # state) — implemented to execute the goal as written and gate it at 96 seeds.
    opening_suppress_pvp: bool = False
    # A/B harness for the "drenagem dupla" fix (commit 8459f7b). When True,
    # plan_lite_waves passes ``source_spend_budget=None`` to _greedy_select, which
    # the planner_core docstring documents reproduces the OLD single-budget
    # behaviour EXACTLY (a source could fund several waves summing past its safe
    # drain). This is the faithful pre-fix INCUMBENT used by the seat-rotated
    # promotion gate (scripts/league_submit_ruler.py via the pgs_hold_prefix
    # agent). Per-instance config only — never a global env — so candidate and
    # incumbent can play head-to-head in the same process without contamination.
    # False (default) == shipped fix, byte-identical to current behaviour.
    disable_drain_fix: bool = False


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)  # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))  # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)  # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye  # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)  # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)  # [P] summed over sources


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs,
        obs_tensors,
        garrison_status,
        cache,
        config=config,
        K_eta=K_eta,
        H=H,
        prod=prod,
        source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]  # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)  # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status,
        source_idx=source_idx,
        source_ships=source_ships,
        H_eff=H_eff,
        player_id=pid,
    )  # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)  # [T]

    # G3.1a: opening capture-and-hold margin. For NEUTRAL targets during the
    # opening, inflate the capture floor by projected enemy counter-pressure so a
    # contestable neutral we can't capture-and-hold is gated out of `clears_floor`.
    reinforcement = None
    rrm = float(getattr(config, "reactive_reinforce_margin", 0.0))
    if rrm > 0.0:
        # Rank-19 threat-aware targeting (general): inflate the capture floor by the
        # enemy mass that can reach the target, for EVERY enemy/neutral target every
        # step, so contestable captures the enemy can over-reinforce are gated out.
        pressure = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
        tgt_c = target_idx.clamp(0, P - 1)
        attackable = obs.is_neutral[tgt_c] | obs.is_enemy[tgt_c]
        margin_t = torch.where(
            attackable,
            pressure[tgt_c] * rrm,
            torch.zeros((), dtype=dtype, device=device).expand(T),
        )  # [T]
        # ρ(k): the enemy can only reinforce by arrival-step k with enough flight time,
        # so don't over-penalize fast/near captures (the rank-19 ρ(eta) timing ramp).
        ks = torch.arange(1, K_eta + 1, dtype=dtype, device=device)  # [K_eta], 1-indexed (rank-19)
        rho_k = reinforcement_timing_factor(
            ks,
            eta_free=float(config.reactive_reinforce_eta_free),
            eta_scale=float(config.reactive_reinforce_eta_scale),
        )  # [K_eta]
        reinforcement = (margin_t.view(T, 1) * rho_k.view(1, K_eta)).contiguous()  # [T, K_eta]
    elif float(config.opening_hold_margin) > 0.0:
        cur_step = float(obs.step.flatten()[0])
        if cur_step < float(config.opening_hold_until_step):
            pressure = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
            tgt_c = target_idx.clamp(0, P - 1)
            margin_t = torch.where(
                obs.is_neutral[tgt_c],
                pressure[tgt_c] * float(config.opening_hold_margin),
                torch.zeros((), dtype=dtype, device=device).expand(T),
            )  # [T]
            reinforcement = margin_t.view(T, 1).expand(T, K_eta).contiguous()  # [T, K_eta]

    floor = capture_floor(
        garrison_status,
        target_idx=target_idx,
        k_max=K_eta,
        capture_overhead=1.0,
        player_id=pid,
        reinforcement=reinforcement,
    )  # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()  # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement,
        source_idx=source_idx,
        target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)  # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),  # [S, 1]
        target_idx.unsqueeze(0),  # [1, T]
        sizes,  # [S, T]
        active=active,
    )
    angle = aim["angle"]  # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = (
            floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        )
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr  # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable
        & clears_floor
        & (sizes >= 1.0)
        & src_neq_tgt
        & source_exists.view(S, 1)
        & target_exists.view(1, T)
    )  # [S, T]

    # G3.1a literal Phase-2 rule: in the opening, drop a source's PvP candidates
    # (enemy-PLAYER targets) while that source still has a viable neutral capture.
    if bool(config.opening_suppress_pvp) and float(obs.step.flatten()[0]) < float(config.opening_hold_until_step):
        tgt_c = target_idx.clamp(0, P - 1)
        is_enemy_t = obs.is_enemy[tgt_c].view(1, T)        # enemy-player target
        is_neutral_t = obs.is_neutral[tgt_c].view(1, T)    # neutral (capture) target
        has_neutral = (valid & is_neutral_t).any(dim=1, keepdim=True)  # [S,1]
        valid = valid & ~(is_enemy_t & has_neutral)

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]  # [C]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=pid,
    )  # [C]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # Offensive spend budget per planet = scattered safe_drain (see "drenagem
    # dupla", todo.md 2026-06-11). Padded shortlist slots contribute zero.
    spend_budget = torch.zeros(P, dtype=dtype, device=device)
    spend_budget.scatter_add_(
        0,
        source_idx.clamp(0, P - 1).to(torch.long),
        torch.where(source_exists, drain.floor().to(dtype), torch.zeros_like(drain, dtype=dtype)),
    )

    wave_entries, leftover = _greedy_select(
        P=P,
        W=W,
        device=device,
        dtype=dtype,
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_angle=cand_angle,
        cand_eta=cand_eta,
        cand_active=cand_active,
        cand_tgt_slot=cand_tgt_slot,
        cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def,
        source_budget=obs.ships.to(dtype).clone(),
        # disable_drain_fix=True -> None reproduces the pre-fix single-budget
        # behaviour exactly (the gate's pre-fix incumbent); see ProducerLiteConfig.
        source_spend_budget=(None if getattr(config, "disable_drain_fix", False)
                             else spend_budget),
        target_exists=target_exists,
        roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    regroup_entries = _plan_regroup(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        garrison_status=garrison_status,
        leftover=leftover,
        original_ships=obs.ships.to(dtype),
        pressure=enemy_mass,
        config=config,
        H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        garrison_status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=config,
        player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors,
        movement=movement,
        entries=entries,
        player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement,
        launches=launches,
        owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default.
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    # G3.1a: opening capture-and-hold. Off by default; the 96-seed sweep toggles
    # it via env so we A/B without editing code (proven before shipping).
    opening_hold_margin=float(os.environ.get("OWL_OPENING_HOLD_MARGIN", "0.0")),
    opening_hold_until_step=int(os.environ.get("OWL_OPENING_HOLD_UNTIL", "50")),
    opening_suppress_pvp=bool(int(os.environ.get("OWL_OPENING_SUPPRESS_PVP", "0"))),
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None,
                 config_override: "ProducerLiteConfig | None" = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()
        # Optional explicit config (overrides _config_for). Used by the G3.1a
        # eval to give ONLY seat 0 the opening-hold margin while opponents stay
        # baseline, without touching global env. None == default behaviour.
        self.config_override = config_override

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.reset()
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = (
            self.config_override
            if self.config_override is not None
            else _config_for(mem.cached_player_count)
        )
        row = run_turn(
            obs_tensors,
            config=config,
            player_count=int(mem.cached_player_count),
            memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _planet_row(planet):
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


def _fleet_row(fleet):
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


def _to_list_observation(obs):
    if not isinstance(obs, dict):
        return obs
    converted = dict(obs)
    converted["planets"] = [_planet_row(planet) for planet in obs.get("planets", [])]
    converted["initial_planets"] = [
        _planet_row(planet) for planet in obs.get("initial_planets", [])
    ]
    converted["fleets"] = [_fleet_row(fleet) for fleet in obs.get("fleets", [])]
    return converted


def _run(runtime, obs):
    obs = _to_list_observation(obs)
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = runtime.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)


def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    return _run(_RUNTIME, obs)


def make_agent():
    """Isolated Producer agent with its own ProducerLiteRuntime memory.

    For batched/vectorized rollouts where concurrent games must not share the
    module-global ``_RUNTIME``'s per-game memory.
    """
    runtime = ProducerLiteRuntime()
    return lambda obs: _run(runtime, obs)
