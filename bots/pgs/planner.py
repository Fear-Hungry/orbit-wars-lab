"""PGS — Portfolio Greedy Search over mission scripts (heuristic front, H-P1/H-P2).

Per-turn plan GENERATION = a per-source-planet portfolio of mission scripts
{PRODUCER, HOLD, SNIPE, CAPTURE, REINFORCE, EVAC}; per-turn SEARCH = greedy
assignment improvement (Portfolio Greedy Search, Churchill & Buro 2013;
hyper-heuristic *generation* framing, Burke et al. 2013), evaluated on the
orbit_lite timeline: production-weighted territory at a fixed horizon plus a
ship-margin tiebreak, against a static Producer-predicted opponent reply.

Floor invariant: the all-PRODUCER assignment IS the Producer plan (same
generator, ``ProducerLiteRuntime``), so the search starts at Producer parity
and only accepts per-planet deviations the timeline scores strictly better.
This is the heuristic-space analogue of BReP's KEEP-init: the plan space
strictly contains Producer instead of merely containing it as one candidate.

Submission-safe: pure Python over ``orbit_lite``/``bots`` (no Rust import).
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from bots.pgs._helpers import (
    _clone_movement,
    _debit_entry_sources,
    _entries_from_sparse_row,
    _to_list_observation,
    _with_tensor_player,
)
from bots.producer._upstream import (
    ProducerLiteRuntime,
    _config_for as _producer_config_for,
    plan_lite_waves,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite.distance_cache import build_distance_cache
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
    _empty_entries,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
)


import os

_DEBUG = os.getenv("PGS_DEBUG", "").strip() not in {"", "0"}


@dataclass(frozen=True)
class PGSConfig:
    value_horizon: int = 40
    max_search_sources: int = 12
    max_passes: int = 2
    max_deviations: int = 3
    epsilon: float = 1e-6
    # value = ship-margin at H + prod_weight * production-territory margin at H.
    # Ships are the war currency; production pays back over ~prod_weight turns.
    # A territory-dominated value (tiny ship weight) overvalues neutral captures
    # that never pay back before the decisive fight (measured: -0.35 by step 60).
    prod_weight: float = 15.0
    # CAPTURE only when ships spent pay back within this many turns of production
    payback_max_turns: float = 20.0
    # reactive arbiter: deviations must beat the all-PRODUCER floor by this much
    # (value units: ships + prod_weight*prod). Myopic 1-ply value misjudges small
    # edges, so only clearly-winning deviations are allowed through.
    arbiter_margin: float = 25.0
    capture_margin: float = 2.0
    min_ships_to_act: float = 5.0
    evac_window: int = 10
    deadline_ms: float = 600.0
    # deviations allowed only while step <= this (0 = no limit). Phase-gated
    # regime: search where it measurably helps, pure Producer elsewhere.
    deviation_max_step: int = 0
    # comma-separated subset of {hold,half,snipe,capture,reinforce,evac} to offer
    # as deviations (ablation knob). "half" = keep Producer's launch/target but
    # send 50% of the ships (re-aimed for the new fleet speed) — the partial veto.
    scripts: str = "hold,snipe,capture,reinforce,evac"
    # WAVE discipline v1 (H-P5, DB ids 138/139): conditional ATTACK-wave merging.
    # Floor launches aimed at ENEMY-owned planets are grouped by target; a group
    # whose total is under wave_min_ships is withheld (garrisons accumulate and
    # the Producer generator re-sizes next turn's proposal) until either the
    # group total crosses the threshold or the target has been pending for
    # wave_max_delay turns (then it releases anyway). Expansion (neutral targets)
    # and defense (own targets) are NEVER filtered — v0's unconditional size veto
    # lost tempo on both rulers (id=139). Elite LB play (~1710) attacks in
    # coordinated >=50-ship waves. 0 = off.
    wave_min_ships: float = 0.0
    wave_start_step: int = 50
    wave_max_delay: int = 8
    # 4p plays the exact Producer floor (no deviations, no wave) when True —
    # the 2026-06-09 tarball behaved this way via its 2p-only early return.
    floor_in_4p: bool = False


def _select_entries(entries: LaunchEntries, mask: Tensor) -> LaunchEntries:
    return LaunchEntries(
        source_slots=entries.source_slots[mask],
        target_slots=entries.target_slots[mask],
        ships=entries.ships[mask],
        angle=entries.angle[mask],
        eta=entries.eta[mask],
        valid=entries.valid[mask],
    )


def _single_entry(
    movement: PlanetMovement, source: int, target: int, ships: float, angle: float, eta: float
) -> LaunchEntries:
    dev, dt = movement.device, movement.dtype
    return LaunchEntries(
        source_slots=torch.tensor([source], dtype=torch.long, device=dev),
        target_slots=torch.tensor([target], dtype=torch.long, device=dev),
        ships=torch.tensor([ships], dtype=dt, device=dev),
        angle=torch.tensor([angle], dtype=dt, device=dev),
        eta=torch.tensor([eta], dtype=dt, device=dev),
        valid=torch.tensor([True], dtype=torch.bool, device=dev),
    )


class PGSRuntime:
    """Per-game PGS planner. Stateless across turns in v0 (fresh Producer plans)."""

    def __init__(self, config: PGSConfig | None = None) -> None:
        self.config = config or PGSConfig()
        # wave v1 cross-turn state: target_slot -> step when its attack group
        # was first withheld (age gate). Reset at step 0.
        self._wave_pending: dict[int, int] = {}
        self._player_count: int | None = None

    # -- agent glue --------------------------------------------------------
    def act(self, obs: Any):
        obs = _to_list_observation(obs)
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        with torch.no_grad():
            row = self.tensor_action(obs_tensors)
        return sparse_action_row_to_moves(row, obs, player_id=player_id)

    # -- producer base plan -------------------------------------------------
    def _producer_entries(
        self, owner_id: int, obs_tensors: dict, movement: PlanetMovement
    ) -> LaunchEntries:
        runtime = ProducerLiteRuntime()
        with torch.no_grad():
            row = runtime.tensor_action(_with_tensor_player(obs_tensors, int(owner_id)))
        return _entries_from_sparse_row(
            row=row, movement=movement, obs_tensors=obs_tensors, player_id=int(owner_id)
        )

    # -- mission scripts (each returns LaunchEntries or None) ---------------
    def _script_take(
        self,
        movement: PlanetMovement,
        status,
        source: int,
        available: float,
        target_mask: Tensor,
        me: int,
    ) -> LaunchEntries | None:
        cfg = self.config
        H = int(cfg.value_horizon)
        targets = torch.where(target_mask)[0]
        if int(targets.numel()) == 0:
            return None
        src = torch.full_like(targets, int(source))
        size = torch.full(targets.shape, float(available), dtype=movement.dtype, device=movement.device)
        aim = intercept_angle(movement, src, targets, size)
        eta = aim["eta"]
        ok = aim["viable"] & torch.isfinite(eta) & (eta <= float(H))
        if not bool(ok.any()):
            return None
        k = eta.ceil().long().clamp(0, H)
        defenders = status.ships[targets, k]
        needed = (defenders + 1.0 + float(cfg.capture_margin)).ceil()
        feasible = ok & (needed <= float(available))
        if not bool(feasible.any()):
            return None
        prod = movement.planet_prod[targets]
        score = torch.where(
            feasible, prod / (eta + 1.0) / needed.clamp(min=1.0), torch.full_like(eta, -math.inf)
        )
        best = int(score.argmax().item())
        target = int(targets[best].item())
        send = float(min(available, math.ceil(float(needed[best].item()) * 1.25)))
        # fleet speed depends on fleet SIZE: re-solve the intercept for the actual
        # send size (aiming with `available`'s angle would miss the moving target)
        for _ in range(2):
            re_aim = intercept_angle(
                movement,
                torch.tensor([source], device=movement.device),
                torch.tensor([target], device=movement.device),
                torch.tensor([send], dtype=movement.dtype, device=movement.device),
            )
            re_eta = float(re_aim["eta"][0].item())
            if not bool(re_aim["viable"][0]) or not math.isfinite(re_eta) or re_eta > float(H):
                return None
            k2 = min(H, int(math.ceil(re_eta)))
            need2 = math.ceil(float(status.ships[target, k2].item()) + 1.0 + float(cfg.capture_margin))
            if need2 <= send:
                tgt_prod = float(movement.planet_prod[target].item())
                if tgt_prod <= 0.0 or send / tgt_prod > float(cfg.payback_max_turns):
                    return None  # capture never pays back in time — junk expansion
                return _single_entry(
                    movement, source, target, send, float(re_aim["angle"][0].item()), re_eta
                )
            if need2 > available:
                return None
            send = float(min(available, math.ceil(need2 * 1.25)))
        return None

    def _script_reinforce(
        self,
        movement: PlanetMovement,
        status,
        source: int,
        available: float,
        me: int,
    ) -> LaunchEntries | None:
        """Defend an own planet the projection (WITH the opponent's predicted
        launches applied) says will flip — reinforcements must arrive before the
        flip step. Defending a known incoming attack is the cheapest sound
        deviation against a deterministic opponent."""
        H = int(self.config.value_horizon)
        owner = status.owner
        mine_now = owner[:, 0] == int(me)
        lost_mask = (owner[:, 1:] >= 0) & (owner[:, 1:] != int(me))
        flips = mine_now & lost_mask.any(dim=1)
        flips[int(source)] = False
        targets = torch.where(flips)[0]
        if int(targets.numel()) == 0:
            return None
        # earliest flip step per candidate target
        first_flip = torch.argmax(lost_mask[targets].long(), dim=1) + 1
        prod = movement.planet_prod[targets]
        order = prod.argsort(descending=True)
        for j in order.tolist():
            best_t = int(targets[j].item())
            flip_k = int(first_flip[j].item())
            send = float(max(1.0, math.floor(available * 0.6)))
            aim = intercept_angle(
                movement,
                torch.tensor([source], device=movement.device),
                torch.tensor([best_t], device=movement.device),
                torch.tensor([send], dtype=movement.dtype, device=movement.device),
            )
            eta = float(aim["eta"][0].item())
            if not bool(aim["viable"][0]) or not math.isfinite(eta):
                continue
            if math.ceil(eta) > flip_k:
                continue  # arrives after the planet already fell
            return _single_entry(
                movement, source, best_t, send, float(aim["angle"][0].item()), eta
            )
        return None

    def _script_evac(
        self,
        movement: PlanetMovement,
        status,
        source: int,
        available: float,
        me: int,
    ) -> LaunchEntries | None:
        cfg = self.config
        H = int(cfg.value_horizon)
        W = min(int(cfg.evac_window), H)
        owner = status.owner
        src_owner = owner[int(source), : W + 1]
        if bool((src_owner == int(me)).all()):
            return None  # source is safe; nothing to evacuate
        safe = (owner == int(me)).all(dim=1)
        safe[int(source)] = False
        targets = torch.where(safe)[0]
        if int(targets.numel()) == 0:
            return None
        src = torch.full_like(targets, int(source))
        size = torch.full(targets.shape, float(available), dtype=movement.dtype, device=movement.device)
        aim = intercept_angle(movement, src, targets, size)
        eta = aim["eta"]
        ok = aim["viable"] & torch.isfinite(eta)
        if not bool(ok.any()):
            return None
        eta_masked = torch.where(ok, eta, torch.full_like(eta, math.inf))
        best = int(eta_masked.argmin().item())
        return _single_entry(
            movement, source, int(targets[best].item()), float(available),
            float(aim["angle"][best].item()), float(eta_masked[best].item()),
        )

    def _script_half(
        self, movement: PlanetMovement, base_for_source: LaunchEntries
    ) -> LaunchEntries | None:
        """Partial veto: keep Producer's launches/targets from this source but send
        HALF the ships, re-aimed for the new fleet speed (size changes speed)."""
        valid = base_for_source.valid & (base_for_source.ships >= 4.0)
        if not bool(valid.any()):
            return None
        ships_new = torch.where(
            valid, (base_for_source.ships * 0.5).floor().clamp(min=1.0), base_for_source.ships
        )
        aim = intercept_angle(
            movement, base_for_source.source_slots, base_for_source.target_slots, ships_new
        )
        ok = aim["viable"] & torch.isfinite(aim["eta"])
        if not bool((ok | ~valid).all()):
            return None  # a halved fleet can no longer intercept its target
        return LaunchEntries(
            source_slots=base_for_source.source_slots,
            target_slots=base_for_source.target_slots,
            ships=ships_new,
            angle=torch.where(valid, aim["angle"].to(movement.dtype), base_for_source.angle),
            eta=torch.where(valid, aim["eta"].to(movement.dtype), base_for_source.eta),
            valid=base_for_source.valid,
        )

    # -- reactive reply (ported from OEP's _reactive_reply_entries) ----------
    def _reactive_reply(
        self,
        my_entries: LaunchEntries,
        opponent_id: int,
        obs_tensors: dict,
        movement: PlanetMovement,
        cache,
        player_count: int,
        me: int,
    ) -> LaunchEntries:
        """Producer re-plans for the opponent AGAINST the world after my launches.

        Static 1-ply prediction inflates deviations (OEP diagnosis B); this is the
        2-ply correction, used as the final arbiter between the deviated plan and
        the all-PRODUCER floor."""
        pcfg = _producer_config_for(player_count)
        h = int(pcfg.horizon)
        reply_movement = _clone_movement(movement)
        if bool(my_entries.valid.any()):
            launches = infer_planned_launches_from_entries(
                obs_tensors=obs_tensors, movement=reply_movement, entries=my_entries, player_id=int(me)
            )
            apply_private_planned_launches(
                movement=reply_movement, launches=launches, owner_id=int(me), obs_tensors=obs_tensors
            )
            _debit_entry_sources(reply_movement, my_entries)
        reply_status = reply_movement.garrison_status(max_horizon=h)
        opp_tensors = _with_tensor_player(obs_tensors, int(opponent_id))
        opp_obs = parse_obs(opp_tensors, player_id=int(opponent_id))
        return plan_lite_waves(
            movement=reply_movement,
            obs=opp_obs,
            obs_tensors=opp_tensors,
            cache=cache,
            garrison_status=reply_status,
            prod=reply_movement.planet_prod,
            alive_by_step=reply_movement.alive_by_step[: h + 1],
            config=pcfg,
            player_count=int(player_count),
        )

    # -- plan value ----------------------------------------------------------
    def _plan_value(
        self,
        movement: PlanetMovement,
        obs_tensors: dict,
        my_entries: LaunchEntries,
        opp_entries_by_owner: list[tuple[int, LaunchEntries]],
        me: int,
    ) -> float:
        cfg = self.config
        H = int(cfg.value_horizon)
        clone = _clone_movement(movement)
        for owner_id, entries in [(me, my_entries)] + opp_entries_by_owner:
            if not bool(entries.valid.any()):
                continue
            launches = infer_planned_launches_from_entries(
                obs_tensors=obs_tensors, movement=clone, entries=entries, player_id=int(owner_id)
            )
            apply_private_planned_launches(
                movement=clone, launches=launches, owner_id=int(owner_id), obs_tensors=obs_tensors
            )
            _debit_entry_sources(clone, entries)
        st = clone.garrison_status(max_horizon=H)
        owner_h = st.owner[:, H]
        prod = clone.planet_prod
        mine = owner_h == int(me)
        enemy = (owner_h >= 0) & (~mine)
        territory = float((prod[mine].sum() - prod[enemy].sum()).item())
        ships_h = st.ships[:, H]
        ship_margin = float((ships_h[mine].sum() - ships_h[enemy].sum()).item())
        return ship_margin + float(cfg.prod_weight) * territory

    def _wave_merge_filter(
        self, my_base: LaunchEntries, owner0: Tensor, me: int, step_now: int
    ) -> LaunchEntries:
        """Wave v1: withhold under-sized ATTACK groups (same enemy target) until
        they merge into a >= wave_min_ships wave or age out (wave_max_delay)."""
        cfg = self.config
        valid = my_base.valid
        if not bool(valid.any()):
            self._wave_pending = {}
            return my_base
        tgt = my_base.target_slots
        tgt_owner = owner0[tgt]
        attack = valid & (tgt_owner >= 0) & (tgt_owner != int(me))
        keep = valid.clone()
        pending_next: dict[int, int] = {}
        for t_slot in {int(t.item()) for t in tgt[attack]}:
            group = attack & (tgt == t_slot)
            total = float(my_base.ships[group].sum().item())
            first = self._wave_pending.get(t_slot, step_now)
            if total >= float(cfg.wave_min_ships) or (step_now - first) >= int(cfg.wave_max_delay):
                continue  # release the wave (or it aged out)
            keep &= ~group
            pending_next[t_slot] = first
        self._wave_pending = pending_next
        if bool((keep | ~valid).all()):
            return my_base
        return _select_entries(my_base, keep | ~valid)

    # -- main entry ----------------------------------------------------------
    def tensor_action(self, obs_tensors: dict) -> dict[str, Tensor]:
        t_start = time.perf_counter()
        cfg = self.config
        if bool((obs_tensors["step"] == 0).all()):
            self._player_count = None
        if self._player_count is None:
            self._player_count = largest_initial_player_count(obs_tensors)
        player_count = int(self._player_count)
        obs = parse_obs(obs_tensors)
        device = obs_tensors["planets"].device
        if obs.P == 0:
            return empty_action_row(device)
        me = int(obs.player_id)
        H = int(cfg.value_horizon)
        movement = ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=MovementConfig(
                movement_horizon=H, drift_epsilon=1e-3, track_fleets=True,
                player_count=player_count, max_tracked_fleets=128,
            ),
            cached_movement=None,
        )
        planet_ids = obs_tensors["planets"][..., 0].long()

        my_base = self._producer_entries(me, obs_tensors, movement)
        step_now = int(obs_tensors["step"].item())
        if step_now == 0:
            self._wave_pending = {}
        if player_count != 2 and bool(cfg.floor_in_4p):
            return entries_to_sparse_payload(my_base, planet_ids=planet_ids)
        if int(cfg.deviation_max_step) > 0 and step_now > int(cfg.deviation_max_step):
            # out-of-regime steps fall back to the exact Producer plan
            return entries_to_sparse_payload(my_base, planet_ids=planet_ids)

        status = movement.garrison_status(max_horizon=H)
        alive = obs.alive
        owner0 = status.owner[:, 0]
        if float(cfg.wave_min_ships) > 0 and step_now >= int(cfg.wave_start_step):
            my_base = self._wave_merge_filter(my_base, owner0, me, step_now)
        opp_ids = sorted({int(o.item()) for o in owner0[(owner0 >= 0) & (owner0 != me)]})
        opp_entries_by_owner = [
            (oid, self._producer_entries(oid, obs_tensors, movement)) for oid in opp_ids
        ]

        # scripts read the projection WITH the opponent's predicted launches applied
        # (their plan this turn is exactly predictable): flips include the incoming
        # attacks, snipe targets include their reinforcements.
        script_status = status
        if any(bool(e.valid.any()) for _, e in opp_entries_by_owner):
            opp_clone = _clone_movement(movement)
            for oid, entries in opp_entries_by_owner:
                if not bool(entries.valid.any()):
                    continue
                launches = infer_planned_launches_from_entries(
                    obs_tensors=obs_tensors, movement=opp_clone, entries=entries, player_id=int(oid)
                )
                apply_private_planned_launches(
                    movement=opp_clone, launches=launches, owner_id=int(oid), obs_tensors=obs_tensors
                )
                _debit_entry_sources(opp_clone, entries)
            script_status = opp_clone.garrison_status(max_horizon=H)

        def value(my_entries: LaunchEntries) -> float:
            return self._plan_value(movement, obs_tensors, my_entries, opp_entries_by_owner, me)

        # deviation sources: my alive planets with enough garrison, top-K by ships
        ships_now = movement.planet_ships
        my_mask = alive & (owner0 == me)
        # launches must be integer-valued and <= available; keep 1 ship home
        avail = (ships_now - 1.0).clamp(min=0.0).floor()
        candidates = torch.where(my_mask & (avail >= float(cfg.min_ships_to_act)))[0]
        if int(candidates.numel()) > int(cfg.max_search_sources):
            order = avail[candidates].argsort(descending=True)
            candidates = candidates[order[: int(cfg.max_search_sources)]]
        sources = [int(s.item()) for s in candidates]

        enemy_mask = alive & (owner0 >= 0) & (owner0 != me)
        neutral_mask = alive & (owner0 < 0)

        empty = _empty_entries(movement.device, movement.dtype)
        base_by_source: dict[int, LaunchEntries] = {}
        for s in sources:
            base_by_source[s] = _select_entries(my_base, my_base.source_slots == s)
        other_mask = torch.ones_like(my_base.valid)
        for s in sources:
            other_mask &= my_base.source_slots != s
        fixed_base = _select_entries(my_base, other_mask)  # base moves outside search set

        enabled = {t.strip().lower() for t in str(cfg.scripts).split(",") if t.strip()}
        portfolio: dict[int, list[tuple[str, LaunchEntries]]] = {}
        for s in sources:
            a = float(avail[s].item())
            options: list[tuple[str, LaunchEntries]] = [("PROD", base_by_source[s])]
            if "hold" in enabled:
                options.append(("HOLD", empty))
            maybe: list[tuple[str, LaunchEntries | None]] = []
            if "half" in enabled:
                maybe.append(("HALF", self._script_half(movement, base_by_source[s])))
            if "snipe" in enabled:
                maybe.append(("SNIPE", self._script_take(movement, script_status, s, a, enemy_mask, me)))
            if "capture" in enabled:
                maybe.append(("CAPTURE", self._script_take(movement, script_status, s, a, neutral_mask, me)))
            if "reinforce" in enabled:
                maybe.append(("REINFORCE", self._script_reinforce(movement, script_status, s, a, me)))
            if "evac" in enabled:
                maybe.append(("EVAC", self._script_evac(movement, script_status, s, a, me)))
            for name, ent in maybe:
                if ent is not None:
                    options.append((name, ent))
            portfolio[s] = options

        assign: dict[int, int] = {s: 0 for s in sources}  # index into portfolio[s]; 0 == PROD

        def assembled(assign_now: dict[int, int]) -> LaunchEntries:
            parts = [fixed_base] + [portfolio[s][assign_now[s]][1] for s in sources]
            return concat_launch_entries(parts)

        cur_value = value(assembled(assign))
        deviations = 0
        deadline = t_start + float(cfg.deadline_ms) / 1000.0
        for _ in range(int(cfg.max_passes)):
            improved = False
            for s in sources:
                if time.perf_counter() > deadline or deviations >= int(cfg.max_deviations):
                    break
                best_idx, best_val = assign[s], cur_value
                for idx in range(len(portfolio[s])):
                    if idx == assign[s]:
                        continue
                    if time.perf_counter() > deadline:
                        break
                    trial = dict(assign)
                    trial[s] = idx
                    v = value(assembled(trial))
                    if v > best_val + float(cfg.epsilon):
                        best_idx, best_val = idx, v
                if best_idx != assign[s]:
                    if assign[s] == 0:
                        deviations += 1
                    assign[s] = best_idx
                    cur_value = best_val
                    improved = True
            if not improved or time.perf_counter() > deadline:
                break

        deviated = any(assign[s] != 0 for s in sources)
        if deviated and opp_ids:
            # REACTIVE arbiter: the greedy pass used a static opponent prediction,
            # which is known to inflate deviations. Re-score the deviated plan and
            # the all-PRODUCER floor against EVERY opponent's reply to each plan,
            # and only keep the deviation if it still wins. Preserves the floor.
            cache = build_distance_cache(movement, max_k=int(_producer_config_for(player_count).horizon))
            plan_dev = assembled(assign)
            plan_prod = concat_launch_entries(
                [fixed_base] + [portfolio[s][0][1] for s in sources]
            )

            def reactive_value(plan: LaunchEntries) -> float:
                replies = [
                    (oid, self._reactive_reply(plan, oid, obs_tensors, movement, cache, player_count, me))
                    for oid in opp_ids
                ]
                return self._plan_value(movement, obs_tensors, plan, replies, me)

            if reactive_value(plan_dev) <= reactive_value(plan_prod) + float(cfg.arbiter_margin):
                assign = {s: 0 for s in sources}

        if _DEBUG and any(assign[s] != 0 for s in sources):
            import sys

            step_now = int(obs_tensors["step"].item())
            chosen = {s: portfolio[s][assign[s]][0] for s in sources if assign[s] != 0}
            print(f"[pgs] step={step_now} me={me} deviations={chosen}", file=sys.stderr)

        final = disambiguate_duplicate_launches(assembled(assign))
        return entries_to_sparse_payload(final, planet_ids=planet_ids)


_RUNTIME: PGSRuntime | None = None


def make_runtime(config: PGSConfig | None = None) -> PGSRuntime:
    """Fresh, fully isolated PGS runtime (safe for vectorized rollouts)."""
    return PGSRuntime(config)


def agent(obs: Any):
    global _RUNTIME
    if _RUNTIME is None or (isinstance(obs, dict) and int(obs.get("step", 0)) == 0):
        _RUNTIME = PGSRuntime()
    return _RUNTIME.act(obs)
