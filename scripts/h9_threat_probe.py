"""H9 falsification probe — does per-enemy forward threat separate launch/hold?

[118] showed the endpoint value ``margin@H`` is FLAT across launch/hold/defend in
4p (PROD==HOLD). H9 lives or dies on one question: do the forward, per-enemy
trajectory features (``bots/pgs/threat.py``) SEPARATE the same actions where
margin@H is flat? If not, H9 is dead — escalate to H11 (attention, DB 170).

Method: drive producer x4 for ``--warmup`` steps, then on each mid-game 4p state
build, for the agent seat, two post-launch projections that BOTH apply the
predicted opponent launches (so only OUR action differs):
  HOLD  = no launches from us
  PROD  = our Producer plan
and score each with the OLD endpoint value (margin@H) and the NEW threat_value.

Reports per-state relative separation. PASS heuristic: on a clear majority of
states the OLD value is flat (rel < flat_tol) while the NEW value separates
(rel > sep_tol). This is a diagnostic, not a promotion gate.
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np
import torch

from bots.pgs._helpers import _clone_movement, _debit_entry_sources
from bots.pgs.planner import PGSConfig, _single_entry, make_runtime
from bots.pgs.threat import compute_threat_features
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation

ME = 0


def _empty_entries() -> object:
    from orbit_lite.movement_step import LaunchEntries

    return LaunchEntries(
        source_slots=torch.zeros(0, dtype=torch.long),
        target_slots=torch.zeros(0, dtype=torch.long),
        ships=torch.zeros(0),
        angle=torch.zeros(0),
        eta=torch.zeros(0),
        valid=torch.zeros(0, dtype=torch.bool),
    )


def _apply(movement, obs_tensors, owner_entries, H):
    """Clone, apply each (owner, entries) launch set + debit sources, return (clone, status)."""
    clone = _clone_movement(movement)
    for owner_id, entries in owner_entries:
        if not bool(entries.valid.any()):
            continue
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors, movement=clone, entries=entries, player_id=int(owner_id)
        )
        apply_private_planned_launches(
            movement=clone, launches=launches, owner_id=int(owner_id), obs_tensors=obs_tensors
        )
        _debit_entry_sources(clone, entries)
    return clone, clone.garrison_status(max_horizon=H)


def _endpoint_value(clone, status, H, prod_weight) -> float:
    """Replicates bots/pgs/planner._plan_value endpoint readout (the [118] value)."""
    owner_h = status.owner[:, H]
    prod = clone.planet_prod
    mine = owner_h == ME
    enemy = (owner_h >= 0) & (~mine)
    territory = float((prod[mine].sum() - prod[enemy].sum()).item())
    ships_h = status.ships[:, H]
    ship_margin = float((ships_h[mine].sum() - ships_h[enemy].sum()).item())
    return ship_margin + float(prod_weight) * territory


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(a) + abs(b) + 1e-9)


def _forced_attack(movement, status, H):
    """A REAL garrison-depleting deviation: biggest-garrison planet sends half its
    ships at the nearest reachable enemy planet. Forced (no payback gate) so the
    value function is tested against an action that actually changes our garrison —
    unlike the Producer floor, which holds on most 4p turns (DB 118)."""
    dev = movement.device
    owner0 = status.owner[:, 0]
    ships0 = status.ships[:, 0]
    mine = owner0 == ME
    if not bool(mine.any()):
        return None
    my_idx = torch.where(mine)[0]
    src = int(my_idx[ships0[my_idx].argmax()].item())
    src_ships = float(ships0[src].item())
    if src_ships < 2.0:
        return None
    send = max(1.0, math.floor(0.5 * src_ships))
    enemies = torch.where((owner0 >= 0) & (owner0 != ME))[0]
    if int(enemies.numel()) == 0:
        return None
    x0, y0 = movement.x[0], movement.y[0]
    d = (x0[enemies] - x0[src]) ** 2 + (y0[enemies] - y0[src]) ** 2
    order = enemies[d.argsort()]
    for tgt_t in order[:4]:
        tgt = int(tgt_t.item())
        aim = intercept_angle(
            movement,
            torch.tensor([src], device=dev),
            torch.tensor([tgt], device=dev),
            torch.tensor([send], dtype=movement.dtype, device=dev),
        )
        eta = float(aim["eta"][0].item())
        if bool(aim["viable"][0]) and math.isfinite(eta) and eta <= float(H):
            return _single_entry(movement, src, tgt, send, float(aim["angle"][0].item()), eta)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=60, help="producer x4 steps before probing")
    ap.add_argument("--flat-tol", type=float, default=0.02, help="rel below this => OLD value is flat")
    ap.add_argument("--sep-tol", type=float, default=0.05, help="rel above this => value separates HOLD/PROD")
    args = ap.parse_args()
    torch.set_num_threads(1)

    seeds = list(range(1000, 1000 + args.seeds))
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=4, seed=seeds[0],
                               config=RustConfig(enable_comets=True, episode_steps=500))
    backend.reset(seeds[0])
    states = backend.states()

    # drive all four seats with isolated producer agents to reach a live mid-game.
    opp = get_isolated_opponents("producer", n * 4)
    for _ in range(args.warmup):
        rows = []
        for i in range(n):
            for seat in range(4):
                for m in opp[i * 4 + seat](states[i], seat):
                    if len(m) >= 3:
                        rows.append([float(i), float(seat), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, ME)
        states = backend.states()

    cfg = PGSConfig()
    H = int(cfg.value_horizon)
    rel_old: list[float] = []
    rel_new: list[float] = []
    rows_dump = []
    skipped = 0
    for i in range(n):
        obs = to_official_observation(states[i], ME)
        obs_tensors = single_obs_to_tensor(obs, player_id=ME)
        runtime = make_runtime(PGSConfig())
        movement = ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=MovementConfig(movement_horizon=H, drift_epsilon=1e-3, track_fleets=True,
                                        player_count=4, max_tracked_fleets=128),
            cached_movement=None,
        )
        status0 = movement.garrison_status(max_horizon=H)
        owner0 = status0.owner[:, 0]
        if not bool((owner0 == ME).any()):
            skipped += 1
            continue
        my_dev = _forced_attack(movement, status0, H)
        if my_dev is None:
            skipped += 1
            continue
        opp_ids = sorted({int(o.item()) for o in owner0[(owner0 >= 0) & (owner0 != ME)]})
        opp_entries = [(oid, runtime._producer_entries(oid, obs_tensors, movement)) for oid in opp_ids]

        clone_hold, st_hold = _apply(movement, obs_tensors, opp_entries + [(ME, _empty_entries())], H)
        clone_prod, st_prod = _apply(movement, obs_tensors, opp_entries + [(ME, my_dev)], H)

        old_hold = _endpoint_value(clone_hold, st_hold, H, cfg.prod_weight)
        old_prod = _endpoint_value(clone_prod, st_prod, H, cfg.prod_weight)
        new_hold = compute_threat_features(st_hold, ME, horizon=H).threat_value(cfg.prod_weight)
        new_prod = compute_threat_features(st_prod, ME, horizon=H).threat_value(cfg.prod_weight)

        rel_old.append(_rel(old_prod, old_hold))
        rel_new.append(_rel(new_prod, new_hold))
        rows_dump.append((seeds[i], len(opp_ids), old_hold, old_prod, new_hold, new_prod))

    if not rel_old:
        print("[h9-probe] no usable 4p states (all seats lost the agent seat?). skipped:", skipped)
        return

    flat_old = [r < args.flat_tol for r in rel_old]
    sep_new = [r > args.sep_tol for r in rel_new]
    h9_state = [fo and sn for fo, sn in zip(flat_old, sep_new)]

    print(f"[h9-probe] states={len(rel_old)} skipped={skipped} warmup={args.warmup} H={H}")
    print(f"  OLD margin@H rel-sep (PROD vs HOLD): median={statistics.median(rel_old):.4f} "
          f"max={max(rel_old):.4f}  -> flat(<{args.flat_tol}) in {sum(flat_old)}/{len(rel_old)}")
    print(f"  NEW threat_value rel-sep           : median={statistics.median(rel_new):.4f} "
          f"max={max(rel_new):.4f}  -> separates(>{args.sep_tol}) in {sum(sep_new)}/{len(rel_new)}")
    frac = sum(h9_state) / len(rel_old)
    print(f"  H9 SIGNAL (old flat AND new separates): {sum(h9_state)}/{len(rel_old)} = {frac:.0%}")
    verdict = "PASS (threat features see what margin@H cannot)" if frac >= 0.5 else \
              "WEAK/FAIL (threat features mostly flat too -> escalate to H11)"
    print(f"  VERDICT: {verdict}")
    print("\n  seed  nOpp   old_hold   old_prod   new_hold   new_prod")
    for s, no, oh, op, nh, npd in rows_dump[:12]:
        print(f"  {s:5d}  {no:3d}  {oh:9.2f}  {op:9.2f}  {nh:9.2f}  {npd:9.2f}")


if __name__ == "__main__":
    main()
