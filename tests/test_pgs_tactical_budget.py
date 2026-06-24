"""Regression: PGS deviation scripts must budget by safe_drain, not raw garrison.

The old `avail = ships - 1` was PHYSICAL: a source holding 80 ships under
sustained attack may only shed ~25 without the do-nothing projection losing the
planet. Scripts that ADD launches (snipe/capture/reinforce/evac and future
waves) sized by the physical pool overdrain exactly the threatened sources —
the losing regime. tensor_action now caps every script's `available` at
floor(safe_drain) (a doomed source keeps `available = ships`: safe_drain
collapses to source_ships when no turn is held, so EVAC still works).
"""
from __future__ import annotations

import numpy as np
import torch
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import ensure_planet_movement
from orbit_lite.obs import parse_obs
from orbit_lite.planner_core import safe_drain
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation


def _advance(backend, states, policies, steps: int):
    for _ in range(steps):
        rows = []
        for player, pol in enumerate(policies):
            for m in pol(states[0], player):
                if len(m) >= 3:
                    rows.append([0.0, float(player), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
    return states


def _states_along_game(num_players: int, capture: list[int], seed: int = 7):
    """Producer (seat 0) vs rusher (seat 1): guarantees THREATENED sources for
    seat 0 (drain << ships), the regime where the tactical cap must bite."""
    backend = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed, config=RustConfig(enable_comets=True)
    )
    backend.reset(seed)
    states = backend.states()
    producer = get_isolated_opponents("producer", num_players)[0]
    from scripts.league_agents import make as league_make

    rusher = league_make("rusher")
    policies = [producer] + [
        (lambda state, player: rusher(to_official_observation(state, player)))
    ] * (num_players - 1)
    out, last = [], 0
    for step in sorted(capture):
        states = _advance(backend, states, policies, step - last)
        out.append(states[0])
        last = step
    return out


def _drain_by_slot(obs: dict, player_id: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """floor(safe_drain) per planet slot, replicating tensor_action's recipe."""
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    parsed = parse_obs(obs_tensors)
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=MovementConfig(
            movement_horizon=H, drift_epsilon=1e-3, track_fleets=True,
            player_count=2, max_tracked_fleets=128,
        ),
        cached_movement=None,
    )
    status = movement.garrison_status(max_horizon=H)
    owner0 = status.owner[:, 0]
    my_mask = parsed.alive & (owner0 == player_id)
    drain = torch.zeros_like(movement.planet_ships)
    idx = torch.where(my_mask)[0]
    if int(idx.numel()) > 0:
        drain[idx] = safe_drain(
            status,
            source_idx=idx,
            source_ships=movement.planet_ships[idx],
            H_eff=torch.full((), float(H), dtype=movement.dtype, device=movement.device),
            player_id=player_id,
        ).floor().clamp(min=0.0)
    physical = (movement.planet_ships - 1.0).clamp(min=0.0).floor()
    return drain, torch.where(my_mask, physical, torch.zeros_like(physical))


def test_scripts_receive_safe_drain_capped_available() -> None:
    from bots.pgs.planner import PGSConfig, make_runtime

    states = _states_along_game(2, capture=[30, 60, 90, 120])
    config = PGSConfig(scripts="hold,snipe,capture,reinforce,evac")
    H = int(config.value_horizon)

    calls: list[tuple[int, float, float]] = []  # (source, available, entry_ships)
    runtime = make_runtime(config)
    orig_take = runtime._script_take
    orig_rein = runtime._script_reinforce
    orig_evac = runtime._script_evac

    def spy(orig):
        def run(movement, status, source, available, *rest):
            ent = orig(movement, status, source, available, *rest)
            sent = float(ent.ships[ent.valid].sum().item()) if ent is not None else 0.0
            calls.append((int(source), float(available), sent))
            return ent
        return run

    runtime._script_take = spy(orig_take)
    runtime._script_reinforce = spy(orig_rein)
    runtime._script_evac = spy(orig_evac)

    capped_cases = 0
    for state in states:
        for player in range(2):
            obs = to_official_observation(state, player)
            drain, physical = _drain_by_slot(obs, player, H)
            calls.clear()
            runtime.act(obs)
            for source, available, sent in calls:
                assert available <= float(drain[source].item()) + 1e-6, (
                    f"script got available={available} > safe_drain="
                    f"{float(drain[source].item())} at slot {source}"
                )
                assert sent <= available + 1e-6, (
                    f"mission sends {sent} > available={available} at slot {source}"
                )
                if float(drain[source].item()) < float(physical[source].item()):
                    capped_cases += 1
    # the suite must include sources where the tactical cap BITES (drain <
    # ships-1), otherwise this test cannot tell tactical from physical budget
    assert capped_cases > 0, "no threatened source seen; widen capture steps/seed"
