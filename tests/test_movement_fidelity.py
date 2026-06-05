from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig

PRODUCER_DIR = Path(__file__).resolve().parent / "opponents/producer/producer"
sys.path.insert(0, str(PRODUCER_DIR))

from orbit_lite.adapter import single_obs_to_tensor  # noqa: E402
from orbit_lite.garrison_launch import _run_exact_recurrence  # noqa: E402
from orbit_lite.intercept_aim import intercept_angle  # noqa: E402
from orbit_lite.movement import MovementConfig, PlanetMovement  # noqa: E402
from orbit_lite.movement_step import (  # noqa: E402
    LaunchEntries,
    apply_private_planned_launches,
    infer_planned_launches_from_entries,
)
from orbit_lite.planner_core import entries_to_sparse_payload  # noqa: E402

HORIZON = 18
FIDELITY_SEEDS = 200
COMET_FIDELITY_SEEDS = 50
COMET_START_STEP = 55
COMET_HORIZON = 18
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
XY_ATOL = 1.0e-6


def _planet_rows(state: dict) -> list[list[float]]:
    return [
        [
            p["id"],
            p["owner"],
            p["x"],
            p["y"],
            p["radius"],
            p["ships"],
            p["production"],
        ]
        for p in state["planets"]
    ]


def _fleet_rows(state: dict) -> list[list[float]]:
    return [
        [
            f["id"],
            f["owner"],
            f["x"],
            f["y"],
            f["angle"],
            f["from_planet_id"],
            f["ships"],
        ]
        for f in state["fleets"]
    ]


def _state_to_obs_tensors(state: dict, *, player_id: int) -> dict:
    obs = {
        **state,
        "planets": _planet_rows(state),
        "initial_planets": [
            [
                p["id"],
                p["owner"],
                p["x"],
                p["y"],
                p["radius"],
                p["ships"],
                p["production"],
            ]
            for p in state["initial_planets"]
        ],
        "fleets": _fleet_rows(state),
    }
    tensors = single_obs_to_tensor(obs, player_id=player_id)
    # The production adapter intentionally uses float32. The fidelity gate uses
    # float64 so the 1e-6 positional tolerance checks world-model math rather
    # than adapter quantization.
    tensors["planets"] = torch.tensor(obs["planets"], dtype=torch.float64)
    tensors["initial_planets"] = torch.tensor(obs["initial_planets"], dtype=torch.float64)
    fleet_rows = obs["fleets"]
    tensors["fleets"] = torch.tensor(fleet_rows, dtype=torch.float64).reshape(-1, 7)
    tensors["angular_velocity"] = torch.tensor(state["angular_velocity"], dtype=torch.float64)
    tensors["step"] = torch.tensor(state["step"], dtype=torch.int32)
    return tensors


def _rollout_truth(seed: int, horizon: int) -> list[dict]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=seed,
        config=RustConfig(episode_steps=500, enable_comets=False),
    )
    state0 = backend.reset(seed)[0]
    states = [state0]
    for _ in range(horizon):
        _, next_states = backend.step_with_states([[[], []]])
        states.append(next_states[0])
    return states


def _rollout_truth_from_comet_state(
    seed: int,
    start_step: int,
    horizon: int,
    *,
    initial_moves: list[list[float]] | None = None,
) -> list[dict]:
    next_spawn = min(spawn for spawn in COMET_SPAWN_STEPS if spawn > start_step)
    assert start_step + horizon < next_spawn
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=seed,
        config=RustConfig(episode_steps=500, enable_comets=True),
    )
    state = backend.reset(seed)[0]
    while int(state["step"]) < int(start_step):
        _, states = backend.step_with_states([[[], []]])
        state = states[0]
    assert int(state["step"]) == int(start_step)
    assert state["comets"], f"seed={seed} reached step={start_step} without active comets"
    states = [state]
    if initial_moves is not None:
        _, next_states = backend.step_with_states([[initial_moves, []]])
        states.append(next_states[0])
        horizon -= 1
    for _ in range(horizon):
        _, next_states = backend.step_with_states([[[], []]])
        states.append(next_states[0])
    return states


def _comet_seeds(*, needed: int, start_step: int) -> list[int]:
    seeds: list[int] = []
    seed = 0
    while len(seeds) < needed:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=2,
            seed=seed,
            config=RustConfig(episode_steps=500, enable_comets=True),
        )
        state = backend.reset(seed)[0]
        while int(state["step"]) < int(start_step):
            _, states = backend.step_with_states([[[], []]])
            state = states[0]
        if state["comets"]:
            seeds.append(seed)
        seed += 1
    return seeds


def _first_field_diff(seed: int, states: list[dict], movement: PlanetMovement) -> str | None:
    status = movement.garrison_status(max_horizon=HORIZON)
    for step_idx, state in enumerate(states):
        truth_by_id = {int(p["id"]): p for p in state["planets"]}
        for slot, planet_id in enumerate(movement.planet_ids.tolist()):
            if int(planet_id) < 0:
                continue
            truth = truth_by_id.get(int(planet_id))
            projected_alive = bool(movement.alive_by_step[step_idx, slot].item())
            if truth is None:
                if projected_alive:
                    return (
                        f"[FIDELITY-L1] seed={seed} step={step_idx} planet={planet_id} "
                        "field=alive expected=False projected=True"
                    )
                continue
            if not projected_alive:
                return (
                    f"[FIDELITY-L1] seed={seed} step={step_idx} planet={planet_id} "
                    "field=alive expected=True projected=False"
                )

            for field, projected, expected in (
                ("x", float(movement.x[step_idx, slot].item()), float(truth["x"])),
                ("y", float(movement.y[step_idx, slot].item()), float(truth["y"])),
            ):
                if not math.isclose(projected, expected, rel_tol=1.0e-6, abs_tol=XY_ATOL):
                    return (
                        f"[FIDELITY-L1] seed={seed} step={step_idx} planet={planet_id} "
                        f"field={field} expected={expected:.12f} projected={projected:.12f}"
                    )

            expected_owner = int(truth["owner"])
            projected_owner = int(status.owner[slot, step_idx].item())
            if projected_owner != expected_owner:
                return (
                    f"[FIDELITY-L2] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=owner expected={expected_owner} projected={projected_owner}"
                )

            expected_ships = int(truth["ships"])
            projected_ships = int(status.ships[slot, step_idx].item())
            if projected_ships != expected_ships:
                return (
                    f"[FIDELITY-L2] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=ships expected={expected_ships} projected={projected_ships}"
                )
    return None


def _compare_projected_garrison(
    *,
    tag: str,
    seed: int,
    states: list[dict],
    movement: PlanetMovement,
    owner_by_step: torch.Tensor,
    ships_by_step: torch.Tensor,
    start_step: int = 0,
) -> str | None:
    for step_idx, state in enumerate(states[start_step:], start=start_step):
        truth_by_id = {int(p["id"]): p for p in state["planets"]}
        for slot, planet_id in enumerate(movement.planet_ids.tolist()):
            if int(planet_id) < 0:
                continue
            truth = truth_by_id.get(int(planet_id))
            projected_alive = bool(movement.alive_by_step[step_idx, slot].item())
            if truth is None:
                if projected_alive:
                    return (
                        f"[{tag}] seed={seed} step={step_idx} planet={planet_id} "
                        "field=alive expected=False projected=True"
                    )
                continue
            if not projected_alive:
                return (
                    f"[{tag}] seed={seed} step={step_idx} planet={planet_id} "
                    "field=alive expected=True projected=False"
                )
            expected_owner = int(truth["owner"])
            projected_owner = int(owner_by_step[slot, step_idx].item())
            if projected_owner != expected_owner:
                return (
                    f"[{tag}] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=owner expected={expected_owner} projected={projected_owner}"
                )
            expected_ships = int(truth["ships"])
            projected_ships = int(ships_by_step[slot, step_idx].item())
            if projected_ships != expected_ships:
                return (
                    f"[{tag}] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=ships expected={expected_ships} projected={projected_ships}"
                )
    return None


def _comet_planet_ids(state: dict) -> set[int]:
    return {int(pid) for pid in state.get("comet_planet_ids", [])}


def _comet_path_index_by_id(state: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for group in state.get("comets", []):
        path_index = int(group["path_index"])
        for pid in group.get("planet_ids", []):
            out[int(pid)] = path_index
    return out


def _first_comet_diff(seed: int, states: list[dict], movement: PlanetMovement) -> str | None:
    status = movement.garrison_status(max_horizon=COMET_HORIZON)
    initial_comet_ids = _comet_planet_ids(states[0])
    initial_path_index_by_id = _comet_path_index_by_id(states[0])
    if not initial_comet_ids:
        return f"[FIDELITY-L5a] seed={seed} step=0 field=comets expected=present projected=empty"
    comet_slots = [
        slot
        for slot, planet_id in enumerate(movement.planet_ids.tolist())
        if int(planet_id) in initial_comet_ids
    ]
    if len(comet_slots) != len(initial_comet_ids):
        return (
            f"[FIDELITY-L5a] seed={seed} step=0 field=comet_slots "
            f"expected={len(initial_comet_ids)} projected={len(comet_slots)}"
        )
    for step_idx, state in enumerate(states):
        truth_by_id = {int(p["id"]): p for p in state["planets"]}
        comet_ids_now = _comet_planet_ids(state)
        path_index_by_id = _comet_path_index_by_id(state)
        for slot in comet_slots:
            planet_id = int(movement.planet_ids[slot].item())
            truth = truth_by_id.get(planet_id)
            expected_alive = planet_id in comet_ids_now and truth is not None
            projected_alive = bool(movement.alive_by_step[step_idx, slot].item())
            if projected_alive != expected_alive:
                return (
                    f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=alive expected={expected_alive} projected={projected_alive}"
                )
            if not expected_alive:
                continue

            assert truth is not None
            for field, projected, expected in (
                ("x", float(movement.x[step_idx, slot].item()), float(truth["x"])),
                ("y", float(movement.y[step_idx, slot].item()), float(truth["y"])),
            ):
                if not math.isclose(projected, expected, rel_tol=1.0e-6, abs_tol=XY_ATOL):
                    return (
                        f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                        f"field={field} expected={expected:.12f} projected={projected:.12f}"
                    )

            expected_path_index = int(path_index_by_id[planet_id])
            projected_path_index = int(initial_path_index_by_id[planet_id]) + step_idx
            if projected_path_index != expected_path_index:
                return (
                    f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=path_index expected={expected_path_index} projected={projected_path_index}"
                )

            expected_prod = int(truth["production"])
            projected_prod = int(movement.planet_prod[slot].item())
            if projected_prod != expected_prod:
                return (
                    f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=production expected={expected_prod} projected={projected_prod}"
                )

            expected_owner = int(truth["owner"])
            projected_owner = int(status.owner[slot, step_idx].item())
            if projected_owner != expected_owner:
                return (
                    f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=owner expected={expected_owner} projected={projected_owner}"
                )

            expected_ships = int(truth["ships"])
            projected_ships = int(status.ships[slot, step_idx].item())
            if projected_ships != expected_ships:
                return (
                    f"[FIDELITY-L5a] seed={seed} step={step_idx} planet={planet_id} "
                    f"field=ships expected={expected_ships} projected={projected_ships}"
                )
    return None


def _make_comet_capture_entries(state: dict, movement: PlanetMovement) -> LaunchEntries:
    comet_ids = _comet_planet_ids(state)
    if not comet_ids:
        return LaunchEntries(
            source_slots=torch.zeros(0, dtype=torch.long),
            target_slots=torch.zeros(0, dtype=torch.long),
            ships=torch.zeros(0, dtype=torch.float64),
            angle=torch.zeros(0, dtype=torch.float64),
            eta=torch.zeros(0, dtype=torch.float64),
            valid=torch.zeros(0, dtype=torch.bool),
        )
    ships_by_id = {int(p["id"]): int(p["ships"]) for p in state["planets"]}
    owner_by_id = {int(p["id"]): int(p["owner"]) for p in state["planets"]}
    source_slots = [
        slot
        for slot, planet_id in enumerate(movement.planet_ids.tolist())
        if int(planet_id) not in comet_ids
        and int(planet_id) >= 0
        and owner_by_id.get(int(planet_id)) == 0
        and ships_by_id.get(int(planet_id), 0) >= 8
    ]
    target_slots = [
        slot
        for slot, planet_id in enumerate(movement.planet_ids.tolist())
        if int(planet_id) in comet_ids and int(planet_id) >= 0
    ]
    source_slots.sort(key=lambda slot: ships_by_id.get(int(movement.planet_ids[slot].item()), 0), reverse=True)
    for source_slot in source_slots:
        source_pid = int(movement.planet_ids[source_slot].item())
        budget = ships_by_id.get(source_pid, 0)
        for target_slot in target_slots:
            target_pid = int(movement.planet_ids[target_slot].item())
            defenders = ships_by_id.get(target_pid, 0)
            send = min(budget - 1, defenders + 3)
            if send <= defenders or send < 1:
                continue
            src = torch.tensor([[source_slot]], dtype=torch.long)
            tgt = torch.tensor([[target_slot]], dtype=torch.long)
            ships = torch.tensor([[float(send)]], dtype=movement.dtype)
            aim = intercept_angle(
                movement,
                src,
                tgt,
                ships,
                active=torch.ones(1, 1, dtype=torch.bool),
            )
            viable = bool(aim["viable"][0, 0].item())
            eta = float(aim["eta"][0, 0].item())
            if not viable or eta > float(COMET_HORIZON):
                continue
            return LaunchEntries(
                source_slots=torch.tensor([source_slot], dtype=torch.long),
                target_slots=torch.tensor([target_slot], dtype=torch.long),
                ships=torch.tensor([float(send)], dtype=movement.dtype),
                angle=torch.tensor([float(aim["angle"][0, 0].item())], dtype=movement.dtype),
                eta=torch.tensor([eta], dtype=movement.dtype),
                valid=torch.ones(1, dtype=torch.bool),
            )
    return LaunchEntries(
        source_slots=torch.zeros(0, dtype=torch.long),
        target_slots=torch.zeros(0, dtype=torch.long),
        ships=torch.zeros(0, dtype=movement.dtype),
        angle=torch.zeros(0, dtype=movement.dtype),
        eta=torch.zeros(0, dtype=movement.dtype),
        valid=torch.zeros(0, dtype=torch.bool),
    )


def _captured_initial_comet(states: list[dict]) -> bool:
    comet_ids = _comet_planet_ids(states[0])
    if not comet_ids:
        return False
    for state in states[1:]:
        for planet in state["planets"]:
            if int(planet["id"]) in comet_ids and int(planet["owner"]) == 0:
                return True
    return False


def test_movement_l1_l2_matches_rust_do_nothing_rollout() -> None:
    for seed in range(FIDELITY_SEEDS):
        states = _rollout_truth(seed, HORIZON)
        movement = PlanetMovement.from_obs_tensors(
            _state_to_obs_tensors(states[0], player_id=0),
            config=MovementConfig(
                movement_horizon=HORIZON,
                track_fleets=True,
                player_count=2,
            ),
        )
        diff = _first_field_diff(seed, states, movement)
        assert diff is None, diff


def test_movement_l5a_comet_projection() -> None:
    captured_windows = 0
    for seed in _comet_seeds(needed=COMET_FIDELITY_SEEDS, start_step=COMET_START_STEP):
        states_without_moves = _rollout_truth_from_comet_state(seed, COMET_START_STEP, 0)
        movement = PlanetMovement.from_obs_tensors(
            _state_to_obs_tensors(states_without_moves[0], player_id=0),
            config=MovementConfig(
                movement_horizon=COMET_HORIZON,
                track_fleets=True,
                player_count=2,
            ),
        )
        entries = _make_comet_capture_entries(states_without_moves[0], movement)
        moves = _payload_to_moves(entries_to_sparse_payload(entries, planet_ids=movement.planet_ids))
        states = _rollout_truth_from_comet_state(
            seed,
            COMET_START_STEP,
            COMET_HORIZON,
            initial_moves=moves if moves else None,
        )
        if moves:
            launches = infer_planned_launches_from_entries(
                obs_tensors=_state_to_obs_tensors(states[0], player_id=0),
                movement=movement,
                entries=entries,
                player_id=0,
            )
            apply_private_planned_launches(
                movement=movement,
                launches=launches,
                owner_id=0,
                obs_tensors=_state_to_obs_tensors(states[0], player_id=0),
            )
        captured_windows += int(_captured_initial_comet(states))
        diff = _first_comet_diff(seed, states, movement)
        assert diff is None, diff
    assert captured_windows > 0


def _make_random_launch_entries(state: dict, movement: PlanetMovement, *, seed: int) -> LaunchEntries:
    rng = random.Random(104_729 + seed)
    owned_slots = [
        slot
        for slot, planet_id in enumerate(movement.planet_ids.tolist())
        if int(planet_id) >= 0
        and any(int(p["id"]) == int(planet_id) and int(p["owner"]) == 0 and int(p["ships"]) >= 4 for p in state["planets"])
    ]
    target_slots = [
        slot for slot, planet_id in enumerate(movement.planet_ids.tolist()) if int(planet_id) >= 0
    ]
    if not owned_slots or len(target_slots) < 2:
        return LaunchEntries(
            source_slots=torch.zeros(0, dtype=torch.long),
            target_slots=torch.zeros(0, dtype=torch.long),
            ships=torch.zeros(0, dtype=torch.float64),
            angle=torch.zeros(0, dtype=torch.float64),
            eta=torch.zeros(0, dtype=torch.float64),
            valid=torch.zeros(0, dtype=torch.bool),
        )

    ships_by_slot = {
        slot: int(p["ships"])
        for slot, planet_id in enumerate(movement.planet_ids.tolist())
        for p in state["planets"]
        if int(p["id"]) == int(planet_id)
    }
    rng.shuffle(owned_slots)
    launch_count = min(rng.randint(1, 3), len(owned_slots))
    src_rows: list[int] = []
    tgt_rows: list[int] = []
    ship_rows: list[float] = []
    angle_rows: list[float] = []
    for source_slot in owned_slots[:launch_count]:
        budget = ships_by_slot[source_slot]
        ships = max(1, min(budget - 1, rng.randint(1, max(1, budget // 2))))
        candidates = [slot for slot in target_slots if slot != source_slot]
        target_slot = rng.choice(candidates)
        sx = float(movement.x[0, source_slot].item())
        sy = float(movement.y[0, source_slot].item())
        tx = float(movement.x[0, target_slot].item())
        ty = float(movement.y[0, target_slot].item())
        src_rows.append(source_slot)
        tgt_rows.append(target_slot)
        ship_rows.append(float(ships))
        angle_rows.append(math.atan2(ty - sy, tx - sx))
        ships_by_slot[source_slot] -= ships

    return LaunchEntries(
        source_slots=torch.tensor(src_rows, dtype=torch.long),
        target_slots=torch.tensor(tgt_rows, dtype=torch.long),
        ships=torch.tensor(ship_rows, dtype=torch.float64),
        angle=torch.tensor(angle_rows, dtype=torch.float64),
        eta=torch.ones(len(src_rows), dtype=torch.float64),
        valid=torch.ones(len(src_rows), dtype=torch.bool),
    )


def _payload_to_moves(payload: dict[str, torch.Tensor]) -> list[list[float]]:
    count = int(payload["counts"].item())
    return [
        [
            int(payload["from_planet_id"][idx].item()),
            float(payload["angle"][idx].item()),
            int(payload["num_ships"][idx].item()),
        ]
        for idx in range(count)
    ]


def _rollout_truth_with_initial_moves(
    *,
    seed: int,
    horizon: int,
    moves: list[list[float]],
) -> list[dict]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=seed,
        config=RustConfig(episode_steps=500, enable_comets=False),
    )
    state0 = backend.reset(seed)[0]
    states = [state0]
    _, next_states = backend.step_with_states([[moves, []]])
    states.append(next_states[0])
    for _ in range(horizon - 1):
        _, next_states = backend.step_with_states([[[], []]])
        states.append(next_states[0])
    return states


def _project_with_private_launches(
    *,
    movement: PlanetMovement,
    obs_tensors: dict,
    entries: LaunchEntries,
) -> tuple[torch.Tensor, torch.Tensor]:
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors,
        movement=movement,
        entries=entries,
        player_id=0,
    )
    apply_private_planned_launches(
        movement=movement,
        launches=launches,
        owner_id=0,
        obs_tensors=obs_tensors,
    )
    status = movement.garrison_status(max_horizon=HORIZON)
    assert status.arrivals_by_owner is not None

    debit = torch.zeros(movement.P, dtype=movement.dtype, device=movement.device)
    if int(entries.valid.numel()) > 0:
        source_slots = entries.source_slots.to(device=movement.device, dtype=torch.long)
        ships = entries.ships.to(device=movement.device, dtype=movement.dtype)
        valid = entries.valid.to(device=movement.device, dtype=torch.bool)
        debit.index_put_((source_slots[valid],), ships[valid], accumulate=True)

    owner, ships, _, _ = _run_exact_recurrence(
        init_owner=status.owner[:, 0].unsqueeze(0),
        init_ships=(status.ships[:, 0] - debit).clamp(min=0.0).unsqueeze(0),
        prod=movement.planet_prod.unsqueeze(0),
        alive=movement.alive_by_step[: HORIZON + 1].transpose(0, 1).unsqueeze(0),
        arrivals=status.arrivals_by_owner[:, 1:, :].unsqueeze(0),
    )
    return owner.squeeze(0), ships.squeeze(0)


def test_movement_l3_matches_rust_with_random_valid_launches() -> None:
    for seed in range(FIDELITY_SEEDS):
        backend = RustBatchBackend(
            num_envs=1,
            num_players=2,
            seed=seed,
            config=RustConfig(episode_steps=500, enable_comets=False),
        )
        state0 = backend.reset(seed)[0]
        obs_tensors = _state_to_obs_tensors(state0, player_id=0)
        movement = PlanetMovement.from_obs_tensors(
            obs_tensors,
            config=MovementConfig(
                movement_horizon=HORIZON,
                track_fleets=True,
                player_count=2,
            ),
        )
        entries = _make_random_launch_entries(state0, movement, seed=seed)
        payload = entries_to_sparse_payload(entries, planet_ids=movement.planet_ids)
        moves = _payload_to_moves(payload)
        states = _rollout_truth_with_initial_moves(seed=seed, horizon=HORIZON, moves=moves)
        owner, ships = _project_with_private_launches(
            movement=movement,
            obs_tensors=obs_tensors,
            entries=entries,
        )
        diff = _compare_projected_garrison(
            tag="FIDELITY-L3",
            seed=seed,
            states=states,
            movement=movement,
            owner_by_step=owner,
            ships_by_step=ships,
            start_step=1,
        )
        assert diff is None, diff


def _resolve_planet_combat_python(
    *,
    planet_owner: int,
    planet_ships: int,
    arrivals: list[tuple[int, int]],
) -> tuple[int, int]:
    by_owner: dict[int, int] = defaultdict(int)
    for owner, ships in arrivals:
        if ships > 0:
            by_owner[int(owner)] += int(ships)
    if not by_owner:
        return planet_owner, planet_ships

    forces = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    winner_owner, top = forces[0]
    second = forces[1][1] if len(forces) > 1 else 0
    survivor = top - second
    if survivor <= 0:
        return planet_owner, planet_ships
    if planet_owner == winner_owner:
        return planet_owner, planet_ships + survivor
    if survivor > planet_ships:
        return winner_owner, survivor - planet_ships
    return planet_owner, planet_ships - survivor


@pytest.mark.parametrize(
    ("owner", "ships", "arrivals", "expected"),
    [
        (0, 10, [(1, 15)], (1, 5)),
        (0, 10, [(0, 5)], (0, 15)),
        (-1, 2, [(1, 10), (2, 7), (3, 4)], (1, 1)),
        (0, 10, [(1, 8), (2, 8)], (0, 10)),
        (0, 10, [(1, 10)], (0, 0)),
        (0, 10, [(1, 15), (1, 5), (2, 12)], (0, 2)),
    ],
)
def test_l4_python_combat_rule_pins_rust_semantics(
    owner: int,
    ships: int,
    arrivals: list[tuple[int, int]],
    expected: tuple[int, int],
) -> None:
    assert _resolve_planet_combat_python(
        planet_owner=owner,
        planet_ships=ships,
        arrivals=arrivals,
    ) == expected
