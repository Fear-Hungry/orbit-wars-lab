from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig

PRODUCER_DIR = Path(__file__).resolve().parent / "opponents/producer/producer"
sys.path.insert(0, str(PRODUCER_DIR))

from orbit_lite.adapter import single_obs_to_tensor  # noqa: E402
from orbit_lite.movement import MovementConfig, PlanetMovement  # noqa: E402

HORIZON = 18
FIDELITY_SEEDS = 200
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
