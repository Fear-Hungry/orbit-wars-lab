"""Small obs/movement helpers vendored from ``bots.oep.planner`` so the PGS
submission ships without the (3k-line) OEP planner module."""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

from orbit_lite.movement import PlanetMovement
from orbit_lite.movement_step import (
    LaunchEntries,
    infer_planned_launches_from_entries,
)
from orbit_lite.planner_core import _empty_entries


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


def _with_tensor_player(obs_tensors: dict, player_id: int) -> dict:
    copied = dict(obs_tensors)
    player = obs_tensors["player"]
    copied["player"] = torch.full_like(player, int(player_id))
    return copied


def _clone_movement(movement: PlanetMovement) -> PlanetMovement:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(PlanetMovement):
        value = getattr(movement, field.name)
        values[field.name] = value.clone() if isinstance(value, Tensor) else value
    return PlanetMovement(**values)


def _invalidate_garrison_cache(movement: PlanetMovement) -> None:
    movement.garrison_owner_cache = None
    movement.garrison_ships_cache = None
    movement.garrison_pre_combat_owner_cache = None
    movement.garrison_pre_combat_ships_cache = None
    movement.garrison_dirty_from = None


def _debit_entry_sources(movement: PlanetMovement, entries: LaunchEntries) -> None:
    valid = entries.valid & (entries.ships >= 1.0)
    valid_slots = torch.where(valid)[0]
    if int(valid_slots.numel()) == 0:
        return
    ships = movement.planet_ships.clone()
    for entry_idx in valid_slots.tolist():
        source_slot = int(entries.source_slots[entry_idx].item())
        if source_slot < 0 or source_slot >= int(ships.shape[0]):
            continue
        ships[source_slot] = (ships[source_slot] - entries.ships[entry_idx]).clamp(min=0.0)
    movement.planet_ships = ships
    _invalidate_garrison_cache(movement)


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
            raise ValueError(
                f"policy emitted move from unknown planet id: {int(from_ids[idx].item())}"
            )
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
