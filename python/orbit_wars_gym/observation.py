from __future__ import annotations

from typing import Any

OFFICIAL_OBSERVATION_KEYS = (
    "planets",
    "fleets",
    "player",
    "angular_velocity",
    "initial_planets",
    "next_fleet_id",
    "comets",
    "comet_planet_ids",
)


def _planet_row(planet: Any) -> list[Any]:
    """Official planet row: [id, owner, x, y, radius, ships, production]."""
    if isinstance(planet, dict):
        return [
            int(planet["id"]),
            int(planet["owner"]),
            float(planet["x"]),
            float(planet["y"]),
            float(planet["radius"]),
            int(planet["ships"]),
            int(planet["production"]),
        ]
    return list(planet)


def _fleet_row(fleet: Any) -> list[Any]:
    """Official fleet row: [id, owner, x, y, angle, from_planet_id, ships]."""
    if isinstance(fleet, dict):
        return [
            int(fleet["id"]),
            int(fleet["owner"]),
            float(fleet["x"]),
            float(fleet["y"]),
            float(fleet["angle"]),
            int(fleet["from_planet_id"]),
            int(fleet["ships"]),
        ]
    return list(fleet)


def to_official_observation(state: dict[str, Any], player: int) -> dict[str, Any]:
    """Build a Kaggle-shaped per-player observation from simulator state.

    Planets and fleets are emitted as the official list rows the Kaggle
    interpreter uses ([id, owner, x, y, radius, ships, production] and
    [id, owner, x, y, angle, from_planet_id, ships]), not dicts, so a real
    Kaggle-format agent (e.g. the packaged Producer, whose adapter slices
    ``planet[:7]``) runs against the local régua exactly as it would on Kaggle.
    """

    obs = {
        "planets": [_planet_row(p) for p in state.get("planets", [])],
        "fleets": [_fleet_row(f) for f in state.get("fleets", [])],
        "player": int(player),
        "angular_velocity": float(state.get("angular_velocity", 0.0)),
        "initial_planets": [_planet_row(p) for p in state.get("initial_planets", [])],
        "next_fleet_id": int(state.get("next_fleet_id", 0)),
        "comets": list(state.get("comets", [])),
        "comet_planet_ids": list(state.get("comet_planet_ids", [])),
    }
    for key, value in state.items():
        obs.setdefault(key, value)
    return obs
