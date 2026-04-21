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


def to_official_observation(state: dict[str, Any], player: int) -> dict[str, Any]:
    """Build a Kaggle-shaped per-player observation from simulator state."""

    obs = {
        "planets": list(state.get("planets", [])),
        "fleets": list(state.get("fleets", [])),
        "player": int(player),
        "angular_velocity": float(state.get("angular_velocity", 0.0)),
        "initial_planets": list(state.get("initial_planets", [])),
        "next_fleet_id": int(state.get("next_fleet_id", 0)),
        "comets": list(state.get("comets", [])),
        "comet_planet_ids": list(state.get("comet_planet_ids", [])),
    }
    for key, value in state.items():
        obs.setdefault(key, value)
    return obs
