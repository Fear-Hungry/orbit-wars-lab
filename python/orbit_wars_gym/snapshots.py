from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from orbit_wars_gym.observation import OFFICIAL_OBSERVATION_KEYS


def official_initial_snapshot(num_players: int, *, player: int = 0, comet_speed: float = 4.0) -> dict[str, Any]:
    """Materialize the official Kaggle Orbit Wars opening observation."""

    if num_players not in (2, 4):
        raise ValueError("Orbit Wars supports only 2 or 4 players")
    if player < 0 or player >= num_players:
        raise ValueError("player index out of range for snapshot")

    from kaggle_environments.envs.orbit_wars import orbit_wars

    state = [
        SimpleNamespace(observation=SimpleNamespace(), action=None, reward=0, status="ACTIVE")
        for _ in range(num_players)
    ]
    env = SimpleNamespace(configuration=SimpleNamespace(cometSpeed=float(comet_speed)), done=False)
    observations = orbit_wars.interpreter(state, env)
    obs = observations[player].observation
    return {key: deepcopy(getattr(obs, key)) for key in OFFICIAL_OBSERVATION_KEYS}


def official_seeded_initial_snapshot(
    num_players: int,
    seed: int,
    *,
    player: int = 0,
    comet_speed: float = 4.0,
) -> dict[str, Any]:
    previous_state = random.getstate()
    try:
        random.seed(seed)
        return official_initial_snapshot(num_players, player=player, comet_speed=comet_speed)
    finally:
        random.setstate(previous_state)


def snapshot_to_state(
    snapshot: Mapping[str, Any],
    *,
    num_players: int,
    step: int = 0,
    done: bool = False,
) -> dict[str, Any]:
    """Wrap an official per-player observation in the local simulator state shape."""

    if num_players not in (2, 4):
        raise ValueError("Orbit Wars supports only 2 or 4 players")

    return {
        "step": int(step),
        "num_players": int(num_players),
        "angular_velocity": float(snapshot["angular_velocity"]),
        "planets": deepcopy(snapshot["planets"]),
        "initial_planets": deepcopy(snapshot["initial_planets"]),
        "fleets": deepcopy(snapshot["fleets"]),
        "next_fleet_id": int(snapshot["next_fleet_id"]),
        "comets": deepcopy(snapshot["comets"]),
        "comet_planet_ids": list(snapshot["comet_planet_ids"]),
        "done": bool(done),
    }
