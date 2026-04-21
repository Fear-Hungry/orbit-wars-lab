from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .entities import (
    fleet_angle,
    fleet_from_planet_id,
    fleet_owner,
    fleet_ships,
    fleet_x,
    fleet_y,
    planet_id,
    planet_owner,
    planet_production,
    planet_radius,
    planet_ships,
    planet_x,
    planet_y,
)

BOARD_SIZE = 100.0
CENTER = 50.0
MAX_PLAYERS = 4


@dataclass(frozen=True)
class EncoderConfig:
    max_planets: int = 96
    max_fleets: int = 256
    include_fleets: bool = True


DEFAULT_ENCODER_CONFIG = EncoderConfig()


def _owner_rel(owner: int, player: int) -> tuple[float, float, float, float]:
    if owner == -1:
        return 0.0, 0.0, 1.0, 0.0
    if owner == player:
        return 1.0, 0.0, 0.0, 0.0
    return 0.0, 1.0, 0.0, 0.0


def encode_state(
    state: dict[str, Any],
    player: int,
    cfg: EncoderConfig = DEFAULT_ENCODER_CONFIG,
) -> np.ndarray:
    """Fixed-size flat encoding for first PPO baseline.

    Later, replace this with an entity transformer encoder. This flat encoder is
    deliberately simple to make parity, debugging and ablations easier.
    """

    planets = state.get("planets", [])
    fleets = state.get("fleets", [])
    step = float(state.get("step", 0)) / 500.0
    angular = float(state.get("angular_velocity", 0.0))

    own_ships = sum(planet_ships(p) for p in planets if planet_owner(p) == player)
    enemy_ships = sum(planet_ships(p) for p in planets if planet_owner(p) not in (-1, player))
    own_prod = sum(planet_production(p) for p in planets if planet_owner(p) == player)
    enemy_prod = sum(planet_production(p) for p in planets if planet_owner(p) not in (-1, player))

    global_feats = np.array(
        [
            step,
            angular,
            len(planets) / max(cfg.max_planets, 1),
            len(fleets) / max(cfg.max_fleets, 1),
            math.log1p(own_ships) / 8.0,
            math.log1p(enemy_ships) / 8.0,
            own_prod / 64.0,
            enemy_prod / 64.0,
        ],
        dtype=np.float32,
    )

    planet_feats = np.zeros((cfg.max_planets, 14), dtype=np.float32)
    for i, p in enumerate(planets[: cfg.max_planets]):
        pid = planet_id(p)
        owner = planet_owner(p)
        x = planet_x(p)
        y = planet_y(p)
        radius = planet_radius(p)
        ships = planet_ships(p)
        prod = planet_production(p)
        owner_self, owner_enemy, owner_neutral, owner_other = _owner_rel(int(owner), player)
        dx = (x - CENTER) / CENTER
        dy = (y - CENTER) / CENTER
        dist_center = math.sqrt(dx * dx + dy * dy)
        planet_feats[i] = np.array(
            [
                1.0,
                owner_self,
                owner_enemy,
                owner_neutral,
                owner_other,
                x / BOARD_SIZE,
                y / BOARD_SIZE,
                dx,
                dy,
                dist_center,
                radius / 10.0,
                math.log1p(max(float(ships), 0.0)) / 8.0,
                float(prod) / 5.0,
                float(pid) / 512.0,
            ],
            dtype=np.float32,
        )

    fleet_feats = np.zeros((cfg.max_fleets, 10), dtype=np.float32)
    if cfg.include_fleets:
        for i, f in enumerate(fleets[: cfg.max_fleets]):
            owner = fleet_owner(f)
            x = fleet_x(f)
            y = fleet_y(f)
            angle = fleet_angle(f)
            from_pid = fleet_from_planet_id(f)
            ships = fleet_ships(f)
            owner_self, owner_enemy, owner_neutral, owner_other = _owner_rel(int(owner), player)
            fleet_feats[i] = np.array(
                [
                    1.0,
                    owner_self,
                    owner_enemy,
                    owner_neutral,
                    x / BOARD_SIZE,
                    y / BOARD_SIZE,
                    math.cos(angle),
                    math.sin(angle),
                    math.log1p(max(float(ships), 0.0)) / 8.0,
                    float(from_pid) / 512.0,
                ],
                dtype=np.float32,
            )

    return np.concatenate([global_feats, planet_feats.reshape(-1), fleet_feats.reshape(-1)])


def observation_dim(cfg: EncoderConfig = DEFAULT_ENCODER_CONFIG) -> int:
    return 8 + cfg.max_planets * 14 + cfg.max_fleets * 10
