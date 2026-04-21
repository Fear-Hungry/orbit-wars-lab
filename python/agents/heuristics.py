from __future__ import annotations

import math
import random
from typing import Any

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.entities import (
    planet_id,
    planet_owner,
    planet_production,
    planet_ships,
    planet_x,
    planet_y,
)


def greedy_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    return greedy_moves(state, player)


def defensive_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    targets = [p for p in planets if planet_owner(p) == -1]
    if not own or not targets:
        return []
    moves = []
    for src in sorted(own, key=planet_ships, reverse=True):
        reserve = 18 if len(own) <= 2 else 10
        ships = planet_ships(src) - reserve
        if ships <= 0:
            continue
        sx, sy = planet_x(src), planet_y(src)
        tgt = min(targets, key=lambda p: float(planet_ships(p)) + 0.08 * math.hypot(planet_x(p) - sx, planet_y(p) - sy))
        angle = math.atan2(planet_y(tgt) - sy, planet_x(tgt) - sx)
        moves.append([planet_id(src), float(angle), int(ships * 0.5)])
        if len(moves) >= 4:
            break
    return moves


def rush_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    if not own or not enemies:
        return greedy_moves(state, player)
    enemy_home = max(enemies, key=planet_ships)
    moves = []
    for src in sorted(own, key=planet_ships, reverse=True)[:2]:
        ships = max(0, planet_ships(src) - 5)
        if ships <= 0:
            continue
        angle = math.atan2(planet_y(enemy_home) - planet_y(src), planet_x(enemy_home) - planet_x(src))
        moves.append([planet_id(src), float(angle), int(ships)])
    return moves


def anti_meta_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    neutrals = [p for p in planets if planet_owner(p) == -1]
    if not own:
        return []
    if enemies:
        targets = sorted(
            enemies,
            key=lambda p: (
                -planet_ships(p),
                -sum(1 for n in neutrals if math.hypot(planet_x(n) - planet_x(p), planet_y(n) - planet_y(p)) < 20.0),
            ),
        )
        focus = targets[0]
    elif neutrals:
        focus = max(neutrals, key=lambda p: (planet_production(p), -planet_ships(p)))
    else:
        return greedy_moves(state, player)

    moves = []
    for src in sorted(own, key=planet_ships, reverse=True)[:3]:
        ships = max(0, planet_ships(src) - 7)
        if ships <= 0:
            continue
        angle = math.atan2(planet_y(focus) - planet_y(src), planet_x(focus) - planet_x(src))
        moves.append([planet_id(src), float(angle), int(max(1, ships * 0.6))])
    return moves


def weak_random_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    planets = state.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    targets = [p for p in planets if planet_owner(p) != player]
    if not own or not targets:
        return []
    rng = random.Random(int(state.get("step", 0)) + 997 * player + len(planets))
    src = rng.choice(sorted(own, key=planet_ships, reverse=True)[: max(1, min(3, len(own)))])
    tgt = rng.choice(targets)
    ships = max(0, int((planet_ships(src) - 6) * rng.uniform(0.25, 0.55)))
    if ships <= 0:
        return []
    angle = math.atan2(planet_y(tgt) - planet_y(src), planet_x(tgt) - planet_x(src))
    return [[planet_id(src), float(angle), ships]]
