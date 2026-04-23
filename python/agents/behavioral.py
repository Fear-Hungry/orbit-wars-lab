from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from python.orbit_wars_gym.entities import (
    fleet_owner,
    fleet_ships,
    fleet_x,
    fleet_y,
    planet_id,
    planet_owner,
    planet_production,
    planet_ships,
    planet_x,
    planet_y,
)

Policy = Callable[[dict[str, Any], int], list[list[float]]]

MAX_MOVES = 4
MIN_LAUNCH = 1


def planets(state: dict[str, Any]) -> list[Any]:
    return list(state.get("planets", []))


def fleets(state: dict[str, Any]) -> list[Any]:
    return list(state.get("fleets", []))


def own_planets(state: dict[str, Any], player: int) -> list[Any]:
    return [planet for planet in planets(state) if planet_owner(planet) == player]


def neutral_planets(state: dict[str, Any]) -> list[Any]:
    return [planet for planet in planets(state) if planet_owner(planet) == -1]


def enemy_planets(state: dict[str, Any], player: int) -> list[Any]:
    return [planet for planet in planets(state) if planet_owner(planet) not in (-1, player)]


def xy(entity: Any) -> tuple[float, float]:
    return (planet_x(entity), planet_y(entity))


def distance(source: Any, target: Any) -> float:
    sx, sy = xy(source)
    tx, ty = xy(target)
    return math.hypot(tx - sx, ty - sy)


def angle_to(source: Any, target: Any) -> float:
    sx, sy = xy(source)
    tx, ty = xy(target)
    return math.atan2(ty - sy, tx - sx)


def launchable_ships(source: Any, reserve: int, fraction: float = 1.0) -> int:
    spare = max(0, planet_ships(source) - reserve)
    return max(0, int(math.floor(spare * fraction)))


def build_move(source: Any, target: Any, ships: int) -> list[float]:
    return [planet_id(source), float(angle_to(source, target)), int(ships)]


def source_strength_key(source: Any) -> tuple[int, int, int]:
    return (planet_ships(source), planet_production(source), -planet_id(source))


def sorted_sources(candidates: list[Any], reserve: int) -> list[Any]:
    return sorted(
        [source for source in candidates if launchable_ships(source, reserve) >= MIN_LAUNCH],
        key=source_strength_key,
        reverse=True,
    )


def player_strength(state: dict[str, Any], player: int) -> float:
    owned = [planet for planet in planets(state) if planet_owner(planet) == player]
    return sum(float(planet_ships(p)) + 8.0 * float(planet_production(p)) for p in owned)


def leader_player(state: dict[str, Any], player: int) -> int | None:
    opponents = sorted({planet_owner(p) for p in planets(state)} - {-1, player})
    if not opponents:
        return None
    return max(opponents, key=lambda opponent: (player_strength(state, opponent), -opponent))


def nearby_enemy_pressure(state: dict[str, Any], target: Any, player: int) -> float:
    pressure = 0.0
    for planet in enemy_planets(state, player):
        pressure += (planet_ships(planet) + 4.0 * planet_production(planet)) / (distance(planet, target) + 8.0)
    for fleet in fleets(state):
        if fleet_owner(fleet) == player:
            continue
        dx = fleet_x(fleet) - planet_x(target)
        dy = fleet_y(fleet) - planet_y(target)
        pressure += 0.5 * fleet_ships(fleet) / (math.hypot(dx, dy) + 6.0)
    return pressure


def nearest_source_for_target(sources: list[Any], target: Any, reserve: int) -> Any | None:
    viable = [source for source in sources if planet_id(source) != planet_id(target)]
    viable = [source for source in viable if launchable_ships(source, reserve) >= MIN_LAUNCH]
    if not viable:
        return None
    return min(viable, key=lambda source: (distance(source, target), -planet_ships(source), planet_id(source)))


def enemy_pressure_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    """Concentrate surplus ships on vulnerable enemy holdings."""
    own = own_planets(state, player)
    targets = enemy_planets(state, player)
    if not own or not targets:
        return []

    moves: list[list[float]] = []
    for source in sorted_sources(own, reserve=7):
        target = max(
            targets,
            key=lambda planet: (
                14.0 * planet_production(planet)
                - 0.75 * planet_ships(planet)
                - 0.18 * distance(source, planet),
                -planet_id(planet),
            ),
        )
        ships = launchable_ships(source, reserve=7, fraction=0.65)
        if ships >= MIN_LAUNCH:
            moves.append(build_move(source, target, ships))
        if len(moves) >= MAX_MOVES:
            break
    return moves


def neutral_economy_greed_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    """Expand into efficient neutral planets while preserving a small reserve."""
    own = own_planets(state, player)
    neutrals = neutral_planets(state)
    if not own or not neutrals:
        return enemy_pressure_agent(state, player)

    moves: list[list[float]] = []
    claimed: set[int] = set()
    for source in sorted_sources(own, reserve=5):
        candidates = [planet for planet in neutrals if planet_id(planet) not in claimed]
        if not candidates:
            break
        target = max(
            candidates,
            key=lambda planet: (
                18.0 * planet_production(planet)
                - 1.15 * planet_ships(planet)
                - 0.22 * distance(source, planet),
                -planet_id(planet),
            ),
        )
        required = planet_ships(target) + 1
        ships = min(launchable_ships(source, reserve=5, fraction=0.85), required + 3)
        if ships >= MIN_LAUNCH:
            moves.append(build_move(source, target, ships))
            claimed.add(planet_id(target))
        if len(moves) >= MAX_MOVES:
            break
    return moves


def risk_balanced_defense_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    """Reinforce exposed own planets, then use safe leftovers for nearby economy."""
    own = own_planets(state, player)
    if not own:
        return []

    vulnerable = sorted(
        own,
        key=lambda planet: (
            nearby_enemy_pressure(state, planet, player)
            + 0.08 * planet_production(planet)
            - 0.04 * planet_ships(planet),
            -planet_id(planet),
        ),
        reverse=True,
    )
    moves: list[list[float]] = []
    for target in vulnerable[:2]:
        source = nearest_source_for_target(own, target, reserve=9)
        if source is None:
            continue
        pressure_gap = max(0.0, nearby_enemy_pressure(state, target, player) - 0.08 * planet_ships(target))
        ships = min(launchable_ships(source, reserve=9, fraction=0.5), max(2, int(math.ceil(pressure_gap * 3.0))))
        if ships >= MIN_LAUNCH:
            moves.append(build_move(source, target, ships))

    if moves:
        return moves[:MAX_MOVES]
    return neutral_economy_greed_agent(state, player)[:2]


def leader_denial_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    """Attack the strongest opponent's best planets to reduce ranking runaway."""
    own = own_planets(state, player)
    leader = leader_player(state, player)
    if not own or leader is None:
        return neutral_economy_greed_agent(state, player)

    leader_targets = [planet for planet in planets(state) if planet_owner(planet) == leader]
    if not leader_targets:
        return neutral_economy_greed_agent(state, player)

    moves: list[list[float]] = []
    for source in sorted_sources(own, reserve=6):
        target = max(
            leader_targets,
            key=lambda planet: (
                20.0 * planet_production(planet)
                + 0.4 * planet_ships(planet)
                - 0.16 * distance(source, planet),
                -planet_id(planet),
            ),
        )
        ships = launchable_ships(source, reserve=6, fraction=0.72)
        if ships >= MIN_LAUNCH:
            moves.append(build_move(source, target, ships))
        if len(moves) >= MAX_MOVES:
            break
    return moves


BEHAVIORAL_POLICIES: dict[str, Policy] = {
    "enemy_pressure": enemy_pressure_agent,
    "neutral_economy_greed": neutral_economy_greed_agent,
    "risk_balanced_defense": risk_balanced_defense_agent,
    "leader_denial": leader_denial_agent,
}
