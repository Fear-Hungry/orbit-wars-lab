from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Entity = Mapping[str, Any] | Sequence[Any]


def _field(entity: Entity, index: int, key: str) -> Any:
    if isinstance(entity, Mapping):
        return entity[key]
    return entity[index]


def planet_id(planet: Entity) -> int:
    return int(_field(planet, 0, "id"))


def planet_owner(planet: Entity) -> int:
    return int(_field(planet, 1, "owner"))


def planet_x(planet: Entity) -> float:
    return float(_field(planet, 2, "x"))


def planet_y(planet: Entity) -> float:
    return float(_field(planet, 3, "y"))


def planet_radius(planet: Entity) -> float:
    return float(_field(planet, 4, "radius"))


def planet_ships(planet: Entity) -> int:
    return int(_field(planet, 5, "ships"))


def planet_production(planet: Entity) -> int:
    return int(_field(planet, 6, "production"))


def fleet_id(fleet: Entity) -> int:
    return int(_field(fleet, 0, "id"))


def fleet_owner(fleet: Entity) -> int:
    return int(_field(fleet, 1, "owner"))


def fleet_x(fleet: Entity) -> float:
    return float(_field(fleet, 2, "x"))


def fleet_y(fleet: Entity) -> float:
    return float(_field(fleet, 3, "y"))


def fleet_angle(fleet: Entity) -> float:
    return float(_field(fleet, 4, "angle"))


def fleet_from_planet_id(fleet: Entity) -> int:
    return int(_field(fleet, 5, "from_planet_id"))


def fleet_ships(fleet: Entity) -> int:
    return int(_field(fleet, 6, "ships"))
