from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orbit_wars_gym.entities import (
    fleet_angle,
    fleet_id,
    fleet_owner,
    fleet_ships,
    fleet_x,
    fleet_y,
    planet_id,
    planet_owner,
    planet_ships,
    planet_x,
    planet_y,
)

DEFAULT_FLOAT_TOLERANCE = 1e-9

Entity = Mapping[str, Any] | Sequence[Any]


def assert_official_snapshot_parity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    done: bool,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
    float_tolerance_reason: str | None = None,
) -> None:
    """Assert the documented parity contract against an official snapshot."""

    if float_tolerance != DEFAULT_FLOAT_TOLERANCE and not float_tolerance_reason:
        raise ValueError("float_tolerance can only be relaxed with an explicit justification")

    assert bool(actual["done"]) is bool(done)
    _assert_planets_match(actual["planets"], expected["planets"], float_tolerance)
    _assert_fleets_match(actual["fleets"], expected["fleets"], float_tolerance)


def _assert_planets_match(actual: Sequence[Entity], expected: Sequence[Entity], tol: float) -> None:
    assert len(actual) == len(expected)
    for actual_planet, expected_planet in zip(actual, expected, strict=True):
        assert planet_id(actual_planet) == planet_id(expected_planet)
        assert planet_owner(actual_planet) == planet_owner(expected_planet)
        assert planet_ships(actual_planet) == planet_ships(expected_planet)
        assert abs(planet_x(actual_planet) - planet_x(expected_planet)) <= tol
        assert abs(planet_y(actual_planet) - planet_y(expected_planet)) <= tol


def _assert_fleets_match(actual: Sequence[Entity], expected: Sequence[Entity], tol: float) -> None:
    assert len(actual) == len(expected)
    for actual_fleet, expected_fleet in zip(actual, expected, strict=True):
        assert fleet_id(actual_fleet) == fleet_id(expected_fleet)
        assert fleet_owner(actual_fleet) == fleet_owner(expected_fleet)
        assert fleet_ships(actual_fleet) == fleet_ships(expected_fleet)
        assert abs(fleet_x(actual_fleet) - fleet_x(expected_fleet)) <= tol
        assert abs(fleet_y(actual_fleet) - fleet_y(expected_fleet)) <= tol
        assert abs(fleet_angle(actual_fleet) - fleet_angle(expected_fleet)) <= tol
