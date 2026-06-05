from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_upstream():
    module_path = Path(__file__).with_name("_upstream.py")
    spec = importlib.util.spec_from_file_location("_orbit_wars_producer_upstream", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Producer upstream module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_UPSTREAM = _load_upstream()


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


def agent(obs: Any):
    return _UPSTREAM.agent(_to_list_observation(obs))
