"""Board-symmetry transforms for the map-bias audit (todo P5).

Orbit Wars is played on a square ``100x100`` board with the sun fixed at the
centre ``(50, 50)``. The geometry is therefore invariant under the dihedral
symmetries of the square (here we use 180° rotation and reflection across the
vertical mid-axis) and, in 2-player games, under swapping the two players.

A position-agnostic policy should be *equivariant* under these transforms: the
optimal launch/source/target/frac/offset decision is unchanged because the
decoder ranks sources/targets by ships/production (preserved by the transforms)
and angles are relative to the computed base angle (which co-rotates). Any change
in the policy's logits between a state and its transform is spurious sensitivity
to the absolute ``x/y`` and ``planet_id`` features — exactly the map bias P5 is
auditing.

Transforms operate on the dict-entity state form produced by the Rust backend.
"""

from __future__ import annotations

import math
from typing import Any

BOARD_SIZE = 100.0


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _map_planet(planet: dict[str, Any], fx, fy) -> dict[str, Any]:
    out = dict(planet)
    out["x"] = fx(planet["x"], planet["y"])
    out["y"] = fy(planet["x"], planet["y"])
    return out


def _map_fleet(fleet: dict[str, Any], fx, fy, fangle) -> dict[str, Any]:
    out = dict(fleet)
    out["x"] = fx(fleet["x"], fleet["y"])
    out["y"] = fy(fleet["x"], fleet["y"])
    out["angle"] = _wrap_angle(fangle(fleet["angle"]))
    return out


def _transform(state: dict[str, Any], fx, fy, fangle) -> dict[str, Any]:
    out = dict(state)
    out["planets"] = [_map_planet(p, fx, fy) for p in state.get("planets", [])]
    out["fleets"] = [_map_fleet(f, fx, fy, fangle) for f in state.get("fleets", [])]
    return out


def rotate_state_180(state: dict[str, Any]) -> dict[str, Any]:
    """180° rotation about the board centre: ``(x, y) -> (100-x, 100-y)``."""
    return _transform(
        state,
        fx=lambda x, y: BOARD_SIZE - x,
        fy=lambda x, y: BOARD_SIZE - y,
        fangle=lambda a: a + math.pi,
    )


def reflect_state_x(state: dict[str, Any]) -> dict[str, Any]:
    """Reflection across the vertical mid-axis ``x = 50``: ``x -> 100-x``."""
    return _transform(
        state,
        fx=lambda x, y: BOARD_SIZE - x,
        fy=lambda x, y: y,
        fangle=lambda a: math.pi - a,
    )


def swap_players_2p(state: dict[str, Any]) -> dict[str, Any]:
    """Swap owners 0 <-> 1 (neutrals untouched). Valid only for 2-player states.

    ``encode_state(swap_players_2p(s), player=0)`` should mirror
    ``encode_state(s, player=1)`` for a perspective-correct encoder.
    """

    def _swap_owner(entity: dict[str, Any]) -> dict[str, Any]:
        out = dict(entity)
        owner = int(entity["owner"])
        if owner == 0:
            out["owner"] = 1
        elif owner == 1:
            out["owner"] = 0
        return out

    out = dict(state)
    out["planets"] = [_swap_owner(p) for p in state.get("planets", [])]
    out["fleets"] = [_swap_owner(f) for f in state.get("fleets", [])]
    return out
