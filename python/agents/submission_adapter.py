from __future__ import annotations

import math
from typing import Any

from orbit_wars_gym.action_decoder import (
    DEFAULT_DECODER_CONFIG,
    decode_discrete_action,
)
from orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state
from orbit_wars_gym.entities import (
    planet_id,
    planet_owner,
    planet_production,
    planet_ships,
    planet_x,
    planet_y,
)


def encode_submission(obs: dict[str, Any], player: int) -> dict[str, Any]:
    """Build a lightweight submission-oriented feature pack.

    The feature pack keeps a flat numeric encoding for future exported models
    while also exposing aggregated game-state cues for the current lightweight
    policy.
    """

    planets = obs.get("planets", [])
    own = [p for p in planets if planet_owner(p) == player]
    enemies = [p for p in planets if planet_owner(p) not in (-1, player)]
    neutrals = [p for p in planets if planet_owner(p) == -1]
    return {
        "player": player,
        "flat": encode_state(obs, player, DEFAULT_ENCODER_CONFIG),
        "own_count": len(own),
        "enemy_count": len(enemies),
        "neutral_count": len(neutrals),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "own_ships": sum(planet_ships(p) for p in own),
        "enemy_ships": sum(planet_ships(p) for p in enemies),
        "own_prod": sum(planet_production(p) for p in own),
        "enemy_prod": sum(planet_production(p) for p in enemies),
    }


def policy_forward(features: dict[str, Any]) -> list[int]:
    """Return one compact discrete action.

    This is intentionally lightweight and deterministic. It already follows the
    final Kaggle flow `encode -> policy_forward -> decode`, so an exported neural
    policy can replace this function without changing the rest of the pipeline.
    """

    source_rank = 0
    target_rank = 0
    offset_idx = 2

    own_ships = int(features["own_ships"])
    enemy_ships = int(features["enemy_ships"])
    own_prod = int(features["own_prod"])
    enemy_prod = int(features["enemy_prod"])
    neutral_count = int(features["neutral_count"])
    own_count = int(features["own_count"])

    if enemy_prod > own_prod or enemy_ships >= max(own_ships - 4, 1):
        frac_idx = 3
    elif neutral_count > 0 and own_count <= 2:
        frac_idx = 2
    else:
        frac_idx = 1

    angular_velocity = float(features["angular_velocity"])
    if angular_velocity > 0.04:
        offset_idx = 1
    elif angular_velocity < -0.04:
        offset_idx = 3

    return [source_rank, target_rank, frac_idx, offset_idx]


def decode_submission_action(obs: dict[str, Any], player: int, action: list[int]) -> list[list[float]]:
    return decode_discrete_action(obs, player, action, DEFAULT_DECODER_CONFIG)


def _moves_are_legal(obs: dict[str, Any], player: int, moves: list[list[float]]) -> bool:
    own_ids = {planet_id(p) for p in obs.get("planets", []) if planet_owner(p) == player}
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return False
        from_id, angle, ships = move
        if int(from_id) not in own_ids:
            return False
        if not math.isfinite(float(angle)):
            return False
        if int(ships) <= 0:
            return False
    return True


def fallback_greedy(obs: dict[str, Any]) -> list[list[float]]:
    try:
        player = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        own = [p for p in planets if planet_owner(p) == player and planet_ships(p) >= 2]
        if not own:
            return []
        own.sort(key=lambda planet: (planet_ships(planet), planet_production(planet)), reverse=True)
        src = own[0]
        candidates = [planet for planet in planets if planet_id(planet) != planet_id(src)]
        if not candidates:
            return []

        sx, sy = planet_x(src), planet_y(src)

        def target_score(planet: Any) -> float:
            tx, ty = planet_x(planet), planet_y(planet)
            dist = math.hypot(tx - sx, ty - sy)
            owner = planet_owner(planet)
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            return float(planet_production(planet)) * 10.0 + enemy_bonus + neutral_bonus - 0.15 * dist - 0.12 * float(planet_ships(planet))

        target = max(candidates, key=target_score)
        ships = max(2, min(planet_ships(src) - 8, planet_ships(src) // 4))
        if ships <= 0:
            return []
        angle = math.atan2(planet_y(target) - sy, planet_x(target) - sx)
        moves = [[planet_id(src), float(angle), int(ships)]]
        return moves if _moves_are_legal(obs, player, moves) else []
    except Exception:
        return []


def safe_submission_agent(obs: dict[str, Any]) -> list[list[float]]:
    """Local mirror of the Kaggle submission flow that surfaces failures."""

    try:
        player = int(obs.get("player", 0))
        features = encode_submission(obs, player)
        action = policy_forward(features)
        moves = decode_submission_action(obs, player, action)
        if not _moves_are_legal(obs, player, moves):
            raise ValueError(f"submission policy produced illegal moves for player={player}: {moves!r}")
        return list(moves)
    except Exception:
        return fallback_greedy(obs)
