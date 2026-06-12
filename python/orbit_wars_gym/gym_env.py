from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
    decode_discrete_action,
    greedy_moves,
)
from .backend import RustBatchBackend, RustConfig
from .encoding import EncoderConfig, encode_state, observation_dim
from .entities import (
    fleet_angle,
    fleet_id,
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
from .rules import normalized_margin

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
SHIP_SPEED = 6.0


class OrbitWarsGymEnv(gym.Env):
    """Single-learning-agent wrapper.

    Player 0 is controlled by the learner. Other players use a configurable
    opponent callable. This is the fastest route to a first PPO baseline.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 2,
        seed: int = 0,
        encoder_cfg: EncoderConfig | None = None,
        rust_cfg: RustConfig | None = None,
        opponent_policy=None,
        decoder_cfg: DecoderConfig | None = None,
        sun_loss_penalty: float = 0.02,
        border_loss_penalty: float = 0.02,
        ship_margin_scale: float = 0.0,
        normalized_margin_scale: float = 0.0,
        base_shaping_scale: float = 1.0,
        comet_shaping_scale: float = 0.0,
        shaping_gamma: float = 0.99,
        four_player_vulnerability_scale: float = 0.0,
        four_player_leader_scale: float = 0.0,
        four_player_third_player_scale: float = 0.0,
    ):
        super().__init__()
        self.num_players = num_players
        self.encoder_cfg = encoder_cfg or EncoderConfig()
        self.backend = RustBatchBackend(num_envs=1, num_players=num_players, seed=seed, config=rust_cfg)
        self.opponent_policy = opponent_policy or greedy_moves
        self.decoder_cfg = decoder_cfg or DEFAULT_DECODER_CONFIG
        self.sun_loss_penalty = float(sun_loss_penalty)
        self.border_loss_penalty = float(border_loss_penalty)
        self.ship_margin_scale = float(ship_margin_scale)
        self.normalized_margin_scale = float(normalized_margin_scale)
        self.base_shaping_scale = float(base_shaping_scale)
        self.comet_shaping_scale = float(comet_shaping_scale)
        # Discount for potential-based shaping F = γ·Φ(s') − Φ(s). Must match the
        # RL discount used by GAE for the shaping to stay policy-invariant
        # (Ng, Harada & Russell 1999).
        self.shaping_gamma = float(shaping_gamma)
        self.four_player_vulnerability_scale = float(four_player_vulnerability_scale)
        self.four_player_leader_scale = float(four_player_leader_scale)
        self.four_player_third_player_scale = float(four_player_third_player_scale)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_dim(self.encoder_cfg),), dtype=np.float32)
        # [launch, source_rank, target_rank, fraction_idx, offset_idx]; launch==0 passes.
        self.action_space = spaces.MultiDiscrete([2, 16, 32, 4, 5])
        self.state: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        states = self.backend.reset(0 if seed is None else seed)
        self.state = states[0]
        obs = encode_state(self.state, player=0, cfg=self.encoder_cfg)
        return obs, {}

    def step(self, action):
        assert self.state is not None
        previous_state = self.state
        actions = [[[] for _ in range(self.num_players)]]
        actions[0][0] = decode_discrete_action(self.state, 0, action, self.decoder_cfg)
        for pid in range(1, self.num_players):
            actions[0][pid] = self.opponent_policy(self.state, pid)
        outcomes, states = self.backend.step_with_states(actions)
        next_state = states[0]
        self.state = next_state
        done = bool(outcomes[0]["done"])
        obs = encode_state(self.state, player=0, cfg=self.encoder_cfg)
        base_shaping_reward = self._base_shaping_reward(
            previous_state, next_state, player=0, player_moves=actions[0][0], done=done
        )
        ship_margin_reward = self._ship_margin_reward(previous_state, next_state, player=0)
        normalized_margin_reward = self._normalized_margin_reward(
            previous_state,
            next_state,
            player=0,
            done=done,
        )
        comet_shaping_reward = self._comet_auxiliary_reward(previous_state, next_state, player=0)
        vulnerability_reward, leader_reward, third_player_reward = self._four_player_strategic_reward(
            previous_state,
            next_state,
            player=0,
        )
        reward = (
            self.base_shaping_scale * base_shaping_reward
            + ship_margin_reward
            + normalized_margin_reward
            + self.comet_shaping_scale * comet_shaping_reward
            + self.four_player_vulnerability_scale * vulnerability_reward
            + self.four_player_leader_scale * leader_reward
            + self.four_player_third_player_scale * third_player_reward
        )
        if done:
            reward += float(outcomes[0]["rewards"][0])
        terminated = done
        truncated = False
        sun_losses, border_losses = self._loss_counts(previous_state, next_state, player=0, player_moves=actions[0][0])
        info = {
            "scores": outcomes[0].get("scores", []),
            "sun_losses": sun_losses,
            "border_losses": border_losses,
            "base_shaping_reward": base_shaping_reward,
            "ship_margin_reward": ship_margin_reward,
            "normalized_margin_reward": normalized_margin_reward,
            "comet_shaping_reward": comet_shaping_reward,
            "four_player_vulnerability_reward": vulnerability_reward,
            "four_player_leader_reward": leader_reward,
            "four_player_third_player_reward": third_player_reward,
        }
        return obs, reward, terminated, truncated, info

    def _state_potential(self, state: dict[str, Any], *, player: int) -> float:
        """Shaping potential Φ(s): opponent-relative production/territory advantage.

        Pure function of the state (no transition info). Used only inside the
        potential-difference shaping term γ·Φ(s') − Φ(s); see _base_shaping_reward.
        """
        planets = state.get("planets", [])
        own_prod = sum(planet_production(p) for p in planets if planet_owner(p) == player)
        enemy_prod = sum(planet_production(p) for p in planets if planet_owner(p) not in (-1, player))
        own_planets = sum(1 for p in planets if planet_owner(p) == player)
        enemy_planets = sum(1 for p in planets if planet_owner(p) not in (-1, player))
        return 0.002 * (own_prod - enemy_prod) + 0.001 * (own_planets - enemy_planets)

    def _base_shaping_reward(
        self,
        previous_state: dict[str, Any],
        state: dict[str, Any],
        *,
        player: int,
        player_moves: list[list[float]],
        done: bool = False,
    ) -> float:
        # Potential-based shaping in the difference form F = γ·Φ(s') − Φ(s)
        # (Ng, Harada & Russell 1999): this is policy-invariant, unlike the
        # previous raw Φ(s') term which biased the optimum toward the
        # production-greedy (Producer-like) basin. The terminal state takes
        # Φ ≡ 0 by convention, so over an episode the shaping telescopes to a
        # boundary term and adds no spurious value at termination — the true
        # game outcome is added separately by step()/the rollout loop.
        prev_potential = self._state_potential(previous_state, player=player)
        next_potential = 0.0 if done else self._state_potential(state, player=player)
        shaping = self.shaping_gamma * next_potential - prev_potential
        sun_losses, border_losses = self._loss_counts(previous_state, state, player=player, player_moves=player_moves)
        return (
            shaping
            - self.sun_loss_penalty * sun_losses
            - self.border_loss_penalty * border_losses
        )

    def _ship_margin_reward(self, previous_state: dict[str, Any], state: dict[str, Any], *, player: int) -> float:
        if self.ship_margin_scale == 0.0:
            return 0.0
        previous_margin = self._ship_margin(previous_state, player)
        current_margin = self._ship_margin(state, player)
        return self.ship_margin_scale * (current_margin - previous_margin)

    def _normalized_margin_reward(
        self,
        previous_state: dict[str, Any],
        state: dict[str, Any],
        *,
        player: int,
        done: bool = False,
    ) -> float:
        if self.normalized_margin_scale == 0.0:
            return 0.0
        previous_margin = self._normalized_margin_potential(previous_state, player)
        current_margin = 0.0 if done else self._normalized_margin_potential(state, player)
        return self.normalized_margin_scale * (self.shaping_gamma * current_margin - previous_margin)

    def _normalized_margin_potential(self, state: dict[str, Any], player: int) -> float:
        scores = [self._player_total_ships(state, pid) for pid in range(self.num_players)]
        if player >= len(scores):
            return 0.0
        return normalized_margin(scores, player)

    def _ship_margin(self, state: dict[str, Any], player: int) -> float:
        own = self._player_total_ships(state, player)
        enemy = 0.0
        for pid in range(self.num_players):
            if pid != player:
                enemy += self._player_total_ships(state, pid)
        return own - enemy

    def _comet_auxiliary_reward(
        self,
        previous_state: dict[str, Any],
        state: dict[str, Any],
        *,
        player: int,
    ) -> float:
        previous_comets = self._comet_planets(previous_state)
        current_comets = self._comet_planets(state)
        previous_by_id = {planet_id(planet): planet for planet in previous_comets}
        current_by_id = {planet_id(planet): planet for planet in current_comets}

        prev_margin = self._comet_ship_margin(previous_comets, player)
        current_margin = self._comet_ship_margin(current_comets, player)

        capture_delta = 0
        for comet_id in set(previous_by_id) | set(current_by_id):
            prev_owner = planet_owner(previous_by_id[comet_id]) if comet_id in previous_by_id else -1
            current_owner = planet_owner(current_by_id[comet_id]) if comet_id in current_by_id else -1
            if prev_owner != player and current_owner == player:
                capture_delta += 1
            elif prev_owner == player and current_owner != player:
                capture_delta -= 1

        return 0.02 * capture_delta + 0.0005 * (current_margin - prev_margin)

    def _comet_planets(self, state: dict[str, Any]) -> list[Any]:
        comet_ids = set(state.get("comet_planet_ids", []))
        return [planet for planet in state.get("planets", []) if planet_id(planet) in comet_ids]

    def _comet_ship_margin(self, comets: list[Any], player: int) -> int:
        own_ships = sum(planet_ships(planet) for planet in comets if planet_owner(planet) == player)
        enemy_ships = sum(planet_ships(planet) for planet in comets if planet_owner(planet) not in (-1, player))
        return own_ships - enemy_ships

    def _four_player_strategic_reward(
        self,
        previous_state: dict[str, Any],
        state: dict[str, Any],
        *,
        player: int,
    ) -> tuple[float, float, float]:
        if self.num_players < 4:
            return 0.0, 0.0, 0.0
        vulnerability_reward = 0.0005 * (
            self._vulnerability_index(previous_state, player) - self._vulnerability_index(state, player)
        )
        leader_reward = 0.001 * (self._leader_gap(previous_state, player) - self._leader_gap(state, player))
        third_player_reward = 0.0008 * (
            self._third_player_gap(previous_state, player) - self._third_player_gap(state, player)
        )
        return vulnerability_reward, leader_reward, third_player_reward

    def _vulnerability_index(self, state: dict[str, Any], player: int) -> float:
        own_planets = [planet for planet in state.get("planets", []) if planet_owner(planet) == player]
        enemy_planets = [planet for planet in state.get("planets", []) if planet_owner(planet) not in (-1, player)]
        enemy_fleets = [fleet for fleet in state.get("fleets", []) if fleet_owner(fleet) not in (-1, player)]
        exposure = 0.0
        for planet in own_planets:
            px = planet_x(planet)
            py = planet_y(planet)
            pressure = 0.0
            for enemy in enemy_planets:
                pressure += planet_ships(enemy) / max(math.hypot(px - planet_x(enemy), py - planet_y(enemy)), 1.0)
            for fleet in enemy_fleets:
                pressure += fleet_ships(fleet) / max(math.hypot(px - fleet_x(fleet), py - fleet_y(fleet)), 1.0)
            exposure += max(0.0, pressure - 0.1 * planet_ships(planet))
        return exposure

    def _leader_gap(self, state: dict[str, Any], player: int) -> float:
        scores = [self._player_total_ships(state, pid) for pid in range(self.num_players)]
        if player >= len(scores):
            return 0.0
        other_scores = [score for idx, score in enumerate(scores) if idx != player]
        return max(other_scores, default=0.0) - scores[player]

    def _third_player_gap(self, state: dict[str, Any], player: int) -> float:
        scores = [self._player_total_ships(state, pid) for pid in range(self.num_players)]
        if player >= len(scores):
            return 0.0
        other_scores = sorted((score for idx, score in enumerate(scores) if idx != player), reverse=True)
        if not other_scores:
            return 0.0
        reference = other_scores[1] if len(other_scores) > 1 else other_scores[0]
        return reference - scores[player]

    def _player_total_ships(self, state: dict[str, Any], player: int) -> float:
        total = 0.0
        for planet in state.get("planets", []):
            if planet_owner(planet) == player:
                total += float(planet_ships(planet))
        for fleet in state.get("fleets", []):
            if fleet_owner(fleet) == player:
                total += float(fleet_ships(fleet))
        return total

    def _loss_counts(
        self,
        previous_state: dict[str, Any],
        next_state: dict[str, Any],
        *,
        player: int,
        player_moves: list[list[float]],
    ) -> tuple[int, int]:
        next_ids = {fleet_id(fleet) for fleet in next_state.get("fleets", [])}
        sun_losses = 0
        border_losses = 0

        prev_planets = previous_state.get("planets", [])
        for fleet in previous_state.get("fleets", []):
            if fleet_owner(fleet) != player or fleet_id(fleet) in next_ids:
                continue
            start = (fleet_x(fleet), fleet_y(fleet))
            end = self._fleet_endpoint(start, fleet_angle(fleet), fleet_ships(fleet))
            arrived = any(
                math.hypot(end[0] - planet_x(p), end[1] - planet_y(p)) <= planet_radius(p) + 1.0
                for p in prev_planets
            )
            if arrived:
                continue
            if self._hits_border(end):
                border_losses += 1
            elif self._crosses_sun(start, end):
                sun_losses += 1

        planets_by_id = {planet_id(planet): planet for planet in previous_state.get("planets", [])}
        for move in player_moves:
            source = planets_by_id.get(int(move[0]))
            if source is None:
                continue
            start = self._launch_start(source, float(move[1]))
            end = self._fleet_endpoint(start, float(move[1]), int(move[2]))
            if self._hits_border(end):
                border_losses += 1
            elif self._crosses_sun(start, end):
                sun_losses += 1

        return sun_losses, border_losses

    def _launch_start(self, source_planet: Any, angle: float) -> tuple[float, float]:
        radius = planet_radius(source_planet) + 0.1
        return (
            planet_x(source_planet) + math.cos(angle) * radius,
            planet_y(source_planet) + math.sin(angle) * radius,
        )

    def _fleet_endpoint(self, start: tuple[float, float], angle: float, ships: int) -> tuple[float, float]:
        speed = self._fleet_speed(ships)
        return (start[0] + math.cos(angle) * speed, start[1] + math.sin(angle) * speed)

    def _fleet_speed(self, ships: int) -> float:
        scale = math.log(max(ships, 1)) / math.log(1000.0)
        speed = 1.0 + (SHIP_SPEED - 1.0) * scale**1.5
        return min(SHIP_SPEED, max(1.0, speed))

    def _hits_border(self, point: tuple[float, float]) -> bool:
        return not (0.0 <= point[0] <= BOARD_SIZE and 0.0 <= point[1] <= BOARD_SIZE)

    def _crosses_sun(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        return self._point_to_segment_distance((CENTER, CENTER), start, end) < SUN_RADIUS

    def _point_to_segment_distance(
        self,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
        projection = (start[0] + t * dx, start[1] + t * dy)
        return math.hypot(point[0] - projection[0], point[1] - projection[1])
