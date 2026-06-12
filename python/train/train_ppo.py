from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from python.agents.policy import (
    EntityActorCritic,
    FlatActorCritic,
    _action_value_from_heads,
    launch_gated_kl,
)
from python.agents.registry import (
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
    get_isolated_opponents,
)
from python.train.opponent_pool import get_process_opponent_pool

from orbit_wars_gym import OrbitWarsGymEnv
from orbit_wars_gym.action_decoder import DecoderConfig, decode_discrete_action
from orbit_wars_gym.action_masks import build_action_masks, split_masks
from orbit_wars_gym.backend import RustBatchBackend, RustConfig
from orbit_wars_gym.encoding import observation_dim
from orbit_wars_gym.entities import fleet_owner, planet_id, planet_owner
from orbit_wars_gym.observation import to_official_observation
from orbit_wars_gym.rules import normalized_margin

_POLICY_ARCHS = {"flat": FlatActorCritic, "entity": EntityActorCritic}


def _build_policy(arch: str, obs_dim: int):
    if arch not in _POLICY_ARCHS:
        raise ValueError(f"unknown policy arch {arch!r}; valid: {sorted(_POLICY_ARCHS)}")
    return _POLICY_ARCHS[arch](obs_dim)


_HEURISTIC_POLICIES = get_heuristic_policies()
PHASE0_OPPONENTS = {
    name: _HEURISTIC_POLICIES[name]
    for name in ("producer", "oep", "greedy", "defensive", "rush", "anti_meta", "weak_random")
}

LEAGUE_TRAINING_OPPONENTS = frozenset({"pgs_holdwave", "pgs_bigwave", "brep"})


def _league_training_policy(name: str):
    bot = None

    def policy(state: dict[str, Any], player: int) -> list[list[float]]:
        nonlocal bot
        if bot is None:
            from scripts.league_agents import make

            bot = make(name)
        moves = bot(to_official_observation(state, player=player))
        return list(moves) if isinstance(moves, list) else []

    return policy


for _name in sorted(LEAGUE_TRAINING_OPPONENTS):
    PHASE0_OPPONENTS[_name] = _league_training_policy(_name)


def _opponent_parts(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(name).split("+") if part.strip())


def _unknown_opponent_parts(names: Sequence[str]) -> list[str]:
    unknown: set[str] = set()
    for name in names:
        parts = _opponent_parts(name)
        if not parts:
            unknown.add(str(name))
            continue
        unknown.update(part for part in parts if part not in PHASE0_OPPONENTS)
    return sorted(unknown)


def _seat_policy(name: str, seat_index: int):
    if name in STATEFUL_SINGLETON_OPPONENTS:
        return get_isolated_opponents(name, seat_index + 1)[seat_index]
    if name in LEAGUE_TRAINING_OPPONENTS:
        return _league_training_policy(name)
    return PHASE0_OPPONENTS[name]


def _training_opponent_policy(name: str, num_players: int):
    parts = _opponent_parts(name)
    unknown = _unknown_opponent_parts([name])
    if unknown:
        raise ValueError(f"unknown phase-0 opponent parts: {', '.join(unknown)}")
    if not parts:
        raise ValueError("opponent name cannot be empty")

    seat_policies = [
        _seat_policy(parts[(seat - 1) % len(parts)], seat - 1)
        for seat in range(1, max(2, int(num_players)))
    ]

    def policy(state: dict[str, Any], player: int) -> list[list[float]]:
        if player <= 0:
            return []
        seat = (int(player) - 1) % len(seat_policies)
        return seat_policies[seat](state, player)

    policy.__name__ = "+".join(parts)
    return policy


@dataclass(frozen=True)
class Phase0TrainingConfig:
    seed: int = 0
    policy_track: str = "phase0_2p"
    policy_arch: str = "flat"
    num_players: int = 2
    total_timesteps: int = 200_000
    rollout_steps: int = 256
    rollout_num_envs: int = 1
    # Opponent-call parallelism. Default 1 (sequential) — both threads (GIL-bound
    # planners) and processes (per-step state IPC > planner compute) were measured
    # SLOWER than sequential. >1 enables the experimental process pool (opt-in).
    opponent_workers: int = 1
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    early_survival_window: int = 64
    opponents: tuple[str, ...] = ("greedy", "defensive", "rush", "anti_meta", "weak_random")
    checkpoint_in: str | None = None
    checkpoint_out: str | None = None
    device: str = "cpu"
    enable_comets: bool = True
    sun_loss_penalty: float = 0.02
    border_loss_penalty: float = 0.02
    ship_margin_scale: float = 0.0
    normalized_margin_scale_start: float = 0.0
    normalized_margin_scale_end: float = 0.0
    base_shaping_scale_start: float = 1.0
    base_shaping_scale_end: float = 0.15
    comet_shaping_scale_start: float = 0.08
    comet_shaping_scale_end: float = 0.0
    four_player_vulnerability_scale_start: float = 0.06
    four_player_vulnerability_scale_end: float = 0.02
    four_player_leader_scale_start: float = 0.05
    four_player_leader_scale_end: float = 0.02
    four_player_third_player_scale_start: float = 0.04
    four_player_third_player_scale_end: float = 0.015
    decoder_max_moves_per_turn: int = 8
    decoder_min_ships_to_launch: int = 2
    decoder_reserve_home_ships: int = 8
    decoder_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    decoder_angle_offsets: tuple[float, ...] = (-0.261799, -0.130899, 0.0, 0.130899, 0.261799)
    inherit_checkpoint_decoder: bool = True
    # --- Movement 2 (de-anchor the reward + anti-drift on scaling) ---
    # De-anchor: "none" forces the production/territory base shaping to 0 (the
    # potential that equals the Producer's greedy objective). "producer" keeps it.
    shaping_potential: str = "producer"
    # KL-to-reference anchor: penalise divergence from a frozen reference policy
    # (e.g. the BC init / --checkpoint-in) to stop the policy drifting off the good
    # init while it over-optimises the shaping (the P3 620k regression). 0 = off.
    kl_to_ref_coef: float = 0.0
    ref_checkpoint: str | None = None  # default: fall back to checkpoint_in
    # Eval-gating: periodic in-loop margin eval vs an opponent; keep the BEST
    # checkpoint by margin and early-stop on regression. The P3 regression proved
    # "healthy curves" lie — only the paired margin catches the drift. 0 = off.
    eval_every_updates: int = 0
    eval_seeds: int = 8
    eval_opponent: str = "producer"
    eval_max_steps: int = 600
    early_stop_patience: int = 0  # consecutive evals without improvement before stopping; 0 = off


@dataclass
class EpisodeMetrics:
    opponent: str
    length: int = 0
    return_sum: float = 0.0
    neutral_captures: int = 0
    early_alive_steps: int = 0
    early_window_observed: int = 0
    completed: bool = False

    def record_step(self, reward: float, neutral_captures: int, alive: bool, early_window: int) -> None:
        self.length += 1
        self.return_sum += float(reward)
        self.neutral_captures += int(neutral_captures)
        if self.length <= early_window:
            self.early_window_observed += 1
            if alive:
                self.early_alive_steps += 1

    def as_summary(self) -> dict[str, Any]:
        observed = max(self.early_window_observed, 1)
        length = max(self.length, 1)
        return {
            "opponent": self.opponent,
            "length": self.length,
            "return_sum": self.return_sum,
            "neutral_captures": self.neutral_captures,
            "neutral_capture_rate": self.neutral_captures / length,
            "early_survival_rate": self.early_alive_steps / observed,
            "completed": self.completed,
        }


@dataclass
class RolloutSegment:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    opponent: str
    episode_metrics: list[dict[str, Any]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_opponents(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        items = tuple(str(part).strip() for part in raw if str(part).strip())
    if not items:
        raise ValueError("at least one opponent is required")
    if len(set(items)) < 2:
        raise ValueError("training requires at least two distinct opponents")
    unknown = _unknown_opponent_parts(items)
    if unknown:
        raise ValueError(f"unknown phase-0 opponent parts: {', '.join(unknown)}")
    return items


def _player_alive(state: dict[str, Any], player: int) -> bool:
    planets = state.get("planets", [])
    fleets = state.get("fleets", [])
    return any(planet_owner(planet) == player for planet in planets) or any(fleet_owner(fleet) == player for fleet in fleets)


def _neutral_capture_count(previous_state: dict[str, Any], next_state: dict[str, Any], player: int) -> int:
    previous_owners = {planet_id(planet): planet_owner(planet) for planet in previous_state.get("planets", [])}
    captures = 0
    for planet in next_state.get("planets", []):
        pid = planet_id(planet)
        if previous_owners.get(pid) == -1 and planet_owner(planet) == player:
            captures += 1
    return captures


def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)
    next_value = next_value.to(rewards.device, dtype=rewards.dtype).view(())
    for idx in range(rewards.shape[0] - 1, -1, -1):
        if idx == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[idx]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[idx]
            next_values = values[idx + 1]
        delta = rewards[idx] + gamma * next_values * next_non_terminal - values[idx]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[idx] = last_advantage
    returns = advantages + values
    return advantages, returns


def _compute_gae_batched(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    next_value = next_value.to(rewards.device, dtype=rewards.dtype).view(rewards.shape[1])
    for idx in range(rewards.shape[0] - 1, -1, -1):
        if idx == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[idx]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[idx]
            next_values = values[idx + 1]
        delta = rewards[idx] + gamma * next_values * next_non_terminal - values[idx]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[idx] = last_advantage
    returns = advantages + values
    return advantages, returns


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return float(start + (end - start) * progress)


def shaping_scales(training_cfg: Phase0TrainingConfig, progress: float) -> tuple[float, float]:
    # Movement 2 de-anchor: drop the production/territory base shaping entirely
    # (its potential equals the Producer's greedy objective, which biases the
    # policy toward the Producer basin) and lean on the sparse outcome reward.
    base = (
        0.0
        if training_cfg.shaping_potential == "none"
        else _linear_schedule(training_cfg.base_shaping_scale_start, training_cfg.base_shaping_scale_end, progress)
    )
    return (
        base,
        _linear_schedule(training_cfg.comet_shaping_scale_start, training_cfg.comet_shaping_scale_end, progress),
    )


def four_player_shaping_scales(training_cfg: Phase0TrainingConfig, progress: float) -> tuple[float, float, float]:
    return (
        _linear_schedule(
            training_cfg.four_player_vulnerability_scale_start,
            training_cfg.four_player_vulnerability_scale_end,
            progress,
        ),
        _linear_schedule(
            training_cfg.four_player_leader_scale_start,
            training_cfg.four_player_leader_scale_end,
            progress,
        ),
        _linear_schedule(
            training_cfg.four_player_third_player_scale_start,
            training_cfg.four_player_third_player_scale_end,
            progress,
        ),
    )


def normalized_margin_scale(training_cfg: Phase0TrainingConfig, progress: float) -> float:
    return _linear_schedule(
        training_cfg.normalized_margin_scale_start,
        training_cfg.normalized_margin_scale_end,
        progress,
    )


def decoder_config(training_cfg: Phase0TrainingConfig) -> DecoderConfig:
    return DecoderConfig(
        fractions=tuple(float(value) for value in training_cfg.decoder_fractions),
        angle_offsets=tuple(float(value) for value in training_cfg.decoder_angle_offsets),
        max_moves_per_turn=int(training_cfg.decoder_max_moves_per_turn),
        min_ships_to_launch=int(training_cfg.decoder_min_ships_to_launch),
        reserve_home_ships=int(training_cfg.decoder_reserve_home_ships),
    )


def decoder_payload(training_cfg: Phase0TrainingConfig) -> dict[str, Any]:
    cfg = decoder_config(training_cfg)
    return {
        "fractions": list(cfg.fractions),
        "angle_offsets": list(cfg.angle_offsets),
        "max_moves_per_turn": cfg.max_moves_per_turn,
        "min_ships_to_launch": cfg.min_ships_to_launch,
        "reserve_home_ships": cfg.reserve_home_ships,
    }


def _checkpoint_decoder_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
    if isinstance(summary.get("decoder"), dict):
        return dict(summary["decoder"])
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    payload: dict[str, Any] = {}
    for source, target in (
        ("decoder_fractions", "fractions"),
        ("decoder_angle_offsets", "angle_offsets"),
        ("decoder_max_moves_per_turn", "max_moves_per_turn"),
        ("decoder_min_ships_to_launch", "min_ships_to_launch"),
        ("decoder_reserve_home_ships", "reserve_home_ships"),
    ):
        if source in config:
            payload[target] = config[source]
    return payload


def _inherit_checkpoint_decoder(
    training_cfg: Phase0TrainingConfig,
    checkpoint: dict[str, Any] | None,
) -> Phase0TrainingConfig:
    if checkpoint is None or not training_cfg.inherit_checkpoint_decoder:
        return training_cfg
    decoder = _checkpoint_decoder_payload(checkpoint)
    if not decoder:
        return training_cfg
    return replace(
        training_cfg,
        decoder_fractions=tuple(float(value) for value in decoder.get("fractions", training_cfg.decoder_fractions)),
        decoder_angle_offsets=tuple(
            float(value) for value in decoder.get("angle_offsets", training_cfg.decoder_angle_offsets)
        ),
        decoder_max_moves_per_turn=int(decoder.get("max_moves_per_turn", training_cfg.decoder_max_moves_per_turn)),
        decoder_min_ships_to_launch=int(
            decoder.get("min_ships_to_launch", training_cfg.decoder_min_ships_to_launch)
        ),
        decoder_reserve_home_ships=int(
            decoder.get("reserve_home_ships", training_cfg.decoder_reserve_home_ships)
        ),
    )


def build_phase5_4p_config(**overrides: Any) -> Phase0TrainingConfig:
    cfg = Phase0TrainingConfig(
        policy_track="phase5_4p",
        num_players=4,
        enable_comets=True,
        normalized_margin_scale_start=0.15,
        normalized_margin_scale_end=0.04,
        four_player_vulnerability_scale_start=0.08,
        four_player_vulnerability_scale_end=0.03,
        four_player_leader_scale_start=0.06,
        four_player_leader_scale_end=0.025,
        four_player_third_player_scale_start=0.05,
        four_player_third_player_scale_end=0.02,
    )
    return replace(cfg, **overrides)


def train_phase5_4p(training_cfg: Phase0TrainingConfig | None = None) -> dict[str, Any]:
    cfg = training_cfg or build_phase5_4p_config()
    if cfg.num_players != 4:
        raise ValueError("phase5_4p requires num_players=4")
    if cfg.policy_track != "phase5_4p":
        cfg = replace(cfg, policy_track="phase5_4p")
    return train_phase0(cfg)


def _collect_single_env_rollout_segment(
    model: FlatActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    rollout_steps: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
) -> RolloutSegment:
    base_shaping_scale, comet_shaping_scale = shaping_scales(training_cfg, progress)
    margin_scale = normalized_margin_scale(training_cfg, progress)
    (
        four_player_vulnerability_scale,
        four_player_leader_scale,
        four_player_third_player_scale,
    ) = four_player_shaping_scales(training_cfg, progress)
    env = build_phase0_env(
        seed=base_seed,
        num_players=training_cfg.num_players,
        opponent_name=opponent_name,
        enable_comets=training_cfg.enable_comets,
        decoder_cfg=decoder_config(training_cfg),
        sun_loss_penalty=training_cfg.sun_loss_penalty,
        border_loss_penalty=training_cfg.border_loss_penalty,
        ship_margin_scale=training_cfg.ship_margin_scale,
        normalized_margin_scale=margin_scale,
        base_shaping_scale=base_shaping_scale,
        comet_shaping_scale=comet_shaping_scale,
        shaping_gamma=training_cfg.gamma,
        four_player_vulnerability_scale=four_player_vulnerability_scale,
        four_player_leader_scale=four_player_leader_scale,
        four_player_third_player_scale=four_player_third_player_scale,
    )
    obs_np, _ = env.reset(seed=base_seed)
    episode = EpisodeMetrics(opponent=opponent_name)
    episode_metrics: list[dict[str, Any]] = []

    obs_buf = []
    action_buf = []
    logprob_buf = []
    value_buf = []
    mask_buf = []
    rewards_np = np.empty(rollout_steps, dtype=np.float32)
    dones_np = np.empty(rollout_steps, dtype=np.float32)

    for reset_idx in range(rollout_steps):
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            build_action_masks(env.state, 0, min_ships_to_launch=training_cfg.decoder_min_ships_to_launch),
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(
                obs_tensor, masks=split_masks(mask_tensor)
            )
        action = action_tensor.squeeze(0).cpu().numpy()
        previous_state = env.state
        next_obs_np, reward, done, _, _ = env.step(action)
        next_state = env.state

        neutral_captures = _neutral_capture_count(previous_state, next_state, player=0)
        alive = _player_alive(next_state, player=0)
        episode.record_step(
            reward=reward,
            neutral_captures=neutral_captures,
            alive=alive,
            early_window=training_cfg.early_survival_window,
        )

        obs_buf.append(obs_tensor.squeeze(0))
        action_buf.append(action_tensor.squeeze(0))
        logprob_buf.append(logprob_tensor.squeeze(0))
        value_buf.append(value_tensor.squeeze(0))
        mask_buf.append(mask_tensor.squeeze(0))
        rewards_np[reset_idx] = float(reward)
        dones_np[reset_idx] = float(done)

        if done:
            episode.completed = True
            episode_metrics.append(episode.as_summary())
            next_seed = base_seed + reset_idx + 1
            obs_np, _ = env.reset(seed=next_seed)
            episode = EpisodeMetrics(opponent=opponent_name)
        else:
            obs_np = next_obs_np

    if episode.length > 0:
        episode_metrics.append(episode.as_summary())

    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
        if rollout_steps > 0 and bool(dones_np[rollout_steps - 1]):
            next_value = torch.zeros(1, device=device, dtype=torch.float32)

    observations = torch.stack(obs_buf)
    actions = torch.stack(action_buf).to(dtype=torch.long)
    logprobs = torch.stack(logprob_buf)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
    values = torch.stack(value_buf)
    advantages, returns = _compute_gae(
        rewards,
        dones,
        values,
        next_value.squeeze(0),
        gamma=training_cfg.gamma,
        gae_lambda=training_cfg.gae_lambda,
    )
    return RolloutSegment(
        observations=observations,
        actions=actions,
        logprobs=logprobs,
        advantages=advantages,
        returns=returns,
        values=values,
        rewards=rewards,
        masks=torch.stack(mask_buf),
        opponent=opponent_name,
        episode_metrics=episode_metrics,
    )


def _moves_to_flat_rows(env_index: int, player_index: int, moves: Sequence[Sequence[float]]) -> list[list[float]]:
    rows: list[list[float]] = []
    for move in moves:
        if len(move) < 3:
            continue
        rows.append([float(env_index), float(player_index), float(move[0]), float(move[1]), float(move[2])])
    return rows


def _batched_rollout_supported(
    training_cfg: Phase0TrainingConfig,
    opponent_name: str | None = None,
) -> bool:
    if int(training_cfg.rollout_num_envs) <= 1 or training_cfg.num_players != 2:
        return False
    if opponent_name is not None and len(_opponent_parts(opponent_name)) != 1:
        return False
    if opponent_name in LEAGUE_TRAINING_OPPONENTS:
        return False
    return True


def _collect_batched_rollout_segment(
    model: FlatActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    rollout_steps: int,
    sample_limit: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
) -> RolloutSegment:
    if training_cfg.num_players != 2:
        raise ValueError("batched PPO rollout currently supports only 2-player training")
    if opponent_name not in PHASE0_OPPONENTS:
        raise ValueError(f"unknown phase-0 opponent: {opponent_name}")

    num_envs = max(1, min(int(training_cfg.rollout_num_envs), int(sample_limit)))
    # One isolated opponent instance per env so concurrent games never share
    # per-game memory (producer/oep singletons). Each instance resets on step==0,
    # so reusing the cached pool across segments is safe. Stateless heuristics
    # reuse one shared callable.
    opponent_policies = get_isolated_opponents(opponent_name, num_envs)
    # Opponent planners (producer/oep) are the per-step bottleneck and are
    # independent per env (isolation makes concurrent calls safe). Threads do NOT
    # help — the planners are GIL-bound pure Python, so a ThreadPoolExecutor was
    # measured ~20x SLOWER. Real parallelism needs separate processes:
    # ``opponent_workers > 1`` spawns a persistent process pool with a fixed
    # env->worker assignment (per-env memory lives in the worker). 0 = auto.
    if (
        int(training_cfg.opponent_workers) == 1
        or num_envs == 1
        or opponent_name not in STATEFUL_SINGLETON_OPPONENTS  # stateless opps are cheap; no pool
    ):
        opponent_pool = None
    else:
        requested = int(training_cfg.opponent_workers)
        workers = min(requested, num_envs) if requested > 0 else min(num_envs, os.cpu_count() or 1)
        # Cached/persistent pool keyed by the configured max env count, reused
        # across segments (workers' per-game memory resets on step==0).
        opponent_pool = (
            get_process_opponent_pool(opponent_name, int(training_cfg.rollout_num_envs), workers)
            if workers > 1
            else None
        )
    base_shaping_scale, comet_shaping_scale = shaping_scales(training_cfg, progress)
    margin_scale = normalized_margin_scale(training_cfg, progress)
    reward_env = build_phase0_env(
        seed=base_seed,
        num_players=training_cfg.num_players,
        opponent_name=opponent_name,
        enable_comets=training_cfg.enable_comets,
        decoder_cfg=decoder_config(training_cfg),
        sun_loss_penalty=training_cfg.sun_loss_penalty,
        border_loss_penalty=training_cfg.border_loss_penalty,
        ship_margin_scale=training_cfg.ship_margin_scale,
        normalized_margin_scale=margin_scale,
        base_shaping_scale=base_shaping_scale,
        comet_shaping_scale=comet_shaping_scale,
        shaping_gamma=training_cfg.gamma,
    )
    backend = RustBatchBackend(
        num_envs=num_envs,
        num_players=training_cfg.num_players,
        seed=base_seed,
        config=RustConfig(enable_comets=training_cfg.enable_comets),
    )
    current_states = backend.reset(base_seed)
    obs_np = backend.encoded_states(0)
    episodes = [EpisodeMetrics(opponent=opponent_name) for _ in range(num_envs)]
    episode_metrics: list[dict[str, Any]] = []
    active = np.ones(num_envs, dtype=bool)

    obs_buf = []
    action_buf = []
    logprob_buf = []
    value_buf = []
    mask_buf = []
    rewards_np = np.empty((rollout_steps, num_envs), dtype=np.float32)
    dones_np = np.empty((rollout_steps, num_envs), dtype=np.float32)

    for step_index in range(rollout_steps):
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(
            np.stack([
                build_action_masks(state, 0, min_ships_to_launch=training_cfg.decoder_min_ships_to_launch)
                for state in current_states
            ]),
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad():
            action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(
                obs_tensor, masks=split_masks(mask_tensor)
            )
        actions_np = action_tensor.cpu().numpy()

        action_rows: list[list[float]] = []
        player_moves_by_env: list[list[list[float]]] = [[] for _ in range(num_envs)]
        active_indices = [i for i in range(num_envs) if active[i]]

        # Opponent moves (the expensive part); via a process pool when enabled,
        # else sequential. Each env has its own isolated opponent instance.
        if opponent_pool is not None:
            opponent_moves_by_env = opponent_pool.moves(current_states, active_indices)
        else:
            opponent_moves_by_env = {
                i: opponent_policies[i](current_states[i], 1) for i in active_indices
            }

        for env_index in active_indices:
            state = current_states[env_index]
            player_moves = decode_discrete_action(state, 0, actions_np[env_index], decoder_config(training_cfg))
            player_moves_by_env[env_index] = player_moves
            action_rows.extend(_moves_to_flat_rows(env_index, 0, player_moves))
            action_rows.extend(_moves_to_flat_rows(env_index, 1, opponent_moves_by_env[env_index]))

        flat_actions = (
            np.asarray(action_rows, dtype=np.float64)
            if action_rows
            else np.zeros((0, 5), dtype=np.float64)
        )
        previous_states = current_states
        outcomes, next_obs_np = backend.step_flat_with_encoded_states(flat_actions, 0)
        next_states = backend.states()

        rewards_row = rewards_np[step_index]
        dones_row = dones_np[step_index]
        rewards_row.fill(0.0)
        dones_row.fill(0.0)
        for env_index, (previous_state, next_state, outcome) in enumerate(
            zip(previous_states, next_states, outcomes, strict=True)
        ):
            if not active[env_index]:
                dones_row[env_index] = 1.0
                continue
            done = bool(outcome.get("done", False))
            base_reward = reward_env._base_shaping_reward(
                previous_state,
                next_state,
                player=0,
                player_moves=player_moves_by_env[env_index],
                done=done,
            )
            ship_margin_reward = reward_env._ship_margin_reward(previous_state, next_state, player=0)
            normalized_margin_reward = reward_env._normalized_margin_reward(
                previous_state,
                next_state,
                player=0,
                done=done,
            )
            comet_reward = reward_env._comet_auxiliary_reward(previous_state, next_state, player=0)
            reward = (
                base_shaping_scale * base_reward
                + ship_margin_reward
                + normalized_margin_reward
                + comet_shaping_scale * comet_reward
            )
            if done:
                rewards = outcome.get("rewards", [])
                reward += float(rewards[0]) if rewards else 0.0

            rewards_row[env_index] = float(reward)
            dones_row[env_index] = float(done)
            neutral_captures = _neutral_capture_count(previous_state, next_state, player=0)
            alive = _player_alive(next_state, player=0)
            episodes[env_index].record_step(
                reward=reward,
                neutral_captures=neutral_captures,
                alive=alive,
                early_window=training_cfg.early_survival_window,
            )
            if done:
                episodes[env_index].completed = True
                episode_metrics.append(episodes[env_index].as_summary())
                active[env_index] = False

        obs_buf.append(obs_tensor)
        action_buf.append(action_tensor)
        logprob_buf.append(logprob_tensor)
        value_buf.append(value_tensor)
        mask_buf.append(mask_tensor)

        current_states = next_states
        obs_np = next_obs_np

    # Pool is cached/persistent (reset on step==0); closed at interpreter exit.

    for env_index, episode in enumerate(episodes):
        if active[env_index] and episode.length > 0:
            episode_metrics.append(episode.as_summary())

    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
        inactive = torch.as_tensor(~active, dtype=torch.bool, device=device)
        next_value = next_value.masked_fill(inactive, 0.0)

    observations = torch.stack(obs_buf)
    actions = torch.stack(action_buf).to(dtype=torch.long)
    logprobs = torch.stack(logprob_buf)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
    values = torch.stack(value_buf)
    advantages, returns = _compute_gae_batched(
        rewards,
        dones,
        values,
        next_value,
        gamma=training_cfg.gamma,
        gae_lambda=training_cfg.gae_lambda,
    )

    masks = torch.stack(mask_buf)
    flat_observations = observations.reshape(-1, observations.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_logprobs = logprobs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_values = values.reshape(-1)
    flat_rewards = rewards.reshape(-1)
    flat_masks = masks.reshape(-1, masks.shape[-1])
    if sample_limit > 0:
        flat_observations = flat_observations[:sample_limit]
        flat_actions = flat_actions[:sample_limit]
        flat_logprobs = flat_logprobs[:sample_limit]
        flat_advantages = flat_advantages[:sample_limit]
        flat_returns = flat_returns[:sample_limit]
        flat_values = flat_values[:sample_limit]
        flat_rewards = flat_rewards[:sample_limit]
        flat_masks = flat_masks[:sample_limit]

    return RolloutSegment(
        observations=flat_observations,
        actions=flat_actions,
        logprobs=flat_logprobs,
        advantages=flat_advantages,
        returns=flat_returns,
        values=flat_values,
        rewards=flat_rewards,
        masks=flat_masks,
        opponent=opponent_name,
        episode_metrics=episode_metrics,
    )


def _collect_rollout_segment(
    model: FlatActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    rollout_steps: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
    sample_limit: int | None = None,
) -> RolloutSegment:
    limit = rollout_steps if sample_limit is None else max(1, int(sample_limit))
    if _batched_rollout_supported(training_cfg, opponent_name):
        per_env_steps = max(1, min(int(rollout_steps), math.ceil(limit / max(1, int(training_cfg.rollout_num_envs)))))
        return _collect_batched_rollout_segment(
            model,
            opponent_name=opponent_name,
            base_seed=base_seed,
            rollout_steps=per_env_steps,
            sample_limit=limit,
            device=device,
            training_cfg=training_cfg,
            progress=progress,
        )
    return _collect_single_env_rollout_segment(
        model,
        opponent_name=opponent_name,
        base_seed=base_seed,
        rollout_steps=limit,
        device=device,
        training_cfg=training_cfg,
        progress=progress,
    )


def _concat_segments(segments: Sequence[RolloutSegment]) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.cat([segment.observations for segment in segments], dim=0),
        "actions": torch.cat([segment.actions for segment in segments], dim=0),
        "logprobs": torch.cat([segment.logprobs for segment in segments], dim=0),
        "advantages": torch.cat([segment.advantages for segment in segments], dim=0),
        "returns": torch.cat([segment.returns for segment in segments], dim=0),
        "values": torch.cat([segment.values for segment in segments], dim=0),
        "masks": torch.cat([segment.masks for segment in segments], dim=0),
    }


def _ppo_update(
    model: FlatActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: Phase0TrainingConfig,
    ref_model: nn.Module | None = None,
) -> dict[str, float]:
    batch_size = batch["observations"].shape[0]
    permutation = np.arange(batch_size)
    use_kl_ref = ref_model is not None and cfg.kl_to_ref_coef > 0.0
    stats: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "kl_to_ref": 0.0,
    }
    update_steps = 0

    for _ in range(cfg.update_epochs):
        np.random.shuffle(permutation)
        for start in range(0, batch_size, min(cfg.minibatch_size, batch_size)):
            indices = permutation[start : start + min(cfg.minibatch_size, batch_size)]
            obs_mb = batch["observations"][indices]
            action_mb = batch["actions"][indices]
            old_logprob_mb = batch["logprobs"][indices]
            advantage_mb = batch["advantages"][indices]
            return_mb = batch["returns"][indices]
            mask_mb = split_masks(batch["masks"][indices])

            advantage_mb = (advantage_mb - advantage_mb.mean()) / (advantage_mb.std(unbiased=False) + 1e-8)

            # Same mask as sampling time -> the PPO importance ratio stays correct.
            # One forward gives both the PPO terms (via _action_value_from_heads)
            # and the head logits reused for the KL-to-reference anchor below.
            cur_out = model(obs_mb)
            _, new_logprob, entropy, new_value = _action_value_from_heads(cur_out, action_mb, mask_mb)
            log_ratio = new_logprob - old_logprob_mb
            ratio = log_ratio.exp()

            loss_unclipped = -advantage_mb * ratio
            loss_clipped = -advantage_mb * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(loss_unclipped, loss_clipped).mean()
            value_loss = F.mse_loss(new_value, return_mb)
            entropy_loss = entropy.mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

            kl_to_ref_val = 0.0
            if use_kl_ref:
                with torch.no_grad():
                    ref_out = ref_model(obs_mb)
                kl_ref = launch_gated_kl(cur_out, ref_out, mask_mb).mean()
                loss = loss + cfg.kl_to_ref_coef * kl_ref
                kl_to_ref_val = float(kl_ref.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                approx_kl = ((ratio - 1.0) - log_ratio).mean().abs().item()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy_loss.item())
            stats["approx_kl"] += approx_kl
            stats["clipfrac"] += clipfrac
            stats["kl_to_ref"] += kl_to_ref_val
            update_steps += 1

    if update_steps:
        for key in stats:
            stats[key] /= update_steps

    # Critic explained variance over the rollout batch (pre-update value targets):
    # ev <= 0 means the value head explains no more than a constant baseline.
    with torch.no_grad():
        y_true = batch["returns"]
        y_pred = batch["values"]
        var_y = torch.var(y_true, unbiased=False)
        stats["explained_variance"] = (
            float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y) if float(var_y) > 0 else 0.0
        )
    return stats


def _aggregate_episode_metrics(episodes: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {
            "episodes_observed": 0.0,
            "completed_episodes": 0.0,
            "mean_return": 0.0,
            "mean_neutral_captures": 0.0,
            "mean_neutral_capture_rate": 0.0,
            "mean_early_survival_rate": 0.0,
        }
    count = float(len(episodes))
    return {
        "episodes_observed": count,
        "completed_episodes": float(sum(1 for episode in episodes if episode["completed"])),
        "mean_return": float(sum(episode["return_sum"] for episode in episodes) / count),
        "mean_neutral_captures": float(sum(episode["neutral_captures"] for episode in episodes) / count),
        "mean_neutral_capture_rate": float(sum(episode["neutral_capture_rate"] for episode in episodes) / count),
        "mean_early_survival_rate": float(sum(episode["early_survival_rate"] for episode in episodes) / count),
    }


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid PPO checkpoint: {path}")
    return checkpoint


def _greedy_action(out: dict[str, torch.Tensor], masks: dict[str, torch.Tensor] | None):
    """Deterministic (argmax) launch-gated action as the env step input
    ``[launch, source, target, frac, offset]``."""

    def pick(key: str) -> torch.Tensor:
        logits = out[key]
        if masks is not None and key in masks:
            logits = logits.masked_fill(~masks[key].bool(), float("-inf"))
        return logits.argmax(dim=-1)

    cols = [pick("launch")] + [pick(k) for k in ("source", "target", "frac", "offset")]
    return torch.stack(cols, dim=-1).squeeze(0).cpu().numpy()


def _evaluate_margin(
    model: nn.Module,
    training_cfg: Phase0TrainingConfig,
    *,
    opponent_name: str,
    seeds: int,
    device: torch.device,
) -> float:
    """Seat-0 in-loop margin proxy vs an opponent, for keep-best / early-stop gating.

    NOT the promotion gate (that stays the separate both-seat 96-seed benchmark) —
    a cheap, consistent relative signal so the loop keeps the best checkpoint and
    early-stops on drift. Greedy actions; final normalized score margin in [-1, 1].
    """
    was_training = model.training
    model.eval()
    margins: list[float] = []
    for seed in range(int(seeds)):
        env = build_phase0_env(
            seed=seed,
            num_players=training_cfg.num_players,
            opponent_name=opponent_name,
            enable_comets=training_cfg.enable_comets,
            decoder_cfg=decoder_config(training_cfg),
            base_shaping_scale=0.0,
        )
        obs_np, _ = env.reset(seed=seed)
        last_scores: list[float] = [0.0, 0.0]
        for _ in range(int(training_cfg.eval_max_steps)):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            masks = split_masks(
                torch.as_tensor(
                    build_action_masks(env.state, 0, min_ships_to_launch=training_cfg.decoder_min_ships_to_launch),
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
            )
            with torch.no_grad():
                out = model(obs_t)
            obs_np, _, done, _, info = env.step(_greedy_action(out, masks))
            scores = info.get("scores") or None
            if scores:
                last_scores = list(scores)
            if done:
                break
        margins.append(normalized_margin(last_scores, 0) if last_scores else 0.0)
    if was_training:
        model.train()
    return float(np.mean(margins)) if margins else 0.0


def train_phase0(training_cfg: Phase0TrainingConfig) -> dict[str, Any]:
    started_at = time.perf_counter()
    _set_seed(training_cfg.seed)
    device = torch.device(training_cfg.device)
    opponents = _parse_opponents(training_cfg.opponents)

    # Producer/OEP keep per-game memory in a module-level singleton. The batched
    # rollout now gives each env its OWN isolated opponent instance (see
    # get_isolated_opponents), so concurrent games no longer cross-contaminate and
    # batched/vectorized rollout is safe with stateful opponents.

    # When warm-starting, adopt the checkpoint's architecture (e.g. an entity-BC
    # init) so the state_dict loads cleanly; otherwise use the configured arch.
    checkpoint = _load_checkpoint(training_cfg.checkpoint_in, device) if training_cfg.checkpoint_in else None
    training_cfg = _inherit_checkpoint_decoder(training_cfg, checkpoint)
    if checkpoint is not None:
        ckpt_summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
        policy_arch = str(ckpt_summary.get("arch", training_cfg.policy_arch))
    else:
        policy_arch = training_cfg.policy_arch

    model = _build_policy(policy_arch, observation_dim()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

    total_timesteps = 0
    update_idx = 0
    all_episode_metrics: list[dict[str, Any]] = []
    opponent_segments = {name: 0 for name in opponents}
    latest_update_stats: dict[str, float] = {}
    update_series: list[dict[str, float]] = []

    # Movement 2: frozen reference policy for the KL anti-drift anchor (defaults to
    # the warm-start / BC checkpoint).
    ref_model: nn.Module | None = None
    ref_path = training_cfg.ref_checkpoint or training_cfg.checkpoint_in
    if training_cfg.kl_to_ref_coef > 0.0 and ref_path:
        ref_ckpt = _load_checkpoint(ref_path, device)
        ref_arch = str((ref_ckpt.get("summary") or {}).get("arch", policy_arch))
        ref_model = _build_policy(ref_arch, observation_dim()).to(device)
        ref_model.load_state_dict(ref_ckpt["model_state_dict"])
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    # Movement 2: eval-gating state (keep-best by margin + early-stop on drift).
    eval_series: list[dict[str, float]] = []
    best_eval_margin = float("-inf")
    best_eval_update = -1
    best_state: dict[str, torch.Tensor] | None = None
    evals_since_best = 0
    early_stopped = False

    while total_timesteps < training_cfg.total_timesteps:
        segments: list[RolloutSegment] = []
        for opponent_idx, opponent_name in enumerate(opponents):
            if total_timesteps >= training_cfg.total_timesteps:
                break
            remaining = training_cfg.total_timesteps - total_timesteps
            segment_steps = min(training_cfg.rollout_steps, remaining)
            seed = training_cfg.seed + update_idx * 10_000 + opponent_idx * 1_000
            progress = total_timesteps / max(float(training_cfg.total_timesteps), 1.0)
            segment = _collect_rollout_segment(
                model,
                opponent_name=opponent_name,
                base_seed=seed,
                rollout_steps=segment_steps,
                device=device,
                training_cfg=training_cfg,
                progress=progress,
                sample_limit=remaining if _batched_rollout_supported(training_cfg, opponent_name) else segment_steps,
            )
            segments.append(segment)
            opponent_segments[opponent_name] += 1
            total_timesteps += int(segment.observations.shape[0])
            all_episode_metrics.extend(segment.episode_metrics)
        batch = _concat_segments(segments)
        latest_update_stats = _ppo_update(model, optimizer, batch, training_cfg, ref_model=ref_model)
        update_series.append(
            {"update": float(update_idx), "timesteps": float(total_timesteps), **latest_update_stats}
        )
        update_idx += 1

        # Movement 2: eval-gating. Periodically measure the paired margin (the only
        # thing that caught the P3 drift); keep the BEST checkpoint and early-stop.
        if training_cfg.eval_every_updates > 0 and update_idx % training_cfg.eval_every_updates == 0:
            eval_margin = _evaluate_margin(
                model,
                training_cfg,
                opponent_name=training_cfg.eval_opponent,
                seeds=training_cfg.eval_seeds,
                device=device,
            )
            eval_series.append(
                {"update": float(update_idx), "timesteps": float(total_timesteps), "eval_margin": eval_margin}
            )
            if eval_margin > best_eval_margin:
                best_eval_margin = eval_margin
                best_eval_update = update_idx
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                evals_since_best = 0
            else:
                evals_since_best += 1
                if training_cfg.early_stop_patience > 0 and evals_since_best >= training_cfg.early_stop_patience:
                    early_stopped = True
                    break

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    summary = {
        "algorithm": "ppo",
        "arch": policy_arch,
        "policy_track": training_cfg.policy_track,
        "num_players": training_cfg.num_players,
        "timesteps": total_timesteps,
        "updates": update_idx,
        "rollout_num_envs": int(training_cfg.rollout_num_envs),
        "rollout_backend": (
            "rust_batch"
            if any(_batched_rollout_supported(training_cfg, name) for name in opponents)
            else "gym_single_env"
        ),
        "training_wall_seconds": elapsed_seconds,
        "env_steps_per_second": total_timesteps / elapsed_seconds,
        "opponents": list(opponents),
        "opponent_segments": opponent_segments,
        "enable_comets": training_cfg.enable_comets,
        "reward_shaping": "annealed_base_plus_normalized_margin_plus_temporal_comet_auxiliary",
        "ship_margin_scale": training_cfg.ship_margin_scale,
        "normalized_margin_scale_start": training_cfg.normalized_margin_scale_start,
        "normalized_margin_scale_end": training_cfg.normalized_margin_scale_end,
        "base_shaping_scale_start": training_cfg.base_shaping_scale_start,
        "base_shaping_scale_end": training_cfg.base_shaping_scale_end,
        "shaping_potential": training_cfg.shaping_potential,
        "kl_to_ref_coef": training_cfg.kl_to_ref_coef,
        "ref_checkpoint": ref_path if training_cfg.kl_to_ref_coef > 0.0 else None,
        "comet_shaping_scale_start": training_cfg.comet_shaping_scale_start,
        "comet_shaping_scale_end": training_cfg.comet_shaping_scale_end,
        "four_player_vulnerability_scale_start": training_cfg.four_player_vulnerability_scale_start,
        "four_player_vulnerability_scale_end": training_cfg.four_player_vulnerability_scale_end,
        "four_player_leader_scale_start": training_cfg.four_player_leader_scale_start,
        "four_player_leader_scale_end": training_cfg.four_player_leader_scale_end,
        "four_player_third_player_scale_start": training_cfg.four_player_third_player_scale_start,
        "four_player_third_player_scale_end": training_cfg.four_player_third_player_scale_end,
        "decoder": decoder_payload(training_cfg),
        "checkpoint_in": training_cfg.checkpoint_in,
        "checkpoint_out": training_cfg.checkpoint_out,
    }
    summary.update(_aggregate_episode_metrics(all_episode_metrics))
    summary.update({f"last_{key}": value for key, value in latest_update_stats.items()})
    summary["update_series"] = update_series
    eval_gated = training_cfg.eval_every_updates > 0 and best_state is not None
    summary["eval_series"] = eval_series
    summary["eval_gated"] = eval_gated
    summary["best_eval_margin"] = best_eval_margin if eval_gated else None
    summary["best_eval_update"] = best_eval_update if eval_gated else None
    summary["early_stopped"] = early_stopped
    summary["checkpoint_selection"] = "best_eval_margin" if eval_gated else "final"

    if training_cfg.checkpoint_out:
        # Movement 2: with eval-gating on, persist the BEST checkpoint by paired
        # margin (the final/drifted model is worse — the whole point of gating);
        # otherwise the final model.
        save_state = best_state if eval_gated else model.state_dict()
        checkpoint_path = Path(training_cfg.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": save_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(training_cfg),
                "summary": summary,
            },
            checkpoint_path,
        )

    return summary


def build_phase0_env(
    *,
    seed: int,
    num_players: int = 2,
    opponent_name: str = "greedy",
    enable_comets: bool = True,
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
    decoder_cfg: DecoderConfig | None = None,
) -> OrbitWarsGymEnv:
    opponent_policy = _training_opponent_policy(opponent_name, num_players)
    rust_cfg = RustConfig(enable_comets=enable_comets)
    return OrbitWarsGymEnv(
        num_players=num_players,
        seed=seed,
        rust_cfg=rust_cfg,
        opponent_policy=opponent_policy,
        decoder_cfg=decoder_cfg,
        sun_loss_penalty=sun_loss_penalty,
        border_loss_penalty=border_loss_penalty,
        ship_margin_scale=ship_margin_scale,
        normalized_margin_scale=normalized_margin_scale,
        base_shaping_scale=base_shaping_scale,
        comet_shaping_scale=comet_shaping_scale,
        shaping_gamma=shaping_gamma,
        four_player_vulnerability_scale=four_player_vulnerability_scale,
        four_player_leader_scale=four_player_leader_scale,
        four_player_third_player_scale=four_player_third_player_scale,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-track", choices=("phase0_2p", "phase5_4p"), default="phase0_2p")
    parser.add_argument("--policy-arch", choices=("flat", "entity"), default="flat",
                        help="policy architecture for training from scratch; a --checkpoint-in overrides it with the checkpoint's arch")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--opponents", default="greedy,defensive,rush,anti_meta,weak_random")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollout-num-envs", type=int, default=1)
    parser.add_argument("--opponent-workers", type=int, default=1,
                        help="opponent-call parallelism in batched rollout. 1=sequential (recommended/default). "
                             ">1 = experimental process pool (measured SLOWER: planner is too cheap vs per-step IPC). "
                             "0 = auto (also slower). Threads were tried too and are GIL-bound.")
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--early-survival-window", type=int, default=64)
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable-comets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sun-loss-penalty", type=float, default=0.02)
    parser.add_argument("--border-loss-penalty", type=float, default=0.02)
    parser.add_argument("--ship-margin-scale", type=float, default=0.0)
    parser.add_argument("--normalized-margin-scale-start", type=float, default=0.0)
    parser.add_argument("--normalized-margin-scale-end", type=float, default=0.0)
    parser.add_argument("--base-shaping-scale-start", type=float, default=1.0)
    parser.add_argument("--base-shaping-scale-end", type=float, default=0.15)
    parser.add_argument("--comet-shaping-scale-start", type=float, default=0.08)
    parser.add_argument("--comet-shaping-scale-end", type=float, default=0.0)
    parser.add_argument("--four-player-vulnerability-scale-start", type=float, default=0.06)
    parser.add_argument("--four-player-vulnerability-scale-end", type=float, default=0.02)
    parser.add_argument("--four-player-leader-scale-start", type=float, default=0.05)
    parser.add_argument("--four-player-leader-scale-end", type=float, default=0.02)
    parser.add_argument("--four-player-third-player-scale-start", type=float, default=0.04)
    parser.add_argument("--four-player-third-player-scale-end", type=float, default=0.015)
    parser.add_argument("--decoder-max-moves-per-turn", type=int, default=8)
    parser.add_argument("--decoder-min-ships-to-launch", type=int, default=2)
    parser.add_argument("--decoder-reserve-home-ships", type=int, default=8)
    parser.add_argument(
        "--inherit-checkpoint-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when warm-starting, use the decoder payload embedded in --checkpoint-in",
    )
    # --- Movement 2 ---
    parser.add_argument("--shaping-potential", choices=("producer", "none"), default="producer",
                        help="Mov.2 de-anchor: 'none' drops the production/territory base shaping (= Producer objective)")
    parser.add_argument("--kl-to-ref-coef", type=float, default=0.0,
                        help="Mov.2 anti-drift: KL-to-reference penalty coefficient (0=off)")
    parser.add_argument("--ref-checkpoint", default=None,
                        help="Mov.2: reference policy for the KL anchor (default: --checkpoint-in)")
    parser.add_argument("--eval-every-updates", type=int, default=0,
                        help="Mov.2 eval-gating: in-loop margin eval every N updates (0=off)")
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-opponent", default="producer")
    parser.add_argument("--eval-max-steps", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Mov.2 eval-gating: stop after N consecutive evals without improvement (0=off)")
    args = parser.parse_args()

    env: gym.Env = build_phase0_env(
        seed=args.seed,
        num_players=args.num_players,
        opponent_name="greedy",
        enable_comets=args.enable_comets,
        sun_loss_penalty=args.sun_loss_penalty,
        border_loss_penalty=args.border_loss_penalty,
        ship_margin_scale=args.ship_margin_scale,
        normalized_margin_scale=args.normalized_margin_scale_start,
        base_shaping_scale=args.base_shaping_scale_start,
        comet_shaping_scale=args.comet_shaping_scale_start,
        four_player_vulnerability_scale=args.four_player_vulnerability_scale_start,
        four_player_leader_scale=args.four_player_leader_scale_start,
        four_player_third_player_scale=args.four_player_third_player_scale_start,
        decoder_cfg=DecoderConfig(
            max_moves_per_turn=args.decoder_max_moves_per_turn,
            min_ships_to_launch=args.decoder_min_ships_to_launch,
            reserve_home_ships=args.decoder_reserve_home_ships,
        ),
    )
    obs, _ = env.reset(seed=args.seed)
    base_cfg = Phase0TrainingConfig(
        seed=args.seed,
        policy_track=args.training_track,
        policy_arch=args.policy_arch,
        num_players=args.num_players,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        rollout_num_envs=args.rollout_num_envs,
        opponent_workers=args.opponent_workers,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        early_survival_window=args.early_survival_window,
        opponents=_parse_opponents(args.opponents),
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        device=args.device,
        enable_comets=args.enable_comets,
        sun_loss_penalty=args.sun_loss_penalty,
        border_loss_penalty=args.border_loss_penalty,
        ship_margin_scale=args.ship_margin_scale,
        normalized_margin_scale_start=args.normalized_margin_scale_start,
        normalized_margin_scale_end=args.normalized_margin_scale_end,
        base_shaping_scale_start=args.base_shaping_scale_start,
        base_shaping_scale_end=args.base_shaping_scale_end,
        comet_shaping_scale_start=args.comet_shaping_scale_start,
        comet_shaping_scale_end=args.comet_shaping_scale_end,
        four_player_vulnerability_scale_start=args.four_player_vulnerability_scale_start,
        four_player_vulnerability_scale_end=args.four_player_vulnerability_scale_end,
        four_player_leader_scale_start=args.four_player_leader_scale_start,
        four_player_leader_scale_end=args.four_player_leader_scale_end,
        four_player_third_player_scale_start=args.four_player_third_player_scale_start,
        four_player_third_player_scale_end=args.four_player_third_player_scale_end,
        decoder_max_moves_per_turn=args.decoder_max_moves_per_turn,
        decoder_min_ships_to_launch=args.decoder_min_ships_to_launch,
        decoder_reserve_home_ships=args.decoder_reserve_home_ships,
        inherit_checkpoint_decoder=bool(args.inherit_checkpoint_decoder),
        shaping_potential=args.shaping_potential,
        kl_to_ref_coef=args.kl_to_ref_coef,
        ref_checkpoint=args.ref_checkpoint,
        eval_every_updates=args.eval_every_updates,
        eval_seeds=args.eval_seeds,
        eval_opponent=args.eval_opponent,
        eval_max_steps=args.eval_max_steps,
        early_stop_patience=args.early_stop_patience,
    )
    summary = (
        train_phase5_4p(base_cfg)
        if args.training_track == "phase5_4p"
        else train_phase0(base_cfg)
    )
    payload = {
        "obs_dim": int(obs.shape[0]),
        "params": int(sum(p.numel() for p in _build_policy(summary.get("arch", "flat"), observation_dim()).parameters())),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
