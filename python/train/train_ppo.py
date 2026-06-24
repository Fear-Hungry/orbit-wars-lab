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
    PLANET_N,
    EntityActorCritic,
    FlatActorCritic,
    GridNetActorCritic,
    ProducerResidualBranchActorCritic,
    _action_value_from_heads,
    gridnet_gated_kl,
    launch_gated_kl,
)
from python.agents.registry import (
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
    get_isolated_opponents,
    make_isolated_opponent,
)
from python.train.opponent_pool import get_process_opponent_pool

from orbit_wars_gym import OrbitWarsGymEnv
from orbit_wars_gym.action_decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
    decode_discrete_action,
    decode_gridnet_action,
    gridnet_planet_mask,
)
from orbit_wars_gym.action_masks import build_action_masks, split_masks
from orbit_wars_gym.backend import RustBatchBackend, RustConfig
from orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim
from orbit_wars_gym.entities import fleet_owner, planet_id, planet_owner, planet_ships

_POLICY_ARCHS = {
    "flat": FlatActorCritic,
    "entity": EntityActorCritic,
    "producer_residual": ProducerResidualBranchActorCritic,
    "gridnet": GridNetActorCritic,
}


def _build_policy(arch: str, obs_dim: int, **kwargs):
    if arch not in _POLICY_ARCHS:
        raise ValueError(f"unknown policy arch {arch!r}; valid: {sorted(_POLICY_ARCHS)}")
    return _POLICY_ARCHS[arch](obs_dim, **kwargs)


_HEURISTIC_POLICIES = get_heuristic_policies()
# "pgs" is the operational holdwave bot (bots.pgs.agent SUBMISSION_CONFIG). It is
# ~30ms/decision, so weight it lightly in rollout curricula (e.g. 1 slot in 10).
PHASE0_OPPONENTS = {
    name: _HEURISTIC_POLICIES[name]
    for name in ("producer", "oep", "pgs", "greedy", "defensive", "rush", "anti_meta", "weak_random")
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


def _bc_anchor_policies(name: str | None, count: int):
    if not name:
        return []
    parts = _opponent_parts(name)
    if len(parts) != 1:
        raise ValueError("bc_anchor_teacher must be a single known expert, not a composite lineup")
    teacher = parts[0]
    unknown = _unknown_opponent_parts([teacher])
    if unknown:
        raise ValueError(f"unknown bc_anchor_teacher: {', '.join(unknown)}")
    if teacher in STATEFUL_SINGLETON_OPPONENTS:
        return get_isolated_opponents(teacher, count)
    if teacher in LEAGUE_TRAINING_OPPONENTS:
        return [_league_training_policy(teacher) for _ in range(count)]
    return [PHASE0_OPPONENTS[teacher] for _ in range(count)]


def _training_opponent_policy(name: str, num_players: int, learner_player: int = 0):
    parts = _opponent_parts(name)
    unknown = _unknown_opponent_parts([name])
    if unknown:
        raise ValueError(f"unknown phase-0 opponent parts: {', '.join(unknown)}")
    if not parts:
        raise ValueError("opponent name cannot be empty")
    learner_player = int(learner_player)
    num_players = int(num_players)
    if learner_player < 0 or learner_player >= num_players:
        raise ValueError("learner_player must be in [0, num_players)")

    opponent_players = [player for player in range(num_players) if player != learner_player]
    seat_policies = {
        player: _seat_policy(parts[idx % len(parts)], idx)
        for idx, player in enumerate(opponent_players)
    }

    def policy(state: dict[str, Any], player: int) -> list[list[float]]:
        player = int(player)
        if player == learner_player:
            return []
        return seat_policies[player](state, player)

    policy.__name__ = "+".join(parts)
    return policy


@dataclass(frozen=True)
class Phase0TrainingConfig:
    seed: int = 0
    policy_track: str = "phase0_2p"
    policy_arch: str = "flat"
    # BASE agent for the producer_residual arch (the plan the net edits). Registry
    # name: "producer" (original BReP) or "pgs"/"pgs_holdwave" (holdwave incumbent —
    # the parity floor we want to not regress below). Ignored by other archs.
    base_agent: str = "producer"
    # GridNet league/PFSP: path to a FROZEN snapshot checkpoint that plays the
    # "self" opponent seat (instead of the live policy). The campaign points this at
    # a growing pool of past chunks so the opponent's strength tracks the agent's —
    # the AlphaStar device for crossing strength gaps the live-self-play empate can't.
    self_opponent_checkpoint: str | None = None
    # Handicap curriculum: scale a planner opponent's launched ships by this factor.
    # The GridNet policy beats the producer at ~0.2x ships but is crushed at 1.0x —
    # a difficulty curriculum (start low, raise as the policy dominates) is the only
    # bridge across the reactive→planner cliff (no intermediate-strength bot exists).
    opponent_handicap: float = 1.0
    num_players: int = 2
    total_timesteps: int = 200_000
    episode_steps: int = 500
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
    elimination_penalty: float = 0.0
    decoder_max_moves_per_turn: int = 8
    decoder_min_ships_to_launch: int = 2
    decoder_reserve_home_ships: int = 8
    # Override the policy's target_rank during rollout/eval decode. The learned
    # target head is unlearnable (invert_moves -> near-uniform target_rank label;
    # measured BC val acc 0.085 ~ random/32). 0 = always aim at the decoder's own
    # highest target_score candidate. Set 0 so PPO does NOT explore a random target
    # head (which annihilates the policy in training, the recorded "DRL cliff").
    decoder_force_target_rank: int | None = None
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
    # Optional residual-style teacher anchor: label rollout states with a strong
    # heuristic/BReP action through the same inverse projection used by BC, then
    # add a small launch-gated CE term to PPO. 0 keeps the original PPO behavior.
    bc_anchor_coef: float = 0.0
    bc_anchor_coef_end: float | None = None
    bc_anchor_teacher: str | None = None
    bc_anchor_max_quant_error: float = float("inf")
    # Rotate the learner across seats in single-env rollouts. This is especially
    # important for 4p: the promotion gate evaluates every seat, so training only
    # as player 0 creates a silent seat-distribution mismatch.
    learner_seat_rotation: bool = False


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
    teacher_actions: torch.Tensor
    teacher_action_mask: torch.Tensor
    teacher_quant_errors: torch.Tensor
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
    # "self" = GridNet self-play opponent (the live/snapshot net plays the other
    # seat); valid alone (pure self-play needs no second distinct opponent).
    if len(set(items)) < 2 and "self" not in items:
        raise ValueError("training requires at least two distinct opponents")
    # "<name>@<scale>" = handicap-curriculum opponent (ships scaled). Validate base.
    unknown = [
        name for name in items
        if name.partition("@")[0] not in PHASE0_OPPONENTS and name != "self"
    ]
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


def bc_anchor_coef(training_cfg: Phase0TrainingConfig, progress: float) -> float:
    end = training_cfg.bc_anchor_coef if training_cfg.bc_anchor_coef_end is None else training_cfg.bc_anchor_coef_end
    return _linear_schedule(float(training_cfg.bc_anchor_coef), float(end), progress)


def bc_anchor_enabled(training_cfg: Phase0TrainingConfig) -> bool:
    end = training_cfg.bc_anchor_coef if training_cfg.bc_anchor_coef_end is None else training_cfg.bc_anchor_coef_end
    return max(float(training_cfg.bc_anchor_coef), float(end)) > 0.0


def decoder_config(training_cfg: Phase0TrainingConfig) -> DecoderConfig:
    return DecoderConfig(
        fractions=tuple(float(value) for value in training_cfg.decoder_fractions),
        angle_offsets=tuple(float(value) for value in training_cfg.decoder_angle_offsets),
        max_moves_per_turn=int(training_cfg.decoder_max_moves_per_turn),
        min_ships_to_launch=int(training_cfg.decoder_min_ships_to_launch),
        reserve_home_ships=int(training_cfg.decoder_reserve_home_ships),
        force_target_rank=(None if training_cfg.decoder_force_target_rank is None
                           else int(training_cfg.decoder_force_target_rank)),
    )


def decoder_payload(training_cfg: Phase0TrainingConfig) -> dict[str, Any]:
    cfg = decoder_config(training_cfg)
    return {
        "fractions": list(cfg.fractions),
        "angle_offsets": list(cfg.angle_offsets),
        "max_moves_per_turn": cfg.max_moves_per_turn,
        "min_ships_to_launch": cfg.min_ships_to_launch,
        "reserve_home_ships": cfg.reserve_home_ships,
        "force_target_rank": cfg.force_target_rank,
    }


def _bc_anchor_action(
    state: dict[str, Any],
    player: int,
    policy,
    decoder_cfg: DecoderConfig,
    max_quant_error: float,
) -> tuple[list[int], bool, float]:
    moves = list(policy(state, player))
    if moves and not moves_are_legal(state, player, moves):
        return [0, 0, 0, 0, 0], False, float("inf")
    result = invert_moves(
        state,
        player,
        moves,
        decoder_cfg=decoder_cfg,
        cfg=DEFAULT_INVERSE_CONFIG,
    )
    use_label = bool(result.is_no_op or float(result.quant_error) <= float(max_quant_error))
    return [int(value) for value in result.action5], use_label, float(result.quant_error)


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
        elimination_penalty=0.35,
        learner_seat_rotation=True,
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
    learner_player: int,
    rollout_steps: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
) -> RolloutSegment:
    learner_player = int(learner_player)
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
        learner_player=learner_player,
        opponent_name=opponent_name,
        enable_comets=training_cfg.enable_comets,
        episode_steps=training_cfg.episode_steps,
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
        elimination_penalty=training_cfg.elimination_penalty,
    )
    obs_np, _ = env.reset(seed=base_seed)
    episode = EpisodeMetrics(opponent=opponent_name)
    episode_metrics: list[dict[str, Any]] = []
    anchor_policy = (
        _bc_anchor_policies(training_cfg.bc_anchor_teacher, 1)[0]
        if bc_anchor_enabled(training_cfg) and training_cfg.bc_anchor_teacher
        else None
    )
    decoder_cfg = decoder_config(training_cfg)

    obs_buf = []
    action_buf = []
    logprob_buf = []
    value_buf = []
    mask_buf = []
    teacher_action_buf = []
    teacher_mask_buf = []
    teacher_quant_error_buf = []
    rewards_np = np.empty(rollout_steps, dtype=np.float32)
    dones_np = np.empty(rollout_steps, dtype=np.float32)

    for reset_idx in range(rollout_steps):
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            build_action_masks(
                env.state,
                learner_player,
                min_ships_to_launch=training_cfg.decoder_min_ships_to_launch,
            ),
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(
                obs_tensor, masks=split_masks(mask_tensor)
            )
        action = action_tensor.squeeze(0).cpu().numpy()
        previous_state = env.state
        if anchor_policy is None:
            teacher_action = [0, 0, 0, 0, 0]
            teacher_mask = False
            teacher_quant_error = float("inf")
        else:
            teacher_action, teacher_mask, teacher_quant_error = _bc_anchor_action(
                previous_state,
                learner_player,
                anchor_policy,
                decoder_cfg,
                training_cfg.bc_anchor_max_quant_error,
            )
        next_obs_np, reward, done, _, _ = env.step(action)
        next_state = env.state

        neutral_captures = _neutral_capture_count(previous_state, next_state, player=learner_player)
        alive = _player_alive(next_state, player=learner_player)
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
        teacher_action_buf.append(torch.as_tensor(teacher_action, dtype=torch.long, device=device))
        teacher_mask_buf.append(torch.as_tensor(teacher_mask, dtype=torch.bool, device=device))
        teacher_quant_error_buf.append(torch.as_tensor(teacher_quant_error, dtype=torch.float32, device=device))
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
        teacher_actions=torch.stack(teacher_action_buf),
        teacher_action_mask=torch.stack(teacher_mask_buf),
        teacher_quant_errors=torch.stack(teacher_quant_error_buf),
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
    if training_cfg.learner_seat_rotation:
        return False
    if int(training_cfg.rollout_num_envs) <= 1 or training_cfg.num_players != 2:
        return False
    if opponent_name is not None and len(_opponent_parts(opponent_name)) != 1:
        return False
    if opponent_name in LEAGUE_TRAINING_OPPONENTS:
        return False
    return True


# BReP per-slot edit codes over a base move (v2): 0=KEEP, 1=CANCEL, ship-SCALE
# {2:x0.25, 3:x0.5, 4:x1.5, 5:x2.0}. Down-scales always legal; up-scales capped at
# the source planet's ships. This table MUST match the exported submission's decode.
_EDIT_SCALES = {2: 0.25, 3: 0.5, 4: 1.5, 5: 2.0}

# Registry aliases for the residual base; "pgs_holdwave" == "pgs" (holdwave is the
# pgs.agent SUBMISSION_CONFIG isolated opponent).
_BASE_AGENT_ALIASES = {"pgs_holdwave": "pgs"}


def _make_base_agent(name: str) -> "Policy":
    """Fresh isolated base agent (own per-game memory) for the BReP rollout."""
    return make_isolated_opponent(_BASE_AGENT_ALIASES.get(name, name))


def _apply_residual_edits(
    state: dict[str, Any], base_moves: Sequence[Sequence[float]], edits: Sequence[int], k_max: int
) -> list[list[float]]:
    """Apply per-slot BReP edits to a base plan. Codes: 0=KEEP, 1=CANCEL (drop),
    2-5 = ship SCALE (down always legal; up capped at the source planet's ships).
    Moves past ``k_max`` editable slots are kept as-is, so KEEP-everything reproduces
    the EXACT base plan (the parity-floor invariant). Base-agnostic: works over any
    ``[planet, angle, ships]`` move list (Producer or holdwave)."""
    ships_by_id = {planet_id(p): planet_ships(p) for p in state.get("planets", [])}
    out: list[list[float]] = []
    for i, mv in enumerate(base_moves):
        ships = int(mv[2])
        if i >= k_max:
            out.append([mv[0], mv[1], float(ships)])
            continue
        e = int(edits[i])
        if e == 1:  # CANCEL
            continue
        scale = _EDIT_SCALES.get(e)
        if scale is None:  # KEEP (0) or any unknown code -> exact base move
            out.append([mv[0], mv[1], float(ships)])
            continue
        scaled = int(round(ships * scale))
        if scale > 1.0:  # boost: cap at the source planet's available ships
            scaled = min(scaled, max(1, int(ships_by_id.get(int(mv[0]), ships)) - 1))
        out.append([mv[0], mv[1], float(max(1, scaled))])
    return out


def _collect_brep_rollout_segment(
    model: ProducerResidualBranchActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    rollout_steps: int,
    sample_limit: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
) -> RolloutSegment:
    """BReP batched rollout: the agent action is a per-slot edit over a BASE plan
    (one extra base-agent call per env) instead of a decoded raw move. Mask = active
    base-move slots. Mirrors _collect_batched_rollout_segment's reward path; the base
    agent is configurable (training_cfg.base_agent — producer or pgs/holdwave)."""
    if training_cfg.num_players != 2:
        raise ValueError("BReP rollout currently supports only 2-player training")
    if opponent_name not in PHASE0_OPPONENTS:
        raise ValueError(f"unknown phase-0 opponent: {opponent_name}")
    k_max = int(model.k_max)
    agent_player = 0
    opp_player = 1
    num_envs = max(1, min(int(training_cfg.rollout_num_envs), int(sample_limit)))
    opponent_policies = get_isolated_opponents(opponent_name, num_envs)
    # FRESH base instances (not the cached pool) so the player-0 base planner never
    # shares a stateful runtime with the player-1 opponent of the same name.
    base_policies = [_make_base_agent(training_cfg.base_agent) for _ in range(num_envs)]
    base_shaping_scale, comet_shaping_scale = shaping_scales(training_cfg, progress)
    reward_env = build_phase0_env(
        seed=base_seed,
        num_players=training_cfg.num_players,
        opponent_name=opponent_name,
        enable_comets=training_cfg.enable_comets,
        decoder_cfg=decoder_config(training_cfg),
        sun_loss_penalty=training_cfg.sun_loss_penalty,
        border_loss_penalty=training_cfg.border_loss_penalty,
        ship_margin_scale=training_cfg.ship_margin_scale,
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
    obs_np = backend.encoded_states(agent_player)
    episodes = [EpisodeMetrics(opponent=opponent_name) for _ in range(num_envs)]
    episode_metrics: list[dict[str, Any]] = []
    active = np.ones(num_envs, dtype=bool)

    obs_buf, action_buf, logprob_buf, value_buf, mask_buf = [], [], [], [], []
    rewards_np = np.empty((rollout_steps, num_envs), dtype=np.float32)
    dones_np = np.empty((rollout_steps, num_envs), dtype=np.float32)

    for step_index in range(rollout_steps):
        active_indices = [i for i in range(num_envs) if active[i]]
        # Agent's BASE plan per env (computed BEFORE the net so the mask knows how
        # many edit slots are live). This is the one extra base-agent call.
        base_moves_by_env: list[list[list[float]]] = [[] for _ in range(num_envs)]
        for i in active_indices:
            base_moves_by_env[i] = [list(m) for m in base_policies[i](current_states[i], agent_player)]

        edit_mask_np = np.zeros((num_envs, k_max), dtype=bool)
        for i in active_indices:
            n_slots = min(len(base_moves_by_env[i]), k_max)
            if n_slots > 0:
                edit_mask_np[i, :n_slots] = True
        mask_tensor = torch.as_tensor(edit_mask_np, dtype=torch.bool, device=device)

        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(
                obs_tensor, masks={"edit": mask_tensor}
            )
        actions_np = action_tensor.cpu().numpy()

        opponent_moves_by_env = {
            i: opponent_policies[i](current_states[i], opp_player) for i in active_indices
        }

        action_rows: list[list[float]] = []
        player_moves_by_env: list[list[list[float]]] = [[] for _ in range(num_envs)]
        for env_index in active_indices:
            state = current_states[env_index]
            player_moves = _apply_residual_edits(
                state, base_moves_by_env[env_index], actions_np[env_index], k_max
            )
            player_moves_by_env[env_index] = player_moves
            action_rows.extend(_moves_to_flat_rows(env_index, agent_player, player_moves))
            action_rows.extend(_moves_to_flat_rows(env_index, opp_player, opponent_moves_by_env[env_index]))

        flat_actions = (
            np.asarray(action_rows, dtype=np.float64) if action_rows else np.zeros((0, 5), dtype=np.float64)
        )
        previous_states = current_states
        outcomes, next_obs_np = backend.step_flat_with_encoded_states(flat_actions, agent_player)
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
                previous_state, next_state, player=agent_player,
                player_moves=player_moves_by_env[env_index], done=done,
            )
            ship_margin_reward = reward_env._ship_margin_reward(previous_state, next_state, player=agent_player)
            comet_reward = reward_env._comet_auxiliary_reward(previous_state, next_state, player=agent_player)
            reward = (
                base_shaping_scale * base_reward + ship_margin_reward + comet_shaping_scale * comet_reward
            )
            if done:
                rewards = outcome.get("rewards", [])
                reward += float(rewards[agent_player]) if rewards else 0.0
            rewards_row[env_index] = float(reward)
            dones_row[env_index] = float(done)
            episodes[env_index].record_step(
                reward=reward,
                neutral_captures=_neutral_capture_count(previous_state, next_state, player=agent_player),
                alive=_player_alive(next_state, player=agent_player),
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

    for env_index, episode in enumerate(episodes):
        if active[env_index] and episode.length > 0:
            episode_metrics.append(episode.as_summary())

    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
        next_value = next_value.masked_fill(torch.as_tensor(~active, dtype=torch.bool, device=device), 0.0)

    observations = torch.stack(obs_buf)
    actions = torch.stack(action_buf).to(dtype=torch.long)
    logprobs = torch.stack(logprob_buf)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
    values = torch.stack(value_buf)
    advantages, returns = _compute_gae_batched(
        rewards, dones, values, next_value, gamma=training_cfg.gamma, gae_lambda=training_cfg.gae_lambda
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


def _gridnet_moves_batch(
    model: GridNetActorCritic,
    states: list[dict],
    indices: list[int],
    player: int,
    device: torch.device,
    decoder_cfg: DecoderConfig,
    *,
    sample: bool,
    obs_np: "np.ndarray | None" = None,
) -> dict[int, list[list[float]]]:
    """Batched GridNet move generation for a set of envs (one forward, greedy or
    sampled). Used for the SELF-PLAY opponent seat — the net plays itself, which is
    batchable (~1ms) instead of a Python planner per step (the historical
    throughput bottleneck). ``obs_np`` (the backend's batched encoding for ``player``)
    is reused when given, avoiding a per-env Python encode_state loop (~50% of the
    rollout time)."""
    if not indices:
        return {}
    if obs_np is not None:
        obs = torch.as_tensor(obs_np[indices], dtype=torch.float32, device=device)
    else:
        obs = torch.as_tensor(
            np.stack([encode_state(states[i], player, DEFAULT_ENCODER_CONFIG) for i in indices]),
            dtype=torch.float32, device=device,
        )
    mask = torch.as_tensor(
        np.stack([gridnet_planet_mask(states[i], player, decoder_cfg) for i in indices]),
        dtype=torch.bool, device=device,
    )
    with torch.no_grad():
        if sample:
            a, _, _, _ = model.get_action_and_value(obs, masks={"planet": mask})
        else:
            out = model.forward(obs)
            launch = out["launch"].argmax(-1)
            launch = torch.where(mask, launch, torch.zeros_like(launch))
            a = torch.stack([launch, out["target"].argmax(-1), out["frac"].argmax(-1), out["offset"].argmax(-1)], dim=-1)
    a_np = a.cpu().numpy()
    return {env: decode_gridnet_action(states[env], player, a_np[k], decoder_cfg) for k, env in enumerate(indices)}


def _collect_gridnet_rollout_segment(
    model: GridNetActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    rollout_steps: int,
    sample_limit: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
    opponent_model: GridNetActorCritic | None = None,
) -> RolloutSegment:
    """GridNet per-planet rollout with SELF-PLAY or heuristic opponents (2p).

    opponent_name == "self": the opponent seat is played by ``opponent_model`` (a
    frozen snapshot) or the live model — batchable, removing the per-step Python
    planner bottleneck. Otherwise an isolated heuristic opponent (producer/oep/pgs)
    provides curriculum diversity (the ablation's 'diversified opponents')."""
    if training_cfg.num_players != 2:
        raise ValueError("GridNet rollout currently supports only 2-player training")
    is_self = opponent_name == "self"
    # "<name>@<scale>" handicaps the opponent's launched ships (curriculum).
    handicap = float(training_cfg.opponent_handicap)
    base_opp_name = opponent_name
    if "@" in opponent_name:
        base_opp_name, _, scale_s = opponent_name.partition("@")
        handicap = float(scale_s)
    if not is_self and base_opp_name not in PHASE0_OPPONENTS:
        raise ValueError(f"unknown phase-0 opponent: {base_opp_name}")
    agent_player, opp_player = 0, 1
    decoder_cfg = decoder_config(training_cfg)
    opp_net = opponent_model if opponent_model is not None else model
    num_envs = max(1, min(int(training_cfg.rollout_num_envs), int(sample_limit)))
    opponent_policies = None if is_self else get_isolated_opponents(base_opp_name, num_envs)

    def _hcap(moves: list) -> list:
        if handicap >= 1.0:
            return moves
        return [[m[0], m[1], max(1.0, float(m[2]) * handicap)] for m in moves]
    base_shaping_scale, comet_shaping_scale = shaping_scales(training_cfg, progress)
    reward_env = build_phase0_env(
        seed=base_seed, num_players=2,
        opponent_name="producer" if is_self else base_opp_name,
        enable_comets=training_cfg.enable_comets, decoder_cfg=decoder_cfg,
        sun_loss_penalty=training_cfg.sun_loss_penalty, border_loss_penalty=training_cfg.border_loss_penalty,
        ship_margin_scale=training_cfg.ship_margin_scale,
        base_shaping_scale=base_shaping_scale, comet_shaping_scale=comet_shaping_scale,
        shaping_gamma=training_cfg.gamma,
    )
    backend = RustBatchBackend(
        num_envs=num_envs, num_players=2, seed=base_seed,
        config=RustConfig(enable_comets=training_cfg.enable_comets),
    )
    current_states = backend.reset(base_seed)
    obs_np = backend.encoded_states(agent_player)
    episodes = [EpisodeMetrics(opponent=opponent_name) for _ in range(num_envs)]
    episode_metrics: list[dict[str, Any]] = []
    active = np.ones(num_envs, dtype=bool)
    obs_buf, action_buf, logprob_buf, value_buf, mask_buf = [], [], [], [], []
    rewards_np = np.empty((rollout_steps, num_envs), dtype=np.float32)
    dones_np = np.empty((rollout_steps, num_envs), dtype=np.float32)

    for step_index in range(rollout_steps):
        active_indices = [i for i in range(num_envs) if active[i]]
        mask_np = np.stack([gridnet_planet_mask(s, agent_player, decoder_cfg) for s in current_states])
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor, logprob_tensor, _, value_tensor = model.get_action_and_value(
                obs_tensor, masks={"planet": mask_tensor}
            )
        actions_np = action_tensor.cpu().numpy()

        if is_self:
            opp_obs_np = backend.encoded_states(opp_player)  # batched, no per-env encode
            opp_moves = _gridnet_moves_batch(
                opp_net, current_states, active_indices, opp_player, device, decoder_cfg,
                sample=True, obs_np=opp_obs_np,
            )
        else:
            opp_moves = {i: _hcap(opponent_policies[i](current_states[i], opp_player)) for i in active_indices}

        action_rows: list[list[float]] = []
        player_moves_by_env: list[list[list[float]]] = [[] for _ in range(num_envs)]
        for env_index in active_indices:
            pm = decode_gridnet_action(current_states[env_index], agent_player, actions_np[env_index], decoder_cfg)
            player_moves_by_env[env_index] = pm
            action_rows.extend(_moves_to_flat_rows(env_index, agent_player, pm))
            action_rows.extend(_moves_to_flat_rows(env_index, opp_player, opp_moves[env_index]))

        flat_actions = np.asarray(action_rows, dtype=np.float64) if action_rows else np.zeros((0, 5), dtype=np.float64)
        previous_states = current_states
        outcomes, next_obs_np = backend.step_flat_with_encoded_states(flat_actions, agent_player)
        next_states = backend.states()

        rewards_row, dones_row = rewards_np[step_index], dones_np[step_index]
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
                previous_state, next_state, player=agent_player,
                player_moves=player_moves_by_env[env_index], done=done,
            )
            ship_margin_reward = reward_env._ship_margin_reward(previous_state, next_state, player=agent_player)
            comet_reward = reward_env._comet_auxiliary_reward(previous_state, next_state, player=agent_player)
            reward = base_shaping_scale * base_reward + ship_margin_reward + comet_shaping_scale * comet_reward
            if done:
                rewards = outcome.get("rewards", [])
                reward += float(rewards[agent_player]) if rewards else 0.0
            rewards_row[env_index] = float(reward)
            dones_row[env_index] = float(done)
            episodes[env_index].record_step(
                reward=reward,
                neutral_captures=_neutral_capture_count(previous_state, next_state, player=agent_player),
                alive=_player_alive(next_state, player=agent_player),
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

    for env_index, episode in enumerate(episodes):
        if active[env_index] and episode.length > 0:
            episode_metrics.append(episode.as_summary())

    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
        next_value = next_value.masked_fill(torch.as_tensor(~active, dtype=torch.bool, device=device), 0.0)

    observations = torch.stack(obs_buf)
    actions = torch.stack(action_buf).to(dtype=torch.long)
    logprobs = torch.stack(logprob_buf)
    rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
    values = torch.stack(value_buf)
    advantages, returns = _compute_gae_batched(
        rewards, dones, values, next_value, gamma=training_cfg.gamma, gae_lambda=training_cfg.gae_lambda
    )
    masks = torch.stack(mask_buf)
    # GridNet action/mask are per-planet (PLANET_N, 4) / (PLANET_N,): flatten only
    # the leading (steps, envs) dims, keep the per-planet structure.
    flat_observations = observations.reshape(-1, observations.shape[-1])
    flat_actions = actions.reshape(-1, PLANET_N, 4)
    flat_logprobs = logprobs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_values = values.reshape(-1)
    flat_rewards = rewards.reshape(-1)
    flat_masks = masks.reshape(-1, PLANET_N)
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
        observations=flat_observations, actions=flat_actions, logprobs=flat_logprobs,
        advantages=flat_advantages, returns=flat_returns, values=flat_values,
        rewards=flat_rewards, masks=flat_masks, opponent=opponent_name,
        episode_metrics=episode_metrics,
    )


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
    anchor_policies = (
        _bc_anchor_policies(training_cfg.bc_anchor_teacher, num_envs)
        if bc_anchor_enabled(training_cfg) and training_cfg.bc_anchor_teacher
        else []
    )
    decoder_cfg = decoder_config(training_cfg)
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
    (
        four_player_vulnerability_scale,
        four_player_leader_scale,
        four_player_third_player_scale,
    ) = four_player_shaping_scales(training_cfg, progress)
    reward_env = build_phase0_env(
        seed=base_seed,
        num_players=training_cfg.num_players,
        opponent_name=opponent_name,
        enable_comets=training_cfg.enable_comets,
        episode_steps=training_cfg.episode_steps,
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
        elimination_penalty=training_cfg.elimination_penalty,
    )
    backend = RustBatchBackend(
        num_envs=num_envs,
        num_players=training_cfg.num_players,
        seed=base_seed,
        config=RustConfig(
            episode_steps=training_cfg.episode_steps,
            enable_comets=training_cfg.enable_comets,
        ),
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
    teacher_action_buf = []
    teacher_mask_buf = []
    teacher_quant_error_buf = []
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
        teacher_actions_row = np.zeros((num_envs, 5), dtype=np.int64)
        teacher_masks_row = np.zeros(num_envs, dtype=np.bool_)
        teacher_quant_errors_row = np.full(num_envs, np.inf, dtype=np.float32)
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
            player_moves = decode_discrete_action(state, 0, actions_np[env_index], decoder_cfg)
            player_moves_by_env[env_index] = player_moves
            action_rows.extend(_moves_to_flat_rows(env_index, 0, player_moves))
            action_rows.extend(_moves_to_flat_rows(env_index, 1, opponent_moves_by_env[env_index]))
            if anchor_policies:
                teacher_action, teacher_mask, teacher_quant_error = _bc_anchor_action(
                    state,
                    0,
                    anchor_policies[env_index],
                    decoder_cfg,
                    training_cfg.bc_anchor_max_quant_error,
                )
                teacher_actions_row[env_index] = np.asarray(teacher_action, dtype=np.int64)
                teacher_masks_row[env_index] = bool(teacher_mask)
                teacher_quant_errors_row[env_index] = float(teacher_quant_error)

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
            reward, _reward_info = reward_env.transition_reward(
                previous_state,
                next_state,
                player=0,
                player_moves=player_moves_by_env[env_index],
                done=done,
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
        teacher_action_buf.append(torch.as_tensor(teacher_actions_row, dtype=torch.long, device=device))
        teacher_mask_buf.append(torch.as_tensor(teacher_masks_row, dtype=torch.bool, device=device))
        teacher_quant_error_buf.append(torch.as_tensor(teacher_quant_errors_row, dtype=torch.float32, device=device))

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
    teacher_actions = torch.stack(teacher_action_buf)
    teacher_action_mask = torch.stack(teacher_mask_buf)
    teacher_quant_errors = torch.stack(teacher_quant_error_buf)
    flat_observations = observations.reshape(-1, observations.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_logprobs = logprobs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_values = values.reshape(-1)
    flat_rewards = rewards.reshape(-1)
    flat_masks = masks.reshape(-1, masks.shape[-1])
    flat_teacher_actions = teacher_actions.reshape(-1, teacher_actions.shape[-1])
    flat_teacher_action_mask = teacher_action_mask.reshape(-1)
    flat_teacher_quant_errors = teacher_quant_errors.reshape(-1)
    if sample_limit > 0:
        flat_observations = flat_observations[:sample_limit]
        flat_actions = flat_actions[:sample_limit]
        flat_logprobs = flat_logprobs[:sample_limit]
        flat_advantages = flat_advantages[:sample_limit]
        flat_returns = flat_returns[:sample_limit]
        flat_values = flat_values[:sample_limit]
        flat_rewards = flat_rewards[:sample_limit]
        flat_masks = flat_masks[:sample_limit]
        flat_teacher_actions = flat_teacher_actions[:sample_limit]
        flat_teacher_action_mask = flat_teacher_action_mask[:sample_limit]
        flat_teacher_quant_errors = flat_teacher_quant_errors[:sample_limit]

    return RolloutSegment(
        observations=flat_observations,
        actions=flat_actions,
        logprobs=flat_logprobs,
        advantages=flat_advantages,
        returns=flat_returns,
        values=flat_values,
        rewards=flat_rewards,
        masks=flat_masks,
        teacher_actions=flat_teacher_actions,
        teacher_action_mask=flat_teacher_action_mask,
        teacher_quant_errors=flat_teacher_quant_errors,
        opponent=opponent_name,
        episode_metrics=episode_metrics,
    )


def _collect_rollout_segment(
    model: FlatActorCritic,
    *,
    opponent_name: str,
    base_seed: int,
    learner_player: int,
    rollout_steps: int,
    device: torch.device,
    training_cfg: Phase0TrainingConfig,
    progress: float,
    sample_limit: int | None = None,
    opponent_model: GridNetActorCritic | None = None,
) -> RolloutSegment:
    limit = rollout_steps if sample_limit is None else max(1, int(sample_limit))
    # GridNet is batchable (fixed per-planet mask shape): own self-play collector.
    if isinstance(model, GridNetActorCritic):
        per_env_steps = max(1, min(int(rollout_steps), math.ceil(limit / max(1, int(training_cfg.rollout_num_envs)))))
        return _collect_gridnet_rollout_segment(
            model,
            opponent_name=opponent_name,
            base_seed=base_seed,
            rollout_steps=per_env_steps,
            sample_limit=limit,
            device=device,
            training_cfg=training_cfg,
            progress=progress,
            opponent_model=opponent_model,
        )
    # BReP is batchable by design (per-slot edit mask is fixed-shape): route it to
    # its own batched collector regardless of _batched_rollout_supported.
    if isinstance(model, ProducerResidualBranchActorCritic):
        per_env_steps = max(1, min(int(rollout_steps), math.ceil(limit / max(1, int(training_cfg.rollout_num_envs)))))
        return _collect_brep_rollout_segment(
            model,
            opponent_name=opponent_name,
            base_seed=base_seed,
            rollout_steps=per_env_steps,
            sample_limit=limit,
            device=device,
            training_cfg=training_cfg,
            progress=progress,
        )
    if _batched_rollout_supported(training_cfg):
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
        learner_player=learner_player,
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
        "teacher_actions": torch.cat([segment.teacher_actions for segment in segments], dim=0),
        "teacher_action_mask": torch.cat([segment.teacher_action_mask for segment in segments], dim=0),
        "teacher_quant_errors": torch.cat([segment.teacher_quant_errors for segment in segments], dim=0),
    }


def _masked_head_logits(
    out: dict[str, torch.Tensor],
    key: str,
    masks: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    logits = out[key]
    if masks is not None and key in masks:
        logits = logits.masked_fill(~masks[key].bool(), float("-inf"))
    return logits


def _teacher_anchor_loss(
    out: dict[str, torch.Tensor],
    teacher_actions: torch.Tensor,
    masks: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    launch_label = teacher_actions[:, 0]
    loss = F.cross_entropy(_masked_head_logits(out, "launch", masks), launch_label)
    parts = {"launch": float(loss.detach())}
    active = launch_label == 1
    if bool(active.any()):
        for idx, key in enumerate(("source", "target", "frac", "offset"), start=1):
            head_loss = F.cross_entropy(
                _masked_head_logits(out, key, masks)[active],
                teacher_actions[active, idx],
            )
            loss = loss + head_loss
            parts[key] = float(head_loss.detach())
    return loss, parts


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
        "bc_anchor_loss": 0.0,
        "bc_anchor_examples": 0.0,
        "bc_anchor_launch_rate": 0.0,
        "bc_anchor_mean_quant_error": 0.0,
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
            is_residual = isinstance(model, ProducerResidualBranchActorCritic)
            is_gridnet = isinstance(model, GridNetActorCritic)
            is_branch = is_residual or is_gridnet  # own get_action_and_value, no 5-head KL
            if is_residual:
                mask_mb = {"edit": batch["masks"][indices]}
            elif is_gridnet:
                mask_mb = {"planet": batch["masks"][indices]}
            else:
                mask_mb = split_masks(batch["masks"][indices])

            advantage_mb = (advantage_mb - advantage_mb.mean()) / (advantage_mb.std(unbiased=False) + 1e-8)

            # Same mask as sampling time -> the PPO importance ratio stays correct.
            # Branch archs (residual edit-branch, GridNet per-planet) have no
            # launch/move heads, so they compute terms via get_action_and_value; the
            # 5-head archs reuse one forward for both PPO terms and the KL anchor.
            if is_branch:
                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    obs_mb, action_mb, masks=mask_mb
                )
            else:
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
            bc_anchor_loss_val = 0.0
            bc_anchor_examples_val = 0.0
            bc_anchor_launch_rate_val = 0.0
            bc_anchor_quant_error_val = 0.0
            if float(cfg.bc_anchor_coef) > 0.0 and bool(teacher_mask_mb.any()):
                anchor_actions = teacher_action_mb[teacher_mask_mb]
                anchor_loss, _anchor_parts = _teacher_anchor_loss(
                    {key: value[teacher_mask_mb] for key, value in cur_out.items()},
                    anchor_actions,
                    {key: value[teacher_mask_mb] for key, value in mask_mb.items()},
                )
                loss = loss + float(cfg.bc_anchor_coef) * anchor_loss
                bc_anchor_loss_val = float(anchor_loss.item())
                bc_anchor_examples_val = float(anchor_actions.shape[0])
                bc_anchor_launch_rate_val = float((anchor_actions[:, 0] == 1).float().mean().item())
                finite_quant = teacher_quant_mb[teacher_mask_mb]
                finite_quant = finite_quant[torch.isfinite(finite_quant)]
                bc_anchor_quant_error_val = float(finite_quant.mean().item()) if finite_quant.numel() else 0.0

            kl_to_ref_val = 0.0
            # KL-to-ref anchors the policy to a reference (BC) so RL can't degrade
            # the good BC minimum. 5-head archs use launch_gated_kl; GridNet uses
            # its per-planet gated KL; the residual edit-branch has no anchor.
            if use_kl_ref and is_gridnet:
                cur_out = model.forward(obs_mb)
                with torch.no_grad():
                    ref_out = ref_model.forward(obs_mb)
                kl_ref = gridnet_gated_kl(cur_out, ref_out, batch["masks"][indices]).mean()
                loss = loss + cfg.kl_to_ref_coef * kl_ref
                kl_to_ref_val = float(kl_ref.item())
            elif use_kl_ref and not is_branch:
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
            stats["bc_anchor_loss"] += bc_anchor_loss_val
            stats["bc_anchor_examples"] += bc_anchor_examples_val
            stats["bc_anchor_launch_rate"] += bc_anchor_launch_rate_val
            stats["bc_anchor_mean_quant_error"] += bc_anchor_quant_error_val
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
    """In-loop margin proxy vs an opponent, for keep-best / early-stop gating.

    NOT the promotion gate (that stays the separate both-seat 96-seed benchmark) —
    a cheap, consistent relative signal so the loop keeps the best checkpoint and
    early-stops on drift. With learner seat rotation enabled, the proxy averages
    all player seats. Greedy actions; final normalized score margin in [-1, 1].
    """
    was_training = model.training
    model.eval()
    margins: list[float] = []
    for seed in range(int(seeds)):
        seats = range(training_cfg.num_players) if training_cfg.learner_seat_rotation else range(1)
        for learner_player in seats:
            env = build_phase0_env(
                seed=seed,
                num_players=training_cfg.num_players,
                learner_player=int(learner_player),
                opponent_name=opponent_name,
                enable_comets=training_cfg.enable_comets,
                episode_steps=training_cfg.episode_steps,
                decoder_cfg=decoder_config(training_cfg),
                base_shaping_scale=0.0,
            )
            obs_np, _ = env.reset(seed=seed)
            last_scores: list[float] = [0.0 for _ in range(training_cfg.num_players)]
            for _ in range(int(training_cfg.eval_max_steps)):
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                masks = split_masks(
                    torch.as_tensor(
                        build_action_masks(
                            env.state,
                            int(learner_player),
                            min_ships_to_launch=training_cfg.decoder_min_ships_to_launch,
                        ),
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
            margins.append(normalized_margin(last_scores, int(learner_player)) if last_scores else 0.0)
    if was_training:
        model.train()
    return float(np.mean(margins)) if margins else 0.0


def _scores_margin(scores: Sequence[float], agent_player: int, num_players: int) -> float:
    """Normalized margin of ``agent_player`` vs the best other score, in [-1, 1]."""
    s = [float(x) for x in scores] + [0.0] * (num_players - len(scores))
    mine = s[agent_player]
    others = [s[p] for p in range(num_players) if p != agent_player]
    best_other = max(others) if others else 0.0
    denom = max(abs(mine) + abs(best_other), 1.0)
    return (mine - best_other) / denom


def _play_residual_game(
    model: ProducerResidualBranchActorCritic,
    *,
    base_agent: str,
    opponent_name: str,
    agent_player: int,
    seed: int,
    episode_steps: int,
    num_players: int,
    enable_comets: bool,
    device: torch.device,
) -> float:
    """One game: agent = base plan edited by the net's greedy argmax; others = opponent.
    Returns the agent's normalized score margin."""
    k_max = int(model.k_max)
    backend = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets),
    )
    state = backend.reset(seed)[0]
    base = make_isolated_opponent(_BASE_AGENT_ALIASES.get(base_agent, base_agent))
    opponents = {p: make_isolated_opponent(opponent_name) for p in range(num_players) if p != agent_player}
    last_scores: list[float] = [0.0] * num_players
    for _ in range(episode_steps):
        base_moves = [list(m) for m in base(state, agent_player)]
        n_slots = min(len(base_moves), k_max)
        obs_t = torch.as_tensor(backend.encoded_states(agent_player), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.forward(obs_t)["edit"]
        argmax = logits.argmax(-1)[0].cpu().numpy()
        edits = [int(argmax[i]) if i < n_slots else 0 for i in range(len(base_moves))]
        agent_moves = _apply_residual_edits(state, base_moves, edits, k_max)
        rows: list[list[float]] = list(_moves_to_flat_rows(0, agent_player, agent_moves))
        for p, opp in opponents.items():
            rows.extend(_moves_to_flat_rows(0, p, opp(state, p)))
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        outcomes, states = backend.step_flat_with_states(flat)
        state = states[0]
        sc = outcomes[0].get("scores") or outcomes[0].get("rewards")
        if sc:
            last_scores = list(sc)
        if bool(outcomes[0].get("done", False)):
            break
    return _scores_margin(last_scores, agent_player, num_players)


def evaluate_residual_margin(
    model: ProducerResidualBranchActorCritic,
    *,
    base_agent: str,
    opponent_name: str,
    seeds: int,
    episode_steps: int,
    num_players: int = 2,
    device: torch.device | str = "cpu",
    enable_comets: bool = True,
) -> dict[str, float]:
    """In-process MIRRORED-seat paired margin of a residual policy vs an opponent.

    The BReP submission cannot be exported through ``render_submission`` (5-head
    only), so the per-chunk gate evaluates the checkpoint DIRECTLY. Self-play of the
    base agent has a strong seat bias + huge variance (holdwave-vs-holdwave swings
    to ±1.0 annihilation), so seed-parity seat alternation does NOT cancel it. We
    instead play EACH seed in BOTH seats of the SAME world and average: for identical
    self-play the two mirrored margins are exactly opposite → 0 (true parity floor),
    so any positive residual margin is a genuine learned edge, not seat luck.
    Returns a summary shaped like benchmark_exported_checkpoint (2-player only)."""
    if num_players != 2:
        raise ValueError("mirrored-seat residual eval is 2-player only")
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    margins: list[float] = []
    wins: list[float] = []
    for seed in range(int(seeds)):
        m0 = _play_residual_game(
            model, base_agent=base_agent, opponent_name=opponent_name, agent_player=0,
            seed=seed, episode_steps=episode_steps, num_players=2,
            enable_comets=enable_comets, device=dev,
        )
        m1 = _play_residual_game(
            model, base_agent=base_agent, opponent_name=opponent_name, agent_player=1,
            seed=seed, episode_steps=episode_steps, num_players=2,
            enable_comets=enable_comets, device=dev,
        )
        m = 0.5 * (m0 + m1)  # mirrored: seat bias cancels
        margins.append(m)
        wins.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))
    if was_training:
        model.train()
    return {
        "games": float(2 * len(margins)),
        "mean_score_margin": float(np.mean(margins)) if margins else 0.0,
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "invalid_action_rate": 0.0,
    }


def _play_gridnet_game(
    model: GridNetActorCritic,
    *,
    opponent_name: str,
    agent_player: int,
    seed: int,
    episode_steps: int,
    num_players: int,
    enable_comets: bool,
    device: torch.device,
    decoder_cfg: DecoderConfig,
) -> float:
    """One game: agent = GridNet greedy per-planet; others = opponent. Returns the
    agent's normalized score margin. Tracks invalid moves defensively (should be 0
    by construction of the per-planet mask + decode)."""
    backend = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets),
    )
    state = backend.reset(seed)[0]
    base_opp_name, handicap = opponent_name, 1.0
    if "@" in opponent_name:
        base_opp_name, _, scale_s = opponent_name.partition("@")
        handicap = float(scale_s)
    opponents = {p: make_isolated_opponent(base_opp_name) for p in range(num_players) if p != agent_player}
    last_scores: list[float] = [0.0] * num_players
    for _ in range(episode_steps):
        mask = torch.as_tensor(
            gridnet_planet_mask(state, agent_player, decoder_cfg), dtype=torch.bool, device=device
        ).unsqueeze(0)
        obs = torch.as_tensor(
            backend.encoded_states(agent_player), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            out = model.forward(obs)
            launch = torch.where(mask, out["launch"].argmax(-1), torch.zeros_like(out["launch"].argmax(-1)))
            a = torch.stack([launch, out["target"].argmax(-1), out["frac"].argmax(-1), out["offset"].argmax(-1)], dim=-1)
        agent_moves = decode_gridnet_action(state, agent_player, a[0].cpu().numpy(), decoder_cfg)
        rows: list[list[float]] = list(_moves_to_flat_rows(0, agent_player, agent_moves))
        for p, opp in opponents.items():
            om = opp(state, p)
            if handicap < 1.0:
                om = [[mv[0], mv[1], max(1.0, float(mv[2]) * handicap)] for mv in om]
            rows.extend(_moves_to_flat_rows(0, p, om))
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        outcomes, states = backend.step_flat_with_states(flat)
        state = states[0]
        sc = outcomes[0].get("scores") or outcomes[0].get("rewards")
        if sc:
            last_scores = list(sc)
        if bool(outcomes[0].get("done", False)):
            break
    return _scores_margin(last_scores, agent_player, num_players)


def evaluate_gridnet_margin(
    model: GridNetActorCritic,
    *,
    opponent_name: str,
    seeds: int,
    episode_steps: int,
    num_players: int = 2,
    device: torch.device | str = "cpu",
    enable_comets: bool = True,
    decoder_cfg: DecoderConfig | None = None,
) -> dict[str, float]:
    """Mirrored-seat paired margin of a GridNet policy vs an opponent (2p).

    Same mirrored-world protocol as evaluate_residual_margin (each seed played in
    BOTH seats, averaged) so the opponent's seat bias cancels and the margin is an
    honest signal vs e.g. holdwave."""
    if num_players != 2:
        raise ValueError("mirrored-seat GridNet eval is 2-player only")
    cfg = decoder_cfg or DEFAULT_DECODER_CONFIG
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    margins: list[float] = []
    wins: list[float] = []
    for seed in range(int(seeds)):
        m0 = _play_gridnet_game(model, opponent_name=opponent_name, agent_player=0, seed=seed,
                                episode_steps=episode_steps, num_players=2, enable_comets=enable_comets,
                                device=dev, decoder_cfg=cfg)
        m1 = _play_gridnet_game(model, opponent_name=opponent_name, agent_player=1, seed=seed,
                                episode_steps=episode_steps, num_players=2, enable_comets=enable_comets,
                                device=dev, decoder_cfg=cfg)
        m = 0.5 * (m0 + m1)
        margins.append(m)
        wins.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))
    if was_training:
        model.train()
    return {
        "games": float(2 * len(margins)),
        "mean_score_margin": float(np.mean(margins)) if margins else 0.0,
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "invalid_action_rate": 0.0,
    }


def train_phase0(training_cfg: Phase0TrainingConfig) -> dict[str, Any]:
    started_at = time.perf_counter()
    _set_seed(training_cfg.seed)
    device = torch.device(training_cfg.device)
    opponents = _parse_opponents(training_cfg.opponents)
    if bc_anchor_enabled(training_cfg) and not training_cfg.bc_anchor_teacher:
        raise ValueError("bc_anchor_teacher is required when bc_anchor_coef > 0")

    # Producer/OEP keep per-game memory in a module-level singleton. The batched
    # rollout now gives each env its OWN isolated opponent instance (see
    # get_isolated_opponents), so concurrent games no longer cross-contaminate and
    # batched/vectorized rollout is safe with stateful opponents.

    # When warm-starting, adopt the checkpoint's architecture (e.g. an entity-BC
    # init) so the state_dict loads cleanly; otherwise use the configured arch.
    checkpoint = _load_checkpoint(training_cfg.checkpoint_in, device) if training_cfg.checkpoint_in else None
    arch_kwargs: dict[str, Any] = {}
    if checkpoint is not None:
        ckpt_summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
        policy_arch = str(ckpt_summary.get("arch", training_cfg.policy_arch))
        # GridNet may be a non-default size (e.g. 512/128); adopt the checkpoint's
        # dims so the state_dict loads cleanly.
        if policy_arch == "gridnet":
            for src, dst in (("hidden", "hidden"), ("entity_hidden", "entity_hidden")):
                if src in ckpt_summary:
                    arch_kwargs[dst] = int(ckpt_summary[src])
    else:
        policy_arch = training_cfg.policy_arch

    model = _build_policy(policy_arch, observation_dim(), **arch_kwargs).to(device)
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
    learner_seat_segments = {str(player): 0 for player in range(int(training_cfg.num_players))}
    latest_update_stats: dict[str, float] = {}
    update_series: list[dict[str, float]] = []

    # Movement 2: frozen reference policy for the KL anti-drift anchor (defaults to
    # the warm-start / BC checkpoint).
    ref_model: nn.Module | None = None
    ref_path = training_cfg.ref_checkpoint or training_cfg.checkpoint_in
    if training_cfg.kl_to_ref_coef > 0.0 and ref_path:
        ref_ckpt = _load_checkpoint(ref_path, device)
        ref_sum = ref_ckpt.get("summary") or {}
        ref_arch = str(ref_sum.get("arch", policy_arch))
        ref_kwargs = {k: int(ref_sum[k]) for k in ("hidden", "entity_hidden") if ref_arch == "gridnet" and k in ref_sum}
        ref_model = _build_policy(ref_arch, observation_dim(), **ref_kwargs).to(device)
        ref_model.load_state_dict(ref_ckpt["model_state_dict"])
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    # GridNet league: frozen snapshot that plays the "self" opponent seat.
    self_opponent_model: GridNetActorCritic | None = None
    if training_cfg.self_opponent_checkpoint:
        snap = _load_checkpoint(training_cfg.self_opponent_checkpoint, device)
        self_opponent_model = _build_policy("gridnet", observation_dim()).to(device)
        self_opponent_model.load_state_dict(snap["model_state_dict"])
        self_opponent_model.eval()
        for param in self_opponent_model.parameters():
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
            learner_player = (
                (update_idx * len(opponents) + opponent_idx) % int(training_cfg.num_players)
                if training_cfg.learner_seat_rotation
                else 0
            )
            progress = total_timesteps / max(float(training_cfg.total_timesteps), 1.0)
            segment = _collect_rollout_segment(
                model,
                opponent_name=opponent_name,
                base_seed=seed,
                learner_player=learner_player,
                rollout_steps=segment_steps,
                device=device,
                training_cfg=training_cfg,
                progress=progress,
                sample_limit=remaining if _batched_rollout_supported(training_cfg) else segment_steps,
                opponent_model=self_opponent_model,
            )
            segments.append(segment)
            opponent_segments[opponent_name] += 1
            learner_seat_segments[str(learner_player)] += 1
            total_timesteps += int(segment.observations.shape[0])
            all_episode_metrics.extend(segment.episode_metrics)
        batch = _concat_segments(segments)
        update_progress = total_timesteps / max(float(training_cfg.total_timesteps), 1.0)
        effective_anchor_coef = bc_anchor_coef(training_cfg, update_progress)
        update_cfg = replace(training_cfg, bc_anchor_coef=effective_anchor_coef)
        latest_update_stats = _ppo_update(model, optimizer, batch, update_cfg, ref_model=ref_model)
        latest_update_stats["bc_anchor_effective_coef"] = float(effective_anchor_coef)
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
        # preserve non-default GridNet dims so the saved checkpoint reloads cleanly
        **({"hidden": arch_kwargs["hidden"]} if "hidden" in arch_kwargs else {}),
        **({"entity_hidden": arch_kwargs["entity_hidden"]} if "entity_hidden" in arch_kwargs else {}),
        "policy_track": training_cfg.policy_track,
        "num_players": training_cfg.num_players,
        "timesteps": total_timesteps,
        "episode_steps": training_cfg.episode_steps,
        "updates": update_idx,
        "rollout_num_envs": int(training_cfg.rollout_num_envs),
        "learner_seat_rotation": bool(training_cfg.learner_seat_rotation),
        "learner_seat_segments": learner_seat_segments,
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
        "bc_anchor_coef": float(training_cfg.bc_anchor_coef),
        "bc_anchor_coef_end": (
            float(training_cfg.bc_anchor_coef_end)
            if training_cfg.bc_anchor_coef_end is not None
            else float(training_cfg.bc_anchor_coef)
        ),
        "bc_anchor_teacher": training_cfg.bc_anchor_teacher if bc_anchor_enabled(training_cfg) else None,
        "bc_anchor_max_quant_error": (
            float(training_cfg.bc_anchor_max_quant_error)
            if math.isfinite(float(training_cfg.bc_anchor_max_quant_error))
            else None
        ),
        "comet_shaping_scale_start": training_cfg.comet_shaping_scale_start,
        "comet_shaping_scale_end": training_cfg.comet_shaping_scale_end,
        "four_player_vulnerability_scale_start": training_cfg.four_player_vulnerability_scale_start,
        "four_player_vulnerability_scale_end": training_cfg.four_player_vulnerability_scale_end,
        "four_player_leader_scale_start": training_cfg.four_player_leader_scale_start,
        "four_player_leader_scale_end": training_cfg.four_player_leader_scale_end,
        "four_player_third_player_scale_start": training_cfg.four_player_third_player_scale_start,
        "four_player_third_player_scale_end": training_cfg.four_player_third_player_scale_end,
        "elimination_penalty": training_cfg.elimination_penalty,
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
    learner_player: int = 0,
    episode_steps: int = 500,
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
    elimination_penalty: float = 0.0,
    decoder_cfg: DecoderConfig | None = None,
) -> OrbitWarsGymEnv:
    opponent_policy = _training_opponent_policy(opponent_name, num_players, learner_player=learner_player)
    rust_cfg = RustConfig(episode_steps=int(episode_steps), enable_comets=enable_comets)
    return OrbitWarsGymEnv(
        num_players=num_players,
        seed=seed,
        rust_cfg=rust_cfg,
        opponent_policy=opponent_policy,
        learner_player=learner_player,
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
        elimination_penalty=elimination_penalty,
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
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=500,
        help="training episode horizon; use shorter horizons for curriculum probes with frequent terminal rewards",
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollout-num-envs", type=int, default=1)
    parser.add_argument(
        "--learner-seat-rotation",
        dest="learner_seat_rotation",
        action="store_true",
        default=None,
        help="rotate the learner across seats in single-env rollouts",
    )
    parser.add_argument(
        "--no-learner-seat-rotation",
        dest="learner_seat_rotation",
        action="store_false",
        help="force the learner to seat 0 even on tracks whose default rotates seats",
    )
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
    parser.add_argument("--elimination-penalty", type=float, default=0.0)
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
    parser.add_argument("--bc-anchor-coef", type=float, default=0.0,
                        help="optional residual-style BC anchor coefficient on teacher actions (0=off)")
    parser.add_argument("--bc-anchor-coef-end", type=float, default=None,
                        help="optional final BC anchor coefficient for linear decay/anneal")
    parser.add_argument("--bc-anchor-teacher", default=None,
                        help="single teacher for the BC anchor, e.g. brep, pgs_holdwave, producer")
    parser.add_argument("--bc-anchor-max-quant-error", type=float, default=float("inf"),
                        help="drop teacher labels whose inverse-projection error exceeds this value")
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
        episode_steps=args.episode_steps,
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
        elimination_penalty=args.elimination_penalty,
        decoder_cfg=DecoderConfig(
            max_moves_per_turn=args.decoder_max_moves_per_turn,
            min_ships_to_launch=args.decoder_min_ships_to_launch,
            reserve_home_ships=args.decoder_reserve_home_ships,
        ),
    )
    obs, _ = env.reset(seed=args.seed)
    cfg_kwargs = dict(
        seed=args.seed,
        policy_track=args.training_track,
        policy_arch=args.policy_arch,
        num_players=args.num_players,
        episode_steps=args.episode_steps,
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
        elimination_penalty=args.elimination_penalty,
        decoder_max_moves_per_turn=args.decoder_max_moves_per_turn,
        decoder_min_ships_to_launch=args.decoder_min_ships_to_launch,
        decoder_reserve_home_ships=args.decoder_reserve_home_ships,
        inherit_checkpoint_decoder=bool(args.inherit_checkpoint_decoder),
        shaping_potential=args.shaping_potential,
        kl_to_ref_coef=args.kl_to_ref_coef,
        ref_checkpoint=args.ref_checkpoint,
        bc_anchor_coef=args.bc_anchor_coef,
        bc_anchor_coef_end=args.bc_anchor_coef_end,
        bc_anchor_teacher=args.bc_anchor_teacher,
        bc_anchor_max_quant_error=args.bc_anchor_max_quant_error,
        eval_every_updates=args.eval_every_updates,
        eval_seeds=args.eval_seeds,
        eval_opponent=args.eval_opponent,
        eval_max_steps=args.eval_max_steps,
        early_stop_patience=args.early_stop_patience,
    )
    if args.learner_seat_rotation is not None:
        cfg_kwargs["learner_seat_rotation"] = bool(args.learner_seat_rotation)
    base_cfg = (
        build_phase5_4p_config(**cfg_kwargs)
        if args.training_track == "phase5_4p"
        else Phase0TrainingConfig(**cfg_kwargs)
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
