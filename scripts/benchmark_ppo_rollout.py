from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.agents.policy import FlatActorCritic  # noqa: E402
from python.train.train_ppo import build_phase0_env  # noqa: E402

from orbit_wars_gym.backend import RustBatchBackend, RustConfig  # noqa: E402
from orbit_wars_gym.encoding import EncoderConfig, observation_dim  # noqa: E402


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_model(device: torch.device, encoder_cfg: EncoderConfig) -> FlatActorCritic:
    model = FlatActorCritic(observation_dim(encoder_cfg)).to(device)
    model.eval()
    return model


def _load_checkpoint(model: FlatActorCritic, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)


def _timed_current_single_env(
    *,
    model: FlatActorCritic,
    device: torch.device,
    rollout_steps: int,
    warmup_steps: int,
    seed: int,
    opponent: str,
    num_players: int,
    enable_comets: bool,
) -> dict[str, Any]:
    env = build_phase0_env(
        seed=seed,
        num_players=num_players,
        opponent_name=opponent,
        enable_comets=enable_comets,
    )
    obs_np, _ = env.reset(seed=seed)

    def step_once(obs: np.ndarray, step_seed: int) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _, _, _ = model.get_action_and_value(obs_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()
        next_obs, _, done, _, _ = env.step(action)
        if done:
            next_obs, _ = env.reset(seed=seed + step_seed + 1)
        return next_obs

    for idx in range(warmup_steps):
        obs_np = step_once(obs_np, idx)

    _sync(device)
    started = time.perf_counter()
    for idx in range(rollout_steps):
        obs_np = step_once(obs_np, warmup_steps + idx)
    _sync(device)
    elapsed = time.perf_counter() - started

    return {
        "mode": "current_single_env_gym",
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "env_steps": rollout_steps,
        "model_forwards": rollout_steps,
        "seconds": elapsed,
        "env_steps_per_sec": rollout_steps / elapsed if elapsed > 0 else float("inf"),
        "model_forwards_per_sec": rollout_steps / elapsed if elapsed > 0 else float("inf"),
        "notes": [
            "Matches the current PPO collector shape: batch=1, action_tensor.cpu().numpy() every step.",
            "Includes Python opponent policy, full-state step_with_states, Python encode_state, and reward shaping.",
        ],
    }


def _timed_batched_fast_path(
    *,
    model: FlatActorCritic,
    device: torch.device,
    rollout_steps: int,
    warmup_steps: int,
    seed: int,
    num_envs: int,
    num_players: int,
    enable_comets: bool,
    encoder_cfg: EncoderConfig,
    sync_actions: bool,
) -> dict[str, Any]:
    backend = RustBatchBackend(
        num_envs=num_envs,
        num_players=num_players,
        seed=seed,
        config=RustConfig(enable_comets=enable_comets),
    )
    backend.reset(seed)
    obs_np = backend.encoded_states(
        0,
        max_planets=encoder_cfg.max_planets,
        max_fleets=encoder_cfg.max_fleets,
        include_fleets=encoder_cfg.include_fleets,
    )
    empty_actions = np.zeros((0, 5), dtype=np.float64)

    def step_once(obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor, _, _, _ = model.get_action_and_value(obs_tensor)
        if sync_actions:
            _ = action_tensor.cpu().numpy()
        elif device.type == "cuda":
            # Include queued GPU inference while still measuring the best-case
            # no-action-copy fast path separately from the real decoder cost.
            torch.cuda.synchronize(device)
        _, next_obs = backend.step_flat_with_encoded_states(
            empty_actions,
            0,
            max_planets=encoder_cfg.max_planets,
            max_fleets=encoder_cfg.max_fleets,
            include_fleets=encoder_cfg.include_fleets,
        )
        return next_obs

    for _ in range(warmup_steps):
        obs_np = step_once(obs_np)

    _sync(device)
    started = time.perf_counter()
    for _ in range(rollout_steps):
        obs_np = step_once(obs_np)
    _sync(device)
    elapsed = time.perf_counter() - started
    env_steps = rollout_steps * num_envs

    return {
        "mode": "batched_rust_encoded_noop_actions",
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "env_steps": env_steps,
        "model_forwards": rollout_steps,
        "seconds": elapsed,
        "env_steps_per_sec": env_steps / elapsed if elapsed > 0 else float("inf"),
        "model_forwards_per_sec": rollout_steps / elapsed if elapsed > 0 else float("inf"),
        "sync_actions_to_cpu": sync_actions,
        "uses_step_flat_with_encoded_states": True,
        "notes": [
            "Measures the intended high-throughput training API with obs batch shape (N, obs_dim).",
            "Uses empty flat actions, so it excludes discrete-action decoding and Python opponent policies.",
            "This is an upper-bound backend+policy throughput benchmark, not a drop-in PPO collector yet.",
        ],
    }


def _speedup(batched: dict[str, Any], current: dict[str, Any]) -> float:
    base = float(current["env_steps_per_sec"])
    if base <= 0:
        return float("inf")
    return float(batched["env_steps_per_sec"]) / base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark current PPO rollout throughput against the batched Rust encoded-state API."
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--opponent", default="greedy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--enable-comets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-planets", type=int, default=96)
    parser.add_argument("--max-fleets", type=int, default=256)
    parser.add_argument("--include-fleets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")

    device = _device(args.device)
    encoder_cfg = EncoderConfig(
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        include_fleets=args.include_fleets,
    )
    model = _build_model(device, encoder_cfg)
    _load_checkpoint(model, args.checkpoint_in, device)

    current = _timed_current_single_env(
        model=model,
        device=device,
        rollout_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        opponent=args.opponent,
        num_players=args.num_players,
        enable_comets=args.enable_comets,
    )
    batched = _timed_batched_fast_path(
        model=model,
        device=device,
        rollout_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        num_envs=args.num_envs,
        num_players=args.num_players,
        enable_comets=args.enable_comets,
        encoder_cfg=encoder_cfg,
        sync_actions=args.sync_actions,
    )
    report = {
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_in": args.checkpoint_in,
        "current": current,
        "batched": batched,
        "batched_vs_current_env_steps_speedup": _speedup(batched, current),
        "limitations": [
            "The batched path does not yet decode PPO actions into legal moves.",
            "The batched path does not run Python opponent policies; those must be vectorized or moved to Rust before replacing train_ppo.py.",
            "Use this script as a throughput baseline before changing the collector, not as a training-quality metric.",
        ],
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
