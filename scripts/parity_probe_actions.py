"""Action-driven step-by-step parity probe: Rust simulator vs official Kaggle env.

The passive parity probe (``scripts.parity_probe``) only advances with empty
actions, so it never exercises fleet launch, movement, swept collision, combat,
capture or reinforcement. This probe issues IDENTICAL pseudo-random legal
launches to both the official ``kaggle_environments`` Orbit Wars interpreter and
the local Rust ``RustBatchBackend`` (loaded from the official initial state) and
diffs the full state every step.

Comet spawns use a hidden seed and are not reproducible, so windows must not
cross a spawn step (see ``scripts.parity_probe``). Run within [50k+1, 50(k+1)-1]
windows, optionally with comets enabled to exercise fleet-vs-comet collisions.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Any

from kaggle_environments import make
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.snapshots import snapshot_to_state  # noqa: F401  (kept for parity API)
from scripts.parity_probe import (
    COMET_SPAWN_STEPS,
    _compare_state,
    _state_from_official_observation,
)


def _legal_moves(observation: Any, player: int, rng: random.Random, launch_prob: float) -> list:
    moves = []
    for planet in observation.planets:
        pid, owner, _x, _y, _r, ships, _prod = planet
        if owner == player and int(ships) >= 2 and rng.random() < launch_prob:
            count = max(1, int(ships) // 2)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            moves.append([int(pid), float(angle), int(count)])
    return moves


def run(
    *,
    num_players: int,
    episodes: int,
    start_step: int,
    steps: int,
    enable_comets: bool,
    launch_prob: float,
    atol: float,
) -> dict[str, Any]:
    final_step = start_step + steps
    crossed = [s for s in COMET_SPAWN_STEPS if start_step < s <= final_step]
    if crossed:
        raise ValueError(f"window crosses hidden-seed comet spawns {crossed}")
    episode_steps = final_step + 2
    checked = 0
    failures: list[dict[str, Any]] = []
    for seed in range(episodes):
        env = make(
            "orbit_wars",
            configuration={"seed": seed, "episodeSteps": episode_steps},
            debug=True,
        )
        env.reset(num_players)
        obs = env.state[0].observation
        empty = [[] for _ in range(num_players)]
        while not env.done and int(obs.step) < start_step:
            env.step(empty)
            obs = env.state[0].observation

        rust = RustBatchBackend(
            num_envs=1,
            num_players=num_players,
            seed=seed,
            config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets),
        )
        rust.reset_from_states([_state_from_official_observation(obs, num_players=num_players)])

        rng = random.Random(seed * 7919 + num_players)
        while not env.done and int(obs.step) < final_step:
            actions = [_legal_moves(obs, p, rng, launch_prob) for p in range(num_players)]
            env.step(actions)
            _, rust_states = rust.step_with_states([actions])
            obs = env.state[0].observation
            checked += 1
            diff = _compare_state(
                official_observation=obs,
                rust_state=rust_states[0],
                seed=seed,
                atol=atol,
            )
            if diff is not None:
                failures.append({"seed": seed, "step": int(obs.step), "diff": diff})
                break
    return {
        "passed": not failures,
        "num_players": num_players,
        "episodes": episodes,
        "window": [start_step, final_step],
        "enable_comets": enable_comets,
        "checked_steps": checked,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--num-players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--enable-comets", action="store_true")
    parser.add_argument("--launch-prob", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    report = run(
        num_players=args.num_players,
        episodes=args.episodes,
        start_step=args.start_step,
        steps=args.steps,
        enable_comets=args.enable_comets,
        launch_prob=args.launch_prob,
        atol=args.atol,
    )
    for failure in report["failures"]:
        print(f"  DIVERGENCE {report['num_players']}p {failure['diff']}")
    print(
        f"{report['num_players']}p window={report['window']} comets={report['enable_comets']} "
        f"checked={report['checked_steps']} -> {'PASS' if report['passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
