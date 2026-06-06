from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from scripts.benchmark_submission import (
    _load_submission_agent,
    _resolve_opponent,
    _win_points,
)


def _move_signature(moves: list[list[float]]) -> list[tuple[int, int, int, float]]:
    sig: list[tuple[int, int, int, float]] = []
    for move in moves:
        if len(move) < 4:
            continue
        sig.append(
            (
                int(move[0]),
                int(move[1]),
                int(round(float(move[2]))),
                round(float(move[3]), 3),
            )
        )
    return sorted(sig)


def _run_game(
    *,
    submission_path: Path,
    opponent_spec: str,
    seed: int,
    submission_first: bool,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> dict[str, Any]:
    submission_agent = _load_submission_agent(submission_path)
    _, opponent_policy = _resolve_opponent(opponent_spec)
    players = [submission_agent, opponent_policy] if submission_first else [opponent_policy, submission_agent]
    submission_idx = 0 if submission_first else 1
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(
            episode_steps=int(episode_steps),
            enable_comets=bool(enable_comets),
            act_timeout=float(act_timeout),
        ),
    )
    state = backend.reset(seed)[0]
    actions_trace: list[dict[str, Any]] = []
    decision_turns = 0.0
    elapsed_seconds = 0.0
    crashes = 0.0
    timeouts = 0.0
    invalid_actions = 0.0
    outcome = {"scores": [0.0, 0.0], "done": False}

    while True:
        actions: list[list[list[float]]] = [[], []]
        for player_idx, policy in enumerate(players):
            obs = to_official_observation(state, player=player_idx)
            try:
                start = perf_counter()
                moves = policy(obs) if player_idx == submission_idx else policy(state, player_idx)
                elapsed = perf_counter() - start
                if not isinstance(moves, list):
                    moves = []
            except Exception:
                if player_idx == submission_idx:
                    crashes += 1.0
                moves = []
                elapsed = 0.0

            if player_idx == submission_idx:
                decision_turns += 1.0
                elapsed_seconds += elapsed
                if elapsed > float(act_timeout):
                    timeouts += 1.0
                    moves = []
                if not moves_are_legal(state, player_idx, moves):
                    invalid_actions += 1.0
                    moves = []
                actions_trace.append(
                    {
                        "step": int(state["step"]),
                        "signature": _move_signature(moves),
                    }
                )
            elif not moves_are_legal(state, player_idx, moves):
                moves = []
            actions[player_idx] = moves

        outcomes, states = backend.step_with_states([actions])
        outcome = outcomes[0]
        state = states[0]
        if outcome["done"]:
            break

    scores = [float(score) for score in outcome["scores"]]
    return {
        "seed": int(seed),
        "submission_player": int(submission_idx),
        "win_points": _win_points(scores, submission_idx),
        "normalized_margin": normalized_margin(scores, submission_idx),
        "decision_turns": decision_turns,
        "elapsed_seconds": elapsed_seconds,
        "mean_decision_ms": 1000.0 * elapsed_seconds / max(1.0, decision_turns),
        "crashes": crashes,
        "timeouts": timeouts,
        "invalid_actions": invalid_actions,
        "actions": actions_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace submission action signatures vs an opponent.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--opponent", default="producer")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    games = [
        _run_game(
            submission_path=args.submission,
            opponent_spec=str(args.opponent),
            seed=seed,
            submission_first=submission_first,
            episode_steps=int(args.episode_steps),
            enable_comets=not bool(args.disable_comets),
            act_timeout=float(args.act_timeout),
        )
        for seed in range(max(1, int(args.seeds)))
        for submission_first in (True, False)
    ]
    decisions = sum(float(game["decision_turns"]) for game in games)
    report = {
        "submission": str(args.submission),
        "opponent": str(args.opponent),
        "seeds": list(range(max(1, int(args.seeds)))),
        "episode_steps": int(args.episode_steps),
        "games": games,
        "summary": {
            "games": len(games),
            "win_rate": sum(float(game["win_points"]) for game in games) / max(1, len(games)),
            "mean_score_margin": sum(float(game["normalized_margin"]) for game in games)
            / max(1, len(games)),
            "mean_decision_ms": 1000.0
            * sum(float(game["elapsed_seconds"]) for game in games)
            / max(1.0, decisions),
            "timeout_rate": sum(float(game["timeouts"]) for game in games) / max(1.0, decisions),
            "crash_rate": sum(float(game["crashes"]) for game in games) / max(1.0, decisions),
            "invalid_action_rate": sum(float(game["invalid_actions"]) for game in games)
            / max(1.0, decisions),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
