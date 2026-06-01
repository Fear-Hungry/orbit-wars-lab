from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from python.agents.registry import get_heuristic_policies
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from scripts.benchmark_submission import _load_submission_agent, _submission_runtime, _win_points

HEURISTIC_POLICIES = get_heuristic_policies()


def _empty_runtime_stats() -> dict[str, float]:
    return {
        "crashes": 0.0,
        "timeouts": 0.0,
        "invalid_actions": 0.0,
        "decision_turns": 0.0,
        "elapsed_seconds": 0.0,
    }


def _owner_totals(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for owner in (-1, 0, 1, 2, 3):
        totals[str(owner)] = {"planets": 0, "production": 0, "ships": 0, "fleets": 0, "fleet_ships": 0}
    for planet in state.get("planets", []):
        owner = str(int(planet["owner"] if isinstance(planet, dict) else planet[1]))
        totals.setdefault(owner, {"planets": 0, "production": 0, "ships": 0, "fleets": 0, "fleet_ships": 0})
        totals[owner]["planets"] += 1
        totals[owner]["production"] += int(planet["production"] if isinstance(planet, dict) else planet[6])
        totals[owner]["ships"] += int(planet["ships"] if isinstance(planet, dict) else planet[5])
    for fleet in state.get("fleets", []):
        owner = str(int(fleet["owner"] if isinstance(fleet, dict) else fleet[1]))
        totals.setdefault(owner, {"planets": 0, "production": 0, "ships": 0, "fleets": 0, "fleet_ships": 0})
        totals[owner]["fleets"] += 1
        totals[owner]["fleet_ships"] += int(fleet["ships"] if isinstance(fleet, dict) else fleet[6])
    return totals


def _act(policy, state: dict[str, Any], player: int, *, act_timeout: float, stats: dict[str, float]) -> list[list[float]]:
    stats["decision_turns"] += 1.0
    try:
        start = perf_counter()
        moves = policy(state, player)
        elapsed = perf_counter() - start
        stats["elapsed_seconds"] += elapsed
        if elapsed > act_timeout:
            stats["timeouts"] += 1.0
            return []
        if not isinstance(moves, list) or not moves_are_legal(state, player, moves):
            stats["invalid_actions"] += 1.0
            return []
        return moves
    except Exception:
        stats["crashes"] += 1.0
        return []


def diagnose_match(
    submission_path: Path,
    *,
    opponent_name: str,
    seed: int,
    submission_player: int,
    episode_steps: int = 500,
    enable_comets: bool = True,
    act_timeout: float = 1.0,
    early_turns: int = 20,
    sample_every: int = 25,
) -> dict[str, Any]:
    if opponent_name not in HEURISTIC_POLICIES:
        raise ValueError(f"unknown opponent: {opponent_name}")
    if submission_player not in (0, 1):
        raise ValueError("submission_player must be 0 or 1")

    submission = _submission_runtime(_load_submission_agent(submission_path))
    opponent = HEURISTIC_POLICIES[opponent_name]
    players = [submission, opponent] if submission_player == 0 else [opponent, submission]
    runtime_stats = [_empty_runtime_stats(), _empty_runtime_stats()]

    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout),
    )
    state = backend.reset(seed)[0]
    timeline: list[dict[str, Any]] = []
    outcome = {"scores": [0.0, 0.0], "done": False}

    for turn in range(episode_steps):
        actions = [
            _act(players[0], state, 0, act_timeout=act_timeout, stats=runtime_stats[0]),
            _act(players[1], state, 1, act_timeout=act_timeout, stats=runtime_stats[1]),
        ]
        should_sample = turn < early_turns or (sample_every > 0 and turn % sample_every == 0)
        if should_sample:
            timeline.append(
                {
                    "turn": turn,
                    "totals_before": _owner_totals(state),
                    "submission_actions": actions[submission_player],
                    "opponent_actions": actions[1 - submission_player],
                }
            )

        outcome = backend.step([actions])[0]
        state = backend.states()[0]
        if outcome["done"]:
            if not should_sample:
                timeline.append(
                    {
                        "turn": turn + 1,
                        "totals_before": _owner_totals(state),
                        "submission_actions": [],
                        "opponent_actions": [],
                    }
                )
            break

    scores = [float(score) for score in outcome["scores"]]
    return {
        "submission": str(submission_path),
        "opponent": opponent_name,
        "seed": seed,
        "submission_player": submission_player,
        "episode_steps": episode_steps,
        "enable_comets": enable_comets,
        "scores": scores,
        "done": bool(outcome["done"]),
        "win_points": _win_points(scores, submission_player),
        "normalized_margin": normalized_margin(scores, submission_player),
        "runtime_stats": runtime_stats[submission_player],
        "final_totals": _owner_totals(state),
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="python/submission/submission_template.py")
    parser.add_argument("--opponent", required=True, choices=sorted(HEURISTIC_POLICIES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--submission-player", type=int, choices=(0, 1), required=True)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--early-turns", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = diagnose_match(
        Path(args.submission),
        opponent_name=args.opponent,
        seed=args.seed,
        submission_player=args.submission_player,
        episode_steps=args.episode_steps,
        enable_comets=not args.disable_comets,
        act_timeout=args.act_timeout,
        early_turns=args.early_turns,
        sample_every=args.sample_every,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
