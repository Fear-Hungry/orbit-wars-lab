from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from python.agents.registry import Policy, get_heuristic_policies
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin

HEURISTIC_POLICIES = get_heuristic_policies()
_POLICY_CACHE: dict[str, Policy] = {}


def _opponent_label(path: Path) -> str:
    return path.stem


def _load_submission_agent(path: Path) -> Callable[[dict[str, Any]], list[list[float]]]:
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load submission module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise ValueError(f"submission module at {path} does not define callable agent(obs)")
    return agent


def _submission_runtime(agent: Callable[[dict[str, Any]], list[list[float]]]) -> Policy:
    def act(state: dict[str, Any], player: int) -> list[list[float]]:
        obs = to_official_observation(state, player=player)
        moves = agent(obs)
        return list(moves) if isinstance(moves, list) else []

    return act


def _resolve_opponent(spec: str) -> tuple[str, Policy]:
    if spec in HEURISTIC_POLICIES:
        return spec, HEURISTIC_POLICIES[spec]
    path = Path(spec)
    if path.exists() and path.is_file():
        return _opponent_label(path), _submission_runtime(_load_submission_agent(path))
    raise ValueError(f"unknown opponent: {spec}")


def _cached_submission_runtime(path: str) -> Policy:
    key = f"submission:{Path(path).resolve()}"
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _submission_runtime(_load_submission_agent(Path(path)))
    return _POLICY_CACHE[key]


def _cached_opponent_runtime(spec: str) -> tuple[str, Policy]:
    path = Path(spec)
    key = f"opponent:{path.resolve()}" if path.exists() else f"opponent:{spec}"
    if key not in _POLICY_CACHE:
        name, policy = _resolve_opponent(spec)
        _POLICY_CACHE[key] = policy
        return name, policy
    name = _opponent_label(path) if path.exists() and path.is_file() else spec
    return name, _POLICY_CACHE[key]


def _win_points(scores: list[float], player: int) -> float:
    top = max(float(score) for score in scores)
    winners = [idx for idx, score in enumerate(scores) if float(score) >= top]
    return 1.0 / len(winners) if player in winners else 0.0


def _empty_runtime_stats() -> dict[str, float]:
    return {
        "crashes": 0.0,
        "timeouts": 0.0,
        "invalid_actions": 0.0,
        "decision_turns": 0.0,
        "elapsed_seconds": 0.0,
    }


def _run_match(
    players: list[Policy],
    *,
    seed: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> tuple[list[float], list[dict[str, float]]]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=len(players),
        seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout),
    )
    state = backend.reset(seed)[0]
    runtime_stats = [_empty_runtime_stats() for _ in players]
    outcome = {"scores": [0.0 for _ in players], "done": False}

    while True:
        actions = [[] for _ in players]
        for idx, policy in enumerate(players):
            stats = runtime_stats[idx]
            stats["decision_turns"] += 1.0
            try:
                start = perf_counter()
                moves = policy(state, idx)
                elapsed = perf_counter() - start
                stats["elapsed_seconds"] += elapsed
                if elapsed > act_timeout:
                    stats["timeouts"] += 1.0
                    moves = []
                if not isinstance(moves, list) or not moves_are_legal(state, idx, moves):
                    stats["invalid_actions"] += 1.0
                    moves = []
            except Exception:
                stats["crashes"] += 1.0
                moves = []
            actions[idx] = moves

        outcomes, states = backend.step_with_states([actions])
        outcome = outcomes[0]
        state = states[0]
        if outcome["done"]:
            break

    return [float(score) for score in outcome["scores"]], runtime_stats


def _parallel_map(fn, tasks: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    if jobs <= 1 or len(tasks) <= 1:
        return [fn(task) for task in tasks]
    workers = min(int(jobs), len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, tasks))


def _two_player_task(task: dict[str, Any]) -> dict[str, Any]:
    submission = _cached_submission_runtime(str(task["submission"]))
    opponent_name, opponent = _cached_opponent_runtime(str(task["opponent"]))
    submission_first = bool(task["submission_first"])
    players = [submission, opponent] if submission_first else [opponent, submission]
    submission_idx = 0 if submission_first else 1
    scores, runtime_stats = _run_match(
        players,
        seed=int(task["seed"]),
        episode_steps=int(task["episode_steps"]),
        enable_comets=bool(task["enable_comets"]),
        act_timeout=float(task["act_timeout"]),
    )
    stats = runtime_stats[submission_idx]
    return {
        "seed": float(task["seed"]),
        "submission_player": float(submission_idx),
        "win_points": _win_points(scores, submission_idx),
        "normalized_margin": normalized_margin(scores, submission_idx),
        "opponent": opponent_name,
        **stats,
    }


def _four_player_task(task: dict[str, Any]) -> dict[str, Any]:
    submission = _cached_submission_runtime(str(task["submission"]))
    opponent_specs = [str(spec) for spec in task["opponents"]]
    opponents = [_cached_opponent_runtime(spec) for spec in opponent_specs]
    seed = int(task["seed"])
    rng = random.Random(7_919 * (seed + 1))
    picks = [rng.choice(opponents) for _ in range(3)]
    players: list[Policy] = [submission] + [policy for _, policy in picks]
    scores, runtime_stats = _run_match(
        players,
        seed=seed,
        episode_steps=int(task["episode_steps"]),
        enable_comets=bool(task["enable_comets"]),
        act_timeout=float(task["act_timeout"]),
    )
    return {
        "seed": float(seed),
        "win_points": _win_points(scores, 0),
        "normalized_margin": normalized_margin(scores, 0),
        "crashes": runtime_stats[0]["crashes"],
        "timeouts": runtime_stats[0]["timeouts"],
        "invalid_actions": runtime_stats[0]["invalid_actions"],
        "decision_turns": runtime_stats[0]["decision_turns"],
        "elapsed_seconds": runtime_stats[0]["elapsed_seconds"],
        "lineup": [name for name, _ in picks],
    }


def _summary_from_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "games": 0,
            "win_rate": 0.0,
            "mean_score_margin": 0.0,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
            "invalid_action_rate": 0.0,
            "mean_decision_ms": 0.0,
        }
    decisions = sum(record["decision_turns"] for record in records)
    elapsed = sum(record["elapsed_seconds"] for record in records)
    return {
        "games": len(records),
        "win_rate": fmean(record["win_points"] for record in records),
        "mean_score_margin": fmean(record["normalized_margin"] for record in records),
        "crash_rate": sum(record["crashes"] for record in records) / max(decisions, 1.0),
        "timeout_rate": sum(record["timeouts"] for record in records) / max(decisions, 1.0),
        "invalid_action_rate": sum(record["invalid_actions"] for record in records) / max(decisions, 1.0),
        "mean_decision_ms": 1000.0 * elapsed / max(decisions, 1.0),
    }


def benchmark_two_player(
    submission: Policy,
    opponent_name: str,
    opponent: Policy,
    *,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> dict[str, Any]:
    records: list[dict[str, float]] = []
    for seed in seeds:
        for submission_first in (True, False):
            players = [submission, opponent] if submission_first else [opponent, submission]
            submission_idx = 0 if submission_first else 1
            scores, runtime_stats = _run_match(
                players,
                seed=seed,
                episode_steps=episode_steps,
                enable_comets=enable_comets,
                act_timeout=act_timeout,
            )
            stats = runtime_stats[submission_idx]
            records.append(
                {
                    "seed": float(seed),
                    "submission_player": float(submission_idx),
                    "win_points": _win_points(scores, submission_idx),
                    "normalized_margin": normalized_margin(scores, submission_idx),
                    **stats,
                }
            )
    return {
        "format": "2p",
        "opponent": opponent_name,
        "summary": _summary_from_records(records),
        "records": records,
    }


def benchmark_two_player_spec(
    submission_path: Path,
    opponent_spec: str,
    *,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    jobs: int = 1,
) -> dict[str, Any]:
    opponent_name, _ = _resolve_opponent(opponent_spec)
    tasks = [
        {
            "submission": str(submission_path),
            "opponent": opponent_spec,
            "seed": seed,
            "submission_first": submission_first,
            "episode_steps": episode_steps,
            "enable_comets": enable_comets,
            "act_timeout": act_timeout,
        }
        for seed in seeds
        for submission_first in (True, False)
    ]
    records = _parallel_map(_two_player_task, tasks, jobs)
    for record in records:
        record.pop("opponent", None)
    return {
        "format": "2p",
        "opponent": opponent_name,
        "summary": _summary_from_records(records),
        "records": records,
    }


def benchmark_four_player(
    submission: Policy,
    opponents: list[tuple[str, Policy]],
    *,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> dict[str, Any]:
    if not opponents:
        return {
            "format": "4p",
            "opponents": [],
            "summary": _summary_from_records([]),
            "records": [],
        }

    records: list[dict[str, float]] = []
    for seed in seeds:
        rng = random.Random(7_919 * (seed + 1))
        picks = [rng.choice(opponents) for _ in range(3)]
        players: list[Policy] = [submission] + [policy for _, policy in picks]
        scores, runtime_stats = _run_match(
            players,
            seed=seed,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
        )
        records.append(
            {
                "seed": float(seed),
                "win_points": _win_points(scores, 0),
                "normalized_margin": normalized_margin(scores, 0),
                "crashes": runtime_stats[0]["crashes"],
                "timeouts": runtime_stats[0]["timeouts"],
                "invalid_actions": runtime_stats[0]["invalid_actions"],
                "decision_turns": runtime_stats[0]["decision_turns"],
                "elapsed_seconds": runtime_stats[0]["elapsed_seconds"],
                "lineup": [name for name, _ in picks],
            }
        )
    return {
        "format": "4p",
        "opponents": [name for name, _ in opponents],
        "summary": _summary_from_records(records),
        "records": records,
    }


def benchmark_four_player_spec(
    submission_path: Path,
    opponent_specs: list[str],
    *,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    jobs: int = 1,
) -> dict[str, Any]:
    opponents = [_resolve_opponent(spec) for spec in opponent_specs]
    if not opponents:
        return {
            "format": "4p",
            "opponents": [],
            "summary": _summary_from_records([]),
            "records": [],
        }
    tasks = [
        {
            "submission": str(submission_path),
            "opponents": opponent_specs,
            "seed": seed,
            "episode_steps": episode_steps,
            "enable_comets": enable_comets,
            "act_timeout": act_timeout,
        }
        for seed in seeds
    ]
    records = _parallel_map(_four_player_task, tasks, jobs)
    return {
        "format": "4p",
        "opponents": [name for name, _ in opponents],
        "summary": _summary_from_records(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--opponents", nargs="+", default=list(HEURISTIC_POLICIES))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-2p", action="store_true")
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    submission_path = Path(args.submission)
    seeds = list(range(max(1, int(args.seeds))))
    jobs = max(1, int(args.jobs))

    try:
        for spec in args.opponents:
            _resolve_opponent(spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report: dict[str, Any] = {
        "submission": str(submission_path),
        "seeds": seeds,
        "episode_steps": int(args.episode_steps),
        "enable_comets": not bool(args.disable_comets),
        "jobs": jobs,
        "formats": [],
    }

    if not args.skip_2p:
        report["formats"].append(
            {
                "format": "2p",
                "opponents": [
                    benchmark_two_player_spec(
                        submission_path,
                        spec,
                        seeds=seeds,
                        episode_steps=int(args.episode_steps),
                        enable_comets=not bool(args.disable_comets),
                        act_timeout=float(args.act_timeout),
                        jobs=jobs,
                    )
                    for spec in args.opponents
                ],
            }
        )

    if not args.skip_4p:
        report["formats"].append(
            benchmark_four_player_spec(
                submission_path,
                list(args.opponents),
                seeds=seeds,
                episode_steps=int(args.episode_steps),
                enable_comets=not bool(args.disable_comets),
                act_timeout=float(args.act_timeout),
                jobs=jobs,
            )
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
