from __future__ import annotations
# ruff: noqa: E402,I001

import argparse
import importlib.util
import json
import os
import random
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.agents.registry import (
    PRODUCER_AGENT_PATH,
    PRODUCER_SETUP_COMMAND,
    Policy,
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
    get_isolated_opponents,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin

HEURISTIC_POLICIES = get_heuristic_policies()
DEFAULT_OPPONENTS = ["producer"]
DEFAULT_SEEDS = 16
DEFAULT_JOBS = max(1, os.cpu_count() or 1)
_POLICY_CACHE: dict[str, Policy] = {}
TECHNICAL_FAILURE_METRICS = (
    "crash_rate",
    "timeout_rate",
    "invalid_action_rate",
    "fallback_rate",
    "policy_illegal_move_rate",
    "fallback_error_rate",
)
INSTRUMENTATION_FAILURE_METRIC = "instrumentation_missing_rate"


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
    def _stats_snapshot() -> dict[str, float] | None:
        raw = getattr(agent, "__globals__", {}).get("SUBMISSION_STATS", None)
        if not isinstance(raw, dict):
            return None
        return {
            "fallbacks": float(raw.get("fallbacks", 0.0)),
            "illegal_moves": float(raw.get("illegal_moves", 0.0)),
            "fallback_errors": float(raw.get("fallback_errors", 0.0)),
        }

    def act(state: dict[str, Any], player: int) -> list[list[float]]:
        obs = to_official_observation(state, player=player)
        before = _stats_snapshot()
        try:
            moves = agent(obs)
        finally:
            after = _stats_snapshot()
            missing = before is None or after is None
            act._last_submission_stats_delta = {"instrumentation_missing": 1.0 if missing else 0.0}
            if not missing:
                act._last_submission_stats_delta.update({
                    name: max(0.0, float(after.get(name, 0.0)) - float(before.get(name, 0.0)))
                    for name in ("fallbacks", "illegal_moves", "fallback_errors")
                })
        return moves

    return act


def _resolve_opponent(spec: str) -> tuple[str, Policy]:
    if spec == "producer" and not PRODUCER_AGENT_PATH.exists():
        raise RuntimeError(
            f"Producer bot is missing. Restore bots/producer or run "
            f"`{PRODUCER_SETUP_COMMAND}`."
        )
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


def _opponent_instances(specs: list[str]) -> list[tuple[str, Policy]]:
    """Build live-seat opponent instances for a single 4p game.

    Some built-in opponents keep game memory in a module singleton. In 4p, a
    lineup like producer,producer,producer needs three independent runtimes,
    not three references to one callable.
    """
    stateful_counts = {
        name: specs.count(name)
        for name in set(specs)
        if name in STATEFUL_SINGLETON_OPPONENTS
    }
    stateful_pools = {
        name: iter(get_isolated_opponents(name, count))
        for name, count in stateful_counts.items()
    }
    out: list[tuple[str, Policy]] = []
    for spec in specs:
        path = Path(spec)
        if spec in stateful_pools:
            out.append((spec, next(stateful_pools[spec])))
        elif path.exists() and path.is_file():
            # File-based agents can also be stateful; load one module per seat.
            out.append((_opponent_label(path), _submission_runtime(_load_submission_agent(path))))
        else:
            out.append(_cached_opponent_runtime(spec))
    return out


def _win_points(scores: list[float], player: int) -> float:
    top = max(float(score) for score in scores)
    winners = [idx for idx, score in enumerate(scores) if float(score) >= top]
    return 1.0 / len(winners) if player in winners else 0.0


def _empty_runtime_stats() -> dict[str, float]:
    return {
        "crashes": 0.0,
        "timeouts": 0.0,
        "invalid_actions": 0.0,
        "fallbacks": 0.0,
        "policy_illegal_moves": 0.0,
        "fallback_errors": 0.0,
        "instrumentation_missing": 0.0,
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
        config=RustConfig(
            episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout
        ),
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
                submission_delta = getattr(policy, "_last_submission_stats_delta", {})
                if isinstance(submission_delta, dict):
                    stats["fallbacks"] += float(submission_delta.get("fallbacks", 0.0))
                    stats["policy_illegal_moves"] += float(
                        submission_delta.get("illegal_moves", 0.0)
                    )
                    stats["fallback_errors"] += float(
                        submission_delta.get("fallback_errors", 0.0)
                    )
                    stats["instrumentation_missing"] += float(
                        submission_delta.get("instrumentation_missing", 0.0)
                    )
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
    seed = int(task["seed"])
    rng = random.Random(7_919 * (seed + 1))
    pick_specs = [rng.choice(opponent_specs) for _ in range(3)]
    picks = _opponent_instances(pick_specs)
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
        "fallbacks": runtime_stats[0]["fallbacks"],
        "policy_illegal_moves": runtime_stats[0]["policy_illegal_moves"],
        "fallback_errors": runtime_stats[0]["fallback_errors"],
        "instrumentation_missing": runtime_stats[0]["instrumentation_missing"],
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
            "fallback_rate": 0.0,
            "policy_illegal_move_rate": 0.0,
            "fallback_error_rate": 0.0,
            "instrumentation_missing_rate": 0.0,
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
        "invalid_action_rate": sum(record["invalid_actions"] for record in records)
        / max(decisions, 1.0),
        "fallback_rate": sum(record["fallbacks"] for record in records) / max(decisions, 1.0),
        "policy_illegal_move_rate": sum(record["policy_illegal_moves"] for record in records)
        / max(decisions, 1.0),
        "fallback_error_rate": sum(record["fallback_errors"] for record in records)
        / max(decisions, 1.0),
        "instrumentation_missing_rate": sum(
            record.get("instrumentation_missing", 0.0) for record in records
        ) / max(decisions, 1.0),
        "mean_decision_ms": 1000.0 * elapsed / max(decisions, 1.0),
    }


def _iter_summaries(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    summaries: list[tuple[str, dict[str, Any]]] = []
    for item in report.get("formats", []):
        game_format = str(item.get("format", "unknown"))
        if game_format == "2p":
            for opponent in item.get("opponents", []):
                label = f"2p:{opponent.get('opponent', 'unknown')}"
                summaries.append((label, dict(opponent.get("summary", {}))))
        else:
            summaries.append((game_format, dict(item.get("summary", {}))))
    return summaries


def technical_failures(
    report: dict[str, Any], *, require_submission_stats: bool = False
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    metrics = list(TECHNICAL_FAILURE_METRICS)
    if require_submission_stats:
        metrics.append(INSTRUMENTATION_FAILURE_METRIC)
    for label, summary in _iter_summaries(report):
        for metric in metrics:
            value = float(summary.get(metric, 0.0))
            if value > 0.0:
                failures.append({"label": label, "metric": metric, "value": value})
    return failures


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
                "fallbacks": runtime_stats[0]["fallbacks"],
                "policy_illegal_moves": runtime_stats[0]["policy_illegal_moves"],
                "fallback_errors": runtime_stats[0]["fallback_errors"],
                "instrumentation_missing": runtime_stats[0]["instrumentation_missing"],
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
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--skip-2p", action="store_true")
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument(
        "--allow-technical-failures",
        action="store_true",
        help="write the report even when crash/timeout/invalid/fallback rates are nonzero",
    )
    parser.add_argument(
        "--require-submission-stats",
        action="store_true",
        help="also fail if the submission does not expose SUBMISSION_STATS instrumentation",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    submission_path = Path(args.submission)
    seeds = list(range(max(1, int(args.seeds))))
    jobs = max(1, int(args.jobs))

    try:
        for spec in args.opponents:
            _resolve_opponent(spec)
    except (RuntimeError, ValueError) as exc:
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
    failures = technical_failures(report, require_submission_stats=args.require_submission_stats)
    if failures and not args.allow_technical_failures:
        raise SystemExit(
            "technical failures detected; rerun with --allow-technical-failures "
            f"for exploratory measurement: {failures}"
        )


if __name__ == "__main__":
    main()
