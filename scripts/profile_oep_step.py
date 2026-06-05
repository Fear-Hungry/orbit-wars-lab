from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from bots.oep.planner import OEPLiteRuntime, _env_config
from bots.producer.agent import agent as producer_agent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _producer_policy(state: dict[str, Any], player: int) -> list[list[float]]:
    obs = to_official_observation(state, player=player)
    moves = producer_agent(obs)
    return list(moves) if isinstance(moves, list) else []


def _oep_policy(runtime: OEPLiteRuntime):
    def act(state: dict[str, Any], player: int) -> list[list[float]]:
        obs = to_official_observation(state, player=player)
        moves = runtime.act(obs)
        return list(moves) if isinstance(moves, list) else []

    return act


def _merge_profiles(
    target_totals: dict[str, float],
    target_counts: dict[str, float],
    profile: dict[str, dict[str, float]],
) -> None:
    for name, row in profile.items():
        target_totals[name] = target_totals.get(name, 0.0) + float(row["total_ms"])
        target_counts[name] = target_counts.get(name, 0.0) + float(row["calls"])


def _run_match(
    *,
    seed: int,
    submission_player: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    opponent_response_mode: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    runtime = OEPLiteRuntime(
        seed_policy=producer_agent,
        opponent_policy=producer_agent,
        config=dataclasses.replace(
            _env_config(),
            profile_stages=True,
            opponent_response_mode=str(opponent_response_mode),
        ),
    )
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
    players = [_producer_policy, _producer_policy]
    players[int(submission_player)] = _oep_policy(runtime)
    decision_turns = 0.0
    elapsed_seconds = 0.0
    crashes = 0.0
    timeouts = 0.0
    invalid_actions = 0.0
    decision_ms: list[float] = []
    outcome = {"scores": [0.0, 0.0], "done": False}

    while True:
        actions = [[], []]
        for player_idx, policy in enumerate(players):
            try:
                start = perf_counter()
                moves = policy(state, player_idx)
                elapsed = perf_counter() - start
                if player_idx == int(submission_player):
                    decision_turns += 1.0
                    elapsed_seconds += elapsed
                    decision_ms.append(1000.0 * elapsed)
                    if elapsed > float(act_timeout):
                        timeouts += 1.0
                        moves = []
                if not isinstance(moves, list) or not moves_are_legal(state, player_idx, moves):
                    if player_idx == int(submission_player):
                        invalid_actions += 1.0
                    moves = []
            except Exception:
                if player_idx == int(submission_player):
                    crashes += 1.0
                moves = []
            actions[player_idx] = moves

        outcomes, states = backend.step_with_states([actions])
        outcome = outcomes[0]
        state = states[0]
        if outcome["done"]:
            break

    scores = [float(score) for score in outcome["scores"]]
    record = {
        "seed": float(seed),
        "submission_player": float(submission_player),
        "decision_turns": decision_turns,
        "elapsed_seconds": elapsed_seconds,
        "mean_decision_ms": 1000.0 * elapsed_seconds / max(1.0, decision_turns),
        "p50_decision_ms": _percentile(decision_ms, 0.50),
        "p95_decision_ms": _percentile(decision_ms, 0.95),
        "p99_decision_ms": _percentile(decision_ms, 0.99),
        "max_decision_ms": max(decision_ms) if decision_ms else 0.0,
        "normalized_margin": normalized_margin(scores, int(submission_player)),
        "crashes": crashes,
        "timeouts": timeouts,
        "invalid_actions": invalid_actions,
    }
    return record, runtime.profile_summary()


def _summarize(records: list[dict[str, float]], profile: dict[str, dict[str, float]]) -> dict[str, Any]:
    decisions = sum(record["decision_turns"] for record in records)
    elapsed = sum(record["elapsed_seconds"] for record in records)
    action_total = profile.get("action_total", {})
    action_total_ms = float(action_total.get("total_ms", 0.0))
    stage_rows = []
    stage_total_ms = 0.0
    for name, row in sorted(profile.items(), key=lambda item: item[1]["total_ms"], reverse=True):
        if name == "action_total":
            continue
        total_ms = float(row["total_ms"])
        stage_total_ms += total_ms
        stage_rows.append({
            "stage": name,
            "calls": row["calls"],
            "total_ms": total_ms,
            "mean_ms": row["mean_ms"],
            "pct_action_total": 100.0 * total_ms / action_total_ms if action_total_ms > 0 else 0.0,
        })
    return {
        "records": records,
        "summary": {
            "games": len(records),
            "decision_turns": decisions,
            "mean_decision_ms": 1000.0 * elapsed / max(1.0, decisions),
            "mean_match_p95_decision_ms": fmean(record["p95_decision_ms"] for record in records) if records else 0.0,
            "max_match_p95_decision_ms": max((record["p95_decision_ms"] for record in records), default=0.0),
            "max_decision_ms": max((record["max_decision_ms"] for record in records), default=0.0),
            "mean_score_margin": fmean(record["normalized_margin"] for record in records) if records else 0.0,
            "crash_rate": sum(record["crashes"] for record in records) / max(1.0, decisions),
            "timeout_rate": sum(record["timeouts"] for record in records) / max(1.0, decisions),
            "invalid_action_rate": sum(record["invalid_actions"] for record in records) / max(1.0, decisions),
            "profile_action_total_ms": action_total_ms,
            "profile_stage_sum_ms": stage_total_ms,
            "profile_coverage_pct": 100.0 * stage_total_ms / action_total_ms if action_total_ms > 0 else 0.0,
        },
        "stages": stage_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile OEP decision time by planner stage.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument(
        "--opponent-response-mode",
        choices=("cheap", "producer", "none"),
        default="cheap",
    )
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--single-side", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records: list[dict[str, float]] = []
    profile_totals: dict[str, float] = {}
    profile_counts: dict[str, float] = {}
    sides = (0,) if bool(args.single_side) else (0, 1)
    for seed in range(max(1, int(args.seeds))):
        for submission_player in sides:
            record, profile = _run_match(
                seed=seed,
                submission_player=int(submission_player),
                episode_steps=int(args.episode_steps),
                enable_comets=not bool(args.disable_comets),
                act_timeout=float(args.act_timeout),
                opponent_response_mode=str(args.opponent_response_mode),
            )
            records.append(record)
            _merge_profiles(profile_totals, profile_counts, profile)

    profile = {
        name: {
            "calls": float(profile_counts.get(name, 0.0)),
            "total_ms": float(total_ms),
            "mean_ms": float(total_ms) / max(1.0, profile_counts.get(name, 0.0)),
        }
        for name, total_ms in sorted(profile_totals.items())
    }
    report = _summarize(records, profile)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
