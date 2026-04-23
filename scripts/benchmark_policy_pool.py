from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.agents import (
    anti_meta_agent,
    coalition_field_control_agent,
    defensive_agent,
    field_control_agent,
    greedy_agent,
    opening_gate_anti_meta_meta_agent,
    opening_gate_meta_agent,
    opening_gate_rush_meta_agent,
    rush_agent,
    weak_random_agent,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.train.evaluate_population import _moves_are_legal, _normalized_margin

Policy = Callable[[dict[str, Any], int], list[list[float]]]


BASE_POLICIES: dict[str, Policy] = {
    "greedy": greedy_agent,
    "defensive": defensive_agent,
    "rush": rush_agent,
    "anti_meta": anti_meta_agent,
    "weak_random": weak_random_agent,
    "field_control": field_control_agent,
    "coalition_field_control": coalition_field_control_agent,
    "opening_gate_anti_meta_meta": opening_gate_anti_meta_meta_agent,
    "opening_gate_meta": opening_gate_meta_agent,
    "opening_gate_rush_meta": opening_gate_rush_meta_agent,
}


def _load_policies() -> dict[str, Policy]:
    policies = dict(BASE_POLICIES)
    try:
        from python.agents.behavioral import BEHAVIORAL_POLICIES
    except ImportError:
        return policies
    policies.update(BEHAVIORAL_POLICIES)
    return policies


def _win_points(scores: list[float], player: int) -> float:
    top = max(float(score) for score in scores)
    winners = [idx for idx, score in enumerate(scores) if float(score) >= top]
    return 1.0 / len(winners) if player in winners else 0.0


def _worst_decile(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    count = max(1, len(ordered) // 10)
    return fmean(ordered[:count])


def _run_match(
    players: list[Policy],
    *,
    seed: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> list[float]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=len(players),
        seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout),
    )
    state = backend.reset(seed)[0]
    outcome = {"scores": [0.0 for _ in players], "done": False}
    while not outcome["done"]:
        actions = []
        for player, policy in enumerate(players):
            try:
                moves = policy(state, player)
            except Exception:
                moves = []
            if not isinstance(moves, list) or not _moves_are_legal(state, player, moves):
                moves = []
            actions.append(moves)
        outcome = backend.step([actions])[0]
        state = backend.states()[0]
    return [float(score) for score in outcome["scores"]]


def evaluate_candidate(
    candidate_name: str,
    candidate: Policy,
    opponents: dict[str, Policy],
    *,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> dict[str, Any]:
    all_margins: list[float] = []
    all_win_points: list[float] = []
    per_opponent: dict[str, dict[str, float]] = {}

    for opponent_name, opponent in opponents.items():
        margins: list[float] = []
        win_points: list[float] = []
        for seed in seeds:
            for candidate_first in (True, False):
                players = [candidate, opponent] if candidate_first else [opponent, candidate]
                candidate_idx = 0 if candidate_first else 1
                scores = _run_match(
                    players,
                    seed=seed,
                    episode_steps=episode_steps,
                    enable_comets=enable_comets,
                    act_timeout=act_timeout,
                )
                margins.append(_normalized_margin(scores, candidate_idx))
                win_points.append(_win_points(scores, candidate_idx))
        all_margins.extend(margins)
        all_win_points.extend(win_points)
        per_opponent[opponent_name] = {
            "games": float(len(margins)),
            "win_rate": fmean(win_points),
            "mean_margin": fmean(margins),
            "worst_decile_margin": _worst_decile(margins),
            "min_margin": min(margins),
        }

    return {
        "candidate": candidate_name,
        "summary": {
            "games": float(len(all_margins)),
            "win_rate": fmean(all_win_points) if all_win_points else 0.0,
            "mean_margin": fmean(all_margins) if all_margins else 0.0,
            "worst_decile_margin": _worst_decile(all_margins),
            "min_margin": min(all_margins) if all_margins else 0.0,
        },
        "per_opponent": per_opponent,
    }


def _parse_seeds(raw: str) -> list[int]:
    if "," in raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return list(range(max(1, int(raw))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--seeds", default="4")
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--disable-comets", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    policies = _load_policies()
    missing = [name for name in [*args.candidates, *args.opponents] if name not in policies]
    if missing:
        raise SystemExit(f"unknown policies: {', '.join(sorted(set(missing)))}")

    seeds = _parse_seeds(args.seeds)
    opponents = {name: policies[name] for name in args.opponents}
    report = {
        "seeds": seeds,
        "episode_steps": args.episode_steps,
        "candidates": [
            evaluate_candidate(
                name,
                policies[name],
                opponents,
                seeds=seeds,
                episode_steps=args.episode_steps,
                enable_comets=not args.disable_comets,
                act_timeout=args.act_timeout,
            )
            for name in args.candidates
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
