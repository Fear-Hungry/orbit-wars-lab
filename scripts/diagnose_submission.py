from __future__ import annotations

import argparse
import json
from math import atan2, ceil, cos, hypot, log, sin
from pathlib import Path
from time import perf_counter
from typing import Any

from python.agents.registry import get_heuristic_policies
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from scripts.benchmark_submission import _load_submission_agent, _submission_runtime, _win_points

HEURISTIC_POLICIES = get_heuristic_policies()
PROFILE_RAY_MAX_ANGLE = 0.36
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED = 6.0


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


def _angle_delta(left: float, right: float) -> float:
    return abs(atan2(sin(left - right), cos(left - right)))


def _planet_field(planet: Any, key: str, index: int) -> Any:
    return planet[key] if isinstance(planet, dict) else planet[index]


def _fleet_speed(ships: int) -> float:
    scale = log(max(int(ships), 1)) / log(1000.0)
    speed = 1.0 + (SHIP_SPEED - 1.0) * scale**1.5
    return min(SHIP_SPEED, max(1.0, speed))


def _rotate_about_center(point: tuple[float, float], angle: float) -> tuple[float, float]:
    dx = point[0] - CENTER
    dy = point[1] - CENTER
    c = cos(angle)
    s = sin(angle)
    return (CENTER + dx * c - dy * s, CENTER + dx * s + dy * c)


def _is_rotating_planet(planet: Any) -> bool:
    x = float(_planet_field(planet, "x", 2))
    y = float(_planet_field(planet, "y", 3))
    radius = float(_planet_field(planet, "radius", 4))
    return hypot(x - CENTER, y - CENTER) + radius < ROTATION_RADIUS_LIMIT


def _predict_target_xy(state: dict[str, Any], source_xy: tuple[float, float], target: Any, ships: int) -> tuple[float, float]:
    target_xy = (float(_planet_field(target, "x", 2)), float(_planet_field(target, "y", 3)))
    if not _is_rotating_planet(target):
        return target_xy
    distance = hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1])
    travel_steps = max(1, ceil(distance / _fleet_speed(ships)))
    return _rotate_about_center(target_xy, float(state.get("angular_velocity", 0.0)) * travel_steps)


def _action_target(state: dict[str, Any], action: list[float]) -> dict[str, Any]:
    if not isinstance(action, list) or len(action) != 3:
        return {"action": action, "target_id": None, "score": None}

    source_id = int(action[0])
    angle = float(action[1])
    source = next(
        (planet for planet in state.get("planets", []) if int(_planet_field(planet, "id", 0)) == source_id),
        None,
    )
    if source is None:
        return {"action": action, "target_id": None, "score": None}

    source_xy = (float(_planet_field(source, "x", 2)), float(_planet_field(source, "y", 3)))
    ships = max(1, int(action[2]))
    best: tuple[float, Any] | None = None
    for planet in state.get("planets", []):
        target_xy = _predict_target_xy(state, source_xy, planet, ships)
        distance = hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1])
        if distance <= 0.0:
            continue
        target_angle = atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])
        delta = _angle_delta(angle, target_angle)
        if delta > PROFILE_RAY_MAX_ANGLE:
            continue
        score = delta + 0.004 * distance
        if best is None or score < best[0]:
            best = (score, planet)

    if best is None:
        return {"action": action, "target_id": None, "score": None}

    target = best[1]
    return {
        "action": action,
        "target_id": int(_planet_field(target, "id", 0)),
        "target_owner": int(_planet_field(target, "owner", 1)),
        "target_ships": int(_planet_field(target, "ships", 5)),
        "target_production": int(_planet_field(target, "production", 6)),
        "score": float(best[0]),
    }


def _action_targets(state: dict[str, Any], actions: list[list[float]]) -> list[dict[str, Any]]:
    return [_action_target(state, action) for action in actions]


def _planet_snapshots(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for planet in state.get("planets", []):
        planet_id = int(_planet_field(planet, "id", 0))
        snapshots[planet_id] = {
            "planet_id": planet_id,
            "owner": int(_planet_field(planet, "owner", 1)),
            "x": float(_planet_field(planet, "x", 2)),
            "y": float(_planet_field(planet, "y", 3)),
            "radius": float(_planet_field(planet, "radius", 4)),
            "ships": int(_planet_field(planet, "ships", 5)),
            "production": int(_planet_field(planet, "production", 6)),
            "rotating": _is_rotating_planet(planet),
        }
    return snapshots


def _capture_events(before: dict[str, Any], after: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    previous = _planet_snapshots(before)
    current = _planet_snapshots(after)
    events = []
    for planet_id, now in current.items():
        old = previous.get(planet_id)
        if old is None or int(old["owner"]) == int(now["owner"]):
            continue
        events.append(
            {
                "turn": turn,
                "planet_id": planet_id,
                "old_owner": int(old["owner"]),
                "new_owner": int(now["owner"]),
                "ships": int(now["ships"]),
                "production": int(now["production"]),
                "rotating": bool(now["rotating"]),
                "x": float(now["x"]),
                "y": float(now["y"]),
            }
        )
    return events


def _launch_events(
    state: dict[str, Any],
    actions: list[list[list[float]]],
    *,
    turn: int,
    submission_player: int,
) -> list[dict[str, Any]]:
    events = []
    for player, moves in enumerate(actions):
        actor = "submission" if player == submission_player else "opponent"
        for target in _action_targets(state, moves):
            action = target.get("action")
            if not isinstance(action, list) or len(action) != 3:
                continue
            events.append(
                {
                    "turn": turn,
                    "actor": actor,
                    "player": player,
                    "source_id": int(action[0]),
                    "ships": int(action[2]),
                    "target_id": target.get("target_id"),
                    "target_owner": target.get("target_owner"),
                    "target_ships": target.get("target_ships"),
                    "target_production": target.get("target_production"),
                    "target_score": target.get("score"),
                }
            )
    return events


def _economy_summary(captures: list[dict[str, Any]], players: int = 2) -> dict[str, Any]:
    summary: dict[str, Any] = {
        str(player): {
            "first_capture_turn": None,
            "neutral_captures": 0,
            "neutral_production": 0,
            "enemy_captures": 0,
            "enemy_production": 0,
        }
        for player in range(players)
    }
    for event in captures:
        owner = int(event["new_owner"])
        if owner < 0:
            continue
        item = summary.setdefault(
            str(owner),
            {
                "first_capture_turn": None,
                "neutral_captures": 0,
                "neutral_production": 0,
                "enemy_captures": 0,
                "enemy_production": 0,
            },
        )
        if item["first_capture_turn"] is None:
            item["first_capture_turn"] = int(event["turn"])
        if int(event["old_owner"]) == -1:
            item["neutral_captures"] += 1
            item["neutral_production"] += int(event["production"])
        else:
            item["enemy_captures"] += 1
            item["enemy_production"] += int(event["production"])
    return summary


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
    event_turns: int = 120,
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
    launches: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    outcome = {"scores": [0.0, 0.0], "done": False}

    for turn in range(episode_steps):
        state_before = state
        actions = [
            _act(players[0], state, 0, act_timeout=act_timeout, stats=runtime_stats[0]),
            _act(players[1], state, 1, act_timeout=act_timeout, stats=runtime_stats[1]),
        ]
        if turn < event_turns:
            launches.extend(_launch_events(state, actions, turn=turn, submission_player=submission_player))
        should_sample = turn < early_turns or (sample_every > 0 and turn % sample_every == 0)
        if should_sample:
            timeline.append(
                {
                    "turn": turn,
                    "totals_before": _owner_totals(state),
                    "submission_actions": actions[submission_player],
                    "opponent_actions": actions[1 - submission_player],
                    "submission_action_targets": _action_targets(state, actions[submission_player]),
                    "opponent_action_targets": _action_targets(state, actions[1 - submission_player]),
                }
            )

        outcome = backend.step([actions])[0]
        state = backend.states()[0]
        captures.extend(_capture_events(state_before, state, turn + 1))
        if outcome["done"]:
            if not should_sample:
                timeline.append(
                    {
                        "turn": turn + 1,
                        "totals_before": _owner_totals(state),
                        "submission_actions": [],
                        "opponent_actions": [],
                        "submission_action_targets": [],
                        "opponent_action_targets": [],
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
        "economy_summary": _economy_summary(captures, players=2),
        "launch_events": launches,
        "capture_events": captures,
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
    parser.add_argument("--event-turns", type=int, default=120)
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
        event_turns=args.event_turns,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
