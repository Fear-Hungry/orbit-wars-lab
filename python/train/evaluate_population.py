from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

import torch
import yaml
from python.agents.policy import FlatActorCritic
from python.agents.registry import get_heuristic_policies
from python.league.elo import EloRating, update_elo
from python.league.evaluation import AgentSpec, default_hparams, load_population_manifest
from python.league.hall_of_fame import load_hall_of_fame
from python.orbit_wars_gym.action_decoder import DecoderConfig, decode_discrete_action
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import encode_state, observation_dim
from python.orbit_wars_gym.entities import planet_id, planet_owner
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from rich import print

HEURISTIC_POLICIES = get_heuristic_policies()


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: list[int]
    games_per_pair: int
    include_2p: bool
    include_4p: bool
    episode_steps: int = 500
    enable_comets: bool = True
    act_timeout: float = 1.0


@dataclass
class AggregateStats:
    games: int = 0
    league_games: int = 0
    hall_of_fame_games: int = 0
    win_points_vs_league: float = 0.0
    win_points_vs_hall_of_fame: float = 0.0
    normalized_margins: list[float] = field(default_factory=list)
    crashes: int = 0
    timeouts: int = 0
    invalid_actions: int = 0
    decision_turns: int = 0
    launched_ships: int = 0
    launched_moves: int = 0
    neutral_captures: int = 0
    enemy_captures: int = 0
    final_planet_shares: list[float] = field(default_factory=list)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    eval_cfg = payload["eval"]
    raw_seeds = eval_cfg.get("seeds", 16)
    seeds = list(range(int(raw_seeds))) if isinstance(raw_seeds, int) else [int(seed) for seed in raw_seeds]
    if not seeds:
        raise ValueError("evaluation config requires at least one seed")
    return EvaluationConfig(
        seeds=seeds,
        games_per_pair=int(eval_cfg.get("games_per_pair", 1)),
        include_2p=bool(eval_cfg.get("include_2p", True)),
        include_4p=bool(eval_cfg.get("include_4p", False)),
        episode_steps=int(eval_cfg.get("episode_steps", 500)),
        enable_comets=bool(eval_cfg.get("enable_comets", True)),
        act_timeout=float(eval_cfg.get("act_timeout", 1.0)),
    )


def attach_hall_of_fame_snapshots(
    manifest: dict[str, list[AgentSpec]],
    hall_of_fame_state_path: str | Path | None,
) -> dict[str, list[AgentSpec]]:
    if hall_of_fame_state_path is None:
        return manifest
    hall_of_fame = load_hall_of_fame(hall_of_fame_state_path, max_size=10_000)
    existing_ids = {spec.id for group in manifest.values() for spec in group}
    merged = {
        "population": list(manifest["population"]),
        "hall_of_fame": list(manifest["hall_of_fame"]),
        "heuristics": list(manifest["heuristics"]),
    }
    for entry in hall_of_fame.entries:
        if entry.checkpoint is None or entry.member_id in existing_ids:
            continue
        merged["hall_of_fame"].append(
            AgentSpec(
                id=entry.member_id,
                kind="ppo",
                role="hall_of_fame",
                checkpoint=entry.checkpoint,
                rating=entry.rating,
                hparams=default_hparams(),
                decoder=dict(entry.decoder),
            )
        )
    return merged


def _checkpoint_decoder_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if isinstance(summary.get("decoder"), dict):
        return dict(summary["decoder"])
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    if not config:
        return {}
    out: dict[str, Any] = {}
    mapping = {
        "decoder_fractions": "fractions",
        "decoder_angle_offsets": "angle_offsets",
        "decoder_max_moves_per_turn": "max_moves_per_turn",
        "decoder_min_ships_to_launch": "min_ships_to_launch",
        "decoder_reserve_home_ships": "reserve_home_ships",
    }
    for source, target in mapping.items():
        if source in config:
            out[target] = config[source]
    return out


def _decoder_config(spec: AgentSpec, payload: dict[str, Any] | None = None) -> DecoderConfig:
    decoder = dict(spec.decoder)
    if payload is not None and not decoder:
        decoder = _checkpoint_decoder_payload(payload)
    return DecoderConfig(
        fractions=tuple(float(value) for value in decoder.get("fractions", (0.10, 0.25, 0.50, 0.75))),
        angle_offsets=tuple(float(value) for value in decoder.get("angle_offsets", (-0.261799, -0.130899, 0.0, 0.130899, 0.261799))),
        max_moves_per_turn=int(decoder.get("max_moves_per_turn", 8)),
        min_ships_to_launch=int(decoder.get("min_ships_to_launch", 2)),
        reserve_home_ships=int(decoder.get("reserve_home_ships", 8)),
    )


def _moves_are_legal(state: dict[str, Any], player: int, moves: list[list[float]]) -> bool:
    return moves_are_legal(state, player, moves)


def _policy_runtime(spec: AgentSpec) -> Any:
    if spec.kind == "heuristic":
        try:
            return HEURISTIC_POLICIES[spec.policy or ""]
        except KeyError as exc:
            raise ValueError(f"unknown heuristic policy: {spec.policy}") from exc

    checkpoint_path = Path(spec.checkpoint or "")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FlatActorCritic(observation_dim())
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    decoder_cfg = _decoder_config(spec, payload)

    def act(state: dict[str, Any], player: int) -> list[list[float]]:
        obs = torch.as_tensor(encode_state(state, player), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model.forward(obs)
        action = [int(out[key].argmax(dim=-1).item()) for key in ("source", "target", "frac", "offset")]
        return decode_discrete_action(state, player, action, decoder_cfg)

    return act


def _capture_counts(previous_state: dict[str, Any], next_state: dict[str, Any], player: int) -> tuple[int, int]:
    previous = {planet_id(planet): planet_owner(planet) for planet in previous_state.get("planets", [])}
    neutral_captures = 0
    enemy_captures = 0
    for planet in next_state.get("planets", []):
        pid = planet_id(planet)
        owner_before = previous.get(pid, -1)
        owner_after = planet_owner(planet)
        if owner_after != player or owner_before == player:
            continue
        if owner_before == -1:
            neutral_captures += 1
        else:
            enemy_captures += 1
    return neutral_captures, enemy_captures


def _final_planet_share(state: dict[str, Any], player: int) -> float:
    planets = state.get("planets", [])
    if not planets:
        return 0.0
    owned = sum(1 for planet in planets if planet_owner(planet) == player)
    return owned / len(planets)


def _normalized_margin(scores: list[float], player: int) -> float:
    return normalized_margin(scores, player)


def _sample_opponents(anchor: AgentSpec, opponents: list[AgentSpec], seed: int, count: int) -> list[AgentSpec]:
    if not opponents:
        raise ValueError(f"agent {anchor.id!r} requires opponents for 4p evaluation")
    rng = random.Random((seed + 1) * 104_729 + sum(ord(ch) for ch in anchor.id))
    pool = [opponent for opponent in opponents if opponent.id != anchor.id]
    if not pool:
        raise ValueError(f"agent {anchor.id!r} has no distinct opponents")
    picks: list[AgentSpec] = []
    while len(picks) < count:
        candidate = rng.choice(pool)
        if candidate.id == anchor.id:
            continue
        picks.append(candidate)
    return picks


def _bucket_for_match(players: list[AgentSpec], player_index: int) -> str:
    other_roles = {player.role for idx, player in enumerate(players) if idx != player_index}
    return "hall_of_fame" if "hall_of_fame" in other_roles else "league"


def _run_match(players: list[AgentSpec], seed: int, cfg: EvaluationConfig) -> tuple[list[float], dict[str, dict[str, float]], list[float]]:
    runtime = {spec.id: _policy_runtime(spec) for spec in players}
    backend = RustBatchBackend(
        num_envs=1,
        num_players=len(players),
        seed=seed,
        config=RustConfig(episode_steps=cfg.episode_steps, enable_comets=cfg.enable_comets),
    )
    state = backend.reset(seed)[0]
    per_player: dict[str, dict[str, float]] = {
        spec.id: {
            "crashes": 0.0,
            "timeouts": 0.0,
            "invalid_actions": 0.0,
            "decision_turns": 0.0,
            "launched_ships": 0.0,
            "launched_moves": 0.0,
            "neutral_captures": 0.0,
            "enemy_captures": 0.0,
        }
        for spec in players
    }
    outcome = {"scores": [0.0 for _ in players], "done": False}

    while True:
        actions = [[] for _ in players]
        for idx, spec in enumerate(players):
            stats = per_player[spec.id]
            stats["decision_turns"] += 1.0
            try:
                start = perf_counter()
                moves = runtime[spec.id](state, idx)
                elapsed = perf_counter() - start
                if elapsed > cfg.act_timeout:
                    stats["timeouts"] += 1.0
                    moves = []
                if not isinstance(moves, list) or not _moves_are_legal(state, idx, moves):
                    stats["invalid_actions"] += 1.0
                    moves = []
            except Exception:
                stats["crashes"] += 1.0
                moves = []
            stats["launched_moves"] += float(len(moves))
            stats["launched_ships"] += float(sum(int(move[2]) for move in moves))
            actions[idx] = moves

        previous_state = state
        outcome = backend.step([actions])[0]
        state = backend.states()[0]
        for idx, spec in enumerate(players):
            neutral_captures, enemy_captures = _capture_counts(previous_state, state, idx)
            per_player[spec.id]["neutral_captures"] += float(neutral_captures)
            per_player[spec.id]["enemy_captures"] += float(enemy_captures)
        if outcome["done"]:
            break

    final_shares = [_final_planet_share(state, idx) for idx in range(len(players))]
    return [float(score) for score in outcome["scores"]], per_player, final_shares


def evaluate_population(manifest: dict[str, list[AgentSpec]], cfg: EvaluationConfig) -> dict[str, Any]:
    population = manifest["population"]
    hall_of_fame = manifest["hall_of_fame"]
    heuristics = manifest["heuristics"]
    fixed_opponents = hall_of_fame + heuristics
    stats = {spec.id: AggregateStats() for spec in population}
    seed_stats = {spec.id: {seed: AggregateStats() for seed in cfg.seeds} for spec in population}
    ratings = {spec.id: float(spec.rating) for spec in population + hall_of_fame + heuristics}

    for seed in cfg.seeds:
        if cfg.include_2p:
            for game_index in range(cfg.games_per_pair):
                for left_idx, left in enumerate(population):
                    for right in population[left_idx + 1 :]:
                        match_seed = seed * 10_000 + game_index * 1_000 + left_idx
                        players = [left, right]
                        scores, per_player, final_shares = _run_match(players, match_seed, cfg)
                        _update_aggregates(players, scores, per_player, final_shares, stats, ratings)
                        _update_seed_aggregates(players, scores, per_player, final_shares, seed_stats, seed)
                    for offset, opponent in enumerate(fixed_opponents):
                        match_seed = seed * 10_000 + game_index * 1_000 + 500 + left_idx * 17 + offset
                        players = [left, opponent]
                        scores, per_player, final_shares = _run_match(players, match_seed, cfg)
                        _update_aggregates(players, scores, per_player, final_shares, stats, ratings)
                        _update_seed_aggregates(players, scores, per_player, final_shares, seed_stats, seed)

        if cfg.include_4p:
            opponent_pool = population + fixed_opponents
            for game_index in range(cfg.games_per_pair):
                for offset, anchor in enumerate(population):
                    sampled = _sample_opponents(anchor, opponent_pool, seed + game_index * 101 + offset, count=3)
                    players = [anchor, *sampled]
                    match_seed = seed * 100_000 + game_index * 1_000 + offset
                    scores, per_player, final_shares = _run_match(players, match_seed, cfg)
                    _update_aggregates(players, scores, per_player, final_shares, stats, ratings)
                    _update_seed_aggregates(players, scores, per_player, final_shares, seed_stats, seed)

    members = []
    for spec in population:
        summary = _summarize_stats(stats[spec.id])
        summary["seed_stats"] = {
            str(seed): _summarize_stats(seed_stats[spec.id][seed])
            for seed in cfg.seeds
        }
        members.append(
            {
                "id": spec.id,
                "kind": spec.kind,
                "role": spec.role,
                "checkpoint": spec.checkpoint,
                "policy": spec.policy,
                "rating": ratings[spec.id],
                "hparams": spec.hparams,
                "decoder": spec.decoder,
                "metrics": summary,
            }
        )
    return {
        "members": members,
        "summary": {
            "population_size": len(population),
            "hall_of_fame_size": len(hall_of_fame),
            "heuristics_size": len(heuristics),
            "include_2p": cfg.include_2p,
            "include_4p": cfg.include_4p,
            "games_per_pair": cfg.games_per_pair,
            "seeds": cfg.seeds,
        },
    }


def _update_aggregates(
    players: list[AgentSpec],
    scores: list[float],
    per_player: dict[str, dict[str, float]],
    final_shares: list[float],
    stats: dict[str, AggregateStats],
    ratings: dict[str, float],
) -> None:
    _update_member_stats(players, scores, per_player, final_shares, stats)
    for left_idx, left in enumerate(players):
        for right_idx, right in enumerate(players[left_idx + 1 :], start=left_idx + 1):
            left_rating, right_rating = update_elo(
                EloRating(ratings[left.id]),
                EloRating(ratings[right.id]),
                _pairwise_score(scores[left_idx], scores[right_idx]),
            )
            ratings[left.id] = left_rating.rating
            ratings[right.id] = right_rating.rating


def _update_seed_aggregates(
    players: list[AgentSpec],
    scores: list[float],
    per_player: dict[str, dict[str, float]],
    final_shares: list[float],
    seed_stats: dict[str, dict[int, AggregateStats]],
    seed: int,
) -> None:
    scoped_stats = {member_id: seed_map[seed] for member_id, seed_map in seed_stats.items()}
    _update_member_stats(players, scores, per_player, final_shares, scoped_stats)


def _update_member_stats(
    players: list[AgentSpec],
    scores: list[float],
    per_player: dict[str, dict[str, float]],
    final_shares: list[float],
    stats: dict[str, AggregateStats],
) -> None:
    max_score = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == max_score]
    for idx, spec in enumerate(players):
        if spec.id not in stats:
            continue
        bucket = _bucket_for_match(players, idx)
        player_stats = stats[spec.id]
        player_stats.games += 1
        if bucket == "hall_of_fame":
            player_stats.hall_of_fame_games += 1
            player_stats.win_points_vs_hall_of_fame += 1.0 / len(winners) if idx in winners else 0.0
        else:
            player_stats.league_games += 1
            player_stats.win_points_vs_league += 1.0 / len(winners) if idx in winners else 0.0
        player_stats.normalized_margins.append(_normalized_margin(scores, idx))
        player_stats.crashes += int(per_player[spec.id]["crashes"])
        player_stats.timeouts += int(per_player[spec.id]["timeouts"])
        player_stats.invalid_actions += int(per_player[spec.id]["invalid_actions"])
        player_stats.decision_turns += int(per_player[spec.id]["decision_turns"])
        player_stats.launched_ships += int(per_player[spec.id]["launched_ships"])
        player_stats.launched_moves += int(per_player[spec.id]["launched_moves"])
        player_stats.neutral_captures += int(per_player[spec.id]["neutral_captures"])
        player_stats.enemy_captures += int(per_player[spec.id]["enemy_captures"])
        player_stats.final_planet_shares.append(final_shares[idx])


def _pairwise_score(left: float, right: float) -> float:
    if left > right:
        return 1.0
    if left < right:
        return 0.0
    return 0.5


def _summarize_stats(stats: AggregateStats) -> dict[str, float]:
    league_games = max(stats.league_games, 1)
    hall_of_fame_games = max(stats.hall_of_fame_games, 1)
    decision_turns = max(stats.decision_turns, 1)
    games = max(stats.games, 1)
    launched_moves = max(stats.launched_moves, 1)
    normalized_margin = fmean(stats.normalized_margins) if stats.normalized_margins else 0.0
    robustness = max(0.0, 1.0 - min(1.0, pstdev(stats.normalized_margins) if len(stats.normalized_margins) > 1 else 0.0))
    neutral_capture_rate = stats.neutral_captures / games
    enemy_capture_rate = stats.enemy_captures / games
    mean_planet_share = fmean(stats.final_planet_shares) if stats.final_planet_shares else 0.0
    attack_density = stats.launched_ships / launched_moves
    novelty = min(1.0, 0.4 * neutral_capture_rate + 0.4 * enemy_capture_rate + 0.2 * min(1.0, attack_density / 32.0))
    return {
        "win_rate_vs_league": stats.win_points_vs_league / league_games,
        "normalized_score_margin": normalized_margin,
        "win_rate_vs_hall_of_fame": stats.win_points_vs_hall_of_fame / hall_of_fame_games if stats.hall_of_fame_games else 0.0,
        "robustness_across_seeds": robustness,
        "novelty_bonus": novelty,
        "crash_rate": stats.crashes / games,
        "timeout_rate": stats.timeouts / decision_turns,
        "invalid_action_rate": stats.invalid_actions / decision_turns,
        "expansion": min(1.0, mean_planet_share + 0.1 * neutral_capture_rate),
        "aggression": min(1.0, enemy_capture_rate + 0.1 * min(1.0, attack_density / 32.0)),
        "defense": max(0.0, 1.0 - stats.crashes / games - 0.5 * stats.invalid_actions / decision_turns),
        "fleet_size": min(1.0, attack_density / 64.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/eval_final.yaml")
    parser.add_argument("--hall-of-fame-state")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_population_manifest(args.manifest)
    manifest = attach_hall_of_fame_snapshots(manifest, args.hall_of_fame_state)
    cfg = load_evaluation_config(args.config)
    report = evaluate_population(manifest, cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("[bold]Population evaluation[/bold]")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
