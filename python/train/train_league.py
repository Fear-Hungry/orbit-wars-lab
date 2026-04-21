from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from python.league.evaluation import load_evaluation_report, member_from_evaluation_entry
from python.league.hall_of_fame import (
    HallOfFame,
    HallOfFameEntry,
    load_hall_of_fame,
    save_hall_of_fame,
)
from python.league.map_elites import (
    Behavior,
    MapElitesArchive,
    load_map_elites_archive,
    save_map_elites_archive,
)
from python.league.matchmaking import make_elo_diverse_pairs
from python.league.pbt import Member, exploit_explore
from rich import print


@dataclass
class LeagueIterationResult:
    archive_cells: int
    competitive_population: dict[str, list[str]]
    hall_of_fame_size: int
    hall_of_fame_ids: list[str]
    opponent_pool: dict[str, list[str]]
    pairs: list[tuple[str, str]]
    population_size: int
    promotion_decisions: list[dict[str, Any]]
    promoted_ids: list[str]
    mutated_ids: list[str]


def load_population_from_report(report_path: str | Path, cfg: dict[str, Any]) -> list[Member]:
    payload = load_evaluation_report(report_path)
    return [member_from_evaluation_entry(record, cfg["fitness"]) for record in payload["members"]]


def derive_behavior(member: Member, bins: dict[str, int]) -> Behavior:
    def bucket(metric: str) -> int:
        total_bins = max(1, int(bins[metric]))
        value = float(member.metrics.get(metric, 0.0))
        return min(total_bins - 1, max(0, int(value * total_bins)))

    return Behavior(
        expansion_bin=bucket("expansion"),
        aggression_bin=bucket("aggression"),
        defense_bin=bucket("defense"),
        fleet_size_bin=bucket("fleet_size"),
    )


def build_opponent_pool(
    population: list[Member],
    hall_of_fame: HallOfFame,
    archive: MapElitesArchive,
    cfg: dict[str, Any],
) -> dict[str, list[str]]:
    sample_size = int(cfg["hall_of_fame"]["sample_size"])
    map_elites_sample_size = int(cfg["map_elites"].get("sample_size", sample_size))
    heuristics = list(cfg.get("heuristics", ["greedy", "defensive", "rush"]))
    return {
        "current_population": [member.id for member in population[: max(1, min(4, len(population)))]],
        "hall_of_fame": [entry.member_id for entry in hall_of_fame.sample(sample_size)],
        "map_elites": [str(payload) for payload in archive.elites(limit=map_elites_sample_size)],
        "heuristics": heuristics,
    }


def build_competitive_population(
    population: list[Member],
    mutated_ids: list[str],
    cfg: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        "ppo_policies": [member.id for member in population],
        "mutants": mutated_ids,
        "heuristics": list(cfg.get("heuristics", ["greedy", "defensive", "rush"])),
    }


def _behavior_cell_is_empty(archive: MapElitesArchive, behavior: Behavior) -> bool:
    return behavior not in archive.cells


def _best_hall_of_fame_entry(entries: list[HallOfFameEntry]) -> HallOfFameEntry | None:
    if not entries:
        return None
    return max(entries, key=lambda entry: (entry.fitness, entry.rating, entry.generation))


def _baseline_entry_for_promotion(
    behavior: Behavior,
    hall_of_fame: HallOfFame,
    archive: MapElitesArchive,
) -> HallOfFameEntry | None:
    hof_by_id = {entry.member_id: entry for entry in hall_of_fame.entries}
    archived = archive.cells.get(behavior)
    if archived is not None:
        payload = str(archived[1])
        archived_entry = hof_by_id.get(payload)
        if archived_entry is not None:
            return archived_entry

    matching_behavior = [entry for entry in hall_of_fame.entries if entry.behavior == behavior]
    if matching_behavior:
        return _best_hall_of_fame_entry(matching_behavior)
    return _best_hall_of_fame_entry(hall_of_fame.entries)


def _seed_margin_profile(metrics: dict[str, Any]) -> dict[int, float]:
    seed_stats = metrics.get("seed_stats", {})
    if not isinstance(seed_stats, dict):
        return {}
    profile: dict[int, float] = {}
    for seed, summary in seed_stats.items():
        if not isinstance(summary, dict):
            continue
        try:
            profile[int(seed)] = float(summary.get("normalized_score_margin", 0.0))
        except (TypeError, ValueError):
            continue
    return profile


def _seed_gate(
    candidate: Member,
    baseline: HallOfFameEntry | None,
    cfg: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    threshold = float(cfg["hall_of_fame"].get("min_seed_margin_improvement", 0.0))
    min_fraction = float(cfg["hall_of_fame"].get("min_seed_improved_fraction", 0.5))
    if baseline is None:
        return True, {
            "baseline_id": None,
            "mean_seed_margin_delta": None,
            "seed_improved_fraction": None,
            "used_fallback": False,
        }

    candidate_profile = _seed_margin_profile(candidate.metrics)
    baseline_profile = _seed_margin_profile(baseline.metrics)
    shared_seeds = sorted(set(candidate_profile) & set(baseline_profile))
    if shared_seeds:
        deltas = [candidate_profile[seed] - baseline_profile[seed] for seed in shared_seeds]
        mean_delta = sum(deltas) / len(deltas)
        improved_fraction = sum(delta >= threshold for delta in deltas) / len(deltas)
        return mean_delta >= threshold and improved_fraction >= min_fraction, {
            "baseline_id": baseline.member_id,
            "mean_seed_margin_delta": mean_delta,
            "seed_improved_fraction": improved_fraction,
            "used_fallback": False,
        }

    return False, {
        "baseline_id": baseline.member_id,
        "mean_seed_margin_delta": None,
        "seed_improved_fraction": 0.0,
        "used_fallback": False,
    }


def _promotion_decision(
    member: Member,
    behavior: Behavior,
    hall_of_fame: HallOfFame,
    archive: MapElitesArchive,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    baseline = _baseline_entry_for_promotion(behavior, hall_of_fame, archive)
    league_win_rate = float(member.metrics.get("win_rate_vs_league", 0.0))
    crash_rate = float(member.metrics.get("crash_rate", 0.0))
    timeout_rate = float(member.metrics.get("timeout_rate", 0.0))
    baseline_crash_rate = float((baseline.metrics if baseline else {}).get("crash_rate", 0.0))
    baseline_timeout_rate = float((baseline.metrics if baseline else {}).get("timeout_rate", 0.0))
    empty_cell = _behavior_cell_is_empty(archive, behavior)
    league_gate = league_win_rate >= float(cfg["hall_of_fame"].get("min_league_win_rate", 0.5))
    safety_gate = (
        crash_rate <= baseline_crash_rate + float(cfg["hall_of_fame"].get("max_crash_rate_increase", 0.0))
        and timeout_rate <= baseline_timeout_rate + float(cfg["hall_of_fame"].get("max_timeout_rate_increase", 0.0))
    )
    seed_gate, seed_details = _seed_gate(member, baseline, cfg)
    promoted = league_gate and safety_gate and (seed_gate or empty_cell)
    return {
        "member_id": member.id,
        "baseline_id": seed_details["baseline_id"],
        "league_win_rate": league_win_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
        "baseline_crash_rate": baseline_crash_rate,
        "baseline_timeout_rate": baseline_timeout_rate,
        "empty_map_elites_cell": empty_cell,
        "league_gate": league_gate,
        "safety_gate": safety_gate,
        "seed_gate": seed_gate,
        "seed_gate_used_fallback": seed_details["used_fallback"],
        "mean_seed_margin_delta": seed_details["mean_seed_margin_delta"],
        "seed_improved_fraction": seed_details["seed_improved_fraction"],
        "promoted": promoted,
    }


def run_league_iteration(
    population: list[Member],
    cfg: dict[str, Any],
    *,
    generation: int,
    hall_of_fame: HallOfFame | None = None,
    archive: MapElitesArchive | None = None,
) -> tuple[list[Member], HallOfFame, MapElitesArchive, LeagueIterationResult]:
    if not population:
        raise ValueError("league iteration requires at least one evaluated member")
    if hall_of_fame is None:
        hall_of_fame = HallOfFame(max_size=int(cfg["population"]["hall_of_fame_size"]))
    if archive is None:
        archive = MapElitesArchive()

    elite_frac = float(cfg["population"]["elite_frac"])
    promote_top_k = int(cfg["hall_of_fame"]["promote_top_k"])
    map_elites_enabled = bool(cfg["map_elites"].get("enabled", True))
    mutate_hparams_enabled = bool(cfg["pbt"].get("mutate_hparams", True))
    mutate_decoder_enabled = bool(cfg["pbt"].get("mutate_decoder", True))
    ranked = sorted(population, key=lambda member: member.fitness, reverse=True)
    bins = cfg["map_elites"]["bins"]

    promoted_ids: list[str] = []
    promotion_decisions: list[dict[str, Any]] = []
    for member in ranked[:promote_top_k]:
        behavior = derive_behavior(member, bins)
        decision = _promotion_decision(member, behavior, hall_of_fame, archive, cfg)
        promotion_decisions.append(decision)
        if decision["promoted"]:
            if map_elites_enabled:
                archive.insert(behavior, member.fitness, member.id)
            hall_of_fame.add(
                HallOfFameEntry(
                    member_id=member.id,
                    checkpoint=member.checkpoint,
                    fitness=member.fitness,
                    rating=member.rating,
                    generation=generation,
                    behavior=behavior,
                    decoder=dict(member.decoder),
                    metrics=dict(member.metrics),
                )
            )
            promoted_ids.append(member.id)

    if map_elites_enabled:
        promoted_set = set(promoted_ids)
        for member in ranked:
            if member.id in promoted_set:
                continue
            archive.insert(derive_behavior(member, bins), member.fitness, member.id)

    behaviors = {member.id: derive_behavior(member, bins) for member in ranked}
    pairs = make_elo_diverse_pairs(
        [member.id for member in ranked],
        {member.id: member.rating for member in ranked},
        behaviors,
        pairs_per_agent=int(cfg["matchmaking"]["pairs_per_agent"]),
        seed=generation,
    )
    opponent_pool = build_opponent_pool(ranked, hall_of_fame, archive, cfg)
    next_population = exploit_explore(
        ranked,
        elite_frac=elite_frac,
        seed=generation,
        mutate_hparams_enabled=mutate_hparams_enabled,
        mutate_decoder_enabled=mutate_decoder_enabled,
    )
    mutated_ids = [member.id for member in next_population if member.parent_id is not None]
    competitive_population = build_competitive_population(next_population, mutated_ids, cfg)

    summary = LeagueIterationResult(
        archive_cells=archive.cell_count(),
        competitive_population=competitive_population,
        hall_of_fame_size=len(hall_of_fame.entries),
        hall_of_fame_ids=[entry.member_id for entry in hall_of_fame.entries],
        opponent_pool=opponent_pool,
        pairs=pairs,
        population_size=len(next_population),
        promotion_decisions=promotion_decisions,
        promoted_ids=promoted_ids,
        mutated_ids=mutated_ids,
    )
    return next_population, hall_of_fame, archive, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/league.yaml")
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--hall-of-fame-state")
    parser.add_argument("--map-elites-state")
    parser.add_argument("--generation", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    population = load_population_from_report(args.evaluation_report, cfg)
    hall_of_fame = None
    archive = None
    if args.hall_of_fame_state:
        hall_of_fame = load_hall_of_fame(
            args.hall_of_fame_state,
            max_size=int(cfg["population"]["hall_of_fame_size"]),
        )
    if args.map_elites_state:
        archive = load_map_elites_archive(args.map_elites_state)
    _, hall_of_fame, archive, summary = run_league_iteration(
        population,
        cfg,
        generation=args.generation,
        hall_of_fame=hall_of_fame,
        archive=archive,
    )
    if args.hall_of_fame_state:
        save_hall_of_fame(hall_of_fame, args.hall_of_fame_state)
    if args.map_elites_state:
        save_map_elites_archive(archive, args.map_elites_state)

    print("[bold]League iteration[/bold]")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
