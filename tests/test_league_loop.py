from __future__ import annotations

import json
from pathlib import Path

import torch
from python.agents.policy import FlatActorCritic
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
from python.league.pbt import Member, exploit_explore
from python.orbit_wars_gym.encoding import observation_dim
from python.train.competitive_cycle import (
    CompetitiveCycleConfig,
    manifest_from_members,
    run_competitive_cycle,
    update_trainable_population,
)
from python.train.evaluate_population import EvaluationConfig
from python.train.train_league import load_population_from_report, run_league_iteration


def _cfg() -> dict:
    return {
        "population": {"elite_frac": 0.5, "hall_of_fame_size": 8, "size": 4},
        "hall_of_fame": {
            "promote_top_k": 2,
            "sample_size": 2,
            "min_league_win_rate": 0.5,
            "max_crash_rate_increase": 0.0,
            "max_timeout_rate_increase": 0.0,
            "min_seed_margin_improvement": 0.0,
            "min_seed_improved_fraction": 0.5,
        },
        "matchmaking": {"pairs_per_agent": 2},
        "pbt": {"mutate_hparams": True, "mutate_decoder": True},
        "map_elites": {"enabled": True, "sample_size": 2, "bins": {"expansion": 4, "aggression": 4, "defense": 4, "fleet_size": 4}},
        "heuristics": ["greedy", "defensive", "rush", "anti_meta", "weak_random"],
        "fitness": {
            "win_rate": 2.0,
            "score_margin": 0.8,
            "hof_win_rate": 0.5,
            "robustness": 0.25,
            "novelty": 0.20,
            "crash_penalty": 1.0,
            "invalid_action_penalty": 0.20,
        },
    }


def _member(idx: int, fitness: float, checkpoint: str) -> Member:
    return Member(
        id=f"agent_{idx:03d}",
        fitness=fitness,
        rating=1000.0 + fitness * 100.0,
        checkpoint=checkpoint,
        hparams={"learning_rate": 2.5e-4, "gamma": 0.99, "gae_lambda": 0.95},
        decoder={
            "reserve_home_ships": 8,
            "min_ships_to_launch": 2,
            "max_moves_per_turn": 8,
            "fractions": [0.10, 0.25, 0.50, 0.75],
            "angle_offsets": [-0.26, -0.13, 0.0, 0.13, 0.26],
        },
        metrics={
            "win_rate_vs_league": fitness,
            "normalized_score_margin": 0.1 * idx,
            "win_rate_vs_hall_of_fame": 0.2,
            "robustness_across_seeds": 0.7,
            "novelty_bonus": 0.1,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
            "invalid_action_rate": 0.0,
            "expansion": 0.2 * idx,
            "aggression": 0.1 * idx,
            "defense": 0.15 * idx,
            "fleet_size": 0.25 * idx,
            "seed_stats": {
                "1": {"normalized_score_margin": 0.05 * idx},
                "2": {"normalized_score_margin": 0.05 * idx + 0.02},
            },
        },
    )


def _write_report(path: Path) -> Path:
    payload = {
        "members": [
            {
                "id": "agent_000",
                "kind": "ppo",
                "checkpoint": "agent_0.pt",
                "rating": 1040.0,
                "hparams": {"learning_rate": 2.5e-4, "entropy_coef": 0.01, "clip_coef": 0.2, "gamma": 0.99, "gae_lambda": 0.95},
                "decoder": {
                    "reserve_home_ships": 8,
                    "min_ships_to_launch": 2,
                    "max_moves_per_turn": 8,
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.26, -0.13, 0.0, 0.13, 0.26],
                },
                "metrics": {
                    "win_rate_vs_league": 0.90,
                    "normalized_score_margin": 0.25,
                    "win_rate_vs_hall_of_fame": 0.50,
                    "robustness_across_seeds": 0.90,
                    "novelty_bonus": 0.10,
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "invalid_action_rate": 0.0,
                    "expansion": 0.2,
                    "aggression": 0.3,
                    "defense": 0.4,
                    "fleet_size": 0.5,
                    "seed_stats": {
                        "1": {"normalized_score_margin": 0.22},
                        "2": {"normalized_score_margin": 0.28},
                    },
                },
            },
            {
                "id": "agent_001",
                "kind": "ppo",
                "checkpoint": "agent_1.pt",
                "rating": 1020.0,
                "hparams": {"learning_rate": 2.8e-4, "entropy_coef": 0.01, "clip_coef": 0.2, "gamma": 0.99, "gae_lambda": 0.95},
                "decoder": {
                    "reserve_home_ships": 8,
                    "min_ships_to_launch": 2,
                    "max_moves_per_turn": 8,
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.26, -0.13, 0.0, 0.13, 0.26],
                },
                "metrics": {
                    "win_rate_vs_league": 0.75,
                    "normalized_score_margin": 0.20,
                    "win_rate_vs_hall_of_fame": 0.45,
                    "robustness_across_seeds": 0.80,
                    "novelty_bonus": 0.12,
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "invalid_action_rate": 0.0,
                    "expansion": 0.5,
                    "aggression": 0.2,
                    "defense": 0.3,
                    "fleet_size": 0.4,
                    "seed_stats": {
                        "1": {"normalized_score_margin": 0.18},
                        "2": {"normalized_score_margin": 0.21},
                    },
                },
            },
            {
                "id": "agent_002",
                "kind": "ppo",
                "checkpoint": "agent_2.pt",
                "rating": 980.0,
                "hparams": {"learning_rate": 3.0e-4, "entropy_coef": 0.01, "clip_coef": 0.2, "gamma": 0.99, "gae_lambda": 0.95},
                "decoder": {
                    "reserve_home_ships": 8,
                    "min_ships_to_launch": 2,
                    "max_moves_per_turn": 8,
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.26, -0.13, 0.0, 0.13, 0.26],
                },
                "metrics": {
                    "win_rate_vs_league": 0.40,
                    "normalized_score_margin": -0.05,
                    "win_rate_vs_hall_of_fame": 0.15,
                    "robustness_across_seeds": 0.55,
                    "novelty_bonus": 0.05,
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "invalid_action_rate": 0.0,
                    "expansion": 0.1,
                    "aggression": 0.1,
                    "defense": 0.2,
                    "fleet_size": 0.2,
                    "seed_stats": {
                        "1": {"normalized_score_margin": -0.03},
                        "2": {"normalized_score_margin": -0.07},
                    },
                },
            },
            {
                "id": "agent_003",
                "kind": "ppo",
                "checkpoint": "agent_3.pt",
                "rating": 960.0,
                "hparams": {"learning_rate": 3.2e-4, "entropy_coef": 0.01, "clip_coef": 0.2, "gamma": 0.99, "gae_lambda": 0.95},
                "decoder": {
                    "reserve_home_ships": 8,
                    "min_ships_to_launch": 2,
                    "max_moves_per_turn": 8,
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.26, -0.13, 0.0, 0.13, 0.26],
                },
                "metrics": {
                    "win_rate_vs_league": 0.20,
                    "normalized_score_margin": -0.10,
                    "win_rate_vs_hall_of_fame": 0.05,
                    "robustness_across_seeds": 0.45,
                    "novelty_bonus": 0.02,
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "invalid_action_rate": 0.0,
                    "expansion": 0.0,
                    "aggression": 0.0,
                    "defense": 0.1,
                    "fleet_size": 0.1,
                    "seed_stats": {
                        "1": {"normalized_score_margin": -0.09},
                        "2": {"normalized_score_margin": -0.11},
                    },
                },
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exploit_explore_mutates_only_hparams_and_decoder():
    population = [
        _member(0, fitness=1.0, checkpoint="elite.pt"),
        _member(1, fitness=0.2, checkpoint="loser.pt"),
    ]

    next_population = exploit_explore(population, elite_frac=0.5, seed=3)

    assert len(next_population) == len(population)
    mutated = [member for member in next_population if member.parent_id is not None]
    assert len(mutated) == 1
    child = mutated[0]
    assert child.checkpoint != "elite.pt"
    assert child.checkpoint.endswith(".pt")
    assert child.hparams != population[0].hparams or child.decoder != population[0].decoder


def test_run_league_iteration_consumes_evaluation_report(tmp_path: Path):
    report_path = _write_report(tmp_path / "report.json")
    population = load_population_from_report(report_path, _cfg())

    next_population, hall_of_fame, archive, summary = run_league_iteration(population, _cfg(), generation=2)

    assert len(next_population) == 4
    assert summary.hall_of_fame_size == 2
    assert summary.hall_of_fame_ids == ["agent_000", "agent_001"]
    assert summary.archive_cells >= 2
    assert set(summary.opponent_pool["hall_of_fame"]) == {"agent_000", "agent_001"}
    assert set(summary.opponent_pool["heuristics"]) >= {"anti_meta", "weak_random"}
    assert len(summary.opponent_pool["map_elites"]) == 2
    assert set(summary.competitive_population["heuristics"]) >= {"anti_meta", "weak_random"}
    assert set(summary.competitive_population["mutants"]) == set(summary.mutated_ids)
    assert len(summary.mutated_ids) == 2
    assert len(summary.promotion_decisions) == 2
    assert all(decision["promoted"] for decision in summary.promotion_decisions)
    assert len(hall_of_fame.entries) == 2
    assert all(isinstance(entry.behavior, Behavior) for entry in hall_of_fame.entries)
    assert archive.cell_count() == summary.archive_cells
    assert population[0].fitness > population[-1].fitness


def test_hall_of_fame_persists_across_iterations(tmp_path: Path):
    report_path = _write_report(tmp_path / "report.json")
    hall_path = tmp_path / "hall_of_fame.json"
    baseline_hof = HallOfFame(max_size=8)
    baseline_hof.add(
        HallOfFameEntry(
            member_id="legacy_champion",
            checkpoint="legacy.pt",
            fitness=3.0,
            rating=1110.0,
            generation=1,
            behavior=Behavior(expansion_bin=0, aggression_bin=1, defense_bin=2, fleet_size_bin=3),
        )
    )
    save_hall_of_fame(baseline_hof, hall_path)

    population = load_population_from_report(report_path, _cfg())
    persisted_hof = load_hall_of_fame(hall_path, max_size=8)
    _, hall_of_fame, _, summary = run_league_iteration(population, _cfg(), generation=2, hall_of_fame=persisted_hof)
    save_hall_of_fame(hall_of_fame, hall_path)
    reloaded = load_hall_of_fame(hall_path, max_size=8)

    assert reloaded.entries[0].member_id == "legacy_champion"
    assert "legacy_champion" in summary.opponent_pool["hall_of_fame"]
    assert {entry.member_id for entry in reloaded.entries} >= {"legacy_champion", "agent_000", "agent_001"}


def test_map_elites_persists_across_iterations(tmp_path: Path):
    report_path = _write_report(tmp_path / "report.json")
    archive_path = tmp_path / "map_elites.json"
    archive = MapElitesArchive()
    archive.insert(Behavior(expansion_bin=3, aggression_bin=3, defense_bin=3, fleet_size_bin=3), 9.0, "rare_style")
    save_map_elites_archive(archive, archive_path)

    population = load_population_from_report(report_path, _cfg())
    persisted_archive = load_map_elites_archive(archive_path)
    _, _, archive, summary = run_league_iteration(population, _cfg(), generation=3, archive=persisted_archive)
    save_map_elites_archive(archive, archive_path)
    reloaded = load_map_elites_archive(archive_path)

    assert "rare_style" in summary.opponent_pool["map_elites"]
    assert reloaded.cell_count() >= 1


def test_update_trainable_population_runs_real_ppo(tmp_path: Path):
    population = [_member(0, fitness=1.0, checkpoint="initial.pt")]

    updated = update_trainable_population(
        population,
        CompetitiveCycleConfig(iterations=1, total_timesteps_per_member=16, rollout_steps=8),
        output_dir=tmp_path / "checkpoints",
    )

    assert updated[0].checkpoint is not None
    assert Path(updated[0].checkpoint).exists()


def test_competitive_cycle_repeats_with_trainable_population_and_heuristics(tmp_path: Path):
    checkpoint_a = tmp_path / "initial.pt"
    checkpoint_b = tmp_path / "initial_b.pt"
    torch.save({"model_state_dict": FlatActorCritic(observation_dim()).state_dict()}, checkpoint_a)
    torch.save({"model_state_dict": FlatActorCritic(observation_dim()).state_dict()}, checkpoint_b)
    population = [_member(0, fitness=1.0, checkpoint=str(checkpoint_a)), _member(1, fitness=0.8, checkpoint=str(checkpoint_b))]
    manifest = manifest_from_members(population, HallOfFame(max_size=8), ["greedy", "defensive"])
    eval_cfg = EvaluationConfig(
        seeds=[1],
        games_per_pair=1,
        include_2p=True,
        include_4p=False,
        episode_steps=16,
        enable_comets=False,
    )

    result = run_competitive_cycle(
        manifest,
        eval_cfg,
        _cfg(),
        CompetitiveCycleConfig(iterations=2, total_timesteps_per_member=8, rollout_steps=4),
        output_dir=tmp_path / "cycle",
    )

    assert result.iterations == 2
    assert len(result.summaries) == 2
    assert len(result.manifests) == 3
    assert all(summary.competitive_population["heuristics"] == ["greedy", "defensive", "rush", "anti_meta", "weak_random"] for summary in result.summaries)
    assert any(summary.mutated_ids for summary in result.summaries)


def test_league_promotion_requires_league_fraction_and_seed_improvement():
    cfg = _cfg()
    baseline = HallOfFame(
        max_size=8,
        entries=[
            HallOfFameEntry(
                member_id="incumbent",
                checkpoint="incumbent.pt",
                fitness=1.2,
                rating=1100.0,
                generation=1,
                behavior=Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0),
                metrics={
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "normalized_score_margin": 0.12,
                    "seed_stats": {
                        "1": {"normalized_score_margin": 0.10},
                        "2": {"normalized_score_margin": 0.14},
                    },
                },
            )
        ],
    )
    archive = MapElitesArchive()
    archive.insert(Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0), 1.2, "incumbent")
    challenger = _member(0, fitness=0.9, checkpoint="challenger.pt")
    challenger.metrics.update(
        {
            "win_rate_vs_league": 0.45,
            "normalized_score_margin": 0.20,
            "seed_stats": {
                "1": {"normalized_score_margin": 0.18},
                "2": {"normalized_score_margin": 0.22},
            },
        }
    )

    _, hall_of_fame, _, summary = run_league_iteration([challenger], cfg, generation=2, hall_of_fame=baseline, archive=archive)

    decision = summary.promotion_decisions[0]
    assert decision["league_gate"] is False
    assert decision["seed_gate"] is True
    assert decision["promoted"] is False
    assert [entry.member_id for entry in hall_of_fame.entries] == ["incumbent"]


def test_league_promotion_blocks_crash_and_timeout_regressions():
    cfg = _cfg()
    baseline = HallOfFame(
        max_size=8,
        entries=[
            HallOfFameEntry(
                member_id="stable_incumbent",
                checkpoint="stable.pt",
                fitness=1.1,
                rating=1090.0,
                generation=1,
                behavior=Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0),
                metrics={
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "normalized_score_margin": 0.10,
                    "seed_stats": {
                        "1": {"normalized_score_margin": 0.08},
                        "2": {"normalized_score_margin": 0.12},
                    },
                },
            )
        ],
    )
    archive = MapElitesArchive()
    archive.insert(Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0), 1.1, "stable_incumbent")
    risky = _member(0, fitness=0.95, checkpoint="risky.pt")
    risky.metrics.update(
        {
            "win_rate_vs_league": 0.70,
            "crash_rate": 0.10,
            "timeout_rate": 0.05,
            "normalized_score_margin": 0.25,
            "seed_stats": {
                "1": {"normalized_score_margin": 0.20},
                "2": {"normalized_score_margin": 0.30},
            },
        }
    )

    _, hall_of_fame, _, summary = run_league_iteration([risky], cfg, generation=2, hall_of_fame=baseline, archive=archive)

    decision = summary.promotion_decisions[0]
    assert decision["league_gate"] is True
    assert decision["seed_gate"] is True
    assert decision["safety_gate"] is False
    assert decision["promoted"] is False
    assert [entry.member_id for entry in hall_of_fame.entries] == ["stable_incumbent"]


def test_league_promotion_allows_empty_map_elites_cell_without_seed_improvement():
    cfg = _cfg()
    baseline = HallOfFame(
        max_size=8,
        entries=[
            HallOfFameEntry(
                member_id="generalist",
                checkpoint="generalist.pt",
                fitness=1.0,
                rating=1080.0,
                generation=1,
                behavior=Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0),
                metrics={
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "normalized_score_margin": 0.15,
                    "seed_stats": {
                        "1": {"normalized_score_margin": 0.14},
                        "2": {"normalized_score_margin": 0.16},
                    },
                },
            )
        ],
    )
    archive = MapElitesArchive()
    archive.insert(Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0), 1.0, "generalist")
    diverse = _member(3, fitness=0.85, checkpoint="diverse.pt")
    diverse.metrics.update(
        {
            "win_rate_vs_league": 0.65,
            "normalized_score_margin": 0.05,
            "seed_stats": {
                "1": {"normalized_score_margin": 0.01},
                "2": {"normalized_score_margin": 0.04},
            },
        }
    )

    _, hall_of_fame, archive, summary = run_league_iteration([diverse], cfg, generation=2, hall_of_fame=baseline, archive=archive)

    decision = summary.promotion_decisions[0]
    assert decision["league_gate"] is True
    assert decision["safety_gate"] is True
    assert decision["seed_gate"] is False
    assert decision["empty_map_elites_cell"] is True
    assert decision["promoted"] is True
    assert {entry.member_id for entry in hall_of_fame.entries} == {"generalist", "agent_003"}
    assert archive.cell_count() >= 2


def test_league_promotion_rejects_missing_shared_seed_evidence():
    cfg = _cfg()
    baseline = HallOfFame(
        max_size=8,
        entries=[
            HallOfFameEntry(
                member_id="seeded_incumbent",
                checkpoint="seeded.pt",
                fitness=1.0,
                rating=1085.0,
                generation=1,
                behavior=Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0),
                metrics={
                    "crash_rate": 0.0,
                    "timeout_rate": 0.0,
                    "seed_stats": {
                        "7": {"normalized_score_margin": 0.05},
                        "8": {"normalized_score_margin": 0.07},
                    },
                },
            )
        ],
    )
    archive = MapElitesArchive()
    archive.insert(Behavior(expansion_bin=0, aggression_bin=0, defense_bin=0, fleet_size_bin=0), 1.0, "seeded_incumbent")
    challenger = _member(0, fitness=0.9, checkpoint="challenger.pt")
    challenger.metrics.update(
        {
            "win_rate_vs_league": 0.75,
            "normalized_score_margin": 0.40,
            "seed_stats": {
                "1": {"normalized_score_margin": 0.30},
                "2": {"normalized_score_margin": 0.50},
            },
        }
    )

    _, hall_of_fame, _, summary = run_league_iteration([challenger], cfg, generation=2, hall_of_fame=baseline, archive=archive)

    decision = summary.promotion_decisions[0]
    assert decision["league_gate"] is True
    assert decision["safety_gate"] is True
    assert decision["seed_gate"] is False
    assert decision["seed_gate_used_fallback"] is False
    assert decision["promoted"] is False
    assert [entry.member_id for entry in hall_of_fame.entries] == ["seeded_incumbent"]
