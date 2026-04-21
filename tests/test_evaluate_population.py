from __future__ import annotations

from pathlib import Path

import torch
import yaml
from python.agents.policy import FlatActorCritic
from python.league.evaluation import load_population_manifest
from python.league.hall_of_fame import HallOfFame, HallOfFameEntry, save_hall_of_fame
from python.orbit_wars_gym.encoding import observation_dim
from python.train.evaluate_population import (
    attach_hall_of_fame_snapshots,
    evaluate_population,
    load_evaluation_config,
)


def test_evaluate_population_runs_real_backend_matches(tmp_path: Path):
    manifest_path = tmp_path / "population.yaml"
    config_path = tmp_path / "eval.yaml"

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "population": [
                    {
                        "id": "greedy_league",
                        "kind": "heuristic",
                        "policy": "greedy",
                        "role": "league",
                    },
                    {
                        "id": "defensive_league",
                        "kind": "heuristic",
                        "policy": "defensive",
                        "role": "league",
                    },
                ],
                "hall_of_fame": [
                    {
                        "id": "rush_hof",
                        "kind": "heuristic",
                        "policy": "rush",
                        "role": "hall_of_fame",
                    }
                ],
                "heuristics": [
                    {
                        "id": "greedy_probe",
                        "kind": "heuristic",
                        "policy": "greedy",
                        "role": "heuristic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "seeds": [3],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": True,
                    "episode_steps": 32,
                    "enable_comets": False,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_population_manifest(manifest_path)
    cfg = load_evaluation_config(config_path)
    report = evaluate_population(manifest, cfg)

    assert report["summary"]["population_size"] == 2
    assert report["summary"]["include_4p"] is True
    assert {member["id"] for member in report["members"]} == {"greedy_league", "defensive_league"}
    for member in report["members"]:
        metrics = member["metrics"]
        assert 0.0 <= metrics["win_rate_vs_league"] <= 1.0
        assert -1.0 <= metrics["normalized_score_margin"] <= 1.0
        assert 0.0 <= metrics["win_rate_vs_hall_of_fame"] <= 1.0
        assert 0.0 <= metrics["robustness_across_seeds"] <= 1.0
        assert 0.0 <= metrics["novelty_bonus"] <= 1.0
        assert 0.0 <= metrics["timeout_rate"] <= 1.0
        assert 0.0 <= metrics["invalid_action_rate"] <= 1.0
        assert 0.0 <= metrics["expansion"] <= 1.0
        assert 0.0 <= metrics["aggression"] <= 1.0
        assert 0.0 <= metrics["defense"] <= 1.0
        assert 0.0 <= metrics["fleet_size"] <= 1.0
        assert set(metrics["seed_stats"]) == {"3"}
        assert "normalized_score_margin" in metrics["seed_stats"]["3"]


def test_evaluate_population_can_attach_previous_policy_snapshots(tmp_path: Path):
    manifest_path = tmp_path / "population.yaml"
    config_path = tmp_path / "eval.yaml"
    checkpoint_path = tmp_path / "current.pt"
    hall_path = tmp_path / "hall_of_fame.json"

    torch.save({"model_state_dict": FlatActorCritic(observation_dim()).state_dict()}, checkpoint_path)
    save_hall_of_fame(
        HallOfFame(
            max_size=4,
            entries=[
                HallOfFameEntry(
                    member_id="snapshot_prev",
                    checkpoint=str(checkpoint_path),
                    fitness=1.0,
                    rating=1005.0,
                    generation=1,
                    decoder={
                        "reserve_home_ships": 8,
                        "min_ships_to_launch": 2,
                        "max_moves_per_turn": 8,
                        "fractions": [0.10, 0.25, 0.50, 0.75],
                        "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
                    },
                )
            ],
        ),
        hall_path,
    )

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "population": [
                    {
                        "id": "current_policy",
                        "kind": "ppo",
                        "checkpoint": str(checkpoint_path),
                        "role": "league",
                        "hparams": {
                            "learning_rate": 2.5e-4,
                            "entropy_coef": 0.01,
                            "clip_coef": 0.2,
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                        },
                        "decoder": {
                            "reserve_home_ships": 8,
                            "min_ships_to_launch": 2,
                            "max_moves_per_turn": 8,
                            "fractions": [0.10, 0.25, 0.50, 0.75],
                            "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
                        },
                    }
                ],
                "hall_of_fame": [],
                "heuristics": [],
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "seeds": [1],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": False,
                    "episode_steps": 16,
                    "enable_comets": False,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_population_manifest(manifest_path)
    manifest = attach_hall_of_fame_snapshots(manifest, hall_path)
    report = evaluate_population(manifest, load_evaluation_config(config_path))

    assert {member["id"] for member in report["members"]} == {"current_policy"}
    assert report["members"][0]["metrics"]["win_rate_vs_hall_of_fame"] >= 0.0
    assert "seed_stats" in report["members"][0]["metrics"]
