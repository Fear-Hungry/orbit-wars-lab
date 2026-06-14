from __future__ import annotations

from pathlib import Path

import yaml
from python.league.evaluation import load_population_manifest
from python.train.final_selection import load_final_selection_config
from python.train.objective_validation import (
    _export_runtime_validation,
    load_objective_validation_config,
    run_objective_validation,
)


def test_objective_validation_exports_two_candidates_and_checks_self_play(tmp_path: Path):
    manifest_path = tmp_path / "population.yaml"
    selection_config_path = tmp_path / "final_selection.yaml"
    validation_config_path = tmp_path / "objective_validation.yaml"
    out_dir = tmp_path / "out"

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "population": [
                    {"id": "greedy_candidate", "kind": "heuristic", "policy": "greedy", "role": "league"},
                    {"id": "defensive_candidate", "kind": "heuristic", "policy": "defensive", "role": "league"},
                    {"id": "rush_candidate", "kind": "heuristic", "policy": "rush", "role": "league"},
                    {"id": "anti_meta_candidate", "kind": "heuristic", "policy": "anti_meta", "role": "league"},
                ],
                "hall_of_fame": [],
                "heuristics": [],
            }
        ),
        encoding="utf-8",
    )
    selection_config_path.write_text(
        yaml.safe_dump(
            {
                "final_selection": {
                    "retained_seeds": [3],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": True,
                    "episode_steps": 16,
                    "enable_comets": False,
                    "bad_replay_count": 2,
                    "export_top_k": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    validation_config_path.write_text(
        yaml.safe_dump(
            {
                "objective_validation": {
                    "max_crash_rate": 0.0,
                    "max_timeout_rate": 0.0,
                    "max_invalid_action_rate": 0.25,
                    "min_selection_win_rate": 0.0,
                    "min_mean_score_margin": -1.0,
                    "min_worst_decile_score_margin": -1.0,
                    "require_2p_self_play": True,
                    "require_4p_self_play": True,
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_objective_validation(
        load_population_manifest(manifest_path),
        load_final_selection_config(selection_config_path),
        load_objective_validation_config(validation_config_path),
        out_dir=out_dir,
    )

    assert report["objective_ready"] is True
    assert len(report["exports"]) == 2
    assert len(report["candidate_checks"]) == 2
    assert all(check["self_play_ok"] for check in report["candidate_checks"])
    assert all(check["selection_win_rate_ok"] for check in report["candidate_checks"])
    assert all(check["mean_score_margin_ok"] for check in report["candidate_checks"])
    assert (out_dir / "candidate_1_submission.py").exists()
    assert (out_dir / "candidate_2_submission.py").exists()
    assert (out_dir / "objective_validation_report.json").exists()


def test_objective_runtime_validation_stops_crashed_seats(tmp_path: Path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """
def agent(obs):
    raise RuntimeError("boom")
""",
        encoding="utf-8",
    )

    report = _export_runtime_validation(
        candidate,
        num_players=2,
        seeds=[0],
        episode_steps=8,
        enable_comets=False,
    )

    assert report["decision_turns"] == 2
    assert report["crash_rate"] == 1.0
