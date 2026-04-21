from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from python.league.evaluation import load_population_manifest
from python.train.final_selection import (
    _analyze_bad_replay,
    _worst_decile,
    load_final_selection_config,
    run_final_selection,
)


def _write_manifest(path: Path, policies: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "population": [
                    {
                        "id": f"{policy}_candidate",
                        "kind": "heuristic",
                        "policy": policy,
                        "role": "league",
                    }
                    for policy in policies
                ],
                "hall_of_fame": [],
                "heuristics": [],
            }
        ),
        encoding="utf-8",
    )


def test_final_selection_uses_retained_seeds_worst_decile_and_exports_two_candidates(tmp_path: Path):
    manifest_path = tmp_path / "population.yaml"
    config_path = tmp_path / "final_selection.yaml"
    out_dir = tmp_path / "out"
    _write_manifest(manifest_path, ["greedy", "defensive", "rush"])
    config_path.write_text(
        yaml.safe_dump(
            {
                "final_selection": {
                    "retained_seeds": [3, 7],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": False,
                    "episode_steps": 16,
                    "enable_comets": False,
                    "bad_replay_count": 2,
                    "export_top_k": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_final_selection(
        load_population_manifest(manifest_path),
        load_final_selection_config(config_path),
        out_dir=out_dir,
    )

    assert report["summary"]["retained_seeds"] == [3, 7]
    assert report["summary"]["match_counts"]["2p_matches"] == 6
    assert report["summary"]["match_counts"]["4p_matches"] == 0
    assert len(report["ranking"]) == 3
    assert all("worst_decile_score_margin" in item for item in report["ranking"])
    assert len(report["bad_replays"]) == 2
    assert len(report["exports"]) == 2
    assert [item["candidate_id"] for item in report["exports"]] == [
        report["ranking"][0]["candidate_id"],
        report["ranking"][1]["candidate_id"],
    ]
    assert all(any(char.isdigit() for char in replay["analysis"]) for replay in report["bad_replays"])
    assert (out_dir / "candidate_1_submission.py").exists()
    assert (out_dir / "candidate_2_submission.py").exists()
    saved_report = json.loads((out_dir / "final_selection_report.json").read_text(encoding="utf-8"))
    assert saved_report["summary"]["candidate_count"] == 3


def test_final_selection_supports_four_player_round_robin(tmp_path: Path):
    manifest_path = tmp_path / "population.yaml"
    config_path = tmp_path / "final_selection.yaml"
    out_dir = tmp_path / "out"
    _write_manifest(manifest_path, ["greedy", "defensive", "rush", "anti_meta"])
    config_path.write_text(
        yaml.safe_dump(
            {
                "final_selection": {
                    "retained_seeds": [5],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": True,
                    "episode_steps": 16,
                    "enable_comets": False,
                    "bad_replay_count": 1,
                    "export_top_k": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_final_selection(
        load_population_manifest(manifest_path),
        load_final_selection_config(config_path),
        out_dir=out_dir,
    )

    assert report["summary"]["match_counts"]["2p_matches"] == 6
    assert report["summary"]["match_counts"]["4p_matches"] == 1
    assert any(replay["mode"] == "4p" for replay in report["bad_replays"])


def test_final_selection_worst_decile_and_trace_analysis_are_concrete():
    assert _worst_decile([-0.9, -0.7, -0.2, 0.1, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]) == -0.9
    assert _worst_decile([-0.9, -0.7, -0.2, 0.1, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0] * 2) == -0.9

    record = {
        "player_index": 0,
        "scores": [0.0, 12.0],
        "normalized_margin": -0.75,
        "trace": [
            {"scores": [12.0, 10.0], "move_counts": [1, 1], "launched_ships": [8, 6]},
            {"scores": [0.0, 14.0], "move_counts": [0, 1], "launched_ships": [0, 4]},
        ],
    }
    analysis = _analyze_bad_replay(record)

    assert "pico=12.0" in analysis
    assert "zerou" in analysis


def test_final_selection_requires_exactly_two_exports(tmp_path: Path):
    config_path = tmp_path / "final_selection.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "final_selection": {
                    "retained_seeds": [5],
                    "games_per_pair": 1,
                    "include_2p": True,
                    "include_4p": False,
                    "episode_steps": 16,
                    "enable_comets": False,
                    "bad_replay_count": 1,
                    "export_top_k": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="export_top_k == 2"):
        load_final_selection_config(config_path)
