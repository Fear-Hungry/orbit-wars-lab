from __future__ import annotations

import json
from pathlib import Path

import yaml
from python.league.evaluation import AgentSpec, load_population_manifest
from python.train import objective_validation as objective_validation_module
from python.train.final_selection import load_final_selection_config
from python.train.objective_validation import (
    LiveBaselineConfig,
    ObjectiveValidationConfig,
    _same_agent_identity,
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
                    {
                        "id": "greedy_candidate",
                        "kind": "heuristic",
                        "policy": "greedy",
                        "role": "league",
                    },
                    {
                        "id": "defensive_candidate",
                        "kind": "heuristic",
                        "policy": "defensive",
                        "role": "league",
                    },
                    {
                        "id": "rush_candidate",
                        "kind": "heuristic",
                        "policy": "rush",
                        "role": "league",
                    },
                    {
                        "id": "anti_meta_candidate",
                        "kind": "heuristic",
                        "policy": "anti_meta",
                        "role": "league",
                    },
                ],
                "hall_of_fame": [
                    {
                        "id": "weak_random_hof_1",
                        "kind": "heuristic",
                        "policy": "weak_random",
                        "role": "hall_of_fame",
                    },
                    {
                        "id": "weak_random_hof_2",
                        "kind": "heuristic",
                        "policy": "weak_random",
                        "role": "hall_of_fame",
                    },
                    {
                        "id": "weak_random_hof_3",
                        "kind": "heuristic",
                        "policy": "weak_random",
                        "role": "hall_of_fame",
                    },
                ],
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
                    "candidate_ids": [
                        "greedy_candidate",
                        "defensive_candidate",
                        "rush_candidate",
                        "anti_meta_candidate",
                    ],
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
                    "holdout_seeds": [11],
                    "min_holdout_win_rate": 0.0,
                    "min_holdout_mean_score_margin": -1.0,
                    "min_holdout_worst_decile_score_margin": -1.0,
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
    assert report["holdout_summary"] == {
        "seeds": [11],
        "seed_source": "objective_validation.holdout_seeds",
        "hall_of_fame_size": 3,
    }
    assert all(check["holdout_hall_of_fame_ok"] for check in report["candidate_checks"])
    assert all(check["holdout_vs_hall_of_fame_ok"] for check in report["candidate_checks"])
    assert all(check["generalization_gap_ok"] for check in report["candidate_checks"])
    assert all(check["per_opponent_holdout_ok"] for check in report["candidate_checks"])
    assert all(check["position_balance_ok"] for check in report["candidate_checks"])
    assert all(
        check["holdout_vs_hall_of_fame"]["enabled"] is True for check in report["candidate_checks"]
    )
    assert all(
        check["holdout_vs_hall_of_fame"]["holdout_seeds"] == [11]
        for check in report["candidate_checks"]
    )
    assert all(
        check["holdout_vs_hall_of_fame"]["games"] == 7 for check in report["candidate_checks"]
    )
    assert all(
        sorted(check["holdout_vs_hall_of_fame"]["per_opponent_2p"])
        == ["weak_random_hof_1", "weak_random_hof_2", "weak_random_hof_3"]
        for check in report["candidate_checks"]
    )
    assert all(
        sorted(check["holdout_vs_hall_of_fame"]["two_player_position_summary"]["positions"])
        == ["player_0", "player_1"]
        for check in report["candidate_checks"]
    )
    assert all(
        validation["seeds"] == [11]
        for check in report["candidate_checks"]
        for validation in check["runtime_validations"]
    )
    assert (out_dir / "candidate_1_submission.py").exists()
    assert (out_dir / "candidate_2_submission.py").exists()
    assert (out_dir / "objective_validation_report.json").exists()
    persisted_report = json.loads(
        (out_dir / "objective_validation_report.json").read_text(encoding="utf-8")
    )
    assert persisted_report["holdout_summary"] == report["holdout_summary"]
    assert persisted_report["candidate_checks"][0]["holdout_vs_hall_of_fame"]["holdout_seeds"] == [
        11
    ]


def test_objective_validation_fails_closed_when_holdout_hall_of_fame_is_required_but_missing(
    tmp_path: Path,
):
    manifest_path = tmp_path / "population.yaml"
    selection_config_path = tmp_path / "final_selection.yaml"
    validation_config_path = tmp_path / "objective_validation.yaml"
    out_dir = tmp_path / "out"

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "population": [
                    {
                        "id": "greedy_candidate",
                        "kind": "heuristic",
                        "policy": "greedy",
                        "role": "league",
                    },
                    {
                        "id": "defensive_candidate",
                        "kind": "heuristic",
                        "policy": "defensive",
                        "role": "league",
                    },
                    {
                        "id": "rush_candidate",
                        "kind": "heuristic",
                        "policy": "rush",
                        "role": "league",
                    },
                    {
                        "id": "anti_meta_candidate",
                        "kind": "heuristic",
                        "policy": "anti_meta",
                        "role": "league",
                    },
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
                    "holdout_seeds": [11],
                    "min_holdout_mean_score_margin": -1.0,
                    "min_holdout_worst_decile_score_margin": -1.0,
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

    assert report["objective_ready"] is False
    assert all(check["holdout_hall_of_fame_ok"] is False for check in report["candidate_checks"])
    assert all(check["holdout_vs_hall_of_fame_ok"] is False for check in report["candidate_checks"])
    assert all(
        check["holdout_validation"]["enabled"] is False for check in report["candidate_checks"]
    )
    assert all(
        check["holdout_vs_hall_of_fame"]["games"] == 0 for check in report["candidate_checks"]
    )


def test_load_objective_validation_config_parses_holdout_seed_ranges_and_default_gate(
    tmp_path: Path,
):
    validation_config_path = tmp_path / "objective_validation.yaml"
    validation_config_path.write_text(
        yaml.safe_dump(
            {
                "objective_validation": {
                    "holdout_seeds": 3,
                    "require_2p_self_play": False,
                    "require_4p_self_play": False,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_objective_validation_config(validation_config_path)

    assert cfg.holdout_seeds == [0, 1, 2]
    assert cfg.required_export_count == 1
    assert cfg.require_holdout_hall_of_fame is True
    assert cfg.min_holdout_win_rate == 0.5
    assert cfg.max_selection_holdout_win_rate_gap == 1.0
    assert cfg.min_holdout_per_opponent_win_rate == 0.0
    assert cfg.shipping_blind_seeds == []
    assert cfg.live_baseline is None
    assert cfg.max_holdout_position_worst_decile_score_margin_gap == 2.0


def test_same_agent_identity_treats_renamed_heuristic_snapshots_as_self():
    candidate = AgentSpec(id="greedy_candidate", kind="heuristic", role="league", policy="greedy")
    hall_clone = AgentSpec(id="greedy_hof", kind="heuristic", role="hall_of_fame", policy="greedy")
    different = AgentSpec(id="rush_hof", kind="heuristic", role="hall_of_fame", policy="rush")

    assert _same_agent_identity(candidate, hall_clone) is True
    assert _same_agent_identity(candidate, different) is False


def test_repo_objective_validation_config_and_candidate_pool_are_aligned():
    cfg = load_objective_validation_config("configs/objective_validation.yaml")
    manifest = load_population_manifest("configs/final_candidate_pool.yaml")

    assert cfg.holdout_seeds == [509, 601, 701, 809, 907, 1009]
    assert cfg.required_export_count == 1
    assert cfg.require_holdout_hall_of_fame is True
    assert cfg.min_holdout_win_rate == 0.4
    assert cfg.min_holdout_worst_decile_score_margin == -0.25
    assert cfg.max_selection_holdout_win_rate_gap == 0.35
    assert cfg.max_selection_holdout_mean_score_margin_gap == 0.75
    assert cfg.max_selection_holdout_worst_decile_score_margin_gap == 0.75
    assert cfg.min_holdout_per_opponent_win_rate == 0.25
    assert cfg.min_holdout_per_opponent_worst_decile_score_margin == -0.75
    assert cfg.min_holdout_seed_stratum_win_rate == 0.4
    assert cfg.min_holdout_seed_stratum_worst_decile_score_margin == -0.5
    assert cfg.max_holdout_position_win_rate_gap == 0.35
    assert cfg.max_holdout_position_mean_score_margin_gap == 0.75
    assert cfg.max_holdout_position_worst_decile_score_margin_gap == 0.75
    assert cfg.shipping_blind_seeds == [1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801]
    assert cfg.require_shipping_blind_hall_of_fame is True
    assert cfg.min_shipping_blind_win_rate == 0.4
    assert cfg.min_shipping_blind_worst_decile_score_margin == -0.25
    assert cfg.min_shipping_blind_seed_stratum_win_rate == 0.4
    assert cfg.min_shipping_blind_seed_stratum_worst_decile_score_margin == -0.5
    assert cfg.require_live_baseline_challenge is True
    assert cfg.live_baseline is not None
    assert cfg.live_baseline.name == "literature_validated_submission"
    assert cfg.live_baseline.submission_path.endswith(
        "artifacts/literature_validated_submission.py"
    )
    assert cfg.live_baseline.public_score == 479.4
    assert [spec.policy for spec in manifest["hall_of_fame"]] == ["greedy", "rush", "anti_meta"]
    assert [spec.policy for spec in manifest["heuristics"]] == ["anti_meta", "weak_random"]


def test_objective_validation_supports_single_export_with_shipping_blind_and_live_baseline(
    tmp_path: Path,
    monkeypatch,
):
    candidate = AgentSpec(id="candidate_a", kind="heuristic", role="league", policy="greedy")
    manifest = {
        "population": [
            candidate,
            AgentSpec(id="candidate_b", kind="heuristic", role="league", policy="defensive"),
        ],
        "hall_of_fame": [
            AgentSpec(id="rush_hof", kind="heuristic", role="hall_of_fame", policy="rush")
        ],
        "heuristics": [],
    }
    selection_cfg = load_final_selection_config("configs/final_selection.yaml")
    validation_cfg = ObjectiveValidationConfig(
        required_export_count=1,
        holdout_seeds=[509, 601],
        shipping_blind_seeds=[1103, 1201],
        require_shipping_blind_hall_of_fame=True,
        require_live_baseline_challenge=True,
        live_baseline=LiveBaselineConfig(
            name="literature_validated_submission",
            submission_path="artifacts/literature_validated_submission.py",
            public_score=479.4,
            submitted_at="2026-04-22 11:59:39.027000",
        ),
        require_2p_self_play=False,
        require_4p_self_play=False,
        min_selection_win_rate=0.0,
        min_mean_score_margin=-1.0,
        min_worst_decile_score_margin=-1.0,
        min_holdout_win_rate=0.0,
        min_holdout_mean_score_margin=-1.0,
        min_holdout_worst_decile_score_margin=-1.0,
        min_holdout_per_opponent_win_rate=0.0,
        min_holdout_per_opponent_worst_decile_score_margin=-1.0,
        min_holdout_seed_stratum_win_rate=0.0,
        min_holdout_seed_stratum_worst_decile_score_margin=-1.0,
        min_shipping_blind_win_rate=0.0,
        min_shipping_blind_mean_score_margin=-1.0,
        min_shipping_blind_worst_decile_score_margin=-1.0,
        min_shipping_blind_seed_stratum_win_rate=0.0,
        min_shipping_blind_seed_stratum_worst_decile_score_margin=-1.0,
        min_live_baseline_win_rate=0.0,
        min_live_baseline_mean_score_margin=-1.0,
        min_live_baseline_worst_decile_score_margin=-1.0,
    )

    def fake_final_selection(_manifest, _cfg, *, out_dir):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "candidate_1_submission.py").write_text(
            "def agent(obs):\n    return []\n", encoding="utf-8"
        )
        return {
            "summary": {
                "evaluated_candidates": 2,
                "requested_export_top_k": 1,
                "exported_candidate_count": 1,
            },
            "exports": [
                {"candidate_id": "candidate_a"},
            ],
            "ranking": [
                {
                    "candidate_id": "candidate_a",
                    "win_rate": 0.8,
                    "mean_score_margin": 0.3,
                    "worst_decile_score_margin": 0.1,
                },
                {
                    "candidate_id": "candidate_b",
                    "win_rate": 0.5,
                    "mean_score_margin": 0.0,
                    "worst_decile_score_margin": -0.1,
                },
            ],
        }

    def fake_holdout(_export_path, *, candidate_spec, hall_of_fame, holdout_seeds, cfg):
        assert candidate_spec.id == "candidate_a"
        assert hall_of_fame[0].id == "rush_hof"
        return {
            "enabled": True,
            "holdout_seeds": list(holdout_seeds),
            "hall_of_fame_opponents": ["rush_hof"],
            "games": 4,
            "win_rate": 0.75,
            "mean_score_margin": 0.25,
            "worst_decile_score_margin": 0.1,
            "mode_summary": {
                "2p": {
                    "games": 4,
                    "win_rate": 0.75,
                    "mean_score_margin": 0.25,
                    "worst_decile_score_margin": 0.1,
                }
            },
            "per_opponent_2p": {
                "rush_hof": {
                    "games": 4,
                    "win_rate": 0.75,
                    "mean_score_margin": 0.25,
                    "worst_decile_score_margin": 0.1,
                }
            },
            "seed_archetypes": {
                str(seed): {"seed": seed, "archetype": "mid_spin:mixed_opening"}
                for seed in holdout_seeds
            },
            "seed_strata_2p": {
                "mid_spin:mixed_opening": {
                    "games": 4,
                    "win_rate": 0.75,
                    "mean_score_margin": 0.25,
                    "worst_decile_score_margin": 0.1,
                    "seeds": list(holdout_seeds),
                }
            },
            "worst_matchup_2p": {"min_win_rate": 0.75, "min_worst_decile_score_margin": 0.1},
            "two_player_position_summary": {
                "positions": {
                    "player_0": {
                        "games": 2,
                        "win_rate": 1.0,
                        "mean_score_margin": 0.4,
                        "worst_decile_score_margin": 0.2,
                    },
                    "player_1": {
                        "games": 2,
                        "win_rate": 0.5,
                        "mean_score_margin": 0.1,
                        "worst_decile_score_margin": 0.0,
                    },
                },
                "gaps": {
                    "win_rate_gap": 0.5,
                    "mean_score_margin_gap": 0.3,
                    "worst_decile_score_margin_gap": 0.2,
                },
            },
            "records": [],
        }

    def fake_live_baseline(_export_path, *, baseline_cfg, shipping_blind_seeds, cfg):
        assert baseline_cfg is not None
        assert baseline_cfg.name == "literature_validated_submission"
        return {
            "enabled": True,
            "baseline_name": baseline_cfg.name,
            "baseline_submission_path": baseline_cfg.submission_path,
            "baseline_public_score": baseline_cfg.public_score,
            "baseline_submitted_at": baseline_cfg.submitted_at,
            "shipping_blind_seeds": list(shipping_blind_seeds),
            "games": 4,
            "win_rate": 0.75,
            "mean_score_margin": 0.2,
            "worst_decile_score_margin": 0.05,
            "mode_summary": {
                "2p": {
                    "games": 4,
                    "win_rate": 0.75,
                    "mean_score_margin": 0.2,
                    "worst_decile_score_margin": 0.05,
                }
            },
            "seed_archetypes": {
                str(seed): {"seed": seed, "archetype": "mid_spin:mixed_opening"}
                for seed in shipping_blind_seeds
            },
            "seed_strata_2p": {
                "mid_spin:mixed_opening": {
                    "games": 4,
                    "win_rate": 0.75,
                    "mean_score_margin": 0.2,
                    "worst_decile_score_margin": 0.05,
                    "seeds": list(shipping_blind_seeds),
                }
            },
            "two_player_position_summary": {
                "positions": {
                    "player_0": {
                        "games": 2,
                        "win_rate": 1.0,
                        "mean_score_margin": 0.3,
                        "worst_decile_score_margin": 0.1,
                    },
                    "player_1": {
                        "games": 2,
                        "win_rate": 0.5,
                        "mean_score_margin": 0.1,
                        "worst_decile_score_margin": 0.0,
                    },
                },
                "gaps": {
                    "win_rate_gap": 0.5,
                    "mean_score_margin_gap": 0.2,
                    "worst_decile_score_margin_gap": 0.1,
                },
            },
            "records": [],
        }

    monkeypatch.setattr(objective_validation_module, "run_final_selection", fake_final_selection)
    monkeypatch.setattr(
        objective_validation_module, "_holdout_hall_of_fame_validation", fake_holdout
    )
    monkeypatch.setattr(
        objective_validation_module, "_live_baseline_challenge_validation", fake_live_baseline
    )

    report = run_objective_validation(
        manifest,
        selection_cfg,
        validation_cfg,
        out_dir=tmp_path / "out",
    )

    assert report["objective_ready"] is True
    assert len(report["exports"]) == 1
    assert report["shipping_blind_summary"]["seeds"] == [1103, 1201]
    assert report["baseline_live_reference"]["name"] == "literature_validated_submission"
    check = report["candidate_checks"][0]
    assert check["holdout_seed_strata_ok"] is True
    assert check["shipping_blind_hall_of_fame_ok"] is True
    assert check["shipping_blind_seed_strata_ok"] is True
    assert check["live_baseline_challenge_ok"] is True


def test_objective_validation_rejects_candidates_with_generalization_gap_matchup_collapse_and_position_bias(
    tmp_path: Path,
    monkeypatch,
):
    candidate_a = AgentSpec(id="candidate_a", kind="heuristic", role="league", policy="greedy")
    candidate_b = AgentSpec(id="candidate_b", kind="heuristic", role="league", policy="defensive")
    manifest = {
        "population": [candidate_a, candidate_b],
        "hall_of_fame": [
            AgentSpec(id="rush_hof", kind="heuristic", role="hall_of_fame", policy="rush")
        ],
        "heuristics": [],
    }
    selection_cfg = load_final_selection_config("configs/final_selection.yaml")
    validation_cfg = ObjectiveValidationConfig(
        require_2p_self_play=False,
        require_4p_self_play=False,
        min_holdout_win_rate=0.0,
        min_holdout_mean_score_margin=-1.0,
        min_holdout_worst_decile_score_margin=-1.0,
        max_selection_holdout_win_rate_gap=0.2,
        max_selection_holdout_mean_score_margin_gap=0.3,
        max_selection_holdout_worst_decile_score_margin_gap=0.3,
        min_holdout_per_opponent_win_rate=0.4,
        min_holdout_per_opponent_worst_decile_score_margin=-0.2,
        max_holdout_position_win_rate_gap=0.2,
        max_holdout_position_mean_score_margin_gap=0.3,
        max_holdout_position_worst_decile_score_margin_gap=0.3,
    )

    def fake_final_selection(_manifest, _cfg, *, out_dir):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return {
            "summary": {"evaluated_candidates": 2},
            "exports": [
                {"candidate_id": "candidate_a"},
                {"candidate_id": "candidate_b"},
            ],
            "ranking": [
                {
                    "candidate_id": "candidate_a",
                    "win_rate": 0.95,
                    "mean_score_margin": 0.8,
                    "worst_decile_score_margin": 0.6,
                },
                {
                    "candidate_id": "candidate_b",
                    "win_rate": 0.9,
                    "mean_score_margin": 0.7,
                    "worst_decile_score_margin": 0.5,
                },
            ],
        }

    def fake_holdout(_export_path, *, candidate_spec, hall_of_fame, holdout_seeds, cfg):
        assert hall_of_fame[0].id == "rush_hof"
        if not holdout_seeds:
            return {"enabled": False, "reason": "holdout_seeds_empty", "games": 0}
        assert holdout_seeds == selection_cfg.retained_seeds
        return {
            "enabled": True,
            "holdout_seeds": list(holdout_seeds),
            "hall_of_fame_opponents": ["rush_hof"],
            "games": 4,
            "win_rate": 0.3,
            "mean_score_margin": 0.1,
            "worst_decile_score_margin": -0.4,
            "mode_summary": {
                "2p": {
                    "games": 4,
                    "win_rate": 0.3,
                    "mean_score_margin": 0.1,
                    "worst_decile_score_margin": -0.4,
                }
            },
            "per_opponent_2p": {
                "rush_hof": {
                    "games": 4,
                    "win_rate": 0.25 if candidate_spec.id == "candidate_a" else 0.3,
                    "mean_score_margin": -0.1,
                    "worst_decile_score_margin": -0.8,
                }
            },
            "worst_matchup_2p": {"min_win_rate": 0.25, "min_worst_decile_score_margin": -0.8},
            "two_player_position_summary": {
                "positions": {
                    "player_0": {
                        "games": 2,
                        "win_rate": 0.8,
                        "mean_score_margin": 0.5,
                        "worst_decile_score_margin": 0.2,
                    },
                    "player_1": {
                        "games": 2,
                        "win_rate": 0.1,
                        "mean_score_margin": -0.4,
                        "worst_decile_score_margin": -0.6,
                    },
                },
                "gaps": {
                    "win_rate_gap": 0.7,
                    "mean_score_margin_gap": 0.9,
                    "worst_decile_score_margin_gap": 0.8,
                },
            },
            "records": [],
        }

    monkeypatch.setattr(objective_validation_module, "run_final_selection", fake_final_selection)
    monkeypatch.setattr(
        objective_validation_module, "_holdout_hall_of_fame_validation", fake_holdout
    )

    report = run_objective_validation(
        manifest,
        selection_cfg,
        validation_cfg,
        out_dir=tmp_path / "out",
    )

    assert report["objective_ready"] is False
    assert all(check["holdout_hall_of_fame_ok"] is True for check in report["candidate_checks"])
    assert all(check["generalization_gap_ok"] is False for check in report["candidate_checks"])
    assert all(check["per_opponent_holdout_ok"] is False for check in report["candidate_checks"])
    assert all(check["position_balance_ok"] is False for check in report["candidate_checks"])
    assert all(check["passed"] is False for check in report["candidate_checks"])
