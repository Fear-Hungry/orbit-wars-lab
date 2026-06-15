from __future__ import annotations

from pathlib import Path

import pytest
from scripts import drl_promotion_gate as gate
from scripts.league_agents import FACTORIES, make


def test_drl_reference_pool_matches_objective():
    assert {"producer", "oep", "brep", "pgs_bigwave", "greedy", "rush"}.issubset(
        set(gate.DRL_REFERENCES) | {gate.INCUMBENT}
    )
    assert {"producer", "oep", "brep", "pgs_bigwave", "greedy", "rush", gate.INCUMBENT}.issubset(
        set(gate.DRL_REQUIRED_2P)
    )
    assert "greedy" in FACTORIES
    assert "rush" in FACTORIES


def test_league_greedy_and_rush_are_kaggle_obs_agents():
    obs = {
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 30, 3],
            [1, -1, 30.0, 10.0, 2.0, 8, 2],
            [2, 1, 80.0, 80.0, 2.0, 30, 3],
        ],
        "fleets": [],
        "comet_planet_ids": [],
    }

    for name in ("greedy", "rush"):
        moves = make(name)(obs)
        assert isinstance(moves, list)
        assert all(len(move) == 3 for move in moves)


def test_run_drl_promotion_gate_passes_frozen_requirements(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(
        gate,
        "prepare_candidates",
        lambda **kwargs: [gate.PreparedCandidate(name="cand", kind="league_agent")],
    )

    def fake_build_tasks(candidates, **kwargs):
        calls["candidates"] = candidates
        calls["references"] = kwargs["references"]
        calls["four_player_templates"] = kwargs["four_player_templates"]
        return []

    def fake_build_report(candidates, task_results, **kwargs):
        calls["required_2p_winrates"] = kwargs["required_2p_winrates"]
        return {
            "local_veto_passes": ["cand"],
            "selector_candidate": None,
            "selection_status": "VETO_ONLY",
            "promotion_order_valid": False,
            "ranking": [
                {
                    "candidate": "cand",
                    "verdict": "PASS_LOCAL",
                    "field_advantage": 1.0,
                    "score_2p_fixed": 1.0,
                    "score_2p_peer": None,
                    "score_4p_fixed": 1.0,
                    "worst_bucket_score": 1.0,
                    "risk_penalty": 0.0,
                }
            ],
            "candidates": {},
        }

    monkeypatch.setattr(gate, "build_tasks", fake_build_tasks)
    monkeypatch.setattr(gate, "run_tasks", lambda *args, **kwargs: [])
    monkeypatch.setattr(gate, "build_report", fake_build_report)

    report = gate.run_drl_promotion_gate(
        checkpoint_patterns=[],
        league_candidates=["cand"],
        out_dir=tmp_path,
        profile="quick",
        seeds=4,
        seed_base=123,
        steps=10,
        jobs=1,
        match_chunk_size=0,
        skip_run=False,
        required_2p_threshold=0.55,
        min_decisive_2p=None,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.46,
    )

    assert calls["candidates"] == ["cand"]
    assert set(gate.DRL_REFERENCES).issubset(calls["references"])
    assert gate.DRL_4P_TEMPLATES[0] in tuple(calls["four_player_templates"])
    assert calls["required_2p_winrates"] == gate.required_2p_winrates(0.55)
    assert report["promotable"][0]["name"] == "cand"


def test_prepare_candidates_requires_known_league_candidate(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown league candidate"):
        gate.prepare_candidates(
            checkpoint_patterns=[],
            league_candidates=["missing_bot"],
            submissions_dir=tmp_path,
            parity_dir=tmp_path / "parity",
            skip_parity=False,
            parity_seeds=1,
            parity_steps=1,
        )
