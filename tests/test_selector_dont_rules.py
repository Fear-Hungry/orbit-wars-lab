"""Guard tests for the selector don'ts (etapa 20) — each one locks a rule the
project has already paid to learn."""
from __future__ import annotations

import json

from scripts.league_submit_ruler import build_report, build_tasks


def _minimal_report(tmp_path):
    games = []
    for seed in (1, 2):
        games.append({
            "seed": seed, "seats": ["cand", "producer"], "final_ships": [10.0, 5.0],
            "winner": "cand", "winner_seat": 0, "tie": False,
            "died_at": [None, None], "agent_status": ["DONE", "DONE"], "faults": {},
        })
        games.append({
            "seed": seed, "seats": ["producer", "cand"], "final_ships": [5.0, 10.0],
            "winner": "cand", "winner_seat": 1, "tie": False,
            "died_at": [None, None], "agent_status": ["DONE", "DONE"], "faults": {},
        })
    path = tmp_path / "g.json"
    path.write_text(json.dumps({
        "mode": "2p", "games": games,
        "decision_ms_p95": {"cand": 100.0, "producer": 100.0},
    }))
    results = [{
        "label": "cand__2p__producer", "mode": "2p", "candidate": "cand",
        "names": ["cand", "producer"], "out": str(path), "returncode": 0,
        "role": "fixed_2p",
    }]
    return build_report(
        ["cand"], results, incumbent="inc", min_decisive_2p=1,
        min_producer_winrate=0.5, min_incumbent_winrate=0.5, min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.3, weight_2p=0.46,
    )


def test_no_recommended_candidate_and_no_overall_score_anywhere(tmp_path):
    report = _minimal_report(tmp_path)
    blob = json.dumps(report)
    assert "recommended_candidate" not in blob
    assert "overall_score" not in blob


def test_raw_ruler_output_is_veto_only(tmp_path):
    report = _minimal_report(tmp_path)
    assert report["selection_status"] == "VETO_ONLY"
    assert report["promotion_order_valid"] is False
    assert report["selector_candidate"] is None


def test_peers_never_fill_4p_lineups(tmp_path):
    tasks = build_tasks(
        ["pgs_hold", "pgs_wave_s50", "pgs_valuenet"],
        incumbent="pgs_holdwave",
        references=["producer"],
        four_player_templates=[("producer",), ("producer", "oep")],
        seeds=4, seed_base=1, steps=100, out_dir=tmp_path,
    )
    candidates = {"pgs_hold", "pgs_wave_s50", "pgs_valuenet"}
    for task in tasks:
        if task.role == "fixed_4p":
            assert not (set(task.names) - {task.candidate}) & candidates, task.names
