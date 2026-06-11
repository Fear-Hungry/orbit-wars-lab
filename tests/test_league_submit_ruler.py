from __future__ import annotations

import json

import pytest
from scripts.league_submit_ruler import build_report, build_tasks, main, summarize_candidate


def _game(seats, winner, *, mode="2p", faults=None, status=None, died=None):
    if winner is None:
        ships = [10.0] * len(seats)
        winner_seat = -1
        tie = True
    else:
        ships = [5.0] * len(seats)
        ships[seats.index(winner)] = 20.0
        winner_seat = seats.index(winner)
        tie = False
    return {
        "seed": 1,
        "seats": list(seats),
        "final_ships": ships,
        "winner": winner,
        "winner_seat": winner_seat,
        "tie": tie,
        "died_at": died or [None] * len(seats),
        "agent_status": status or ["DONE"] * len(seats),
        "faults": faults or {},
        "mode": mode,
    }


def _write(tmp_path, label, mode, games):
    path = tmp_path / f"{label}.json"
    names = list(games[0]["seats"])
    for game in games:
        game.pop("mode", None)
    path.write_text(json.dumps({"mode": mode, "games": games}))
    return {
        "label": label,
        "mode": mode,
        "candidate": "cand",
        "names": names,
        "out": str(path),
        "returncode": 0,
        "seconds": 0.0,
        "stdout": "",
        "stderr": "",
    }


def _passing_results(tmp_path):
    return [
        _write(tmp_path, "producer", "2p", [
            _game(["cand", "producer"], "cand"),
            _game(["producer", "cand"], "cand"),
            _game(["cand", "producer"], "cand"),
            _game(["producer", "cand"], "producer"),
        ]),
        _write(tmp_path, "inc", "2p", [
            _game(["cand", "inc"], "cand"),
            _game(["inc", "cand"], "inc"),
            _game(["cand", "inc"], "cand"),
            _game(["inc", "cand"], "inc"),
        ]),
        _write(tmp_path, "floor", "2p", [
            _game(["cand", "pgs_allscripts"], "cand"),
            _game(["pgs_allscripts", "cand"], "cand"),
            _game(["cand", "pgs_allscripts"], "cand"),
            _game(["pgs_allscripts", "cand"], "cand"),
        ]),
        _write(tmp_path, "line0", "4p", [
            _game(["cand", "producer", "inc", "pgs_allscripts"], "cand", mode="4p"),
            _game(["cand", "producer", "inc", "pgs_allscripts"], "producer", mode="4p"),
        ]),
    ]


def test_submit_ruler_passes_clean_candidate(tmp_path):
    summary = summarize_candidate(
        "cand",
        _passing_results(tmp_path),
        incumbent="inc",
        min_decisive_2p=4,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert summary["verdict"] == "PASS_LOCAL"
    assert summary["pairwise"]["producer"]["decisive_win_rate"] == 0.75
    assert summary["pairwise"]["inc"]["decisive_win_rate"] == 0.5
    assert summary["four_player"]["win_rate"] == 0.5


def test_submit_ruler_rejects_faults(tmp_path):
    results = _passing_results(tmp_path)
    fault_path = tmp_path / "fault.json"
    fault_path.write_text(json.dumps({"mode": "2p", "games": [
        _game(
            ["cand", "producer"],
            "producer",
            faults={"cand": {"crashes": 1, "timeouts": 0, "invalid_moves": 0}},
            status=["ERROR", "DONE"],
        )
    ]}))
    results.append({
        "label": "fault",
        "mode": "2p",
        "candidate": "cand",
        "names": ["cand", "producer"],
        "out": str(fault_path),
        "returncode": 0,
        "seconds": 0.0,
        "stdout": "",
        "stderr": "",
    })

    summary = summarize_candidate(
        "cand",
        results,
        incumbent="inc",
        min_decisive_2p=1,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert summary["verdict"] == "REJECT_LOCAL"
    assert any(c["name"] == "no_faults" and not c["passed"] for c in summary["checks"])


def test_submit_ruler_marks_low_coverage_inconclusive(tmp_path):
    summary = summarize_candidate(
        "cand",
        _passing_results(tmp_path),
        incumbent="inc",
        min_decisive_2p=10,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert summary["verdict"] == "INCONCLUSIVE"
    assert any(c["severity"] == "inconclusive" and not c["passed"] for c in summary["checks"])


def test_submit_ruler_report_recommends_best_passing_candidate(tmp_path):
    report = build_report(
        ["cand"],
        _passing_results(tmp_path),
        incumbent="inc",
        min_decisive_2p=4,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert report["recommended_candidate"] == "cand"
    assert report["ranking"][0]["candidate"] == "cand"
    assert report["ranking"][0]["verdict"] == "PASS_LOCAL"


def test_build_tasks_pairs_candidates_with_incumbent_and_fixed_lineups(tmp_path):
    tasks = build_tasks(
        ["pgs_hold"],
        incumbent="pgs_holdwave",
        references=["producer", "pgs_allscripts"],
        four_player_templates=[("producer", "pgs_holdwave", "pgs_allscripts")],
        seeds=3,
        seed_base=100,
        steps=500,
        out_dir=tmp_path,
    )

    assert ("pgs_hold", "pgs_holdwave") in {task.names for task in tasks if task.mode == "2p"}
    assert ("pgs_hold", "producer") in {task.names for task in tasks if task.mode == "2p"}
    assert any(task.mode == "4p" and task.names[0] == "pgs_hold" and len(task.names) == 4 for task in tasks)


def test_build_tasks_uses_shared_seed_slices_across_candidates(tmp_path):
    tasks = build_tasks(
        ["pgs_hold", "pgs_wave_s100"],
        incumbent="pgs_holdwave",
        references=["producer", "pgs_allscripts"],
        four_player_templates=[("producer", "pgs_holdwave", "pgs_allscripts")],
        seeds=4,
        seed_base=1234,
        steps=500,
        out_dir=tmp_path,
    )

    producer_tasks = [task for task in tasks if task.mode == "2p" and task.names[1] == "producer"]
    assert {task.candidate for task in producer_tasks} == {"pgs_hold", "pgs_wave_s100"}
    assert {task.seed_base for task in producer_tasks} == {1234 + 100}

    four_player_tasks = [task for task in tasks if task.mode == "4p"]
    assert {task.candidate for task in four_player_tasks} == {"pgs_hold", "pgs_wave_s100"}
    assert {task.seed_base for task in four_player_tasks} == {1234 + 5000}


def test_cli_requires_seed_multiple_of_four_for_balanced_4p_seats(tmp_path):
    with pytest.raises(SystemExit, match="multiple of 4"):
        main([
            "--candidates",
            "pgs_hold",
            "--seeds",
            "3",
            "--skip-run",
            "--out-dir",
            str(tmp_path),
        ])
