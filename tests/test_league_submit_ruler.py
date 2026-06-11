from __future__ import annotations

import json

import pytest
import scripts.league_submit_ruler as ruler
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


def _write(tmp_path, label, mode, games, *, candidate="cand"):
    path = tmp_path / f"{label}.json"
    names = list(games[0]["seats"])
    for game in games:
        game.pop("mode", None)
    path.write_text(json.dumps({"mode": mode, "games": games}))
    return {
        "label": label,
        "mode": mode,
        "candidate": candidate,
        "names": names,
        "out": str(path),
        "returncode": 0,
        "seconds": 0.0,
        "stdout": "",
        "stderr": "",
    }


def _passing_results(tmp_path, *, candidate="cand", label_prefix=""):
    def label(value):
        return f"{label_prefix}{value}"

    return [
        _write(tmp_path, label("producer"), "2p", [
            _game([candidate, "producer"], candidate),
            _game(["producer", candidate], candidate),
            _game([candidate, "producer"], candidate),
            _game(["producer", candidate], "producer"),
        ], candidate=candidate),
        _write(tmp_path, label("inc"), "2p", [
            _game([candidate, "inc"], candidate),
            _game(["inc", candidate], "inc"),
            _game([candidate, "inc"], candidate),
            _game(["inc", candidate], "inc"),
        ], candidate=candidate),
        _write(tmp_path, label("floor"), "2p", [
            _game([candidate, "pgs_allscripts"], candidate),
            _game(["pgs_allscripts", candidate], candidate),
            _game([candidate, "pgs_allscripts"], candidate),
            _game(["pgs_allscripts", candidate], candidate),
        ], candidate=candidate),
        _write(tmp_path, label("line0"), "4p", [
            _game([candidate, "producer", "inc", "pgs_allscripts"], candidate, mode="4p"),
            _game([candidate, "producer", "inc", "pgs_allscripts"], "producer", mode="4p"),
        ], candidate=candidate),
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


def test_submit_ruler_2p_score_counts_ties_as_non_wins(tmp_path):
    result = _write(tmp_path, "producer_ties", "2p", [
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], None),
        _game(["cand", "producer"], None),
        _game(["producer", "cand"], None),
    ])

    summary = summarize_candidate(
        "cand",
        [result],
        incumbent="inc",
        min_decisive_2p=1,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert summary["pairwise"]["producer"]["decisive_win_rate"] == 1.0
    assert summary["pairwise"]["producer"]["win_rate"] == 0.25
    assert summary["score_2p"] == 0.25


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


def test_submit_ruler_ranking_prioritizes_verdict_before_score(tmp_path):
    results = []
    results.extend(_passing_results(tmp_path, candidate="pass", label_prefix="pass_"))
    results.extend([
        _write(tmp_path, "maybe_producer", "2p", [
            _game(["maybe", "producer"], "maybe"),
        ], candidate="maybe"),
        _write(tmp_path, "maybe_inc", "2p", [
            _game(["maybe", "inc"], "maybe"),
        ], candidate="maybe"),
        _write(tmp_path, "maybe_floor", "2p", [
            _game(["maybe", "pgs_allscripts"], "maybe"),
        ], candidate="maybe"),
        _write(tmp_path, "maybe_line0", "4p", [
            _game(["maybe", "producer", "inc", "pgs_allscripts"], "maybe", mode="4p"),
        ], candidate="maybe"),
    ])
    results.extend(_passing_results(tmp_path, candidate="reject", label_prefix="reject_"))
    results.append(_write(tmp_path, "reject_fault", "4p", [
        _game(
            ["reject", "producer", "inc", "pgs_allscripts"],
            "reject",
            mode="4p",
            faults={"reject": {"crashes": 1, "timeouts": 0, "invalid_moves": 0}},
            status=["ERROR", "DONE", "DONE", "DONE"],
        ),
    ], candidate="reject"))

    report = build_report(
        ["reject", "maybe", "pass"],
        results,
        incumbent="inc",
        min_decisive_2p=4,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )

    assert [row["verdict"] for row in report["ranking"]] == [
        "PASS_LOCAL",
        "INCONCLUSIVE",
        "REJECT_LOCAL",
    ]
    assert [row["candidate"] for row in report["ranking"]] == ["pass", "maybe", "reject"]


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
    assert len({task.seed_base for task in producer_tasks}) == 1

    four_player_tasks = [task for task in tasks if task.mode == "4p"]
    assert {task.candidate for task in four_player_tasks} == {"pgs_hold", "pgs_wave_s100"}
    assert {task.seed_base for task in four_player_tasks} == {1234 + 5000}


def test_build_tasks_adds_peer_candidates_to_reference_panel(tmp_path):
    tasks = build_tasks(
        ["pgs_hold", "pgs_holdwave", "pgs_wave_s100"],
        incumbent="pgs_holdwave",
        references=["producer"],
        four_player_templates=[],
        seeds=4,
        seed_base=1234,
        steps=500,
        out_dir=tmp_path,
    )

    opponents_by_candidate = {
        candidate: {
            task.names[1]
            for task in tasks
            if task.mode == "2p" and task.candidate == candidate
        }
        for candidate in ["pgs_hold", "pgs_holdwave", "pgs_wave_s100"]
    }

    assert opponents_by_candidate["pgs_hold"] == {"pgs_holdwave", "pgs_wave_s100", "producer"}
    assert opponents_by_candidate["pgs_holdwave"] == {"pgs_hold", "pgs_wave_s100", "producer"}
    assert opponents_by_candidate["pgs_wave_s100"] == {"pgs_hold", "pgs_holdwave", "producer"}


def test_build_tasks_propagates_match_chunk_size(tmp_path):
    tasks = build_tasks(
        ["pgs_hold"],
        incumbent="pgs_holdwave",
        references=["producer"],
        four_player_templates=[("producer", "pgs_holdwave", "pgs_allscripts")],
        seeds=8,
        seed_base=1234,
        steps=500,
        out_dir=tmp_path,
        match_chunk_size=4,
    )

    assert tasks
    assert {task.chunk_size for task in tasks} == {4}


def test_build_tasks_keeps_reference_seed_slice_stable_across_candidate_panels(tmp_path):
    solo = build_tasks(
        ["pgs_hold"],
        incumbent="pgs_holdwave",
        references=["producer", "pgs_allscripts"],
        four_player_templates=[],
        seeds=4,
        seed_base=1234,
        steps=500,
        out_dir=tmp_path / "solo",
    )
    panel = build_tasks(
        ["pgs_hold", "pgs_wave_s100"],
        incumbent="pgs_holdwave",
        references=["producer", "pgs_allscripts"],
        four_player_templates=[],
        seeds=4,
        seed_base=1234,
        steps=500,
        out_dir=tmp_path / "panel",
    )

    solo_producer = next(task for task in solo if task.mode == "2p" and task.names == ("pgs_hold", "producer"))
    panel_producer = next(task for task in panel if task.mode == "2p" and task.names == ("pgs_hold", "producer"))
    assert solo_producer.seed_base == panel_producer.seed_base


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


def test_run_tasks_writes_incremental_results(monkeypatch, tmp_path):
    tasks = [
        ruler.MatchTask(
            label="a",
            mode="2p",
            candidate="cand",
            names=("cand", "producer"),
            seeds=4,
            seed_base=1,
            steps=10,
            out=tmp_path / "a.json",
        ),
        ruler.MatchTask(
            label="b",
            mode="2p",
            candidate="cand",
            names=("cand", "oep"),
            seeds=4,
            seed_base=2,
            steps=10,
            out=tmp_path / "b.json",
        ),
    ]

    def fake_run_task(task):
        return {
            "label": task.label,
            "mode": task.mode,
            "candidate": task.candidate,
            "names": list(task.names),
            "out": str(task.out),
            "returncode": 0,
            "seconds": 1.0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(ruler, "_run_task", fake_run_task)
    progress_path = tmp_path / "task_results.json"

    results = ruler.run_tasks(tasks, jobs=1, task_results_out=progress_path)

    assert progress_path.exists()
    assert json.loads(progress_path.read_text()) == results
