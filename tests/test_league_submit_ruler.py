from __future__ import annotations

import json

import pytest
import scripts.league_submit_ruler as ruler
from scripts.league_submit_ruler import build_report, build_tasks, main, summarize_candidate


def _game(seats, winner, *, mode="2p", faults=None, status=None, died=None, seed=1):
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
        "seed": seed,
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


def _write(tmp_path, label, mode, games, *, candidate="cand", p95=None):
    path = tmp_path / f"{label}.json"
    names = list(games[0]["seats"])
    for game in games:
        game.pop("mode", None)
    payload = {"mode": mode, "games": games}
    if p95 is not None:
        payload["decision_ms_p95"] = p95
    path.write_text(json.dumps(payload))
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
        # Seat-balanced incumbent H2H: the candidate clears the per-seat floor from
        # BOTH seats (seat0 2/2, seat1 1/2), so it is a real gain, not a seat-split.
        _write(tmp_path, label("inc"), "2p", [
            _game([candidate, "inc"], candidate),
            _game(["inc", candidate], candidate),
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
    assert summary["pairwise"]["inc"]["decisive_win_rate"] == 0.75
    assert summary["four_player"]["win_rate"] == 0.5
    # The clean candidate clears BOTH per-seat incumbent floors.
    seat_checks = {c["name"]: c for c in summary["checks"]}
    assert seat_checks["incumbent_h2h_seat0"]["passed"] is True
    assert seat_checks["incumbent_h2h_seat1"]["passed"] is True


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


def test_submit_ruler_rejects_missing_required_opponents(tmp_path):
    result = _write(tmp_path, "producer_only", "2p", [
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], "cand"),
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

    assert summary["verdict"] == "REJECT_LOCAL"
    failed = {c["name"]: c for c in summary["checks"] if not c["passed"]}
    assert failed["beats_or_ties_incumbent_h2h"]["details"] == {"missing_required_opponent": "inc"}
    assert failed["clears_rejected_floor"]["details"] == {"missing_required_opponent": "pgs_allscripts"}


def test_submit_ruler_rejects_required_2p_reference_loss(tmp_path):
    results = _passing_results(tmp_path)
    results.append(_write(tmp_path, "brep_loss", "2p", [
        _game(["cand", "brep"], "brep"),
        _game(["brep", "cand"], "brep"),
        _game(["cand", "brep"], "cand"),
        _game(["brep", "cand"], "brep"),
    ]))

    summary = summarize_candidate(
        "cand",
        results,
        incumbent="inc",
        min_decisive_2p=4,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        required_2p_winrates={"brep": 0.5},
        weight_2p=0.5,
    )

    assert summary["verdict"] == "REJECT_LOCAL"
    failed = {c["name"]: c for c in summary["checks"] if not c["passed"]}
    assert failed["required_2p_vs_brep"]["details"] == {
        "decisive_win_rate": 0.25,
        "required": 0.5,
    }


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


def test_load_games_rejects_partial_strict_task_payload(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({
        "agents": ["cand", "producer"],
        "mode": "2p",
        "seed_base": 100,
        "seed_count": 2,
        "steps": 500,
        "games": [
            _game(["cand", "producer"], "cand", seed=100),
        ],
    }))
    result = {
        "label": "cand__2p__producer",
        "mode": "2p",
        "candidate": "cand",
        "names": ["cand", "producer"],
        "seeds": 2,
        "seed_base": 100,
        "steps": 500,
        "out": str(path),
        "returncode": 0,
    }

    with pytest.raises(ValueError, match="games 1 != expected 4"):
        ruler._load_games(path, result)


def test_load_games_rejects_missing_agent_status_in_strict_payload(tmp_path):
    games = []
    for seed in (100, 101):
        for seats in (["cand", "producer"], ["producer", "cand"]):
            game = _game(seats, "cand", seed=seed)
            game.pop("agent_status")
            games.append(game)
    path = tmp_path / "missing_status.json"
    path.write_text(json.dumps({
        "agents": ["cand", "producer"],
        "mode": "2p",
        "seed_base": 100,
        "seed_count": 2,
        "steps": 500,
        "games": games,
    }))
    result = {
        "label": "cand__2p__producer",
        "mode": "2p",
        "candidate": "cand",
        "names": ["cand", "producer"],
        "seeds": 2,
        "seed_base": 100,
        "steps": 500,
        "out": str(path),
        "returncode": 0,
    }

    with pytest.raises(ValueError, match="agent_status"):
        ruler._load_games(path, result)


def test_load_games_accepts_complete_strict_2p_payload(tmp_path):
    games = []
    for seed in (100, 101):
        games.append(_game(["cand", "producer"], "cand", seed=seed))
        games.append(_game(["producer", "cand"], "producer", seed=seed))
    path = tmp_path / "complete.json"
    path.write_text(json.dumps({
        "agents": ["cand", "producer"],
        "mode": "2p",
        "seed_base": 100,
        "seed_count": 2,
        "steps": 500,
        "games": games,
    }))
    result = {
        "label": "cand__2p__producer",
        "mode": "2p",
        "candidate": "cand",
        "names": ["cand", "producer"],
        "seeds": 2,
        "seed_base": 100,
        "steps": 500,
        "out": str(path),
        "returncode": 0,
    }

    assert len(ruler._load_games(path, result)) == 4


def test_load_games_accepts_complete_strict_4p_all_rotations_payload(tmp_path):
    games = []
    names = ["cand", "producer", "inc", "pgs_allscripts"]
    for seed in (100, 101):
        for r in range(4):
            seats = names[r:] + names[:r]
            games.append(_game(seats, "cand", mode="4p", seed=seed))
    path = tmp_path / "complete_4p.json"
    path.write_text(json.dumps({
        "agents": names,
        "mode": "4p",
        "seed_base": 100,
        "seed_count": 2,
        "steps": 500,
        "games": games,
    }))
    result = {
        "label": "cand__4p__line0",
        "mode": "4p",
        "candidate": "cand",
        "names": names,
        "seeds": 2,
        "seed_base": 100,
        "steps": 500,
        "out": str(path),
        "returncode": 0,
    }

    assert len(ruler._load_games(path, result)) == 8


def test_load_games_rejects_old_strict_4p_one_rotation_per_seed_payload(tmp_path):
    names = ["cand", "producer", "inc", "pgs_allscripts"]
    games = []
    for idx, seed in enumerate((100, 101, 102, 103)):
        r = idx % 4
        games.append(_game(names[r:] + names[:r], "cand", mode="4p", seed=seed))
    path = tmp_path / "old_4p.json"
    path.write_text(json.dumps({
        "agents": names,
        "mode": "4p",
        "seed_base": 100,
        "seed_count": 4,
        "steps": 500,
        "games": games,
    }))
    result = {
        "label": "cand__4p__line0",
        "mode": "4p",
        "candidate": "cand",
        "names": names,
        "seeds": 4,
        "seed_base": 100,
        "steps": 500,
        "out": str(path),
        "returncode": 0,
    }

    with pytest.raises(ValueError, match="games 4 != expected 16"):
        ruler._load_games(path, result)


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


def test_cli_rejects_unknown_requested_references(tmp_path):
    with pytest.raises(SystemExit, match="unknown references: typo_bot"):
        main([
            "--candidates",
            "pgs_hold",
            "--references",
            "producer,typo_bot",
            "--seeds",
            "4",
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


def _summary(results, **overrides):
    kwargs = dict(
        incumbent="inc",
        min_decisive_2p=4,
        min_producer_winrate=0.5,
        min_incumbent_winrate=0.5,
        min_floor_winrate=0.6,
        max_annihilation_rate_4p=0.35,
        weight_2p=0.5,
    )
    kwargs.update(overrides)
    return summarize_candidate("cand", results, **kwargs)


def _checks_by_name(summary):
    return {c["name"]: c for c in summary["checks"]}


def test_submit_ruler_rejects_seat_split_false_gain(tmp_path):
    # Candidate wins BOTH seat-0 games vs the incumbent but loses BOTH seat-1
    # games: aggregate decisive winrate is exactly 0.5 (clears the aggregate
    # floor) yet it is a false gain. The per-seat guard must reject it.
    results = _passing_results(tmp_path)
    results[1] = _write(tmp_path, "inc", "2p", [
        _game(["cand", "inc"], "cand"),     # seat0 win
        _game(["inc", "cand"], "inc"),      # seat1 loss
        _game(["cand", "inc"], "cand"),     # seat0 win
        _game(["inc", "cand"], "inc"),      # seat1 loss
    ])

    summary = _summary(results)
    checks = _checks_by_name(summary)

    assert checks["beats_or_ties_incumbent_h2h"]["passed"] is True  # aggregate 0.5
    assert checks["incumbent_h2h_seat0"]["passed"] is True
    assert checks["incumbent_h2h_seat1"]["passed"] is False
    assert summary["verdict"] == "REJECT_LOCAL"


def test_submit_ruler_rejects_p95_over_budget(tmp_path):
    # A candidate that is otherwise clean but blows the p95 latency budget on any
    # payload must be rejected: Kaggle enforces a 1s actTimeout at inference.
    results = _passing_results(tmp_path)
    results[0] = _write(tmp_path, "producer", "2p", [
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], "cand"),
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], "producer"),
    ], p95={"cand": 1500.0, "producer": 120.0})

    summary = _summary(results, max_p95_ms=900.0)
    checks = _checks_by_name(summary)

    assert summary["p95_ms"] == 1500.0
    assert checks["p95_within_limit"]["passed"] is False
    assert summary["verdict"] == "REJECT_LOCAL"


def test_submit_ruler_rejects_below_4p_fair_share(tmp_path):
    # Survives 4p (not annihilated) but pulls below the FFA fair share (~0.25):
    # a passenger in the 4p regime that is the majority of the field.
    results = _passing_results(tmp_path)
    lineup = ["cand", "producer", "inc", "pgs_allscripts"]
    results[3] = _write(tmp_path, "line0", "4p", [
        _game(lineup, "producer", mode="4p"),
        _game(lineup, "inc", mode="4p"),
        _game(lineup, "pgs_allscripts", mode="4p"),
        _game(lineup, "producer", mode="4p"),
    ])

    summary = _summary(results)
    checks = _checks_by_name(summary)

    assert summary["four_player"]["decisive_win_rate"] == 0.0
    assert checks["survives_4p"]["passed"] is True
    assert checks["four_player_fair_share"]["passed"] is False
    assert summary["verdict"] == "REJECT_LOCAL"


def test_submit_ruler_p95_within_budget_passes(tmp_path):
    # p95 present and under the limit on every payload => the latency guard passes.
    results = _passing_results(tmp_path)
    results[0] = _write(tmp_path, "producer", "2p", [
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], "cand"),
        _game(["cand", "producer"], "cand"),
        _game(["producer", "cand"], "producer"),
    ], p95={"cand": 410.0, "producer": 120.0})

    summary = _summary(results, max_p95_ms=900.0)
    checks = _checks_by_name(summary)

    assert summary["p95_ms"] == 410.0
    assert checks["p95_within_limit"]["passed"] is True
    assert summary["verdict"] == "PASS_LOCAL"
