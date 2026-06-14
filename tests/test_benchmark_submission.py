from __future__ import annotations

from pathlib import Path

from scripts.benchmark_submission import (
    _opponent_instances,
    benchmark_four_player_spec,
    benchmark_two_player_spec,
    technical_failures,
)


def _drop_timing(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in record.items() if key != "elapsed_seconds"}
        for record in records
    ]


def test_parallel_two_player_matches_serial_results(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text("def agent(obs):\n    return []\n", encoding="utf-8")
    kwargs = {
        "seeds": [0],
        "episode_steps": 8,
        "enable_comets": False,
        "act_timeout": 1.0,
    }

    serial = benchmark_two_player_spec(submission, "greedy", jobs=1, **kwargs)
    parallel = benchmark_two_player_spec(submission, "greedy", jobs=2, **kwargs)

    assert serial["opponent"] == parallel["opponent"] == "greedy"
    assert _drop_timing(serial["records"]) == _drop_timing(parallel["records"])


def test_benchmark_accepts_league_agent_opponent(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text("def agent(obs):\n    return []\n", encoding="utf-8")

    report = benchmark_two_player_spec(
        submission,
        "pgs_holdwave",
        seeds=[0],
        episode_steps=2,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )

    assert report["opponent"] == "pgs_holdwave"
    assert report["summary"]["games"] == 2


def test_benchmark_reports_submission_fallback_rate(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text(
        """
SUBMISSION_STATS = {"fallbacks": 0, "illegal_moves": 0, "fallback_errors": 0}

def agent(obs):
    SUBMISSION_STATS["fallbacks"] += 1
    return []
""",
        encoding="utf-8",
    )

    report = benchmark_two_player_spec(
        submission,
        "greedy",
        seeds=[0],
        episode_steps=8,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )

    assert report["summary"]["fallback_rate"] == 1.0
    assert report["summary"]["policy_illegal_move_rate"] == 0.0
    assert report["summary"]["fallback_error_rate"] == 0.0
    assert report["summary"]["instrumentation_missing_rate"] == 0.0
    assert all(record["fallbacks"] == record["decision_turns"] for record in report["records"])


def test_benchmark_reports_missing_submission_stats(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text("def agent(obs):\n    return []\n", encoding="utf-8")

    report = benchmark_two_player_spec(
        submission,
        "greedy",
        seeds=[0],
        episode_steps=8,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )

    assert report["summary"]["instrumentation_missing_rate"] == 1.0
    failures = technical_failures(
        {"formats": [{"format": "2p", "opponents": [report]}]},
        require_submission_stats=True,
    )
    assert failures == [
        {"label": "2p:greedy", "metric": "instrumentation_missing_rate", "value": 1.0}
    ]


def test_benchmark_crash_kills_submission_for_episode(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text(
        """
CALLS = 0
SUBMISSION_STATS = {"fallbacks": 0, "illegal_moves": 0, "fallback_errors": 0}

def agent(obs):
    global CALLS
    CALLS += 1
    raise RuntimeError("boom")
""",
        encoding="utf-8",
    )

    report = benchmark_two_player_spec(
        submission,
        "greedy",
        seeds=[0],
        episode_steps=8,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )

    record = report["records"][0]
    assert record["decision_turns"] == 1.0
    assert record["crashes"] == 1.0
    assert record["win_points"] == 0.0
    assert report["summary"]["crash_rate"] == 1.0


def test_parallel_four_player_matches_serial_lineups(tmp_path: Path):
    submission = tmp_path / "submission.py"
    submission.write_text("def agent(obs):\n    return []\n", encoding="utf-8")
    kwargs = {
        "seeds": [0, 1],
        "episode_steps": 8,
        "enable_comets": False,
        "act_timeout": 1.0,
    }

    serial = benchmark_four_player_spec(submission, ["greedy", "rush"], jobs=1, **kwargs)
    parallel = benchmark_four_player_spec(submission, ["greedy", "rush"], jobs=2, **kwargs)

    assert serial["opponents"] == parallel["opponents"] == ["greedy", "rush"]
    assert _drop_timing(serial["records"]) == _drop_timing(parallel["records"])


def test_four_player_stateful_specs_get_isolated_live_seats(monkeypatch):
    import scripts.benchmark_submission as bs

    made = []

    def fake_isolated(name, count):
        out = []
        for idx in range(count):
            def policy(state, player, *, marker=(name, idx)):
                return []

            out.append(policy)
            made.append(policy)
        return out

    monkeypatch.setattr(bs, "STATEFUL_SINGLETON_OPPONENTS", frozenset({"stateful"}))
    monkeypatch.setattr(bs, "get_isolated_opponents", fake_isolated)

    first, second = _opponent_instances(["stateful", "stateful"])

    assert first[0] == second[0] == "stateful"
    assert first[1] is not second[1]
    assert made == [first[1], second[1]]


def test_four_player_league_specs_get_fresh_instances():
    first, second = _opponent_instances(["pgs_holdwave", "pgs_holdwave"])

    assert first[0] == second[0] == "pgs_holdwave"
    assert first[1] is not second[1]
