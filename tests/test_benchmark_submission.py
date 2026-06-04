from __future__ import annotations

from pathlib import Path

from scripts.benchmark_submission import benchmark_four_player_spec, benchmark_two_player_spec


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
