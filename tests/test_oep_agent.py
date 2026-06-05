from __future__ import annotations

from pathlib import Path

from scripts.benchmark_submission import benchmark_two_player_spec


def test_oep_agent_runs_legal_smoke_against_producer() -> None:
    report = benchmark_two_player_spec(
        Path("bots/oep/agent.py"),
        "producer",
        seeds=[0],
        episode_steps=32,
        enable_comets=False,
        act_timeout=1.0,
        jobs=1,
    )
    records = report["records"]
    assert len(records) == 2
    assert all(record["crashes"] == 0.0 for record in records)
    assert all(record["timeouts"] == 0.0 for record in records)
    assert all(record["invalid_actions"] == 0.0 for record in records)
