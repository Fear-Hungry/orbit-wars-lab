"""Regression: league eval and dataset collection must not MASK agent crashes.

Locked-down bug (2026-06-10): scripts/league_match.py and
scripts/collect_value_dataset.py swallowed agent exceptions with
``except Exception: moves = []`` and no counter — a crashing candidate (e.g. an
experimental pgs_valuenet) just "passed" every turn, polluting H2H results and
the value-net dataset with games it never actually played. Both now count
crashes per agent; the collector additionally DROPS the captures of a
contaminated game (outcome labels there reflect a pass-bot, not the agent).
"""
from __future__ import annotations

import random


def _fake_make(name):
    if name == "crasher":
        def crasher(obs):
            raise RuntimeError("boom")
        return crasher

    def passer(obs):
        return []
    return passer


def test_league_match_counts_crashes_per_agent(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    decision_ms: dict = {}
    crashes: dict = {}
    games = lm.play_batch(["crasher", "passer"], [123], 5, decision_ms, crashes)
    assert len(games) == 1
    assert crashes["crasher"] == 1, "Kaggle semantics: first crash kills the agent"
    assert crashes.get("passer", 0) == 0


def test_collect_value_dataset_counts_crashes_and_drops_contaminated_game(monkeypatch):
    import scripts.collect_value_dataset as cv

    monkeypatch.setattr(cv, "make", _fake_make)
    obs, lab, crashes = cv.play_and_capture(
        ["crasher", "passer"], seed=0, steps=10, subsample=5, rng=random.Random(0)
    )
    assert crashes["crasher"] == 10
    assert obs.shape[0] == 0 and lab.shape[0] == 0, (
        "a game with agent crashes must contribute NO dataset rows — its outcome "
        "labels describe a pass-bot, not the agent"
    )


def test_benchmark_submission_counts_non_list_return_as_invalid():
    import scripts.benchmark_submission as bs

    policy = bs._submission_runtime(lambda obs: "not-a-move-list")
    scores, runtime_stats = bs._run_match(
        [policy, lambda state, player: []],
        seed=123,
        episode_steps=3,
        enable_comets=True,
        act_timeout=1.0,
    )
    assert len(scores) == 2
    assert runtime_stats[0]["invalid_actions"] > 0
