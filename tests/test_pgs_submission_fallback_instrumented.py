"""Regression: the PGS tarball wrapper must INSTRUMENT every Producer fallback.

Locked-down bug (2026-06-10): MAIN_TEMPLATE swallowed PGS exceptions
(``except Exception: pass``) and returned the Producer plan with no
SUBMISSION_STATS in main.py's namespace. benchmark_submission detects fallbacks
ONLY via ``agent.__globals__["SUBMISSION_STATS"]``, so a tarball whose PGS died
on every step still benchmarked as "0 fallbacks" — exactly the silent
degradation docs/SUBMISSION.md forbids. The template now mirrors the BReP
instrumentation (calls/fallbacks/timeouts/fallback_errors) and pauses PGS while
a timed-out daemon thread is still alive, then resumes once it finishes.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
from scripts.package_pgs_submission import MAIN_TEMPLATE

PRODUCER_SENTINEL = [[0, 0.0, 1]]


def _render_agent(monkeypatch, pgs_fn, budget_s=0.05, notify_fn=None):
    """Exec MAIN_TEMPLATE exactly like Kaggle, with stubbed PGS/Producer."""
    for name in ("bots", "bots.pgs", "bots.producer"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    pgs_mod = types.ModuleType("bots.pgs.agent")
    pgs_mod.agent = pgs_fn
    pgs_mod.notify_fallback_applied = notify_fn or (lambda: None)
    prod_mod = types.ModuleType("bots.producer.agent")
    prod_mod.agent = lambda obs: PRODUCER_SENTINEL
    monkeypatch.setitem(sys.modules, "bots.pgs.agent", pgs_mod)
    monkeypatch.setitem(sys.modules, "bots.producer.agent", prod_mod)
    ns: dict = {}
    exec(compile(MAIN_TEMPLATE.format(budget_s=budget_s), "main.py", "exec"), ns)
    return ns["agent"], ns


def _benchmark_stats_snapshot(agent):
    """Replicates benchmark_submission._submission_runtime's detection logic."""
    raw = getattr(agent, "__globals__", {}).get("SUBMISSION_STATS", {})
    if not isinstance(raw, dict):
        return {}
    return {
        name: float(raw.get(name, 0.0))
        for name in ("fallbacks", "illegal_moves", "fallback_errors")
    }


def test_agent_is_last_callable_and_stats_in_globals(monkeypatch):
    agent, ns = _render_agent(monkeypatch, lambda obs: [[1, 0.0, 2]])
    last = [n for n, v in ns.items() if callable(v) and not n.startswith("__")][-1]
    assert last == "agent", "Kaggle picks the LAST callable in main.py"
    assert isinstance(agent.__globals__.get("SUBMISSION_STATS"), dict)


def test_healthy_pgs_records_zero_fallbacks(monkeypatch):
    pgs_moves = [[1, 0.0, 2]]
    agent, ns = _render_agent(monkeypatch, lambda obs: pgs_moves)
    for _ in range(5):
        assert agent({"player": 0}) is pgs_moves
    assert ns["SUBMISSION_STATS"] == {
        "calls": 5, "fallbacks": 0, "timeouts": 0,
        "timeout_thread_blocks": 0, "fallback_errors": 0,
        "budget_floor_returns": 0,
    }


def test_dead_pgs_counts_every_fallback(monkeypatch):
    def dead(obs):
        raise RuntimeError("PGS dies every step")

    agent, ns = _render_agent(monkeypatch, dead)
    for _ in range(10):
        assert agent({"player": 0}) is PRODUCER_SENTINEL
    assert ns["SUBMISSION_STATS"] == {
        "calls": 10, "fallbacks": 10, "timeouts": 0,
        "timeout_thread_blocks": 0, "fallback_errors": 10,
        "budget_floor_returns": 0,
    }


def test_benchmark_delta_logic_sees_the_fallback(monkeypatch):
    def dead(obs):
        raise RuntimeError("PGS dies every step")

    agent, _ = _render_agent(monkeypatch, dead)
    before = _benchmark_stats_snapshot(agent)
    agent({"player": 0})
    after = _benchmark_stats_snapshot(agent)
    delta = {k: after[k] - before[k] for k in before}
    assert delta["fallbacks"] == 1.0
    assert delta["fallback_errors"] == 1.0


def test_fallback_notifies_pgs_runtime_reset(monkeypatch):
    resets = []

    def dead(obs):
        raise RuntimeError("PGS dies every step")

    agent, _ = _render_agent(monkeypatch, dead, notify_fn=lambda: resets.append(1))

    assert agent({"player": 0}) is PRODUCER_SENTINEL
    assert resets == [1]


def test_timeout_blocks_until_thread_finishes_then_resumes_pgs(monkeypatch):
    launches = []
    release = threading.Event()

    def slow(obs):
        launches.append(1)
        release.wait(timeout=1.0)
        return [[1, 0.0, 2]]

    agent, ns = _render_agent(monkeypatch, slow, budget_s=0.01)
    for _ in range(5):
        assert agent({"player": 0}) is PRODUCER_SENTINEL
    assert len(launches) == 1, "wrapper must not overlap timed-out PGS threads"
    assert ns["SUBMISSION_STATS"] == {
        "calls": 5, "fallbacks": 5, "timeouts": 1,
        "timeout_thread_blocks": 4, "fallback_errors": 0,
        "budget_floor_returns": 0,
    }

    release.set()
    ns["_active_timeout_thread"][0].join(timeout=1.0)
    assert agent({"player": 0}) == [[1, 0.0, 2]]
    assert len(launches) == 2, "wrapper should resume PGS after the overrun finishes"
    assert ns["SUBMISSION_STATS"] == {
        "calls": 6, "fallbacks": 5, "timeouts": 1,
        "timeout_thread_blocks": 4, "fallback_errors": 0,
        "budget_floor_returns": 0,
    }


def test_runtime_budget_floor_counter_is_folded_into_submission_stats(monkeypatch):
    """The wrapper must surface the planner's internal budget_floor_returns —
    including across the step-0 runtime recreation (counter restarts)."""
    counters = iter([1, 3, 1])  # game 1: 1→3; new runtime (reset): 1

    def healthy(obs):
        return [[1, 0.0, 2]]

    agent, ns = _render_agent(monkeypatch, healthy)
    ns["_pgs_agent"].runtime_stats = lambda: {"budget_floor_returns": next(counters)}
    for _ in range(3):
        agent({"player": 0})
    # deltas: +1 (0→1), +2 (1→3), +1 (3→reset→1)
    assert ns["SUBMISSION_STATS"]["budget_floor_returns"] == 4


def test_packager_rejects_unbundled_value_net_config(monkeypatch, tmp_path):
    from scripts.package_pgs_submission import main

    monkeypatch.setattr(sys, "argv", [
        "package_pgs_submission.py",
        "--pgs-config",
        'scripts="hold", value_net_path="artifacts/h7/value_net.pt"',
        "--out",
        str(tmp_path / "submission_pgs.tar.gz"),
    ])
    with pytest.raises(SystemExit, match="value_net_path is not submission-safe"):
        main()
