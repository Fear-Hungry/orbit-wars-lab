"""Regression: the OEP tarball wrapper must INSTRUMENT every Producer fallback.

Locked-down bug (2026-06-11, mirror of the PGS fix from 2026-06-10):
``package_oep_submission.MAIN_TEMPLATE`` swallowed OEP exceptions
(``except Exception: pass``) and returned the Producer plan with no
SUBMISSION_STATS in main.py's namespace. benchmark_submission detects fallbacks
ONLY via ``agent.__globals__["SUBMISSION_STATS"]``, so a tarball whose OEP died
on every step still benchmarked as "0 fallbacks" — the episode COMPLETEs on
Kaggle while secretly running the Producer. The template now mirrors the PGS
instrumentation (calls/fallbacks/timeouts/fallback_errors) and stops launching
OEP after an overrun (timed-out daemon threads keep running and can mutate
runtime state).
"""
from __future__ import annotations

import os
import sys
import time
import types

from scripts.package_oep_submission import MAIN_TEMPLATE

PRODUCER_SENTINEL = [[0, 0.0, 1]]


def _render_agent(monkeypatch, oep_fn, budget_s=0.05, min_advantage=15):
    """Exec MAIN_TEMPLATE exactly like Kaggle, with stubbed OEP/Producer."""
    # Pre-set the env key via monkeypatch so the template's setdefault no-ops
    # and the test process env is restored afterwards.
    monkeypatch.setitem(os.environ, "OEP_MIN_ADVANTAGE", str(min_advantage))
    for name in ("bots", "bots.oep", "bots.producer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    oep_mod = types.ModuleType("bots.oep.agent")
    oep_mod.agent = oep_fn
    prod_mod = types.ModuleType("bots.producer.agent")
    prod_mod.agent = lambda obs: PRODUCER_SENTINEL
    monkeypatch.setitem(sys.modules, "bots.oep.agent", oep_mod)
    monkeypatch.setitem(sys.modules, "bots.producer.agent", prod_mod)
    ns: dict = {}
    src = MAIN_TEMPLATE.format(min_advantage=min_advantage, budget_s=budget_s)
    exec(compile(src, "main.py", "exec"), ns)
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


def test_min_advantage_is_baked_into_env_setdefault(monkeypatch):
    src = MAIN_TEMPLATE.format(min_advantage=15, budget_s=0.6)
    assert 'os.environ.setdefault("OEP_MIN_ADVANTAGE", "15")' in src


def test_healthy_oep_records_zero_fallbacks(monkeypatch):
    oep_moves = [[1, 0.0, 2]]
    agent, ns = _render_agent(monkeypatch, lambda obs: oep_moves)
    for _ in range(5):
        assert agent({"player": 0}) is oep_moves
    assert ns["SUBMISSION_STATS"] == {
        "calls": 5, "fallbacks": 0, "timeouts": 0, "fallback_errors": 0,
    }


def test_dead_oep_counts_every_fallback(monkeypatch):
    def dead(obs):
        raise RuntimeError("OEP dies every step")

    agent, ns = _render_agent(monkeypatch, dead)
    for _ in range(10):
        assert agent({"player": 0}) is PRODUCER_SENTINEL
    assert ns["SUBMISSION_STATS"] == {
        "calls": 10, "fallbacks": 10, "timeouts": 0, "fallback_errors": 10,
    }


def test_benchmark_delta_logic_sees_the_fallback(monkeypatch):
    def dead(obs):
        raise RuntimeError("OEP dies every step")

    agent, _ = _render_agent(monkeypatch, dead)
    before = _benchmark_stats_snapshot(agent)
    agent({"player": 0})
    after = _benchmark_stats_snapshot(agent)
    delta = {k: after[k] - before[k] for k in before}
    assert delta["fallbacks"] == 1.0
    assert delta["fallback_errors"] == 1.0


def test_timeout_counts_and_killswitch_stops_launching_oep(monkeypatch):
    launches = []

    def slow(obs):
        launches.append(1)
        time.sleep(0.5)
        return [[1, 0.0, 2]]

    agent, ns = _render_agent(monkeypatch, slow, budget_s=0.01)
    for _ in range(5):
        assert agent({"player": 0}) is PRODUCER_SENTINEL
    max_consec = ns["_MAX_CONSEC_TIMEOUTS"]
    assert len(launches) == max_consec, "kill-switch must stop launching OEP"
    assert ns["SUBMISSION_STATS"] == {
        "calls": 5, "fallbacks": 5, "timeouts": 1, "fallback_errors": 0,
    }
