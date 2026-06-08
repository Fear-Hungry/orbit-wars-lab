"""Regression locks for the runtime errors that broke PPO training (2026-06-08).

Each test pins a bug that previously crashed or silently wasted a training run,
so the "treino sem erros" guarantee can't quietly regress.
"""

from __future__ import annotations

import scripts.run_campaign as rc
from scripts.trace_submission_actions import _move_signature

# --- #5: tracer dropped real 3-element moves (reported "never launches") -------

def test_move_signature_keeps_three_element_moves() -> None:
    # Real moves are [source_planet_id, angle, ships] (3 elements). The tracer
    # used to require len >= 4 and dropped every move → false "never launches".
    moves = [[27, -1.889, 49], [3, 0.5, 12]]
    sig = _move_signature(moves)
    assert len(sig) == 2
    assert (27, 49, -1.889) in sig  # (source, ships, angle_rounded)


def test_move_signature_drops_malformed_moves() -> None:
    assert _move_signature([]) == []
    assert _move_signature([[1, 2]]) == []  # < 3 elements → dropped, no IndexError


# --- #1: memory guard (anti-OOM on a shared box) -------------------------------

def test_mem_available_gb_returns_positive_float() -> None:
    val = rc._mem_available_gb()
    assert isinstance(val, float)
    assert val > 0.0  # /proc/meminfo readable on Linux


def test_wait_for_memory_returns_immediately_when_above_floor(monkeypatch) -> None:
    monkeypatch.setattr(rc, "_mem_available_gb", lambda: 100.0)
    calls = {"sleep": 0}
    monkeypatch.setattr(rc.time, "sleep", lambda _s: calls.__setitem__("sleep", calls["sleep"] + 1))
    rc._wait_for_memory(4.0, chunk=0)
    assert calls["sleep"] == 0  # plenty of RAM → never waits


def test_wait_for_memory_waits_then_resumes_when_mem_recovers(monkeypatch) -> None:
    # Exercises the wait-then-resume path the smoke never triggered (RAM stayed high).
    seq = iter([1.0, 1.0, 10.0])  # low, low, then recovered
    monkeypatch.setattr(rc, "_mem_available_gb", lambda: next(seq))
    calls = {"sleep": 0}
    monkeypatch.setattr(rc.time, "sleep", lambda _s: calls.__setitem__("sleep", calls["sleep"] + 1))
    rc._wait_for_memory(4.0, chunk=0)
    assert calls["sleep"] == 2  # waited twice (mem < 4), then resumed at 10GB


# --- #4: keep-best froze cumulative training (every chunk restarted from c00) ---

def test_no_improve_does_not_accrue_on_floor_tie() -> None:
    # The bug: ties at the loss floor counted as "no progress" and early-stopped
    # before cumulative training could climb out. A tie at/below the floor must NOT
    # increment the counter.
    assert rc._accrue_no_improve(-1.0, -1.0, 2, floor_margin=-0.99) == 2  # tie at floor → frozen
    assert rc._accrue_no_improve(-0.5, -0.7, 3, floor_margin=-0.99) == 0  # improved → reset
    assert rc._accrue_no_improve(-0.6, -0.5, 2, floor_margin=-0.99) == 3  # above floor, worse → +1


def test_reset_to_best_only_on_real_regression() -> None:
    # The bug: prev reset to best.pt every chunk on ties → zero cumulative training.
    assert rc._reset_to_best(-1.0, -1.0, 0.05, has_best=True) is False  # tie → continue cumulatively
    assert rc._reset_to_best(-0.50, -0.75, 0.05, has_best=True) is False  # improved → continue
    assert rc._reset_to_best(-0.83, -0.75, 0.05, has_best=True) is True   # real regression → reset
    assert rc._reset_to_best(-0.83, -0.75, 0.05, has_best=False) is False  # no best yet → continue
