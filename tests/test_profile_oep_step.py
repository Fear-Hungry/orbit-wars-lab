from __future__ import annotations

from scripts.profile_oep_step import _percentile


def test_percentile_interpolates_sorted_values() -> None:
    assert _percentile([10.0, 20.0, 30.0], 0.0) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 1.0) == 30.0
    assert _percentile([10.0, 20.0], 0.25) == 12.5


def test_percentile_handles_empty_and_singleton_values() -> None:
    assert _percentile([], 0.95) == 0.0
    assert _percentile([42.0], 0.95) == 42.0
