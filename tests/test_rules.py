from __future__ import annotations

from python.orbit_wars_gym.rules import moves_are_legal

STATE = {
    "planets": [
        [1, 0, 0.0, 0.0, 1.0, 10, 1],
        [2, 1, 5.0, 0.0, 1.0, 10, 1],
    ],
    "fleets": [],
}


def test_moves_are_legal_rejects_aggregate_source_overbudget() -> None:
    assert moves_are_legal(STATE, 0, [[1, 0.0, 6], [1, 0.1, 4]])
    assert not moves_are_legal(STATE, 0, [[1, 0.0, 6], [1, 0.1, 5]])


def test_moves_are_legal_rejects_malformed_numeric_fields() -> None:
    assert not moves_are_legal(STATE, 0, [["not-a-planet", 0.0, 1]])
    assert not moves_are_legal(STATE, 0, [[1, float("nan"), 1]])
    assert not moves_are_legal(STATE, 0, [[1, 0.0, 0]])
