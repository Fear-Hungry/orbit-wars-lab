from __future__ import annotations

from copy import deepcopy

import pytest

from orbit_wars_gym.parity import assert_official_snapshot_parity
from orbit_wars_gym.snapshots import official_initial_snapshot, snapshot_to_state


def test_parity_assertion_uses_exact_and_default_float_rules():
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=False)

    assert_official_snapshot_parity(state, snapshot, done=False)


def test_parity_assertion_rejects_default_tolerance_drift() -> None:
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=False)
    state["planets"][0][2] += 1e-6

    with pytest.raises(AssertionError):
        assert_official_snapshot_parity(state, snapshot, done=False)


def test_parity_assertion_requires_justification_to_relax_float_tolerance() -> None:
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=False)
    state["planets"][0][2] += 1e-6

    with pytest.raises(ValueError, match="explicit justification"):
        assert_official_snapshot_parity(state, snapshot, done=False, float_tolerance=1e-6)


def test_parity_assertion_allows_justified_float_relaxation() -> None:
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=False)
    state["planets"][0][2] += 1e-6

    assert_official_snapshot_parity(
        state,
        snapshot,
        done=False,
        float_tolerance=1e-6,
        float_tolerance_reason="official float accumulation differs from local serialization",
    )


def test_parity_assertion_keeps_done_exact() -> None:
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=True)

    with pytest.raises(AssertionError):
        assert_official_snapshot_parity(state, snapshot, done=False)


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, 999),
        (1, 7),
        (5, 123),
    ],
)
def test_parity_assertion_keeps_planet_id_owner_and_ships_exact(index: int, value: int) -> None:
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2, done=False)
    state["planets"][0][index] = value

    with pytest.raises(AssertionError):
        assert_official_snapshot_parity(state, snapshot, done=False)


def test_parity_assertion_keeps_fleet_id_owner_ships_y_and_angle_exact() -> None:
    snapshot = {
        "planets": [[0, -1, 10.0, 20.0, 2.0, 5, 1]],
        "fleets": [[7, 1, 11.0, 21.0, 0.25, 0, 6]],
    }
    state = {
        "done": False,
        "planets": deepcopy(snapshot["planets"]),
        "fleets": deepcopy(snapshot["fleets"]),
    }

    for index, value in ((0, 8), (1, 0), (6, 9)):
        broken = deepcopy(state)
        broken["fleets"][0][index] = value
        with pytest.raises(AssertionError):
            assert_official_snapshot_parity(broken, snapshot, done=False)

    for index in (3, 4):
        drifted = deepcopy(state)
        drifted["fleets"][0][index] += 1e-6
        with pytest.raises(AssertionError):
            assert_official_snapshot_parity(drifted, snapshot, done=False)
