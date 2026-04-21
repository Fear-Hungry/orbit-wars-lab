from __future__ import annotations

from python.agents.submission_adapter import safe_submission_agent

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.encoding import encode_state, observation_dim
from orbit_wars_gym.observation import OFFICIAL_OBSERVATION_KEYS, to_official_observation
from orbit_wars_gym.snapshots import official_initial_snapshot, snapshot_to_state


def test_official_initial_snapshot_round_trips_through_local_state():
    snapshot = official_initial_snapshot(num_players=2, player=0)
    state = snapshot_to_state(snapshot, num_players=2)
    round_trip = to_official_observation(state, player=0)

    assert tuple(round_trip.keys())[: len(OFFICIAL_OBSERVATION_KEYS)] == OFFICIAL_OBSERVATION_KEYS
    for key in OFFICIAL_OBSERVATION_KEYS:
        assert round_trip[key] == snapshot[key]


def test_official_initial_snapshot_is_usable_by_local_consumers():
    snapshot = official_initial_snapshot(num_players=4, player=0)
    state = snapshot_to_state(snapshot, num_players=4)

    obs = encode_state(state, player=0)
    greedy = greedy_moves(state, player=0)
    submission_moves = safe_submission_agent(to_official_observation(state, player=0))

    assert obs.shape == (observation_dim(),)
    assert isinstance(greedy, list)
    assert isinstance(submission_moves, list)
