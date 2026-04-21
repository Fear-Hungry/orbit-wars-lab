from __future__ import annotations

import pytest
from python.agents.submission_adapter import safe_submission_agent

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.backend import BackendUnavailable, RustBatchBackend
from orbit_wars_gym.encoding import encode_state, observation_dim
from orbit_wars_gym.gym_env import OrbitWarsGymEnv
from orbit_wars_gym.observation import OFFICIAL_OBSERVATION_KEYS, to_official_observation


def test_rust_backend_states_work_with_python_consumers():
    try:
        sim = RustBatchBackend(num_envs=1, num_players=2, seed=0)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    state = sim.reset(123)[0]

    assert isinstance(state["planets"][0], dict)

    obs = encode_state(state, 0)
    moves = greedy_moves(state, 0)

    assert obs.shape == (observation_dim(),)
    assert isinstance(moves, list)


def test_backend_can_materialize_official_observation_schema():
    try:
        sim = RustBatchBackend(num_envs=1, num_players=2, seed=0)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    state = sim.reset(123)[0]
    obs = to_official_observation(state, player=0)

    assert all(key in obs for key in OFFICIAL_OBSERVATION_KEYS)
    assert obs["player"] == 0
    assert isinstance(obs["initial_planets"], list)
    assert isinstance(obs["next_fleet_id"], int)
    assert isinstance(obs["comets"], list)
    assert isinstance(obs["comet_planet_ids"], list)

    moves = safe_submission_agent(obs)
    assert isinstance(moves, list)


def test_backend_step_accepts_official_move_triplets():
    try:
        sim = RustBatchBackend(num_envs=1, num_players=2, seed=0)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    sim.reset(123)
    outcomes = sim.step([[[[0, 0.0, 5]], []]])

    assert isinstance(outcomes, list)
    assert len(outcomes) == 1
    assert "rewards" in outcomes[0]


def test_backend_rejects_invalid_player_count():
    with pytest.raises(ValueError, match="2 or 4 players"):
        RustBatchBackend(num_envs=1, num_players=3, seed=0)


def test_gym_env_steps_with_rust_backend_dict_entities():
    try:
        env = OrbitWarsGymEnv(num_players=2, seed=0)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = env.step([0, 0, 1, 2])

    assert obs.shape == (observation_dim(),)
    assert next_obs.shape == (observation_dim(),)
    assert isinstance(info, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "scores" in step_info
