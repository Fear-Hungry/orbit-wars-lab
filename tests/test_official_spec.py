from __future__ import annotations

from kaggle_environments import make

from orbit_wars_gym.backend import RustConfig
from orbit_wars_gym.observation import OFFICIAL_OBSERVATION_KEYS


def _orbit_wars_spec() -> dict:
    env = make("orbit_wars", debug=True)
    return dict(env.specification)


def test_official_target_is_kaggle_orbit_wars():
    spec = _orbit_wars_spec()
    assert spec["name"] == "orbit_wars"
    assert spec["version"] == "1.0.9"


def test_local_defaults_match_official_runtime_constants():
    spec = _orbit_wars_spec()
    cfg = RustConfig()
    config = spec["configuration"]

    assert config["episodeSteps"]["default"] == cfg.episode_steps == 500
    assert config["actTimeout"]["default"] == cfg.act_timeout == 1
    assert spec["agents"] == [2, 4]
    assert config["shipSpeed"]["default"] == cfg.ship_speed == 6.0
    assert config["cometSpeed"]["default"] == cfg.comet_speed == 4.0


def test_official_observation_keys_match_local_contract():
    spec = _orbit_wars_spec()
    observation = spec["observation"]

    for key in OFFICIAL_OBSERVATION_KEYS:
        assert key in observation


def test_official_action_schema_matches_move_triplets():
    spec = _orbit_wars_spec()
    action = spec["action"]

    assert action["type"] == "array"
    assert action["default"] == []
    assert "[from_planet_id, direction_angle, num_ships]" in action["description"]
