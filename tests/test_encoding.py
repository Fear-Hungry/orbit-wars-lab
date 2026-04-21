from orbit_wars_gym.encoding import encode_state, observation_dim


def test_encoding_dim():
    state = {
        "planets": [[0, 0, 20.0, 20.0, 2.0, 20, 2]],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.03,
    }
    obs = encode_state(state, 0)
    assert obs.shape == (observation_dim(),)


def test_encoding_accepts_dict_entities():
    state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 1, "owner": -1, "x": 40.0, "y": 30.0, "radius": 2.0, "ships": 5, "production": 3},
        ],
        "fleets": [
            {"id": 0, "owner": 0, "x": 21.0, "y": 20.5, "angle": 0.25, "from_planet_id": 0, "ships": 8},
        ],
        "step": 3,
        "angular_velocity": 0.03,
    }
    obs = encode_state(state, 0)
    assert obs.shape == (observation_dim(),)
