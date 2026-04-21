import math

from orbit_wars_gym.action_decoder import decode_discrete_action, greedy_moves


def test_greedy_moves_returns_legal_shape():
    state = {
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 20, 2],
            [1, -1, 30.0, 20.0, 2.0, 5, 3],
        ],
        "fleets": [],
        "step": 0,
    }
    moves = greedy_moves(state, 0)
    assert isinstance(moves, list)
    assert len(moves[0]) == 3
    assert moves[0][0] == 0
    assert moves[0][2] > 0


def test_greedy_moves_accepts_dict_entities():
    state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 1, "owner": -1, "x": 30.0, "y": 20.0, "radius": 2.0, "ships": 5, "production": 3},
        ],
        "fleets": [],
        "step": 0,
    }
    moves = greedy_moves(state, 0)
    assert isinstance(moves, list)
    assert len(moves[0]) == 3
    assert moves[0][0] == 0
    assert moves[0][2] > 0


def test_decode_discrete_action_accepts_dict_entities():
    state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 20, "production": 2},
            {"id": 1, "owner": -1, "x": 30.0, "y": 20.0, "radius": 2.0, "ships": 5, "production": 3},
            {"id": 2, "owner": 1, "x": 80.0, "y": 75.0, "radius": 2.0, "ships": 18, "production": 3},
        ],
        "fleets": [],
        "step": 0,
    }
    moves = decode_discrete_action(state, 0, [0, 0, 1, 2])
    assert isinstance(moves, list)
    assert len(moves[0]) == 3
    assert moves[0][0] == 0
    assert moves[0][2] > 0


def test_decode_discrete_action_leads_rotating_targets():
    state = {
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 40, "production": 2},
            {"id": 1, "owner": -1, "x": 50.0, "y": 30.0, "radius": 2.0, "ships": 5, "production": 5},
            {"id": 2, "owner": 1, "x": 80.0, "y": 75.0, "radius": 2.0, "ships": 18, "production": 3},
        ],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.08,
    }
    static_state = {**state, "angular_velocity": 0.0}

    rotating_moves = decode_discrete_action(state, 0, [0, 0, 3, 2])
    static_moves = decode_discrete_action(static_state, 0, [0, 0, 3, 2])

    assert rotating_moves[0][0] == 0
    assert rotating_moves[0][2] > 0
    assert not math.isclose(rotating_moves[0][1], static_moves[0][1])
