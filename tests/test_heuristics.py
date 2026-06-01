import math

from python.agents.heuristics import anti_meta_agent, defensive_agent, rush_agent, weak_random_agent
from python.orbit_wars_gym.rules import moves_are_legal

STATE_DICT = {
    "planets": [
        {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 30, "production": 2},
        {"id": 1, "owner": 0, "x": 24.0, "y": 22.0, "radius": 2.0, "ships": 16, "production": 2},
        {"id": 2, "owner": -1, "x": 36.0, "y": 25.0, "radius": 2.0, "ships": 5, "production": 3},
        {"id": 3, "owner": 1, "x": 80.0, "y": 75.0, "radius": 2.0, "ships": 18, "production": 3},
        {"id": 4, "owner": 1, "x": 74.0, "y": 70.0, "radius": 2.0, "ships": 22, "production": 4},
    ],
    "fleets": [],
    "step": 7,
}


def _assert_legal_shape(moves):
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3
        assert move[2] > 0


def test_defensive_agent_accepts_dict_entities():
    moves = defensive_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves_are_legal(STATE_DICT, 0, moves)
    assert moves


def test_rush_agent_accepts_dict_entities():
    moves = rush_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves_are_legal(STATE_DICT, 0, moves)
    assert moves


def test_anti_meta_agent_accepts_dict_entities():
    moves = anti_meta_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves_are_legal(STATE_DICT, 0, moves)
    assert moves


def test_weak_random_agent_accepts_dict_entities():
    moves = weak_random_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves_are_legal(STATE_DICT, 0, moves)
    assert moves


def _angle_to(source, target):
    return math.atan2(target[1] - source[1], target[0] - source[0])


def _angle_distance(left, right):
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def test_defensive_agent_prefers_high_production_neutral_over_cheap_low_value_target():
    state = {
        "angular_velocity": 0.0,
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 70, "production": 2},
            {"id": 1, "owner": -1, "x": 28.0, "y": 20.0, "radius": 2.0, "ships": 2, "production": 1},
            {"id": 2, "owner": -1, "x": 42.0, "y": 40.0, "radius": 2.0, "ships": 8, "production": 5},
            {"id": 3, "owner": 1, "x": 82.0, "y": 78.0, "radius": 2.0, "ships": 20, "production": 3},
        ],
        "fleets": [],
        "step": 0,
    }

    move = defensive_agent(state, 0)[0]
    high_value_angle = _angle_to((20.0, 20.0), (42.0, 40.0))
    cheap_angle = _angle_to((20.0, 20.0), (28.0, 20.0))

    assert _angle_distance(move[1], high_value_angle) < _angle_distance(move[1], cheap_angle)


def test_rush_agent_focuses_high_economy_leader_instead_of_largest_side_planet():
    state = {
        "angular_velocity": 0.0,
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 80, "production": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 20.0, "radius": 2.0, "ships": 90, "production": 1},
            {"id": 2, "owner": 2, "x": 20.0, "y": 80.0, "radius": 2.0, "ships": 35, "production": 5},
            {"id": 3, "owner": 2, "x": 28.0, "y": 78.0, "radius": 2.0, "ships": 20, "production": 4},
        ],
        "fleets": [],
        "step": 0,
    }

    move = rush_agent(state, 0)[0]
    leader_angle = _angle_to((20.0, 20.0), (20.0, 80.0))
    side_angle = _angle_to((20.0, 20.0), (80.0, 20.0))

    assert _angle_distance(move[1], leader_angle) < _angle_distance(move[1], side_angle)


def test_anti_meta_agent_presses_ffa_leader_when_multiple_enemy_players_exist():
    state = {
        "angular_velocity": 0.0,
        "planets": [
            {"id": 0, "owner": 0, "x": 18.0, "y": 18.0, "radius": 2.0, "ships": 76, "production": 2},
            {"id": 1, "owner": 1, "x": 82.0, "y": 18.0, "radius": 2.0, "ships": 80, "production": 1},
            {"id": 2, "owner": 2, "x": 18.0, "y": 82.0, "radius": 2.0, "ships": 28, "production": 5},
            {"id": 3, "owner": 3, "x": 82.0, "y": 82.0, "radius": 2.0, "ships": 25, "production": 2},
            {"id": 4, "owner": -1, "x": 45.0, "y": 18.0, "radius": 2.0, "ships": 4, "production": 1},
        ],
        "fleets": [],
        "step": 0,
    }

    move = anti_meta_agent(state, 0)[0]
    leader_angle = _angle_to((18.0, 18.0), (18.0, 82.0))
    side_angle = _angle_to((18.0, 18.0), (82.0, 18.0))

    assert _angle_distance(move[1], leader_angle) < _angle_distance(move[1], side_angle)
