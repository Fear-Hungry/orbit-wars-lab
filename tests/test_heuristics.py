from python.agents.heuristics import anti_meta_agent, defensive_agent, rush_agent, weak_random_agent

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
    assert moves


def test_rush_agent_accepts_dict_entities():
    moves = rush_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_anti_meta_agent_accepts_dict_entities():
    moves = anti_meta_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_weak_random_agent_accepts_dict_entities():
    moves = weak_random_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves
