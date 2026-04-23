import math

from python.agents.behavioral import BEHAVIORAL_POLICIES

STATE_DICT = {
    "planets": [
        {"id": 0, "owner": 0, "x": 18.0, "y": 18.0, "radius": 2.0, "ships": 42, "production": 3},
        {"id": 1, "owner": 0, "x": 27.0, "y": 24.0, "radius": 2.0, "ships": 24, "production": 2},
        {"id": 2, "owner": -1, "x": 40.0, "y": 28.0, "radius": 2.0, "ships": 6, "production": 4},
        {"id": 3, "owner": -1, "x": 52.0, "y": 42.0, "radius": 2.0, "ships": 12, "production": 2},
        {"id": 4, "owner": 1, "x": 78.0, "y": 76.0, "radius": 2.0, "ships": 30, "production": 4},
        {"id": 5, "owner": 2, "x": 70.0, "y": 30.0, "radius": 2.0, "ships": 34, "production": 5},
    ],
    "fleets": [
        {"id": 20, "owner": 1, "x": 38.0, "y": 36.0, "angle": 0.0, "from_planet_id": 4, "ships": 8}
    ],
    "step": 12,
}


def _assert_legal_shape(state: dict, player: int, moves: list[list[float]]) -> None:
    own_ids = {planet["id"] for planet in state["planets"] if planet["owner"] == player}
    assert isinstance(moves, list)
    assert moves
    for move in moves:
        assert isinstance(move, list)
        assert len(move) == 3
        assert int(move[0]) in own_ids
        assert math.isfinite(float(move[1]))
        assert int(move[2]) > 0


def test_behavioral_policy_mapping_smoke_returns_legal_moves_for_dict_state():
    assert set(BEHAVIORAL_POLICIES) == {
        "enemy_pressure",
        "neutral_economy_greed",
        "frontier_economy",
        "risk_balanced_defense",
        "leader_denial",
    }

    for policy in BEHAVIORAL_POLICIES.values():
        moves = policy(STATE_DICT, 0)
        _assert_legal_shape(STATE_DICT, 0, moves)


def test_behavioral_policies_accept_list_entities():
    state = {
        "planets": [
            [planet["id"], planet["owner"], planet["x"], planet["y"], planet["radius"], planet["ships"], planet["production"]]
            for planet in STATE_DICT["planets"]
        ],
        "fleets": [
            [
                fleet["id"],
                fleet["owner"],
                fleet["x"],
                fleet["y"],
                fleet["angle"],
                fleet["from_planet_id"],
                fleet["ships"],
            ]
            for fleet in STATE_DICT["fleets"]
        ],
        "step": STATE_DICT["step"],
    }

    for policy in BEHAVIORAL_POLICIES.values():
        moves = policy(state, 0)
        assert isinstance(moves, list)
        assert moves
        for move in moves:
            assert len(move) == 3
            assert int(move[0]) in {0, 1}
            assert math.isfinite(float(move[1]))
            assert int(move[2]) > 0
