from python.agents.heuristics import (
    anti_meta_agent,
    coalition_field_control_agent,
    defensive_agent,
    field_control_agent,
    greedy_agent,
    opening_gate_anti_meta_meta_agent,
    opening_gate_meta_agent,
    opening_gate_rush_meta_agent,
    rush_agent,
    tail_expansion_agent,
    weak_random_agent,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig

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


def _state_for_seed(seed: int) -> dict:
    backend = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig())
    return backend.reset(seed)[0]


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


def test_field_control_agent_accepts_dict_entities():
    moves = field_control_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_coalition_field_control_agent_accepts_dict_entities():
    ffa_state = {
        "planets": STATE_DICT["planets"]
        + [
            {
                "id": 5,
                "owner": 2,
                "x": 68.0,
                "y": 28.0,
                "radius": 2.0,
                "ships": 14,
                "production": 3,
            },
            {
                "id": 6,
                "owner": 3,
                "x": 72.0,
                "y": 32.0,
                "radius": 2.0,
                "ships": 19,
                "production": 4,
            },
        ],
        "fleets": [],
        "step": 9,
    }
    moves = coalition_field_control_agent(ffa_state, 0)
    _assert_legal_shape(moves)
    assert moves


def test_opening_gate_meta_agent_accepts_dict_entities():
    moves = opening_gate_meta_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_opening_gate_rush_meta_agent_accepts_dict_entities():
    moves = opening_gate_rush_meta_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_opening_gate_anti_meta_meta_agent_accepts_dict_entities():
    moves = opening_gate_anti_meta_meta_agent(STATE_DICT, 0)
    _assert_legal_shape(moves)
    assert moves


def test_opening_gate_rush_meta_uses_rush_on_close_axis_ffa_opening():
    ffa_state = {
        "planets": [
            {
                "id": 4,
                "owner": 0,
                "x": 98.8,
                "y": 56.2,
                "radius": 2.0,
                "ships": 10,
                "production": 1,
            },
            {"id": 5, "owner": 1, "x": 1.2, "y": 56.2, "radius": 2.0, "ships": 10, "production": 1},
            {
                "id": 6,
                "owner": 2,
                "x": 98.8,
                "y": 43.8,
                "radius": 2.0,
                "ships": 10,
                "production": 1,
            },
            {"id": 7, "owner": 3, "x": 1.2, "y": 43.8, "radius": 2.0, "ships": 10, "production": 1},
            {
                "id": 20,
                "owner": -1,
                "x": 82.5,
                "y": 65.1,
                "radius": 2.0,
                "ships": 19,
                "production": 2,
            },
        ],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.0387,
    }
    assert opening_gate_rush_meta_agent(ffa_state, 0) == rush_agent(ffa_state, 0)


def test_opening_gate_rush_meta_uses_anti_meta_on_localized_2p_rush_state():
    rush_state = {
        "planets": [
            {
                "id": 4,
                "owner": 0,
                "x": 59.1,
                "y": 97.7,
                "radius": 2.0,
                "ships": 10,
                "production": 2,
            },
            {"id": 7, "owner": 1, "x": 40.9, "y": 2.3, "radius": 2.0, "ships": 10, "production": 2},
            {
                "id": 28,
                "owner": -1,
                "x": 76.0,
                "y": 91.8,
                "radius": 2.0,
                "ships": 14,
                "production": 4,
            },
            {
                "id": 16,
                "owner": -1,
                "x": 74.4,
                "y": 78.7,
                "radius": 2.0,
                "ships": 12,
                "production": 1,
            },
        ],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.0465,
    }
    assert opening_gate_rush_meta_agent(rush_state, 0) == anti_meta_agent(rush_state, 0)


def test_opening_gate_rush_meta_locks_field_control_opening_signature():
    state = _state_for_seed(1490000)
    assert opening_gate_rush_meta_agent(state, 0) == field_control_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_rush_meta_agent(follow_up, 0) == field_control_agent(follow_up, 0)


def test_opening_gate_rush_meta_locks_anti_meta_opening_signature():
    state = _state_for_seed(170000)
    assert opening_gate_rush_meta_agent(state, 0) == anti_meta_agent(state, 0)


def test_opening_gate_rush_meta_locks_greedy_opening_signature():
    state = _state_for_seed(2330000)
    assert opening_gate_rush_meta_agent(state, 0) == greedy_agent(state, 0)


def test_opening_gate_rush_meta_locks_rush_then_greedy_signature():
    state = _state_for_seed(2330237)
    assert opening_gate_rush_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 5
    assert opening_gate_rush_meta_agent(follow_up, 0) == greedy_agent(follow_up, 0)


def test_opening_gate_rush_meta_uses_player0_rush_then_defensive_signature():
    state = _state_for_seed(530146)
    assert opening_gate_rush_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_rush_meta_agent(follow_up, 0) == defensive_agent(follow_up, 0)


def test_opening_gate_rush_meta_uses_player0_rush_then_greedy_one_signature():
    state = _state_for_seed(1010060)
    assert opening_gate_rush_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_rush_meta_agent(follow_up, 0) == greedy_agent(follow_up, 0)


def test_opening_gate_rush_meta_uses_player0_rush_then_field_control_one_signature():
    state = _state_for_seed(601)
    assert opening_gate_rush_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_rush_meta_agent(follow_up, 0) == field_control_agent(follow_up, 0)


def test_opening_gate_rush_meta_uses_player0_field_control_signature():
    state = _state_for_seed(1010176)
    assert opening_gate_rush_meta_agent(state, 0) == field_control_agent(state, 0)


def test_opening_gate_rush_meta_uses_player1_field_control_on_live_low_spin_rich_heavy():
    state = _state_for_seed(1301)
    assert opening_gate_rush_meta_agent(state, 1) == field_control_agent(state, 1)


def test_opening_gate_rush_meta_locks_high_spin_rich_cheap_close_field_control():
    state = _state_for_seed(1409)
    assert opening_gate_rush_meta_agent(state, 0) == field_control_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_rush_meta_agent(follow_up, 0) == field_control_agent(follow_up, 0)


def test_opening_gate_rush_meta_uses_tail_expansion_on_low_spin_rich_cheap_far():
    state = _state_for_seed(1103)
    assert opening_gate_rush_meta_agent(state, 0) == tail_expansion_agent(state, 0)
    assert opening_gate_rush_meta_agent(state, 1) == tail_expansion_agent(state, 1)


def test_opening_gate_rush_meta_uses_tail_expansion_on_low_spin_cheap_cluster():
    state = _state_for_seed(1801)
    assert opening_gate_rush_meta_agent(state, 0) == tail_expansion_agent(state, 0)
    assert opening_gate_rush_meta_agent(state, 1) == tail_expansion_agent(state, 1)


def test_opening_gate_rush_meta_uses_player0_greedy_signature():
    state = _state_for_seed(1490090)
    assert opening_gate_rush_meta_agent(state, 0) == greedy_agent(state, 0)


def test_opening_gate_anti_meta_meta_uses_anti_meta_on_close_axis_ffa_opening():
    ffa_state = {
        "planets": [
            {
                "id": 4,
                "owner": 0,
                "x": 98.8,
                "y": 56.2,
                "radius": 2.0,
                "ships": 10,
                "production": 1,
            },
            {"id": 5, "owner": 1, "x": 1.2, "y": 56.2, "radius": 2.0, "ships": 10, "production": 1},
            {
                "id": 6,
                "owner": 2,
                "x": 98.8,
                "y": 43.8,
                "radius": 2.0,
                "ships": 10,
                "production": 1,
            },
            {"id": 7, "owner": 3, "x": 1.2, "y": 43.8, "radius": 2.0, "ships": 10, "production": 1},
            {
                "id": 20,
                "owner": -1,
                "x": 82.5,
                "y": 65.1,
                "radius": 2.0,
                "ships": 19,
                "production": 2,
            },
        ],
        "fleets": [],
        "step": 0,
        "angular_velocity": 0.0387,
    }
    assert opening_gate_anti_meta_meta_agent(ffa_state, 0) == anti_meta_agent(ffa_state, 0)


def test_opening_gate_anti_meta_meta_locks_field_control_opening_signature():
    state = _state_for_seed(1010000)
    assert opening_gate_anti_meta_meta_agent(state, 0) == field_control_agent(state, 0)


def test_opening_gate_anti_meta_meta_locks_rush_then_greedy_signature():
    state = _state_for_seed(2330121)
    assert opening_gate_anti_meta_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 5
    assert opening_gate_anti_meta_meta_agent(follow_up, 0) == greedy_agent(follow_up, 0)


def test_opening_gate_anti_meta_meta_uses_field_control_on_response_meta_low_prod_trap():
    state = _state_for_seed(530030)
    assert opening_gate_anti_meta_meta_agent(state, 0) == field_control_agent(state, 0)


def test_opening_gate_anti_meta_meta_uses_defensive_on_response_meta_high_prod_trap():
    state = _state_for_seed(4010302)
    assert opening_gate_anti_meta_meta_agent(state, 0) == defensive_agent(state, 0)


def test_opening_gate_anti_meta_meta_locks_short_rush_then_greedy_response_signature():
    state = _state_for_seed(1490211)
    assert opening_gate_anti_meta_meta_agent(state, 0) == rush_agent(state, 0)
    follow_up = dict(state)
    follow_up["step"] = 3
    assert opening_gate_anti_meta_meta_agent(follow_up, 0) == greedy_agent(follow_up, 0)


def test_opening_gate_anti_meta_meta_uses_player1_rush_then_defensive_signature():
    state = _state_for_seed(530030)
    assert opening_gate_anti_meta_meta_agent(state, 1) == rush_agent(state, 1)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_anti_meta_meta_agent(follow_up, 1) == defensive_agent(follow_up, 1)


def test_opening_gate_anti_meta_meta_uses_player1_greedy_signature():
    state = _state_for_seed(1010060)
    assert opening_gate_anti_meta_meta_agent(state, 1) == greedy_agent(state, 1)


def test_opening_gate_anti_meta_meta_uses_player1_defensive_signature():
    state = _state_for_seed(1490090)
    assert opening_gate_anti_meta_meta_agent(state, 1) == defensive_agent(state, 1)


def test_opening_gate_anti_meta_meta_locks_player1_defensive_then_field_control_signature():
    state = _state_for_seed(1010176)
    assert opening_gate_anti_meta_meta_agent(state, 1) == defensive_agent(state, 1)
    follow_up = dict(state)
    follow_up["step"] = 1
    assert opening_gate_anti_meta_meta_agent(follow_up, 1) == field_control_agent(follow_up, 1)
