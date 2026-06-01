from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from time import perf_counter

import pytest
from kaggle_environments import make
from python.agents.submission_adapter import safe_submission_agent
from python.orbit_wars_gym.entities import planet_id, planet_owner
from scripts.export_submission import render_submission, validate_submission_template

SAMPLE_OBS = {
    "player": 0,
    "angular_velocity": 0.03,
    "planets": [
        [0, 0, 20.0, 20.0, 2.0, 24, 3],
        [1, 0, 26.0, 24.0, 2.0, 12, 2],
        [2, -1, 40.0, 35.0, 2.0, 8, 3],
        [3, 1, 76.0, 72.0, 2.0, 20, 4],
    ],
    "fleets": [],
}


def _assert_moves_are_legal(obs: dict, moves: list[list[float]]) -> None:
    own_ids = {planet_id(p) for p in obs["planets"] if planet_owner(p) == int(obs["player"])}
    assert isinstance(moves, list)
    for move in moves:
        assert isinstance(move, list)
        assert len(move) == 3
        assert int(move[0]) in own_ids
        assert math.isfinite(float(move[1]))
        assert int(move[2]) > 0


def _load_rendered_submission(tmp_path: Path, module_name: str):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / f"{module_name}.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_safe_submission_agent_returns_legal_moves():
    moves = safe_submission_agent(SAMPLE_OBS)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_safe_submission_agent_accepts_dict_entities():
    obs = {
        "player": 0,
        "angular_velocity": 0.03,
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 24, "production": 3},
            {"id": 1, "owner": 0, "x": 26.0, "y": 24.0, "radius": 2.0, "ships": 12, "production": 2},
            {"id": 2, "owner": -1, "x": 40.0, "y": 35.0, "radius": 2.0, "ships": 8, "production": 3},
            {"id": 3, "owner": 1, "x": 76.0, "y": 72.0, "radius": 2.0, "ships": 20, "production": 4},
        ],
        "fleets": [],
    }
    moves = safe_submission_agent(obs)
    _assert_moves_are_legal(obs, moves)


def test_submission_template_exports_self_contained_agent(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "submission.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("submission_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    moves = module.agent(SAMPLE_OBS)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_export_refuses_checkpoint_until_neural_export_exists():
    with pytest.raises(NotImplementedError, match="PPO checkpoint export is not implemented"):
        render_submission(
            Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
            checkpoint="artifacts/ppo/current.pt",
        )


def test_submission_template_accepts_dict_entities(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "submission.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("submission_dict_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    obs = {
        "player": 0,
        "angular_velocity": 0.03,
        "planets": [
            {"id": 0, "owner": 0, "x": 20.0, "y": 20.0, "radius": 2.0, "ships": 24, "production": 3},
            {"id": 1, "owner": 0, "x": 26.0, "y": 24.0, "radius": 2.0, "ships": 12, "production": 2},
            {"id": 2, "owner": -1, "x": 40.0, "y": 35.0, "radius": 2.0, "ships": 8, "production": 3},
            {"id": 3, "owner": 1, "x": 76.0, "y": 72.0, "radius": 2.0, "ships": 20, "production": 4},
        ],
        "fleets": [],
    }

    moves = module.agent(obs)
    _assert_moves_are_legal(obs, moves)


def test_exported_submission_profiles_enemy_expansion_to_neutrals(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_profile_expand")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 10,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 40.0, 2.0, 35, 3],
            [1, 1, 30.0, 40.0, 2.0, 25, 3],
            [2, -1, 70.0, 40.0, 2.0, 6, 4],
        ],
        "fleets": [[8, 1, 42.0, 40.0, 0.0, 1, 18]],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)

    assert features["to_neutral_ratio"] > 0.95
    assert action["fsm_state"] == "OPENING_EXPAND"


def test_exported_submission_fsm_defends_under_ffa_pressure(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_profile_pressure")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 70,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 40, 4],
            [1, -1, 40.0, 40.0, 2.0, 5, 4],
            [2, 1, 82.0, 20.0, 2.0, 40, 3],
            [3, 2, 82.0, 82.0, 2.0, 30, 3],
            [4, 3, 20.0, 82.0, 2.0, 30, 3],
        ],
        "fleets": [[10, 1, 70.0, 20.0, math.pi, 2, 24]],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert features["to_me_ratio"] > 0.95
    assert action["fsm_state"] == "DEFEND_UNDER_PRESSURE"
    assert not action["expand"]
    _assert_moves_are_legal(obs, moves)


def test_exported_submission_tracks_recent_weak_enemy_captures(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_profile_capture")
    module._PROFILE_STATE.clear()
    before = {
        "player": 0,
        "step": 20,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 45, 3],
            [1, 1, 82.0, 20.0, 2.0, 30, 3],
            [2, -1, 50.0, 20.0, 2.0, 6, 4],
        ],
        "fleets": [],
    }
    after = {
        **before,
        "step": 21,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 45, 3],
            [1, 1, 82.0, 20.0, 2.0, 30, 3],
            [2, 1, 50.0, 20.0, 2.0, 5, 4],
        ],
    }

    module.encode(before)
    features = module.encode(after)
    action = module.policy_forward(features)

    assert 2 in features["recent_enemy_captures"]
    assert action["fsm_state"] == "PUNISH_WEAK_CAPTURE"


def test_exported_submission_throttles_sources_with_many_outgoing_fleets(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_source_throttle")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 20,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 18, 3],
            [1, -1, 40.0, 20.0, 2.0, 5, 3],
            [2, 1, 82.0, 82.0, 2.0, 25, 3],
        ],
        "fleets": [
            [10, 0, 25.0, 20.0, 0.0, 0, 2],
            [11, 0, 26.0, 20.0, 0.0, 0, 2],
            [12, 0, 27.0, 20.0, 0.0, 0, 2],
        ],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert moves == []


def test_exported_submission_prefers_nearby_neutral_in_opening(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_distance")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 8,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 80.0, 50.0, 2.0, 40, 5],
            [1, -1, 70.0, 50.0, 2.0, 8, 5],
            [2, -1, 20.0, 80.0, 2.0, 8, 5],
            [3, 1, 10.0, 90.0, 2.0, 50, 3],
        ],
        "fleets": [],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert action["fsm_state"] == "OPENING_EXPAND"
    assert moves
    source_xy = (80.0, 50.0)
    near_angle = module._angle(source_xy, (70.0, 50.0))
    far_angle = module._angle(source_xy, (20.0, 80.0))
    assert module._angle_delta(moves[0][1], near_angle) < module._angle_delta(moves[0][1], far_angle)


def test_exported_submission_falls_back_on_illegal_output(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "submission.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("submission_error_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.decode = lambda *_args, **_kwargs: [[999, float("nan"), -5]]
    moves = module.agent(SAMPLE_OBS)
    assert isinstance(moves, list)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_exported_submission_avoids_local_runtime_dependencies():
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    assert "orbit_wars_rs" not in rendered
    assert "requests" not in rendered
    assert "torch" not in rendered
    assert "numpy" not in rendered
    assert "fallback_greedy" in rendered


def test_export_refuses_template_without_fallback_guardrail():
    broken_template = """
def agent(obs):
    return []
"""

    with pytest.raises(ValueError, match="fallback_greedy"):
        validate_submission_template(broken_template)


def test_submission_agent_falls_back_on_illegal_output(monkeypatch: pytest.MonkeyPatch):
    def _illegal_decode(*_args, **_kwargs):
        return [[999, float("nan"), -5]]

    monkeypatch.setattr("python.agents.submission_adapter.decode_submission_action", _illegal_decode)
    moves = safe_submission_agent(SAMPLE_OBS)
    assert isinstance(moves, list)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_submission_agent_always_returns_list_and_stays_within_timeout():
    start = perf_counter()
    moves = safe_submission_agent(SAMPLE_OBS)
    elapsed = perf_counter() - start

    assert isinstance(moves, list)
    assert elapsed < 1.0


def test_exported_submission_fallback_returns_list_within_timeout(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "submission.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("submission_timeout_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.encode = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    start = perf_counter()
    moves = module.agent(SAMPLE_OBS)
    elapsed = perf_counter() - start

    assert isinstance(moves, list)
    assert elapsed < 1.0
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_exported_submission_self_play_2p_and_4p(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "submission.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("submission_runtime_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for agents in (2, 4):
        env = make("orbit_wars", configuration={"agents": agents}, debug=True)
        env.run([module.agent] * agents)
        assert all(state.status == "DONE" for state in env.state)
