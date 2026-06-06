from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter

import pytest
import torch
from kaggle_environments import make
from python.agents.policy import FlatActorCritic
from python.agents.submission_adapter import safe_submission_agent
from python.orbit_wars_gym.encoding import observation_dim
from python.orbit_wars_gym.entities import planet_id, planet_owner
from scripts.export_submission import render_submission, validate_submission_template

NATIVE_RUNTIME_MODULES = {"orbit_wars_core", "orbit_wars_py", "orbit_wars_rs"}

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


def _native_runtime_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in NATIVE_RUNTIME_MODULES:
                    imports.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".", 1)[0]
            if root in NATIVE_RUNTIME_MODULES:
                imports.add(root)
    return imports


def _assert_no_native_runtime_imports(source: str, *, label: str) -> None:
    imports = _native_runtime_imports(source)
    assert imports == set(), f"{label} imports native Rust runtime modules: {sorted(imports)}"


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
            {
                "id": 0,
                "owner": 0,
                "x": 20.0,
                "y": 20.0,
                "radius": 2.0,
                "ships": 24,
                "production": 3,
            },
            {
                "id": 1,
                "owner": 0,
                "x": 26.0,
                "y": 24.0,
                "radius": 2.0,
                "ships": 12,
                "production": 2,
            },
            {
                "id": 2,
                "owner": -1,
                "x": 40.0,
                "y": 35.0,
                "radius": 2.0,
                "ships": 8,
                "production": 3,
            },
            {
                "id": 3,
                "owner": 1,
                "x": 76.0,
                "y": 72.0,
                "radius": 2.0,
                "ships": 20,
                "production": 4,
            },
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


def test_submission_template_exports_neural_checkpoint_without_runtime_torch(tmp_path: Path):
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": FlatActorCritic(observation_dim()).state_dict(),
            "summary": {
                "decoder": {
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
                    "max_moves_per_turn": 4,
                    "min_ships_to_launch": 2,
                    "reserve_home_ships": 8,
                }
            },
        },
        checkpoint,
    )

    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=str(checkpoint),
    )
    assert "import torch" not in rendered
    assert "import numpy" not in rendered
    assert "_NEURAL_POLICY" in rendered

    out = tmp_path / "submission_neural.py"
    out.write_text(rendered, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("submission_neural_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    moves = module.agent(SAMPLE_OBS)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


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
            {
                "id": 0,
                "owner": 0,
                "x": 20.0,
                "y": 20.0,
                "radius": 2.0,
                "ships": 24,
                "production": 3,
            },
            {
                "id": 1,
                "owner": 0,
                "x": 26.0,
                "y": 24.0,
                "radius": 2.0,
                "ships": 12,
                "production": 2,
            },
            {
                "id": 2,
                "owner": -1,
                "x": 40.0,
                "y": 35.0,
                "radius": 2.0,
                "ships": 8,
                "production": 3,
            },
            {
                "id": 3,
                "owner": 1,
                "x": 76.0,
                "y": 72.0,
                "radius": 2.0,
                "ships": 20,
                "production": 4,
            },
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


def test_exported_submission_detects_fleet_pressure_ratio(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_fleet_pressure")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 90,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 48, 4],
            [1, 1, 82.0, 20.0, 2.0, 15, 3],
            [2, -1, 50.0, 50.0, 2.0, 5, 2],
        ],
        "fleets": [[10, 1, 60.0, 20.0, math.pi, 1, 45]],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)

    assert features["enemy_fleet_ratio"] > 0.60
    assert features["aggression_ratio"] == features["enemy_fleet_ratio"]
    assert features["to_me_ratio"] > 0.95
    assert features["enemy_fleet_ships"] >= 0.85 * features["own_ships"]
    assert action["aggression_ratio"] == features["aggression_ratio"]
    assert action["fleet_pressure"]
    assert action["pressure"]


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


def test_exported_submission_projects_future_incoming_combat(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_future_projection")
    obs = {
        "player": 0,
        "step": 40,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 40, 3],
            [1, 1, 35.0, 20.0, 2.0, 6, 2],
        ],
        "fleets": [[20, 2, 30.0, 20.0, 0.0, 9, 12]],
    }

    owner, ships = module._project_planet_state(obs, obs["planets"][1], 4, cache={})

    assert owner == 2
    assert ships > 0


def test_exported_submission_skips_late_neutral_without_breakeven(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_neutral_breakeven")
    obs = {
        "player": 0,
        "step": 498,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 80, 3],
            [1, -1, 28.0, 20.0, 2.0, 20, 1],
        ],
        "fleets": [],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert moves == []


def test_exported_submission_skips_expiring_comet_target(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_expiring_comet_target")
    obs = {
        "player": 0,
        "step": 80,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 90, 3],
            [9, -1, 35.0, 20.0, 1.0, 2, 1],
        ],
        "fleets": [],
        "comet_planet_ids": [9],
        "comets": [
            {
                "planet_ids": [9],
                "paths": [[[35.0, 20.0], [39.0, 20.0]]],
                "path_index": 1,
            }
        ],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    score, _required, _target_xy = module._target_value(
        obs,
        obs["planets"][0],
        obs["planets"][1],
        0,
        action,
        [obs["planets"][0]],
        [],
    )

    assert score == -999.0


def test_exported_submission_does_not_launch_from_expiring_comet(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_expiring_comet_source")
    obs = {
        "player": 0,
        "step": 80,
        "angular_velocity": 0.0,
        "planets": [
            [9, 0, 20.0, 20.0, 1.0, 90, 1],
            [1, -1, 30.0, 20.0, 2.0, 5, 3],
            [2, 1, 80.0, 80.0, 2.0, 20, 3],
        ],
        "fleets": [],
        "comet_planet_ids": [9],
        "comets": [
            {
                "planet_ids": [9],
                "paths": [[[20.0, 20.0], [24.0, 20.0]]],
                "path_index": 1,
            }
        ],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert moves == []


def test_exported_submission_evacuates_owned_comet_before_expiration(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_comet_evacuation")
    obs = {
        "player": 0,
        "step": 90,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 18, 3],
            [9, 0, 30.0, 20.0, 1.0, 28, 1],
            [2, 1, 80.0, 80.0, 2.0, 20, 3],
        ],
        "fleets": [],
        "comet_planet_ids": [9],
        "comets": [
            {
                "planet_ids": [9],
                "paths": [[[30.0, 20.0], [34.0, 20.0], [38.0, 20.0]]],
                "path_index": 1,
            }
        ],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert moves
    assert int(moves[0][0]) == 9
    assert int(moves[0][2]) == 27
    _assert_moves_are_legal(obs, moves)


def test_exported_submission_moves_surplus_toward_frontier(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_frontier_reinforcement")
    obs = {
        "player": 0,
        "step": 120,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 70, 2],
            [1, 0, 72.0, 70.0, 2.0, 12, 4],
            [2, 0, 45.0, 40.0, 2.0, 18, 2],
            [4, 0, 35.0, 25.0, 2.0, 22, 2],
            [5, 0, 25.0, 42.0, 2.0, 20, 2],
            [3, 1, 84.0, 72.0, 2.0, 40, 4],
            [6, 2, 85.0, 60.0, 2.0, 35, 3],
        ],
        "fleets": [],
    }
    features = module.encode(obs)
    action = module.policy_forward(features)
    own = [planet for planet in obs["planets"] if planet[1] == 0]
    enemies = [planet for planet in obs["planets"] if planet[1] == 1]

    moves = module._frontier_reinforcement_moves(obs, own, enemies, action, {}, 2)

    assert moves
    assert int(moves[0][0]) == 0
    _assert_moves_are_legal(obs, moves)


def test_exported_submission_moves_surplus_toward_frontier_in_two_player(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_frontier_reinforcement_2p")
    obs = {
        "player": 0,
        "step": 140,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 12.0, 12.0, 2.0, 90, 2],
            [1, 0, 72.0, 70.0, 2.0, 14, 4],
            [2, 0, 45.0, 40.0, 2.0, 18, 2],
            [4, 0, 35.0, 25.0, 2.0, 22, 2],
            [5, 0, 25.0, 42.0, 2.0, 20, 2],
            [7, 0, 18.0, 35.0, 2.0, 18, 2],
            [3, 1, 84.0, 72.0, 2.0, 40, 4],
        ],
        "fleets": [],
    }
    features = module.encode(obs)
    action = module.policy_forward(features)
    own = [planet for planet in obs["planets"] if planet[1] == 0]
    enemies = [planet for planet in obs["planets"] if planet[1] == 1]

    moves = module._frontier_reinforcement_moves(obs, own, enemies, action, {}, 2)

    assert not action["ffa"]
    assert moves
    assert int(moves[0][0]) == 0
    _assert_moves_are_legal(obs, moves)


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
    assert action["opening_stage"] == "SAFE_NEUTRALS"
    assert moves
    source_xy = (80.0, 50.0)
    near_angle = module._angle(source_xy, (70.0, 50.0))
    far_angle = module._angle(source_xy, (20.0, 80.0))
    assert module._angle_delta(moves[0][1], near_angle) < module._angle_delta(
        moves[0][1], far_angle
    )


def test_exported_submission_prioritizes_safe_high_production_opening_neutral(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_safe_neutral")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 6,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 42, 3],
            [1, -1, 28.0, 28.0, 2.0, 4, 1],
            [2, -1, 45.0, 20.0, 2.0, 8, 5],
            [3, 1, 82.0, 82.0, 2.0, 40, 3],
        ],
        "fleets": [],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert action["fsm_state"] == "OPENING_EXPAND"
    assert moves
    source_xy = (20.0, 20.0)
    weak_angle = module._angle(source_xy, (28.0, 28.0))
    strong_angle = module._angle(source_xy, (45.0, 20.0))
    assert module._angle_delta(moves[0][1], strong_angle) < module._angle_delta(
        moves[0][1], weak_angle
    )


def test_exported_submission_rejects_opening_neutral_with_enemy_snipe_eta(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_snipe_eta")
    source = [0, 0, 20.0, 20.0, 2.0, 42, 3]
    target = [1, -1, 45.0, 20.0, 2.0, 8, 5]
    enemy = [2, 1, 66.0, 20.0, 2.0, 70, 4]

    assert not module._safe_opening_neutral(source, target, [enemy])
    assert module._opening_neutral_snipable(source, target, [enemy])


def test_exported_submission_skips_snipable_opening_neutral_when_no_safe_neutral(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_skip_snipable")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 6,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 50, 3],
            [1, -1, 42.0, 20.0, 2.0, 8, 5],
            [2, -1, 20.0, 34.0, 2.0, 7, 3],
            [3, 1, 52.0, 20.0, 2.0, 70, 4],
        ],
        "fleets": [],
    }

    action = module.policy_forward(module.encode(obs))
    moves = module.decode(action, obs)

    assert action["opening_stage"] == "SAFE_NEUTRALS"
    assert moves
    source_xy = (20.0, 20.0)
    snipable_angle = module._angle(source_xy, (42.0, 20.0))
    fallback_angle = module._angle(source_xy, (20.0, 34.0))
    assert module._angle_delta(moves[0][1], fallback_angle) < module._angle_delta(
        moves[0][1], snipable_angle
    )


def test_exported_submission_scores_opening_safety_with_snipe_eta(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_safety_score")
    source = [0, 0, 20.0, 20.0, 2.0, 50, 3]
    target = [1, -1, 42.0, 20.0, 2.0, 8, 5]
    far_enemy = [2, 1, 85.0, 85.0, 2.0, 70, 4]
    close_enemy = [2, 1, 52.0, 20.0, 2.0, 70, 4]
    base_obs = {"player": 0, "step": 6, "angular_velocity": 0.0, "fleets": []}
    action = {
        "ffa": False,
        "expand": True,
        "opening_stage": "SAFE_NEUTRALS",
        "orbital_opening_window": False,
        "adaptive_opening_expand": False,
        "fsm_state": "OPENING_EXPAND",
        "leader_owner": 1,
        "neutral_count": 1,
    }

    safe_score, _, _ = module._target_value(
        {**base_obs, "planets": [source, target, far_enemy]},
        source,
        target,
        0,
        dict(action),
        [source],
        [far_enemy],
    )
    snipable_score, _, _ = module._target_value(
        {**base_obs, "planets": [source, target, close_enemy]},
        source,
        target,
        0,
        dict(action),
        [source],
        [close_enemy],
    )

    assert module._safe_opening_neutral(source, target, [far_enemy])
    assert not module._safe_opening_neutral(source, target, [close_enemy])
    assert safe_score > snipable_score + 50.0


def test_exported_submission_expands_when_behind_on_production_in_opening(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_adaptive_opening_expand")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 45,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 55, 2],
            [1, 0, 25.0, 20.0, 2.0, 18, 1],
            [2, 0, 20.0, 28.0, 2.0, 15, 1],
            [3, 0, 28.0, 28.0, 2.0, 14, 1],
            [4, -1, 42.0, 20.0, 2.0, 8, 4],
            [5, 1, 82.0, 82.0, 2.0, 20, 7],
        ],
        "fleets": [],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)

    assert features["own_count"] > 3
    assert action["behind_on_econ"]
    assert action["opening_stage"] == "ADAPTIVE_PRODUCTION"
    assert not action["pressure"]
    assert action["adaptive_opening_expand"]
    assert action["expand"]


def test_exported_submission_explores_safe_orbital_neutral_during_opening_window(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_opening_orbital_window")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 12,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 50, 3],
            [1, -1, 35.0, 20.0, 2.0, 9, 5],
            [2, -1, 20.0, 42.0, 2.0, 4, 1],
            [3, 1, 85.0, 85.0, 2.0, 28, 4],
        ],
        "fleets": [],
    }

    features = module.encode(obs)
    action = module.policy_forward(features)
    moves = module.decode(action, obs)

    assert action["orbital_opening_window"]
    assert action["opening_stage"] == "ORBITAL"
    assert moves
    source_xy = (20.0, 20.0)
    orbital_xy = module._predict_target_xy(obs, source_xy, obs["planets"][1], moves[0][2])
    weak_xy = module._predict_target_xy(obs, source_xy, obs["planets"][2], moves[0][2])
    assert module._angle_delta(
        moves[0][1], module._angle(source_xy, orbital_xy)
    ) < module._angle_delta(moves[0][1], module._angle(source_xy, weak_xy))
    assert 19 <= moves[0][2] <= 22


def test_exported_submission_penalizes_exposed_depth_two_response(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_depth_two_guard")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 70,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 34, 4],
            [1, -1, 42.0, 20.0, 2.0, 8, 4],
            [2, 1, 32.0, 20.0, 2.0, 44, 3],
        ],
        "fleets": [],
    }
    action = module.policy_forward(module.encode(obs))
    source, target, enemy = obs["planets"]
    target_xy = module._predict_target_xy(obs, (20.0, 20.0), target, 24)

    exposed_penalty = module._opponent_response_penalty(
        obs, source, target, 24, target_xy, action, [enemy], 10
    )
    safe_penalty = module._opponent_response_penalty(
        obs, source, target, 12, target_xy, action, [enemy], 22
    )

    assert exposed_penalty > 0.0
    assert safe_penalty == 0.0


def test_exported_submission_penalizes_only_new_best_response_risk(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_best_response_delta")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 120,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 60, 4],
            [1, 0, 70.0, 20.0, 2.0, 60, 4],
            [2, 1, 42.0, 20.0, 2.0, 65, 3],
            [3, -1, 26.0, 20.0, 2.0, 5, 4],
        ],
        "fleets": [],
    }
    action = module.policy_forward(module.encode(obs))
    own = obs["planets"][:2]
    enemies = [obs["planets"][2]]
    launched = {}

    baseline = module._opponent_best_response_delta(obs, own, enemies, action, launched, own[0], 0)
    exposed = module._opponent_best_response_delta(obs, own, enemies, action, launched, own[0], 45)
    remote = module._opponent_best_response_delta(obs, own, enemies, action, launched, own[1], 12)

    assert baseline == 0.0
    assert exposed > 0.0
    assert remote == 0.0


def test_exported_submission_hammers_committed_enemy_target_when_enemy_overextended(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_pressure_hammer")
    module._PROFILE_STATE.clear()
    obs = {
        "player": 0,
        "step": 140,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 45, 4],
            [1, 0, 24.0, 20.0, 2.0, 35, 3],
            [2, 1, 42.0, 22.0, 2.0, 22, 4],
            [3, -1, 78.0, 78.0, 2.0, 8, 2],
        ],
        "fleets": [],
    }
    action = module.policy_forward(module.encode(obs))
    action["enemy_overextended"] = True
    own = [obs["planets"][0], obs["planets"][1]]
    enemies = [obs["planets"][2]]
    source = obs["planets"][1]
    target = obs["planets"][2]

    solo_score, _, _ = module._target_value(obs, source, target, 0, action, own, enemies)
    hammer_score, _, _ = module._target_value(obs, source, target, 18, action, own, enemies)

    assert hammer_score > solo_score


def test_exported_submission_enters_total_war_when_far_behind_without_neutrals(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_total_war")
    features = {
        "player": 0,
        "step": 260,
        "own_count": 3,
        "enemy_count": 5,
        "enemy_players": 1,
        "neutral_count": 0,
        "own_ships": 80,
        "enemy_ships": 120,
        "own_fleet_ships": 0,
        "enemy_fleet_ships": 10,
        "enemy_fleet_ratio": 0.08,
        "own_prod": 12,
        "enemy_prod": 20,
        "leader_owner": 1,
        "angular_velocity": 0.0,
        "profile_total": 0.0,
        "to_neutral_ratio": 0.0,
        "to_me_ratio": 0.0,
        "to_leader_ratio": 0.0,
        "recent_enemy_captures": set(),
    }

    action = module.policy_forward(features)

    assert action["total_war"]
    assert action["strategy_phase"] == "TOTAL_WAR"


def test_exported_submission_classifies_adaptive_reserve_phases(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_reserve_phases")
    module._PROFILE_STATE.clear()
    source = [0, 0, 20.0, 20.0, 2.0, 80, 5]
    enemies = [[1, 1, 82.0, 82.0, 2.0, 60, 5]]

    mid_obs = {"player": 0, "step": 160, "planets": [source, *enemies], "fleets": []}
    mid_action = {
        "ffa": False,
        "expand": False,
        "pressure": False,
        "total_war": False,
        "fsm_state": "BASELINE",
    }
    assert module._reserve_phase_for_source(source, 5, mid_action, mid_obs, 0) == "MID"
    assert module._reserve_for_source(source, 5, enemies, mid_action, mid_obs, 0) >= 15

    late_obs = {"player": 0, "step": 370, "planets": [source, *enemies], "fleets": []}
    late_action = {
        "ffa": False,
        "expand": False,
        "pressure": False,
        "total_war": False,
        "fsm_state": "BASELINE",
    }
    assert module._reserve_phase_for_source(source, 5, late_action, late_obs, 0) == "LATE"
    assert module._reserve_for_source(source, 5, enemies, late_action, late_obs, 0) >= 30

    war_obs = {"player": 0, "step": 260, "planets": [source, *enemies], "fleets": []}
    war_action = {
        "ffa": False,
        "expand": False,
        "pressure": False,
        "total_war": True,
        "fsm_state": "BASELINE",
    }
    assert module._reserve_phase_for_source(source, 5, war_action, war_obs, 0) == "TOTAL_WAR"
    assert module._reserve_for_source(source, 5, enemies, war_action, war_obs, 0) == 0


def test_exported_submission_uses_low_enemy_fleet_ratio_for_opportunism(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_low_fleet_ratio_opportunism")
    features = {
        "player": 0,
        "step": 90,
        "own_count": 4,
        "enemy_count": 3,
        "enemy_players": 1,
        "neutral_count": 5,
        "own_ships": 80,
        "enemy_ships": 70,
        "own_fleet_ships": 10,
        "enemy_fleet_ships": 5,
        "enemy_fleet_ratio": 0.08,
        "own_prod": 18,
        "enemy_prod": 20,
        "leader_owner": 1,
        "angular_velocity": 0.0,
        "profile_total": 20.0,
        "to_neutral_ratio": 0.1,
        "to_me_ratio": 0.0,
        "to_leader_ratio": 0.0,
        "recent_enemy_captures": set(),
    }

    action = module.policy_forward(features)

    assert not action["pressure"]
    assert not action["adaptive_opening_expand"]
    assert action["opportunistic_expand"]
    assert action["expand"]
    assert action["strategy_phase"] == "OPPORTUNISTIC"


def test_exported_submission_exits_opening_by_ratios_before_turn_limit(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_ratio_phase_exit")
    features = {
        "player": 0,
        "step": 40,
        "own_count": 4,
        "enemy_count": 3,
        "enemy_players": 1,
        "neutral_count": 5,
        "own_ships": 80,
        "enemy_ships": 70,
        "own_fleet_ships": 8,
        "enemy_fleet_ships": 4,
        "enemy_fleet_ratio": 0.08,
        "own_prod": 22,
        "enemy_prod": 20,
        "leader_owner": 1,
        "angular_velocity": 0.0,
        "profile_total": 20.0,
        "to_neutral_ratio": 0.1,
        "to_me_ratio": 0.0,
        "to_leader_ratio": 0.0,
        "recent_enemy_captures": set(),
    }

    action = module.policy_forward(features)

    assert action["fsm_state"] == "BASELINE"
    assert action["opportunistic_expand"]
    assert action["strategy_phase"] == "OPPORTUNISTIC"


def test_exported_submission_detects_ratio_pressure_before_extreme_fleet_pressure(tmp_path: Path):
    module = _load_rendered_submission(tmp_path, "submission_ratio_pressure")
    features = {
        "player": 0,
        "step": 120,
        "own_count": 5,
        "enemy_count": 4,
        "enemy_players": 1,
        "neutral_count": 3,
        "own_ships": 100,
        "enemy_ships": 60,
        "own_fleet_ships": 8,
        "enemy_fleet_ships": 90,
        "enemy_fleet_ratio": 0.62,
        "aggression_ratio": 0.62,
        "own_prod": 20,
        "enemy_prod": 18,
        "leader_owner": 1,
        "angular_velocity": 0.0,
        "profile_total": 30.0,
        "to_neutral_ratio": 0.1,
        "to_me_ratio": 0.82,
        "to_leader_ratio": 0.0,
        "recent_enemy_captures": set(),
    }

    action = module.policy_forward(features)

    assert action["ratio_pressure"]
    assert action["aggression_ratio"] == 0.62
    assert not action["fleet_pressure"]
    assert action["pressure"]
    assert not action["opportunistic_expand"]


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
    assert module.SUBMISSION_STATS["calls"] == 1
    assert module.SUBMISSION_STATS["illegal_moves"] == 1
    assert module.SUBMISSION_STATS["fallbacks"] == 1
    assert module.SUBMISSION_STATS["fallback_errors"] == 0


def test_exported_submission_avoids_local_runtime_dependencies():
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    _assert_no_native_runtime_imports(rendered, label="rendered submission")
    assert "requests" not in rendered
    assert "torch" not in rendered
    assert "numpy" not in rendered
    assert "fallback_greedy" in rendered


def test_native_runtime_import_detector_rejects_rust_boundary_crossing():
    source = """
import orbit_wars_py
from orbit_wars_rs import PyBatchSimulator
from math import sqrt
"""

    assert _native_runtime_imports(source) == {"orbit_wars_py", "orbit_wars_rs"}


def test_no_native_imports_in_submission_artifacts_and_producer_tarball(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/submission_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    _assert_no_native_runtime_imports(rendered, label="rendered submission")

    submission_path = Path("artifacts/submission.py")
    if submission_path.exists():
        _assert_no_native_runtime_imports(
            submission_path.read_text(encoding="utf-8"),
            label=str(submission_path),
        )

    tarball = tmp_path / "submission_producer.tar.gz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.package_producer_submission",
            "--out",
            str(tarball),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    with tarfile.open(tarball, "r:gz") as tar:
        names = {member.name for member in tar.getmembers()}
        assert "_upstream.py" in names
        upstream = tar.extractfile("_upstream.py")
        assert upstream is not None
        upstream_source = upstream.read().decode("utf-8")
        assert "mem.reset()" in upstream_source
        python_members = [
            member for member in tar.getmembers() if member.isfile() and member.name.endswith(".py")
        ]
        assert python_members
        for member in python_members:
            extracted = tar.extractfile(member)
            assert extracted is not None
            source = extracted.read().decode("utf-8")
            _assert_no_native_runtime_imports(source, label=f"{tarball}:{member.name}")


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

    monkeypatch.setattr(
        "python.agents.submission_adapter.decode_submission_action", _illegal_decode
    )
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
