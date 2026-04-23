from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from time import perf_counter

import pytest
from kaggle_environments import make
from python.agents import (
    anti_meta_agent,
    defensive_agent,
    greedy_agent,
    opening_gate_anti_meta_meta_agent,
    opening_gate_meta_agent,
    opening_gate_rush_meta_agent,
    rush_agent,
)
from python.agents.submission_adapter import safe_submission_agent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import planet_id, planet_owner
from python.orbit_wars_gym.observation import to_official_observation
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


def test_exported_heuristic_submission_supports_policy_specialization(tmp_path: Path):
    template = Path("python/submission/heuristic_submission_template.py").read_text(
        encoding="utf-8"
    )
    greedy_rendered = render_submission(template, heuristic_policy="greedy")
    defensive_rendered = render_submission(template, heuristic_policy="defensive")
    assert 'HEURISTIC_POLICY = "greedy"' in greedy_rendered
    assert 'HEURISTIC_POLICY = "defensive"' in defensive_rendered

    out = tmp_path / "heuristic_submission.py"
    out.write_text(defensive_rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("heuristic_submission_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    moves = module.agent(SAMPLE_OBS)
    _assert_moves_are_legal(SAMPLE_OBS, moves)


def test_opening_gate_meta_template_matches_local_policy_on_reference_states(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/opening_gate_meta_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "opening_gate_meta.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("opening_gate_meta_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cases = [
        (601, [opening_gate_meta_agent, greedy_agent, rush_agent, anti_meta_agent]),
        (509, [opening_gate_meta_agent, greedy_agent]),
    ]
    for seed, players in cases:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=len(players),
            seed=seed,
            config=RustConfig(episode_steps=500, enable_comets=True),
        )
        state = backend.reset(seed)[0]
        for _ in range(4):
            obs = to_official_observation(state, player=0)
            assert module.agent(obs) == opening_gate_meta_agent(state, 0)
            actions = [policy(state, idx) for idx, policy in enumerate(players)]
            outcome = backend.step([actions])[0]
            state = backend.states()[0]
            if outcome["done"]:
                break


def test_opening_gate_rush_meta_template_matches_local_policy_on_reference_states(tmp_path: Path):
    rendered = render_submission(
        Path("python/submission/opening_gate_rush_meta_template.py").read_text(encoding="utf-8"),
        checkpoint=None,
    )
    out = tmp_path / "opening_gate_rush_meta.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("opening_gate_rush_meta_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cases = [
        (509, 0, [opening_gate_rush_meta_agent, greedy_agent, rush_agent, anti_meta_agent]),
        (509, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (1490000, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (170000, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (2330000, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (2330237, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (530146, 0, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (601, 0, [opening_gate_rush_meta_agent, greedy_agent]),
        (1301, 1, [opening_gate_anti_meta_meta_agent, opening_gate_rush_meta_agent]),
        (1409, 0, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1010060, 0, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1010176, 0, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1490090, 0, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
    ]
    for seed, player_idx, players in cases:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=len(players),
            seed=seed,
            config=RustConfig(episode_steps=500, enable_comets=True),
        )
        state = backend.reset(seed)[0]
        for _ in range(4):
            obs = to_official_observation(state, player=player_idx)
            assert module.agent(obs) == opening_gate_rush_meta_agent(state, player_idx)
            actions = [policy(state, idx) for idx, policy in enumerate(players)]
            outcome = backend.step([actions])[0]
            state = backend.states()[0]
            if outcome["done"]:
                break


def test_opening_gate_anti_meta_meta_template_matches_local_policy_on_reference_states(
    tmp_path: Path,
):
    rendered = render_submission(
        Path("python/submission/opening_gate_anti_meta_meta_template.py").read_text(
            encoding="utf-8"
        ),
        checkpoint=None,
    )
    out = tmp_path / "opening_gate_anti_meta_meta.py"
    out.write_text(rendered, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("opening_gate_anti_meta_meta_module", out)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cases = [
        (509, 0, [opening_gate_anti_meta_meta_agent, greedy_agent, rush_agent, anti_meta_agent]),
        (170003, 0, [opening_gate_anti_meta_meta_agent, rush_agent]),
        (1010000, 0, [opening_gate_anti_meta_meta_agent, greedy_agent]),
        (2330121, 0, [opening_gate_anti_meta_meta_agent, greedy_agent]),
        (530030, 0, [opening_gate_anti_meta_meta_agent, opening_gate_rush_meta_agent]),
        (1490211, 0, [opening_gate_anti_meta_meta_agent, greedy_agent]),
        (4010302, 0, [opening_gate_anti_meta_meta_agent, defensive_agent]),
        (530030, 1, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1010060, 1, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1490090, 1, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
        (1010176, 1, [opening_gate_rush_meta_agent, opening_gate_anti_meta_meta_agent]),
    ]
    for seed, player_idx, players in cases:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=len(players),
            seed=seed,
            config=RustConfig(episode_steps=500, enable_comets=True),
        )
        state = backend.reset(seed)[0]
        for _ in range(4):
            obs = to_official_observation(state, player=player_idx)
            assert module.agent(obs) == opening_gate_anti_meta_meta_agent(state, player_idx)
            actions = [policy(state, idx) for idx, policy in enumerate(players)]
            outcome = backend.step([actions])[0]
            state = backend.states()[0]
            if outcome["done"]:
                break


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
