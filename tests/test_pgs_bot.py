"""PGS bot (heuristic front): legality + parity-floor fidelity (todo H-P1)."""
from __future__ import annotations

import numpy as np
import pytest
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal


def _advance(backend, states, policies, steps: int):
    for _ in range(steps):
        rows = []
        for player, pol in enumerate(policies):
            for m in pol(states[0], player):
                if len(m) >= 3:
                    rows.append([0.0, float(player), float(m[0]), float(m[1]), float(m[2])])
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, 0)
        states = backend.states()
    return states


def _states_along_game(num_players: int, capture: list[int], seed: int = 7):
    backend = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed, config=RustConfig(enable_comets=True)
    )
    backend.reset(seed)
    states = backend.states()
    policies = get_isolated_opponents("producer", num_players)
    out, last = [], 0
    for step in sorted(capture):
        states = _advance(backend, states, policies, step - last)
        out.append(states[0])
        last = step
    return out


@pytest.mark.parametrize("num_players", [2, 4])
def test_pgs_returns_legal_moves(num_players: int) -> None:
    from bots.pgs.planner import make_runtime

    states = _states_along_game(num_players, capture=[0, 25, 60])
    runtime = make_runtime()
    for state in states:
        for player in range(num_players):
            moves = runtime.act(to_official_observation(state, player))
            assert isinstance(moves, list)
            for move in moves:
                assert len(move) == 3 and move[2] > 0
            assert moves_are_legal(state, player, moves)


def test_pgs_floor_matches_producer_modulo_float() -> None:
    """All-PRODUCER assignment ≈ the real Producer plan: same (source, ships)
    multiset; angles within float tolerance (the runtime is stateless per turn)."""
    from bots.pgs.planner import PGSConfig, make_runtime

    states = _states_along_game(2, capture=[0, 10])
    floor = make_runtime(PGSConfig(max_deviations=0))
    producer = get_isolated_opponents("producer", 1)[0]
    for state in states:
        real = sorted((round(m[0]), round(m[2]), round(float(m[1]), 3)) for m in producer(state, 0))
        mine = sorted(
            (round(m[0]), round(m[2]), round(float(m[1]), 3))
            for m in floor.act(to_official_observation(state, 0))
        )
        assert mine == real


def test_planner_exposes_no_default_entrypoint() -> None:
    """Regressão id=129/142 (2026-06-11): planner.py expunha agent() sobre os
    DEFAULTS do dataclass (all-scripts, config REJEITADA) — dependendo do
    import, "pgs" significava outro bot (foi assim que a submissão de
    2026-06-09 embarcou o bot errado). Entrypoint único: bots.pgs.agent."""
    import bots.pgs.planner as planner

    assert not hasattr(planner, "agent"), (
        "entrypoint duplicado reintroduzido no planner — use bots.pgs.agent")


def test_pgs_tiny_deadline_returns_legal_producer_floor() -> None:
    from bots.pgs.planner import PGSConfig, make_runtime

    backend = RustBatchBackend(
        num_envs=1, num_players=2, seed=11, config=RustConfig(enable_comets=True)
    )
    state = backend.reset(11)[0]
    obs = to_official_observation(state, 0)
    producer = get_isolated_opponents("producer", 1)[0]
    runtime = make_runtime(PGSConfig(
        scripts="hold",
        wave_min_ships=60.0,
        wave_start_step=0,
        deadline_ms=0.001,
        deadline_guard_ms=0.0,
    ))

    moves = runtime.act(obs)

    assert moves_are_legal(state, 0, moves)
    assert sorted((round(m[0]), round(m[2])) for m in moves) == sorted(
        (round(m[0]), round(m[2])) for m in producer(state, 0)
    )


def test_half_in_2p_only_adds_half_for_two_player_games(monkeypatch) -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime, make_runtime

    calls: list[int] = []

    def spy_half(self, movement, base_entries):
        calls.append(int(self._player_count or -1))
        return None

    monkeypatch.setattr(PGSRuntime, "_script_half", spy_half)

    state_2p = _states_along_game(2, capture=[0])[0]
    runtime_2p = make_runtime(PGSConfig(scripts="hold", half_in_2p=True, max_deviations=0))
    runtime_2p.act(to_official_observation(state_2p, 0))

    assert calls
    assert set(calls) == {2}

    calls.clear()
    state_4p = _states_along_game(4, capture=[0])[0]
    runtime_4p = make_runtime(PGSConfig(scripts="hold", half_in_2p=True, max_deviations=0))
    runtime_4p.act(to_official_observation(state_4p, 0))

    assert calls == []


def test_league_registers_holdwave_half2p_config(monkeypatch) -> None:
    from scripts import league_agents

    captured = {}

    def fake_pgs(**cfg):
        captured.update(cfg)
        return "agent"

    monkeypatch.setattr(league_agents, "_pgs", fake_pgs)

    assert league_agents.FACTORIES["pgs_holdwave_half2p"]() == "agent"
    assert captured == {
        "scripts": "hold",
        "wave_min_ships": 60.0,
        "wave_start_step": 150,
        "half_in_2p": True,
    }


def test_registry_pgs_routes_to_operational_entrypoint(monkeypatch) -> None:
    """O 'pgs' das heurísticas do registry deve rotear pro entrypoint
    operacional (SUBMISSION_CONFIG), o mesmo bot dos isolados."""
    import bots.pgs.agent as op_agent
    import python.orbit_wars_gym.observation as obs_mod
    from python.agents.registry import pgs_agent

    sentinel = [[1.0, 2.0, 3.0]]
    monkeypatch.setattr(op_agent, "agent", lambda obs: sentinel)
    monkeypatch.setattr(obs_mod, "to_official_observation",
                        lambda state, player: {"step": 0})
    assert pgs_agent({}, 0) == sentinel
