"""Mission layer (pgs_v2): multi-source missions over the Producer base.

Locks three contracts: (1) mission_mode defaults OFF — the per-source portfolio
path stays the frozen baseline and SUBMISSION_CONFIG never opts in silently;
(2) mission mode returns legal moves end-to-end (2p and 4p); (3) the assembly
criterion — a selected mission REPLACES the Producer-floor launches of every
source it uses (and of its exclusive target), otherwise the same source could
launch twice in one turn with no strategic control.
"""
from __future__ import annotations

import numpy as np
import torch
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


def test_mission_mode_defaults_off() -> None:
    from bots.pgs.agent import SUBMISSION_CONFIG
    from bots.pgs.planner import PGSConfig

    assert PGSConfig().mission_mode is False
    assert SUBMISSION_CONFIG.mission_mode is False, "frozen control must not opt in"


def test_mission_mode_returns_legal_moves() -> None:
    from bots.pgs.planner import PGSConfig, make_runtime

    for num_players in (2, 4):
        states = _states_along_game(num_players, capture=[0, 30, 70])
        runtime = make_runtime(PGSConfig(
            scripts="hold", mission_mode=True, deadline_ms=5000.0
        ))
        for state in states:
            for player in range(num_players):
                moves = runtime.act(to_official_observation(state, player))
                assert isinstance(moves, list)
                for move in moves:
                    assert len(move) == 3 and move[2] > 0
                assert moves_are_legal(state, player, moves)


def test_selected_mission_replaces_producer_entries_of_used_sources() -> None:
    from bots.pgs.planner import LaunchEntries, MissionCandidate, PGSConfig, make_runtime

    state = _states_along_game(2, capture=[40])[0]
    player = 0
    obs = to_official_observation(state, player)
    planets = obs["planets"]
    mine = [i for i, p in enumerate(planets) if int(p[1]) == player and float(p[5]) >= 10.0]
    enemy = [i for i, p in enumerate(planets) if int(p[1]) >= 0 and int(p[1]) != player]
    assert len(mine) >= 2 and enemy, "seed/step must give 2 own sources and an enemy"
    s0, s1 = mine[0], mine[1]
    tgt = enemy[0]

    # synthetic 2-source mission with unmistakable markers (ships=3, eta=5)
    dt = torch.float32
    entries = LaunchEntries(
        source_slots=torch.tensor([s0, s1], dtype=torch.long),
        target_slots=torch.tensor([tgt, tgt], dtype=torch.long),
        ships=torch.tensor([3.0, 3.0], dtype=dt),
        angle=torch.tensor([0.0, 0.0], dtype=dt),
        eta=torch.tensor([5.0, 5.0], dtype=dt),
        valid=torch.tensor([True, True]),
    )
    mission = MissionCandidate(
        name="test:hammer", entries=entries,
        replace_sources=frozenset({s0, s1}),
        exclusive_targets=frozenset({tgt}), kind="hammer", priority=1.0,
    )

    runtime = make_runtime(PGSConfig(scripts="hold", mission_mode=True, deadline_ms=5000.0))
    runtime._generate_missions = lambda **kw: [mission]

    def marker_value(movement, obs_tensors, my_entries, opp, me):
        mark = (
            my_entries.valid
            & (my_entries.target_slots == tgt)
            & (my_entries.ships == 3.0)
            & (my_entries.eta == 5.0)
        )
        return 1000.0 * float(mark.sum().item())

    runtime._plan_value = marker_value
    from orbit_lite.planner_core import _empty_entries

    runtime._reactive_reply = (
        lambda *a, **k: _empty_entries(torch.device("cpu"), dt)
    )

    moves = runtime.act(obs)
    assert moves_are_legal(state, player, moves)
    pid = {slot: int(planets[slot][0]) for slot in (s0, s1)}
    for slot in (s0, s1):
        from_src = [m for m in moves if int(m[0]) == pid[slot]]
        assert len(from_src) == 1, (
            f"source slot {slot} must carry ONLY its mission launch, got {from_src}"
        )
        assert from_src[0][2] == 3.0, "the single launch must be the mission's"
