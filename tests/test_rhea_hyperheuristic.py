from __future__ import annotations

from statistics import quantiles
from time import perf_counter

import pytest
from bots.oep.candidate_factory import available_families
from bots.oep.hyperheuristic import context_bucket, hyperheuristic_plan, select_family
from bots.oep.rhea import rhea_plan
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal


def _obs_sequence(*, seed: int, num_players: int, steps: int) -> list[tuple[dict, int]]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=steps + 1, enable_comets=True, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    out: list[tuple[dict, int]] = []
    for _ in range(steps):
        for player in range(num_players):
            out.append((to_official_observation(state, player=player), player))
        _, states = backend.step_with_states([[[] for _ in range(num_players)]])
        state = states[0]
    return out


def test_h6_h7_registered() -> None:
    assert "rhea_macro" in available_families()
    assert "hyperheuristic" in available_families()


@pytest.mark.parametrize("plan", [rhea_plan, hyperheuristic_plan])
@pytest.mark.parametrize("num_players", [2, 4])
def test_legal_moves(plan, num_players: int) -> None:
    for obs, player in _obs_sequence(seed=3, num_players=num_players, steps=6):
        moves = plan(obs)
        assert isinstance(moves, list)
        assert moves_are_legal(obs, player, moves), f"illegal: {moves}"


def test_rhea_within_act_timeout_p95() -> None:
    samples = _obs_sequence(seed=5, num_players=2, steps=20)
    timings_ms: list[float] = []
    for obs, _player in samples:
        start = perf_counter()
        rhea_plan(obs)
        timings_ms.append((perf_counter() - start) * 1000.0)
    p95 = quantiles(timings_ms, n=20)[18] if len(timings_ms) >= 20 else max(timings_ms)
    # Hard invariant: every decision and the p95 stay well under actTimeout=1s.
    assert max(timings_ms) < 1000.0, f"RHEA exceeded actTimeout: max={max(timings_ms):.1f}ms"
    assert p95 < 700.0, f"RHEA p95 too close to actTimeout: {p95:.1f}ms"


def test_context_bucket_known_values() -> None:
    assert context_bucket({"planets": [], "fleets": [], "player": 0}) == "expand"
    assert select_family({"planets": [], "fleets": [], "player": 0}) in {
        "production_projected_attack",
        "timeline_risk",
        "hammer_multiprong",
        "regroup_dominance",
    }
