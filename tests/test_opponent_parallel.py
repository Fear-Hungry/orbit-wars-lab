from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from python.agents.registry import _make_isolated_policy
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig


def _states(seeds: list[int]) -> list[dict]:
    out = []
    for s in seeds:
        backend = RustBatchBackend(
            num_envs=1, num_players=2, seed=s,
            config=RustConfig(episode_steps=16, enable_comets=False, act_timeout=1.0),
        )
        out.append(backend.reset(s)[0])
    return out


def _key(moves) -> tuple:
    return tuple((int(m[0]), round(float(m[1]), 6), int(m[2])) for m in moves)


@pytest.mark.parametrize("name", ["producer", "oep"])
def test_threaded_opponent_calls_match_sequential(name: str) -> None:
    """Calling per-env isolated instances concurrently must match sequential.

    Each thread touches a different instance (no shared mutable state), so the
    results must be identical to the sequential path — a race in a shared planner
    global would surface here as a mismatch or crash.
    """
    n = 8
    states = _states(list(range(n)))
    instances = [_make_isolated_policy(name) for _ in range(n)]

    sequential = [_key(instances[i](states[i], 1)) for i in range(n)]

    # Reset by rebuilding instances so the concurrent run starts from the same
    # (fresh, step==0) state as the sequential run did.
    instances = [_make_isolated_policy(name) for _ in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        concurrent = list(ex.map(lambda i: _key(instances[i](states[i], 1)), range(n)))

    assert concurrent == sequential
