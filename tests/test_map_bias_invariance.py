from __future__ import annotations

import math

from python.agents.policy import FlatActorCritic
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import observation_dim
from python.orbit_wars_gym.symmetry import (
    reflect_state_x,
    rotate_state_180,
    swap_players_2p,
)
from scripts.audit_map_bias import audit


def _state(seed: int = 0) -> dict:
    backend = RustBatchBackend(
        num_envs=1, num_players=2, seed=seed,
        config=RustConfig(episode_steps=16, enable_comets=False, act_timeout=1.0),
    )
    return backend.reset(seed)[0]


def _angles_close(a: float, b: float) -> bool:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b))) < 1e-9


def test_rotate_180_is_an_involution() -> None:
    state = _state()
    back = rotate_state_180(rotate_state_180(state))
    for p0, p1 in zip(state["planets"], back["planets"], strict=True):
        assert abs(p0["x"] - p1["x"]) < 1e-9
        assert abs(p0["y"] - p1["y"]) < 1e-9
        assert p0["owner"] == p1["owner"] and p0["ships"] == p1["ships"]


def test_rotate_180_maps_through_center() -> None:
    state = _state()
    rot = rotate_state_180(state)
    for p0, p1 in zip(state["planets"], rot["planets"], strict=True):
        assert abs((p0["x"] + p1["x"]) - 100.0) < 1e-9
        assert abs((p0["y"] + p1["y"]) - 100.0) < 1e-9


def test_reflect_x_is_an_involution() -> None:
    state = _state()
    back = reflect_state_x(reflect_state_x(state))
    for p0, p1 in zip(state["planets"], back["planets"], strict=True):
        assert abs(p0["x"] - p1["x"]) < 1e-9
        assert abs(p0["y"] - p1["y"]) < 1e-9


def test_swap_players_swaps_owners_and_is_involution() -> None:
    state = _state()
    swapped = swap_players_2p(state)
    for p0, p1 in zip(state["planets"], swapped["planets"], strict=True):
        if p0["owner"] == 0:
            assert p1["owner"] == 1
        elif p0["owner"] == 1:
            assert p1["owner"] == 0
        else:
            assert p1["owner"] == p0["owner"]
    back = swap_players_2p(swapped)
    assert [p["owner"] for p in back["planets"]] == [p["owner"] for p in state["planets"]]


def test_audit_measures_finite_gaps_and_documents_bias() -> None:
    model = FlatActorCritic(observation_dim())
    states = [_state(0), _state(1)]
    report = audit(model, states)
    assert report["num_states"] == 2
    for name in ("rotate_180", "reflect_x", "swap_players"):
        for head in ("launch", "source", "target", "frac", "offset"):
            entry = report["transforms"][name][head]
            assert entry["max"] >= 0.0
            assert math.isfinite(entry["mean_max_abs_logit_gap"])
    # The flat encoder uses absolute x/y + planet_id, so a board symmetry such as
    # 180° rotation must produce a *non-zero* logit gap somewhere — the map bias.
    rot = report["transforms"]["rotate_180"]
    assert max(rot[h]["max"] for h in ("launch", "source", "target", "frac", "offset")) > 0.0
