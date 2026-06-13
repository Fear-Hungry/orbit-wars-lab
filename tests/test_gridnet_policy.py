"""GridNet per-planet policy + decoder (GN1).

The direct 1-tuple decoder died at -1.0 because every source shared one
target/frac rank (target head acc 0.04-0.06). GridNet (Huang & Ontañón 2021)
emits an INDEPENDENT action per planet, so multi-source plans are representable.
These tests pin: per-planet shapes, inactive→no-op masking, decode legality on
real states, and the multi-source capability the old decoder lacked.
"""

from __future__ import annotations

import numpy as np
import torch
from python.agents.policy import GridNetActorCritic, PLANET_N
from python.orbit_wars_gym.action_decoder import (
    PLANET_SLOTS,
    decode_gridnet_action,
    gridnet_planet_mask,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim
from python.orbit_wars_gym.rules import moves_are_legal
from python.agents.registry import make_isolated_opponent


def test_policy_shapes_and_inactive_noop() -> None:
    m = GridNetActorCritic(observation_dim())
    obs = torch.zeros(4, observation_dim())
    mask = torch.zeros(4, PLANET_N, dtype=torch.bool)
    mask[:, :3] = True
    a, lp, ent, v = m.get_action_and_value(obs, masks={"planet": mask})
    assert a.shape == (4, PLANET_N, 4)
    assert lp.shape == (4,) and ent.shape == (4,) and v.shape == (4,)
    assert (a[:, 3:, 0] == 0).all()  # inactive planets forced to no-op
    # re-evaluating a given action reproduces the same log-prob
    _, lp2, _, _ = m.get_action_and_value(obs, a, masks={"planet": mask})
    assert torch.allclose(lp, lp2)


def _advance(num_players: int, seed: int, until_own: int, max_steps: int):
    b = RustBatchBackend(
        num_envs=1, num_players=num_players, seed=seed,
        config=RustConfig(episode_steps=max_steps + 1, enable_comets=True),
    )
    s = b.reset(seed)[0]
    pol = {p: make_isolated_opponent("pgs") for p in range(num_players)}
    for _ in range(max_steps):
        if int(gridnet_planet_mask(s, 0).sum()) >= until_own:
            return s
        rows = []
        for p in range(num_players):
            for mv in pol[p](s, p):
                rows.append([0.0, float(p), float(mv[0]), float(mv[1]), float(mv[2])])
        arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        out, st = b.step_flat_with_states(arr)
        s = st[0]
        if out[0].get("done"):
            break
    return s


def test_decode_is_always_legal_on_real_states() -> None:
    m = GridNetActorCritic(observation_dim())
    b = RustBatchBackend(num_envs=1, num_players=2, seed=0, config=RustConfig(episode_steps=80, enable_comets=True))
    s = b.reset(0)[0]
    opp = make_isolated_opponent("pgs")
    tested = 0
    for _ in range(60):
        mask = torch.as_tensor(gridnet_planet_mask(s, 0), dtype=torch.bool).unsqueeze(0)
        obs = torch.as_tensor(encode_state(s, 0, DEFAULT_ENCODER_CONFIG), dtype=torch.float32).unsqueeze(0)
        a, _, _, _ = m.get_action_and_value(obs, masks={"planet": mask})
        moves = decode_gridnet_action(s, 0, a[0].numpy())
        if moves:
            tested += 1
            assert moves_are_legal(s, 0, moves)
        rows = [[0.0, 0.0, float(mv[0]), float(mv[1]), float(mv[2])] for mv in moves]
        rows += [[0.0, 1.0, float(mv[0]), float(mv[1]), float(mv[2])] for mv in opp(s, 1)]
        arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        out, st = b.step_flat_with_states(arr)
        s = st[0]
        if out[0].get("done"):
            break
    assert tested > 0


def test_multi_source_is_representable() -> None:
    s = _advance(4, 1, until_own=2, max_steps=200)
    mask = gridnet_planet_mask(s, 0)
    active = [i for i in range(PLANET_SLOTS) if mask[i]]
    assert len(active) >= 2
    a = np.zeros((PLANET_SLOTS, 4), dtype=np.int64)
    for k, i in enumerate(active):
        a[i, 0] = 1
        a[i, 1] = k % 5
        a[i, 2] = (k + 1) % 4
        a[i, 3] = k % 5
    moves = decode_gridnet_action(s, 0, a)
    assert len({mv[0] for mv in moves}) >= 2  # distinct sources, each independent
    assert moves_are_legal(s, 0, moves)
