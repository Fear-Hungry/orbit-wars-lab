from __future__ import annotations

import numpy as np
import torch
from python.agents.policy import FLEET_F, FLEET_N, GLOBAL_F, PLANET_F, PLANET_N
from python.agents.value_net import (
    OBS_DIM,
    AttnValueNet,
    EntityValueNet,
    build_value_net,
    load_value_net,
)


def _obs(b, n_planets=5, n_fleets=3, seed=0):
    """A flat OBS_DIM batch with `n_planets`/`n_fleets` present (presence flag at
    feature 0), rest absent (zeros)."""
    rng = np.random.default_rng(seed)
    obs = np.zeros((b, OBS_DIM), dtype=np.float32)
    obs[:, :GLOBAL_F] = rng.standard_normal((b, GLOBAL_F))
    p0 = GLOBAL_F
    for k in range(n_planets):
        seg = p0 + k * PLANET_F
        obs[:, seg] = 1.0  # present
        obs[:, seg + 1: seg + PLANET_F] = rng.standard_normal((b, PLANET_F - 1))
    f0 = GLOBAL_F + PLANET_N * PLANET_F
    for k in range(n_fleets):
        seg = f0 + k * FLEET_F
        obs[:, seg] = 1.0
        obs[:, seg + 1: seg + FLEET_F] = rng.standard_normal((b, FLEET_F - 1))
    return obs


def test_attn_value_net_shape_and_range():
    net = AttnValueNet().eval()
    x = torch.as_tensor(_obs(4), dtype=torch.float32)
    with torch.no_grad():
        v = net(x)
    assert v.shape == (4,)
    assert torch.all(v >= -1.0) and torch.all(v <= 1.0)


def test_attn_value_net_ignores_absent_entities():
    # Padding absent entities with garbage must NOT change the value: presence
    # masking + masked-mean pool must isolate the real entities. This is what
    # makes the encoder permutation/padding invariant like the mean baseline.
    net = AttnValueNet().eval()
    base = _obs(2, n_planets=4, n_fleets=2, seed=1)
    poisoned = base.copy()
    # write garbage into ABSENT fleet slots (presence stays 0)
    f0 = GLOBAL_F + PLANET_N * PLANET_F
    for k in range(2, FLEET_N):
        seg = f0 + k * FLEET_F
        poisoned[:, seg + 1: seg + FLEET_F] = 99.0  # feature 0 (presence) left 0
    with torch.no_grad():
        v0 = net(torch.as_tensor(base, dtype=torch.float32))
        v1 = net(torch.as_tensor(poisoned, dtype=torch.float32))
    assert torch.allclose(v0, v1, atol=1e-5), (v0, v1)


def test_attn_value_net_early_game_no_fleets_is_finite():
    # Early game: planets present, ZERO fleets. key_padding_mask must not NaN.
    net = AttnValueNet().eval()
    x = torch.as_tensor(_obs(3, n_planets=6, n_fleets=0), dtype=torch.float32)
    with torch.no_grad():
        v = net(x)
    assert torch.all(torch.isfinite(v))


def test_build_and_roundtrip_arch_tag(tmp_path):
    # attn checkpoint carries its arch and load_value_net reconstructs it.
    net = build_value_net("attn")
    p = tmp_path / "v_attn.pt"
    torch.save({"model": net.state_dict(), "arch": "attn"}, p)
    loaded = load_value_net(str(p), device="cpu")
    assert isinstance(loaded, AttnValueNet)
    x = torch.as_tensor(_obs(2), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(net.eval()(x), loaded(x), atol=1e-6)


def test_load_value_net_backward_compat_mean(tmp_path):
    # Old checkpoints have NO arch key -> must still load as the mean baseline.
    net = EntityValueNet()
    p = tmp_path / "v_old.pt"
    torch.save({"model": net.state_dict()}, p)  # no "arch"
    loaded = load_value_net(str(p), device="cpu")
    assert isinstance(loaded, EntityValueNet)
