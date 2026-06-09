"""Tests for Movement 2: reward de-anchoring + KL-to-reference anti-drift + eval-gating."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from python.agents.policy import launch_gated_kl  # noqa: E402
from python.train.train_ppo import (  # noqa: E402
    Phase0TrainingConfig,
    _evaluate_margin,
    shaping_scales,
    train_phase0,
)


def _heads(b: int = 6, source_n: int = 16, target_n: int = 32, frac_n: int = 4, offset_n: int = 5):
    return {
        "launch": torch.randn(b, 2),
        "source": torch.randn(b, source_n),
        "target": torch.randn(b, target_n),
        "frac": torch.randn(b, frac_n),
        "offset": torch.randn(b, offset_n),
        "value": torch.randn(b),
    }


# --- KL-to-reference anchor (launch-gated, masked) ---


def test_kl_to_identical_policy_is_zero():
    out = _heads()
    kl = launch_gated_kl(out, {k: v.clone() for k, v in out.items()})
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_kl_to_different_policy_is_nonnegative_and_positive():
    kl = launch_gated_kl(_heads(), _heads())
    assert (kl >= -1e-5).all()
    assert float(kl.sum()) > 0.0


def test_kl_with_masks_is_finite_no_nan():
    cur, ref = _heads(), _heads()
    masks = {
        "launch": torch.ones(6, 2, dtype=torch.bool),
        "source": torch.ones(6, 16, dtype=torch.bool),
        "target": torch.ones(6, 32, dtype=torch.bool),
        "frac": torch.ones(6, 4, dtype=torch.bool),
        "offset": torch.ones(6, 5, dtype=torch.bool),
    }
    # forbid a few positions in both policies (same mask, as PPO requires)
    masks["source"][:, 8:] = False
    masks["target"][:, 20:] = False
    kl = launch_gated_kl(cur, ref, masks)
    assert torch.isfinite(kl).all()
    assert (kl >= -1e-5).all()


# --- reward de-anchoring ---


def test_shaping_potential_none_zeros_base():
    cfg_none = Phase0TrainingConfig(shaping_potential="none", base_shaping_scale_start=1.0, base_shaping_scale_end=0.15)
    base_none, _ = shaping_scales(cfg_none, progress=0.0)
    assert base_none == 0.0
    base_none_end, _ = shaping_scales(cfg_none, progress=1.0)
    assert base_none_end == 0.0


def test_shaping_potential_producer_keeps_base():
    cfg = Phase0TrainingConfig(shaping_potential="producer", base_shaping_scale_start=1.0, base_shaping_scale_end=0.15)
    base, _ = shaping_scales(cfg, progress=0.0)
    assert base == 1.0


# --- eval-gating proxy ---


def test_evaluate_margin_returns_bounded_float():
    from python.agents.policy import FlatActorCritic
    from python.orbit_wars_gym.encoding import observation_dim

    model = FlatActorCritic(observation_dim())
    cfg = Phase0TrainingConfig(eval_seeds=2, eval_max_steps=80, eval_opponent="greedy")
    margin = _evaluate_margin(model, cfg, opponent_name="greedy", seeds=2, device=torch.device("cpu"))
    assert isinstance(margin, float)
    assert -1.0 <= margin <= 1.0


# --- end-to-end: KL anchor + eval-gating in a tiny training run ---


def test_mov2_train_end_to_end(tmp_path):
    base = tmp_path / "ref.pt"
    # 1) tiny base run to produce a reference (BC-like) checkpoint
    cfg0 = Phase0TrainingConfig(
        seed=1, policy_arch="flat", total_timesteps=256, rollout_steps=128,
        opponents=("greedy", "rush"), device="cpu", enable_comets=False,
        checkpoint_out=str(base),
    )
    train_phase0(cfg0)
    assert base.exists()

    # 2) Mov.2 run: KL-to-ref anchor + eval-gating, warm-started from the ref
    out = tmp_path / "mov2.pt"
    cfg = Phase0TrainingConfig(
        seed=2, policy_arch="flat", total_timesteps=256, rollout_steps=128,
        opponents=("greedy", "rush"), device="cpu", enable_comets=False,
        checkpoint_in=str(base), checkpoint_out=str(out),
        shaping_potential="none", kl_to_ref_coef=0.1,
        eval_every_updates=1, eval_seeds=1, eval_opponent="greedy", eval_max_steps=80,
    )
    summary = train_phase0(cfg)

    assert summary["shaping_potential"] == "none"
    assert summary["kl_to_ref_coef"] == 0.1
    assert "kl_to_ref" in summary["update_series"][0]
    assert summary["last_kl_to_ref"] >= 0.0  # anchor active
    assert summary["eval_gated"] is True
    assert len(summary["eval_series"]) >= 1
    assert summary["checkpoint_selection"] == "best_eval_margin"
    assert out.exists()
