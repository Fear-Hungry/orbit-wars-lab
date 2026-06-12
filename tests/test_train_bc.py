from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG
from python.orbit_wars_gym.action_inverse import DEFAULT_INVERSE_CONFIG
from python.train.train_bc import _fit_launch_bias_delta, bc_loss, train_bc
from scripts.collect_imitation_dataset import _pack, collect_dataset
from scripts.export_submission import _load_checkpoint_payload


def _tiny_dataset(tmp_path: Path) -> Path:
    examples = collect_dataset(
        "producer_only",
        seeds=[0, 1, 3],  # 0,1 -> train ; 3 -> val (split_for_seed)
        num_players=2,
        episode_steps=24,
        enable_comets=False,
        act_timeout=1.0,
        decoder_cfg=DEFAULT_DECODER_CONFIG,
        inverse_cfg=DEFAULT_INVERSE_CONFIG,
    )
    packed = _pack(examples)
    out = tmp_path / "producer_only.npz"
    np.savez(out, **packed)
    return out


def test_bc_loss_ignores_move_heads_on_pass_turns() -> None:
    out = {
        "launch": torch.tensor([[2.0, -2.0], [-2.0, 2.0]]),  # row0 -> pass, row1 -> launch
        "source": torch.randn(2, 16),
        "target": torch.randn(2, 32),
        "frac": torch.randn(2, 4),
        "offset": torch.randn(2, 5),
    }
    # Two pass rows: move-head losses must not appear (only the launch part).
    pass_only = torch.tensor([[0, 0, 0, 0, 0], [0, 5, 9, 1, 2]])
    _, parts = bc_loss(out, pass_only)
    assert set(parts) == {"launch"}

    mixed = torch.tensor([[0, 0, 0, 0, 0], [1, 3, 7, 2, 1]])
    _, parts_mixed = bc_loss(out, mixed)
    assert set(parts_mixed) == {"launch", "source", "target", "frac", "offset"}


def test_bc_loss_can_weight_launch_positive_class() -> None:
    out = {
        "launch": torch.tensor([[3.0, -3.0], [3.0, -3.0]]),
        "source": torch.randn(2, 16),
        "target": torch.randn(2, 32),
        "frac": torch.randn(2, 4),
        "offset": torch.randn(2, 5),
    }
    action = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])

    unweighted, _ = bc_loss(out, action)
    weighted, _ = bc_loss(out, action, launch_positive_weight=4.0)

    assert weighted > unweighted


def test_launch_bias_calibration_fits_validation_ce() -> None:
    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 0])

    result = _fit_launch_bias_delta(logits, labels, max_abs=4.0, steps=81)

    assert result["delta"] < 0.0
    assert result["loss_after"] < result["loss_before"]


def test_train_bc_learns_and_exports(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    checkpoint = tmp_path / "bc_producer.pt"
    summary = train_bc(
        dataset=dataset,
        epochs=10,
        batch_size=128,
        lr=1e-3,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    metrics = summary["val_metrics"]
    if metrics["examples"]:
        assert 0.0 <= metrics["predicted_pass_rate"] <= 1.0
        assert "launch_f1" in metrics
        assert set(metrics["active_head_top1_acc"]) == {"source", "target", "frac", "offset"}

    # Non-regression: the checkpoint must export through the submission pipeline,
    # which requires the launch head tensors to be present.
    payload = _load_checkpoint_payload(str(checkpoint))
    assert "launch.weight" in payload["weights"]
    assert "launch.bias" in payload["weights"]


def test_train_bc_embeds_dataset_sidecar_decoder(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    decoder_cfg = asdict(DEFAULT_DECODER_CONFIG)
    decoder_cfg["max_moves_per_turn"] = 1
    dataset.with_suffix(".meta.json").write_text(
        json.dumps({"decoder_config": decoder_cfg}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "bc_single_action.pt"

    summary = train_bc(
        dataset=dataset,
        epochs=1,
        batch_size=128,
        lr=1e-3,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert summary["decoder"]["max_moves_per_turn"] == 1
    payload = _load_checkpoint_payload(str(checkpoint))
    assert payload["decoder"]["max_moves_per_turn"] == 1
