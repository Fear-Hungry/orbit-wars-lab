"""Behavioral cloning of Producer/OEP experts (todo P2).

Trains :class:`FlatActorCritic` to imitate the launch-gated action space produced
by ``scripts.collect_imitation_dataset``. The loss follows the launch-conditional
factorisation (todo P1.5):

    loss = CE(launch_logits, launch_label)
    active = launch_label == 1
    loss += CE(source[active], ...) + CE(target[active], ...)
          + CE(frac[active], ...)   + CE(offset[active], ...)

Move-head losses are only applied to launch turns, because a pass turn has no
meaningful source/target/frac/offset. Global accuracy would be dominated by the
~81% pass rate, so we report launch precision/recall/F1, the predicted-vs-expert
pass rate, and active-head top-1 accuracy (launch turns only), per expert.

Checkpoints embed the decoder payload under ``summary.decoder`` so they export
through ``scripts.export_submission`` / ``scripts.benchmark_ppo_submission``
unchanged. The value head is left untrained (BC has no return targets); export
does not use it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from python.agents.policy import EntityActorCritic, FlatActorCritic
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG
from python.orbit_wars_gym.encoding import observation_dim

_ARCHS = {"flat": FlatActorCritic, "entity": EntityActorCritic}

_HEAD_COLS = {"source": 1, "target": 2, "frac": 3, "offset": 4}
_EXPERT_NAMES = {
    0: "producer",
    1: "oep",
    2: "pgs_holdwave",
    3: "pgs_bigwave",
    4: "brep",
    5: "greedy",
    6: "rush",
}


def _decoder_summary(dataset: Path | None = None) -> dict[str, Any]:
    cfg = asdict(DEFAULT_DECODER_CONFIG)
    if dataset is not None:
        meta_path = dataset.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta.get("decoder_config"), dict):
                cfg.update(meta["decoder_config"])
    return {
        "fractions": list(cfg["fractions"]),
        "angle_offsets": list(cfg["angle_offsets"]),
        "max_moves_per_turn": int(cfg["max_moves_per_turn"]),
        "min_ships_to_launch": int(cfg["min_ships_to_launch"]),
        "reserve_home_ships": int(cfg["reserve_home_ships"]),
    }


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in ("obs", "action", "split_id", "expert_id")}


def _split(data: dict[str, np.ndarray], split_id: int) -> dict[str, torch.Tensor]:
    mask = data["split_id"] == split_id
    return {
        "obs": torch.as_tensor(data["obs"][mask], dtype=torch.float32),
        "action": torch.as_tensor(data["action"][mask], dtype=torch.long),
        "expert_id": torch.as_tensor(data["expert_id"][mask], dtype=torch.long),
    }


def bc_loss(
    out: dict[str, torch.Tensor],
    action: torch.Tensor,
    *,
    launch_positive_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    launch_label = action[:, 0]
    launch_weight = None
    if float(launch_positive_weight) != 1.0:
        launch_weight = torch.as_tensor(
            [1.0, float(launch_positive_weight)],
            dtype=out["launch"].dtype,
            device=out["launch"].device,
        )
    loss = F.cross_entropy(out["launch"], launch_label, weight=launch_weight)
    parts = {"launch": float(loss.detach())}

    active = launch_label == 1
    if bool(active.any()):
        for name, col in _HEAD_COLS.items():
            head_loss = F.cross_entropy(out[name][active], action[active, col])
            loss = loss + head_loss
            parts[name] = float(head_loss.detach())
    return loss, parts


def _fit_launch_bias_delta(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_abs: float = 6.0,
    steps: int = 241,
) -> dict[str, float]:
    """Find a scalar class-1 launch-logit offset that minimizes val CE."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"launch logits must have shape (N, 2), got {tuple(logits.shape)}")
    if labels.numel() == 0:
        return {"delta": 0.0, "loss_before": 0.0, "loss_after": 0.0}

    base_loss = F.cross_entropy(logits, labels)
    deltas = torch.linspace(
        -float(max_abs),
        float(max_abs),
        int(steps),
        dtype=logits.dtype,
        device=logits.device,
    )
    best_delta = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    best_loss = base_loss
    offset = torch.zeros_like(logits)
    for delta in deltas:
        offset.zero_()
        offset[:, 1] = delta
        loss = F.cross_entropy(logits + offset, labels)
        if float(loss.detach()) < float(best_loss.detach()):
            best_loss = loss
            best_delta = delta.detach()
    return {
        "delta": float(best_delta.cpu()),
        "loss_before": float(base_loss.detach().cpu()),
        "loss_after": float(best_loss.detach().cpu()),
    }


def _apply_launch_bias_delta(model: FlatActorCritic, delta: float) -> None:
    launch = getattr(model, "launch", None)
    if launch is None:
        launch = model.heads.launch
    with torch.no_grad():
        launch.bias[1] += float(delta)


@torch.no_grad()
def calibrate_launch_bias(model: FlatActorCritic, val: dict[str, torch.Tensor]) -> dict[str, float]:
    if val["obs"].shape[0] == 0:
        return {"delta": 0.0, "loss_before": 0.0, "loss_after": 0.0}
    out = model.forward(val["obs"])
    result = _fit_launch_bias_delta(out["launch"], val["action"][:, 0])
    _apply_launch_bias_delta(model, result["delta"])
    return result


@torch.no_grad()
def evaluate(model: FlatActorCritic, val: dict[str, torch.Tensor]) -> dict[str, Any]:
    if val["obs"].shape[0] == 0:
        return {"examples": 0}
    out = model.forward(val["obs"])
    action = val["action"]
    launch_label = action[:, 0]
    pred_launch = out["launch"].argmax(dim=-1)

    tp = int(((pred_launch == 1) & (launch_label == 1)).sum())
    fp = int(((pred_launch == 1) & (launch_label == 0)).sum())
    fn = int(((pred_launch == 0) & (launch_label == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    active = launch_label == 1
    head_acc: dict[str, float] = {}
    for name, col in _HEAD_COLS.items():
        if bool(active.any()):
            pred = out[name][active].argmax(dim=-1)
            head_acc[name] = float((pred == action[active, col]).float().mean())
        else:
            head_acc[name] = 0.0

    per_expert: dict[str, Any] = {}
    for eid, ename in _EXPERT_NAMES.items():
        emask = val["expert_id"] == eid
        if bool(emask.any()):
            sub = {k: out[k][emask] for k in ("launch", "source", "target", "frac", "offset")}
            eloss, _ = bc_loss(sub, action[emask])
            per_expert[ename] = {"examples": int(emask.sum()), "loss": float(eloss)}

    return {
        "examples": int(val["obs"].shape[0]),
        "launch_precision": precision,
        "launch_recall": recall,
        "launch_f1": f1,
        "predicted_pass_rate": float((pred_launch == 0).float().mean()),
        "expert_pass_rate": float((launch_label == 0).float().mean()),
        "active_head_top1_acc": head_acc,
        "per_expert": per_expert,
    }


def train_bc(
    *,
    dataset: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    checkpoint_out: Path | None,
    seed: int = 0,
    arch: str = "flat",
    launch_positive_weight: float = 1.0,
    calibrate_launch: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_dataset(dataset)
    train = _split(data, 0)
    val = _split(data, 1)
    if train["obs"].shape[0] == 0:
        raise ValueError(f"dataset {dataset} has no training (split_id==0) examples")

    dev = torch.device(device)
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; valid: {sorted(_ARCHS)}")
    model = _ARCHS[arch](observation_dim()).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs = train["obs"].to(dev)
    action = train["action"].to(dev)
    n = obs.shape[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = model.forward(obs[idx])
            loss, _ = bc_loss(out, action[idx], launch_positive_weight=launch_positive_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(batches, 1)})

    val_on_dev = {k: v.to(dev) for k, v in val.items()}
    launch_calibration = None
    if calibrate_launch:
        launch_calibration = calibrate_launch_bias(model, val_on_dev)
    metrics = evaluate(model, val_on_dev)

    decoder = _decoder_summary(dataset)
    summary = {
        "algorithm": "behavioral_cloning",
        "arch": arch,
        "dataset": str(dataset),
        "train_examples": int(n),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "launch_positive_weight": float(launch_positive_weight),
        "calibrate_launch": bool(calibrate_launch),
        "launch_bias_calibration": launch_calibration,
        "decoder": decoder,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "val_metrics": metrics,
    }

    if checkpoint_out is not None:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.to(torch.device("cpu")).state_dict(),
                "summary": summary,
                "config": {f"decoder_{k}": v for k, v in decoder.items()},
            },
            checkpoint_out,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="path to a .npz from collect_imitation_dataset"
    )
    parser.add_argument("--arch", choices=("flat", "entity"), default="flat")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--launch-positive-weight",
        type=float,
        default=1.0,
        help="cross-entropy weight for launch=1 to counter pass-heavy expert datasets",
    )
    parser.add_argument(
        "--calibrate-launch",
        action="store_true",
        help="fit a scalar launch-logit bias on the validation split before saving/evaluating",
    )
    args = parser.parse_args()

    summary = train_bc(
        dataset=Path(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
        checkpoint_out=Path(args.checkpoint_out) if args.checkpoint_out else None,
        seed=args.seed,
        arch=args.arch,
        launch_positive_weight=float(args.launch_positive_weight),
        calibrate_launch=bool(args.calibrate_launch),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
