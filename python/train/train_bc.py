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
_EXPERT_NAMES = {0: "producer", 1: "oep", 2: "pgs", 3: "mahoraga"}


def _decoder_summary() -> dict[str, Any]:
    cfg = asdict(DEFAULT_DECODER_CONFIG)
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


def bc_loss(out: dict[str, torch.Tensor], action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    launch_label = action[:, 0]
    loss = F.cross_entropy(out["launch"], launch_label)
    parts = {"launch": float(loss.detach())}

    active = launch_label == 1
    if bool(active.any()):
        for name, col in _HEAD_COLS.items():
            head_loss = F.cross_entropy(out[name][active], action[active, col])
            loss = loss + head_loss
            parts[name] = float(head_loss.detach())
    return loss, parts


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
            loss, _ = bc_loss(out, action[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(batches, 1)})

    val_on_dev = {k: v.to(dev) for k, v in val.items()}
    metrics = evaluate(model, val_on_dev)

    summary = {
        "algorithm": "behavioral_cloning",
        "arch": arch,
        "dataset": str(dataset),
        "train_examples": int(n),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "decoder": _decoder_summary(),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "val_metrics": metrics,
    }

    if checkpoint_out is not None:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.to(torch.device("cpu")).state_dict(),
                "summary": summary,
                "config": {f"decoder_{k}": v for k, v in _decoder_summary().items()},
            },
            checkpoint_out,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="path to a .npz from collect_imitation_dataset")
    parser.add_argument("--arch", choices=("flat", "entity"), default="flat")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
