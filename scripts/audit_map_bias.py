"""Audit map-bias / feature asymmetry in the policy encoder (todo P5.1/P5.2).

Measures how much the policy's per-head logits change when a state is replaced by
a geometrically equivalent one (180° rotation, vertical reflection) and how well
the encoder is perspective-symmetric under a 2-player swap. A position-agnostic
policy would have ~0 gap under the board symmetries; the current flat encoder
uses absolute ``x/y`` and ``planet_id``, so the gap is the baseline bias we want
to compare future encoder variants against.

For 180°/reflection the optimal action indices are invariant (ranks are by
ships/production; offsets are relative to a co-rotating base angle), so we compare
``logits(state)`` directly with ``logits(transform(state))`` head-by-head.

For the player swap we compare ``logits(swap(state), player=0)`` with
``logits(state, player=1)`` — a perspective-correct encoder makes these equal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from python.agents.policy import FlatActorCritic
from python.agents.registry import get_heuristic_policies
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim
from python.orbit_wars_gym.symmetry import reflect_state_x, rotate_state_180, swap_players_2p

_HEADS = ("launch", "source", "target", "frac", "offset")


def _collect_states(seeds: list[int], steps: int) -> list[dict[str, Any]]:
    producer = get_heuristic_policies()["producer"]
    states: list[dict[str, Any]] = []
    for seed in seeds:
        backend = RustBatchBackend(
            num_envs=1, num_players=2, seed=seed,
            config=RustConfig(episode_steps=steps, enable_comets=True, act_timeout=1.0),
        )
        state = backend.reset(seed)[0]
        for _ in range(steps):
            states.append(state)
            rows: list[list[float]] = []
            for player in range(2):
                for m in producer(state, player):
                    rows.append([0.0, float(player), float(m[0]), float(m[1]), float(m[2])])
            flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            outcomes, next_states = backend.step_flat_with_states(flat)
            state = next_states[0]
            if bool(outcomes[0].get("done", False)):
                break
    return states


@torch.no_grad()
def _logits(model: FlatActorCritic, state: dict[str, Any], player: int) -> dict[str, torch.Tensor]:
    obs = torch.as_tensor(
        encode_state(state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32
    ).unsqueeze(0)
    out = model.forward(obs)
    return {h: out[h].squeeze(0) for h in _HEADS}


def _gap(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    return {h: float((a[h] - b[h]).abs().max()) for h in _HEADS}


def audit(model: FlatActorCritic, states: list[dict[str, Any]]) -> dict[str, Any]:
    transforms = {
        "rotate_180": (rotate_state_180, 0),
        "reflect_x": (reflect_state_x, 0),
        "swap_players": (swap_players_2p, 0),
    }
    # swap_players compares against the player-1 perspective of the original state.
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in transforms}
    for state in states:
        base0 = _logits(model, state, 0)
        base1 = _logits(model, state, 1)
        for name, (fn, _player) in transforms.items():
            transformed = _logits(model, fn(state), 0)
            reference = base1 if name == "swap_players" else base0
            rows[name].append(_gap(transformed, reference))

    report: dict[str, Any] = {"num_states": len(states), "transforms": {}}
    for name, gaps in rows.items():
        per_head: dict[str, Any] = {}
        for h in _HEADS:
            vals = np.array([g[h] for g in gaps]) if gaps else np.zeros(1)
            per_head[h] = {
                "mean_max_abs_logit_gap": float(vals.mean()),
                "p95": float(np.percentile(vals, 95)),
                "max": float(vals.max()),
            }
        report["transforms"][name] = per_head
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="optional policy checkpoint (.pt)")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--out", default="artifacts/map_bias/invariance_report.json")
    args = parser.parse_args()

    model = FlatActorCritic(observation_dim())
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    states = _collect_states(list(range(args.seeds)), args.steps)
    report = audit(model, states)
    report["checkpoint"] = args.checkpoint

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
