"""Behavioral cloning of the holdwave into the GridNet per-planet policy.

The global 1-tuple BC failed because its target head was unlearnable (shared rank
across sources → acc 0.04-0.06). GridNet labels each source INDEPENDENTLY, and
``invert_gridnet_moves`` reproduces the holdwave's sources at ~100% with ~0.015 rad
angle error, so the supervised target signal is clean. This gives the self-play
PPO a strong init (≈ holdwave) instead of a random −1.0 start.

Loss is per-planet launch cross-entropy plus, on launched planets, the
target/frac/offset cross-entropy — the exact gating used at rollout/eval time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from python.agents.policy import GridNetActorCritic, PLANET_N
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.action_decoder import gridnet_planet_mask, invert_gridnet_moves
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim


def collect(seeds: list[int], num_players: int, episode_steps: int) -> dict[str, np.ndarray]:
    obs_l, act_l, mask_l = [], [], []
    for seed in seeds:
        b = RustBatchBackend(
            num_envs=1, num_players=num_players, seed=seed,
            config=RustConfig(episode_steps=episode_steps, enable_comets=True),
        )
        s = b.reset(seed)[0]
        pol = {p: make_isolated_opponent("pgs") for p in range(num_players)}
        for _ in range(episode_steps):
            # label seat 0 from its holdwave plan
            exp = [list(m) for m in pol[0](s, 0)]
            a, _ = invert_gridnet_moves(s, 0, exp)
            obs_l.append(encode_state(s, 0, DEFAULT_ENCODER_CONFIG).astype(np.float32))
            act_l.append(a)
            mask_l.append(gridnet_planet_mask(s, 0))
            rows = []
            for p in range(num_players):
                for m in pol[p](s, p):
                    rows.append([0.0, float(p), float(m[0]), float(m[1]), float(m[2])])
            arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
            out, st = b.step_flat_with_states(arr)
            s = st[0]
            if out[0].get("done"):
                break
    return {
        "obs": np.stack(obs_l), "action": np.stack(act_l), "mask": np.stack(mask_l),
    }


def train(data: dict[str, np.ndarray], *, epochs: int, batch_size: int, lr: float, seed: int) -> tuple[GridNetActorCritic, dict[str, Any]]:
    torch.manual_seed(seed)
    model = GridNetActorCritic(observation_dim())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    act = torch.as_tensor(data["action"], dtype=torch.long)
    mask = torch.as_tensor(data["mask"], dtype=torch.bool)
    n = obs.shape[0]
    metrics: dict[str, Any] = {}
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o, a, m = obs[idx], act[idx], mask[idx]
            out = model.forward(o)
            mf = m.float()
            launched = (a[..., 0] == 1).float() * mf
            # per-planet cross-entropy; only active planets count for launch, only
            # launched ones for the move heads (matches the rollout gating).
            launch_ce = F.cross_entropy(out["launch"].reshape(-1, 2), a[..., 0].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            tgt_ce = F.cross_entropy(out["target"].reshape(-1, out["target"].shape[-1]), a[..., 1].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            frac_ce = F.cross_entropy(out["frac"].reshape(-1, out["frac"].shape[-1]), a[..., 2].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            off_ce = F.cross_entropy(out["offset"].reshape(-1, out["offset"].shape[-1]), a[..., 3].reshape(-1), reduction="none").reshape(o.shape[0], PLANET_N)
            loss = (launch_ce * mf).sum() / mf.sum().clamp_min(1.0)
            loss = loss + ((tgt_ce + frac_ce + off_ce) * launched).sum() / launched.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
        metrics[f"epoch_{ep}_loss"] = tot / n
    # train-set launch accuracy on active planets (sanity)
    with torch.no_grad():
        out = model.forward(obs)
        mf = mask.float()
        launch_pred = out["launch"].argmax(-1)
        launch_acc = ((launch_pred == act[..., 0]).float() * mf).sum() / mf.sum().clamp_min(1.0)
        launched = (act[..., 0] == 1).float() * mf
        tgt_acc = ((out["target"].argmax(-1) == act[..., 1]).float() * launched).sum() / launched.sum().clamp_min(1.0)
    metrics["launch_acc"] = float(launch_acc)
    metrics["target_acc"] = float(tgt_acc)  # the head that was unlearnable globally
    metrics["examples"] = n
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--episode-steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/bc/gridnet_bc.pt")
    args = ap.parse_args()
    lo, hi = (args.seeds.split("-") + [args.seeds])[:2] if "-" in args.seeds else (args.seeds, args.seeds)
    seeds = list(range(int(lo), int(hi) + 1)) if "-" in args.seeds else [int(x) for x in args.seeds.split(",")]
    data = collect(seeds, args.num_players, args.episode_steps)
    model, metrics = train(data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "summary": {"arch": "gridnet", **metrics}}, out)
    print(json.dumps({"out": str(out), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if not k.startswith("epoch_")}}, indent=2))


if __name__ == "__main__":
    main()
