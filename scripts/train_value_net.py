"""H7 E3 — train the EntityValueNet on the self-play (obs, outcome) dataset.

Supervised regression: obs -> tanh value in [-1,1], MSE to the final outcome
(+1/-1/0) from the acting player's perspective. GPU for training; the saved
model runs CPU-only at inference (E4/submission invariant).

E3 verification criterion: on a held-out split, the learned value must RANK
positions better than the hand-coded baseline. The fair, obs-computable baseline
is the ship-margin heuristic = log(own_ships) - log(enemy_ships) (global features
[4],[5] of the encoding) — that IS the core term of _plan_value. We compare
rank-correlation (Spearman) of each with the true outcome; the net must win.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_value_net.py --epochs 30
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from python.agents.value_net import EntityValueNet

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "artifacts/h7/value_ds"
OUT = ROOT / "artifacts/h7/value_net.pt"


def load_dataset():
    obs, lab = [], []
    for f in sorted(glob.glob(str(DS / "shard_*.npz"))):
        d = np.load(f)
        obs.append(d["obs"])
        lab.append(d["label"])
    if not obs:
        raise SystemExit("no shards in artifacts/h7/value_ds — run collect_value_dataset.py first")
    return np.concatenate(obs).astype(np.float32), np.concatenate(lab).astype(np.float32)


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    d = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    return float((xr * yr).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    obs, lab = load_dataset()
    n = len(lab)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    obs, lab = obs[perm], lab[perm]
    n_hold = int(n * args.holdout)
    tr_o, tr_l = obs[n_hold:], lab[n_hold:]
    ho_o, ho_l = obs[:n_hold], lab[:n_hold]
    print(f"dataset: {n} states (train {len(tr_l)}, holdout {len(ho_l)}); "
          f"win={int((lab>0).sum())} loss={int((lab<0).sum())} tie={int((lab==0).sum())}")

    # baseline: ship-margin heuristic from global features [4]=log own, [5]=log enemy
    base_ho = ho_o[:, 4] - ho_o[:, 5]
    base_rho = spearman(base_ho, ho_l)
    print(f"BASELINE (ship-margin) holdout Spearman vs outcome: {base_rho:+.3f}")

    dev = args.device
    net = EntityValueNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    tr_o_t = torch.as_tensor(tr_o, device=dev)
    tr_l_t = torch.as_tensor(tr_l, device=dev)
    ho_o_t = torch.as_tensor(ho_o, device=dev)
    t0 = time.perf_counter()
    best_rho = -1.0
    for ep in range(args.epochs):
        net.train()
        idx = torch.randperm(len(tr_l_t), device=dev)
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            opt.zero_grad()
            pred = net(tr_o_t[b])
            loss = loss_fn(pred, tr_l_t[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            ho_pred = net(ho_o_t).cpu().numpy()
        rho = spearman(ho_pred, ho_l)
        acc = float(((ho_pred > 0) == (ho_l > 0))[ho_l != 0].mean()) if (ho_l != 0).any() else 0.0
        if rho > best_rho:
            best_rho = rho
            torch.save({"model": net.state_dict()}, OUT)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} mse={tot/len(tr_l_t):.4f} holdout: Spearman={rho:+.3f} "
                  f"sign-acc={acc:.3f} (best={best_rho:+.3f}) ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n=== E3 VERDICT ===")
    print(f"value-net holdout Spearman = {best_rho:+.3f}  vs  baseline ship-margin = {base_rho:+.3f}")
    print(f"{'PASS' if best_rho > base_rho else 'FAIL'}: net {'>' if best_rho>base_rho else '<='} baseline")
    print(f"saved best to {OUT}")


if __name__ == "__main__":
    main()
