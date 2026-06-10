#!/usr/bin/env python
"""Verification gate for workflow ppo-explore Passo 1 (margin-Φ ablation).

Exit 0 iff the relative_margin treatment checkpoint BREAKS the Producer-parity
ceiling, i.e. BOTH:
  (1) mean_score_margin > MARGIN_MIN vs Producer at 500 steps, seat-averaged,
      over >=SEEDS seeds (the only valid eval per the project's 500-step rule);
  (2) last training entropy > ENTROPY_MIN, i.e. the policy did NOT re-collapse
      into the parity basin (collapse tell ≈ 0.13; the shares-Φ init was 0.129).

Designed to be the checkable condition behind the native /goal Stop hook:
    /goal "uv run --extra dev --extra train python -m scripts.verify_passo1_goal"

Both thresholds must hold: a positive margin with a collapsed entropy would be
suspect (likely an eval artifact, not a real escape), and a healthy entropy with
margin<=0 is just exploration without a win. The escape requires both.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "artifacts/ppo/ablation_relmargin/treatment/best.pt"
MARGIN_MIN = 0.0     # strictly beat Producer (parity = 0.0)
ENTROPY_MIN = 0.25   # clearly above the ~0.13 parity-basin collapse
SEEDS = 96           # 96 seeds decide (16 = triage)
EPISODE_STEPS = 500  # official length; 96/128 misleads


def main() -> int:
    if not CKPT.exists():
        print(f"FAIL: treatment checkpoint not found yet: {CKPT}")
        return 1

    summary = torch.load(CKPT, map_location="cpu", weights_only=False).get("summary", {})
    entropy = float(summary.get("last_entropy", 0.0))

    out_json = CKPT.parent / "goal_eval.json"
    cmd = [
        sys.executable, "-m", "scripts.benchmark_ppo_submission",
        "--checkpoint", str(CKPT),
        "--submission-out", str(CKPT.with_suffix(".goal.sub.py")),
        "--out", str(out_json),
        "--opponents", "producer",
        "--seeds", str(SEEDS),
        "--episode-steps", str(EPISODE_STEPS),
        "--skip-4p", "--jobs", "1",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"FAIL: benchmark exited {proc.returncode}")
        return 1
    margin = float(json.loads(out_json.read_text())["summary"]["mean_score_margin"])

    margin_ok = margin > MARGIN_MIN
    entropy_ok = entropy > ENTROPY_MIN
    print(
        f"margin={margin:+.4f} (need >{MARGIN_MIN}, {'OK' if margin_ok else 'NO'}) | "
        f"entropy={entropy:.4f} (need >{ENTROPY_MIN}, {'OK' if entropy_ok else 'NO'}) | "
        f"seeds={SEEDS} steps={EPISODE_STEPS}"
    )
    if margin_ok and entropy_ok:
        print("PASS: margin-Φ broke the Producer-parity ceiling.")
        return 0
    print("FAIL: parity ceiling NOT broken (see margin/entropy above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
