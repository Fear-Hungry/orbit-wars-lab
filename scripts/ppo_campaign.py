"""Eval-gated PPO campaign orchestrator (todo P3 periodic-eval + stopping rule).

A naive long PPO run regresses: extending entity-init PPO from 120k to 620k
timesteps *worsened* the paired margin vs Producer/OEP (-0.749 -> -0.848) even
though the training curves stayed healthy (EV ~0.94, stable KL). Training reward
is not the objective; the paired score margin is. This orchestrator therefore
trains in chunks, benchmarks each chunk's checkpoint against the real ruler, keeps
the best-by-margin checkpoint, and early-stops when the margin stops improving.

Each chunk warm-starts from the previous chunk's checkpoint (PPO + optimizer
state), so chunking is equivalent to one long run but with a paired-margin gate
between chunks. The "best" checkpoint — not the last — is what gets promoted.

Promotion to a Kaggle submission remains a human decision (96-seed gate); this
script only finds and reports the best local checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from python.train.train_ppo import Phase0TrainingConfig, _parse_opponents, train_phase0
from scripts.benchmark_ppo_submission import benchmark_exported_checkpoint


def _margin(checkpoint: Path, *, opponents: list[str], seeds: list[int], episode_steps: int) -> dict[str, Any]:
    report = benchmark_exported_checkpoint(
        checkpoint,
        submission_out=checkpoint.with_suffix(".sub.py"),
        opponents=opponents,
        seeds=seeds,
        episode_steps=episode_steps,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=False,
        jobs=min(8, len(seeds)),
    )
    return report["summary"]


def run_campaign(
    *,
    init_checkpoint: str,
    out_dir: Path,
    opponents: tuple[str, ...],
    eval_opponents: list[str],
    chunks: int,
    chunk_timesteps: int,
    rollout_steps: int,
    eval_seeds: list[int],
    eval_episode_steps: int,
    ent_coef: float,
    patience: int,
    seed: int,
    device: str = "cpu",
    rollout_num_envs: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = init_checkpoint
    best_margin = float("-inf")
    best_path: Path | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    for chunk in range(chunks):
        ckpt = out_dir / f"chunk{chunk:02d}.pt"
        cfg = Phase0TrainingConfig(
            seed=seed + chunk,
            opponents=opponents,
            total_timesteps=chunk_timesteps,
            rollout_steps=rollout_steps,
            rollout_num_envs=rollout_num_envs,
            ent_coef=ent_coef,
            device=device,
            checkpoint_in=prev,
            checkpoint_out=str(ckpt),
        )
        train_summary = train_phase0(cfg)
        summary = _margin(
            ckpt, opponents=eval_opponents, seeds=eval_seeds, episode_steps=eval_episode_steps
        )
        margin = float(summary["mean_score_margin"])
        record = {
            "chunk": chunk,
            "cumulative_timesteps": (chunk + 1) * chunk_timesteps,
            "margin": margin,
            "win_rate": float(summary["win_rate"]),
            "invalid_action_rate": float(summary["invalid_action_rate"]),
            "explained_variance": float(train_summary.get("last_explained_variance", 0.0)),
            "entropy": float(train_summary.get("last_entropy", 0.0)),
            "checkpoint": str(ckpt),
        }
        history.append(record)

        if margin > best_margin:
            best_margin = margin
            best_path = ckpt
            no_improve = 0
        else:
            no_improve += 1

        print(json.dumps(record))
        if no_improve >= patience:
            print(f"early stop: margin did not improve for {patience} chunks (best={best_margin:.4f})")
            break
        prev = str(ckpt)

    best_out = out_dir / "best.pt"
    if best_path is not None:
        shutil.copyfile(best_path, best_out)

    report = {
        "init_checkpoint": init_checkpoint,
        "opponents": list(opponents),
        "eval_opponents": eval_opponents,
        "eval_seeds": eval_seeds,
        "ent_coef": ent_coef,
        "patience": patience,
        "best_margin": best_margin,
        "best_checkpoint": str(best_out) if best_path is not None else None,
        "best_source": str(best_path) if best_path is not None else None,
        "history": history,
    }
    (out_dir / "campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, help="initial checkpoint (e.g. a BC checkpoint)")
    parser.add_argument("--out-dir", default="artifacts/ppo/campaign")
    parser.add_argument(
        "--opponents",
        default="producer,producer,producer,producer,producer,producer,producer,greedy,greedy,oep",
        help="training curriculum (repeat names to weight)",
    )
    parser.add_argument("--eval-opponents", default="producer,oep")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-timesteps", type=int, default=60000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-episode-steps", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-num-envs", type=int, default=1)
    args = parser.parse_args()

    report = run_campaign(
        init_checkpoint=args.init,
        out_dir=Path(args.out_dir),
        opponents=_parse_opponents(args.opponents),
        eval_opponents=[o.strip() for o in args.eval_opponents.split(",") if o.strip()],
        chunks=args.chunks,
        chunk_timesteps=args.chunk_timesteps,
        rollout_steps=args.rollout_steps,
        eval_seeds=list(range(args.eval_seeds)),
        eval_episode_steps=args.eval_episode_steps,
        ent_coef=args.ent_coef,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        rollout_num_envs=args.rollout_num_envs,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
