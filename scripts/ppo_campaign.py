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
from statistics import fmean
from typing import Any

from python.train.train_ppo import (
    Phase0TrainingConfig,
    _parse_opponents,
    build_phase5_4p_config,
    train_phase0,
    train_phase5_4p,
)
from scripts.benchmark_ppo_submission import benchmark_exported_checkpoint


def _margin_for_format(report: dict[str, Any], fmt: str) -> float | None:
    records: list[dict[str, Any]] = []
    for item in report.get("formats", []):
        if item.get("format") != fmt:
            continue
        if fmt == "2p":
            for opponent in item.get("opponents", []):
                records.extend(opponent.get("records", []))
        else:
            records.extend(item.get("records", []))
    if not records:
        return None
    return float(fmean(float(record["normalized_margin"]) for record in records))


def _margin_residual_inprocess(
    checkpoint: Path,
    *,
    base_agent: str,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
) -> dict[str, Any]:
    """Per-chunk gate for the residual arch: evaluate IN-PROCESS (the BReP arch has
    no render_submission export path). Margin vs the first opponent (the parity-floor
    base, e.g. pgs_holdwave) is the gate; others are reported for context."""
    import torch

    from python.agents.policy import ProducerResidualBranchActorCritic
    from python.train.train_ppo import evaluate_residual_margin
    from orbit_wars_gym.encoding import observation_dim

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ProducerResidualBranchActorCritic(observation_dim())
    model.load_state_dict(ckpt["model_state_dict"])
    per_opp = {
        opp: evaluate_residual_margin(
            model, base_agent=base_agent, opponent_name=opp,
            seeds=len(seeds), episode_steps=episode_steps,
        )
        for opp in opponents
    }
    gate = per_opp[opponents[0]]
    return {
        "games": gate["games"],
        "win_rate": gate["win_rate"],
        "mean_score_margin": gate["mean_score_margin"],
        "invalid_action_rate": 0.0,
        "margin_2p": gate["mean_score_margin"],
        "margin_4p": None,
        "per_opponent": {o: per_opp[o]["mean_score_margin"] for o in opponents},
    }


def _margin_gridnet_inprocess(
    checkpoint: Path,
    *,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
) -> dict[str, Any]:
    """Per-chunk gate for the GridNet arch: evaluate IN-PROCESS (no render_submission
    path). Margin vs the first opponent (e.g. pgs_holdwave) is the gate."""
    import torch

    from python.agents.policy import GridNetActorCritic
    from python.train.train_ppo import evaluate_gridnet_margin
    from orbit_wars_gym.encoding import observation_dim

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GridNetActorCritic(observation_dim())
    model.load_state_dict(ckpt["model_state_dict"])
    per_opp = {
        opp: evaluate_gridnet_margin(model, opponent_name=opp, seeds=len(seeds), episode_steps=episode_steps)
        for opp in opponents
    }
    gate = per_opp[opponents[0]]
    return {
        "games": gate["games"],
        "win_rate": gate["win_rate"],
        "mean_score_margin": gate["mean_score_margin"],
        "invalid_action_rate": 0.0,
        "margin_2p": gate["mean_score_margin"],
        "margin_4p": None,
        "per_opponent": {o: per_opp[o]["mean_score_margin"] for o in opponents},
    }


def _margin(
    checkpoint: Path,
    *,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
    include_4p: bool = False,
    jobs: int = 1,
    policy_arch: str = "flat",
    base_agent: str = "producer",
) -> dict[str, Any]:
    if policy_arch == "producer_residual":
        return _margin_residual_inprocess(
            checkpoint, base_agent=base_agent, opponents=opponents,
            seeds=seeds, episode_steps=episode_steps,
        )
    if policy_arch == "gridnet":
        return _margin_gridnet_inprocess(
            checkpoint, opponents=opponents, seeds=seeds, episode_steps=episode_steps,
        )
    report = benchmark_exported_checkpoint(
        checkpoint,
        submission_out=checkpoint.with_suffix(".sub.py"),
        opponents=opponents,
        seeds=seeds,
        episode_steps=episode_steps,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=include_4p,
        jobs=jobs,
    )
    summary = dict(report["summary"])
    summary["margin_2p"] = _margin_for_format(report, "2p")
    summary["margin_4p"] = _margin_for_format(report, "4p")
    return summary


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
    training_track: str = "phase0_2p",
    policy_arch: str = "flat",
    base_agent: str = "producer",
    eval_jobs: int = 1,
    eval_include_4p: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_track == "phase5_4p" and rollout_num_envs != 1:
        # train_ppo only supports vectorized rollouts with num_players == 2.
        print("phase5_4p: forcing rollout_num_envs=1 (multi-env rollout is 2p-only)")
        rollout_num_envs = 1
    prev = init_checkpoint
    best_margin = float("-inf")
    best_path: Path | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    for chunk in range(chunks):
        ckpt = out_dir / f"chunk{chunk:02d}.pt"
        common = dict(
            seed=seed + chunk,
            policy_arch=policy_arch,
            base_agent=base_agent,
            opponents=opponents,
            total_timesteps=chunk_timesteps,
            rollout_steps=rollout_steps,
            rollout_num_envs=rollout_num_envs,
            ent_coef=ent_coef,
            device=device,
            checkpoint_in=prev,
            checkpoint_out=str(ckpt),
        )
        if training_track == "phase5_4p":
            train_summary = train_phase5_4p(build_phase5_4p_config(**common))
        else:
            train_summary = train_phase0(Phase0TrainingConfig(**common))
        summary = _margin(
            ckpt,
            opponents=eval_opponents,
            seeds=eval_seeds,
            episode_steps=eval_episode_steps,
            include_4p=eval_include_4p,
            jobs=eval_jobs,
            policy_arch=policy_arch,
            base_agent=base_agent,
        )
        # The gate margin must reflect the track's goal: a 4p campaign is judged
        # on the 4p format, not on an aggregate diluted by 2p games.
        if training_track == "phase5_4p" and summary.get("margin_4p") is not None:
            margin = float(summary["margin_4p"])
        else:
            margin = float(summary["mean_score_margin"])
        record = {
            "chunk": chunk,
            "cumulative_timesteps": (chunk + 1) * chunk_timesteps,
            "margin": margin,
            "margin_2p": summary.get("margin_2p"),
            "margin_4p": summary.get("margin_4p"),
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
        "training_track": training_track,
        "policy_arch": policy_arch,
        "opponents": list(opponents),
        "eval_opponents": eval_opponents,
        "eval_seeds": eval_seeds,
        "eval_include_4p": eval_include_4p,
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
    parser.add_argument(
        "--init",
        default="",
        help="initial checkpoint (e.g. a BC checkpoint); empty = fresh init "
        "(producer_residual trains from KEEP-init = parity floor)",
    )
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
    parser.add_argument(
        "--training-track",
        choices=("phase0_2p", "phase5_4p"),
        default="phase0_2p",
        help="phase5_4p trains 4-player games (single-env rollout only)",
    )
    parser.add_argument(
        "--policy-arch", choices=("flat", "entity", "producer_residual", "gridnet"), default="flat"
    )
    parser.add_argument(
        "--base-agent",
        default="producer",
        help="base plan for producer_residual: producer or pgs_holdwave (incumbent floor)",
    )
    parser.add_argument(
        "--eval-jobs",
        type=int,
        default=1,
        help="parallel benchmark games per eval; keep 1 on WSL (pool deadlocks)",
    )
    parser.add_argument(
        "--eval-include-4p",
        action="store_true",
        help="also benchmark the 4p format each chunk (gate margin for phase5_4p)",
    )
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
        training_track=args.training_track,
        policy_arch=args.policy_arch,
        base_agent=args.base_agent,
        eval_jobs=max(1, args.eval_jobs),
        eval_include_4p=args.eval_include_4p,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
