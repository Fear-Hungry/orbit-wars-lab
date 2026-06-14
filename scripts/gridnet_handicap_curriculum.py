"""Handicap curriculum to cross the reactive→planner cliff (GridNet).

The strength ladder has a cliff: GridNet beats every REACTIVE bot (greedy/rush/
defensive, +0.7..+0.9) but is crushed by every PLANNER (producer/oep/holdwave,
−1.0). There is no intermediate-strength bot to bridge it. But a HANDICAPPED
planner is winnable: GridNet beats the producer at ~0.2x ships. So we build the
missing bridge by difficulty: train vs producer@scale, and raise the scale each
time the policy dominates the current level — an AlphaStar-style difficulty
curriculum. If the policy can climb scale 0.2 → 1.0 and beat the unhandicapped
planner, the cliff is crossed; if it stalls, the cliff is fundamental at this
compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.train.train_ppo import (
    Phase0TrainingConfig,
    evaluate_gridnet_margin,
    train_phase0,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="artifacts/bc/gridnet_bc_2p_big.pt")
    ap.add_argument("--planner", default="producer")
    ap.add_argument("--out-dir", default="artifacts/ppo/gridnet_handicap")
    ap.add_argument("--start-scale", type=float, default=0.2)
    ap.add_argument("--scale-step", type=float, default=0.1)
    ap.add_argument("--promote-margin", type=float, default=0.15, help="dominate level → raise scale")
    ap.add_argument("--chunk-timesteps", type=int, default=40000)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--max-chunks-per-level", type=int, default=6)
    ap.add_argument("--rollout-num-envs", type=int, default=48)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.0005)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-ratio", type=int, default=0,
                    help="self segments per planner segment. 0 = NO self (self-play degrades the "
                    "BC competence; gentle PPO vs the handicapped planner alone refines it).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prev = args.init
    scale = args.start_scale
    history: list[dict] = []
    level_chunks = 0
    best_full_margin = -1.0  # margin vs the UNHANDICAPPED planner (the real target)

    for chunk in range(args.max_chunks):
        scale_next = min(1.0, scale + args.scale_step)
        opp = f"{args.planner}@{scale:.2f}"
        ckpt = out / f"chunk{chunk:02d}.pt"
        # GENTLE refine vs the handicapped planner — NO self-play (it degrades the
        # BC's planner-competence). Train on the current level + the next one so the
        # policy is already pushing into the harder level it's about to be promoted
        # to. ent/lr low so PPO refines the BC instead of exploring it away.
        if args.self_ratio <= 0:
            opponents = (opp, f"{args.planner}@{scale_next:.2f}")
        else:
            opponents = tuple([opp] + ["self"] * args.self_ratio)
        cfg = Phase0TrainingConfig(
            seed=args.seed + chunk, policy_arch="gridnet",
            opponents=opponents,
            total_timesteps=args.chunk_timesteps, rollout_steps=args.rollout_steps,
            rollout_num_envs=args.rollout_num_envs, ent_coef=args.ent_coef,
            learning_rate=args.learning_rate, device=args.device,
            checkpoint_in=prev, checkpoint_out=str(ckpt),
            self_opponent_checkpoint=(
                str(prev) if args.self_ratio > 0 and prev and prev.endswith(".pt") else None
            ),
        )
        train_phase0(cfg)
        prev = str(ckpt)
        level_chunks += 1

        # gate: margin vs the CURRENT handicapped level AND vs the full planner
        import torch

        from python.agents.policy import GridNetActorCritic
        from orbit_wars_gym.encoding import observation_dim

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = GridNetActorCritic(observation_dim())
        model.load_state_dict(ck["model_state_dict"])
        m_level = evaluate_gridnet_margin(model, opponent_name=opp, seeds=args.eval_seeds, episode_steps=args.eval_steps)
        m_full = evaluate_gridnet_margin(model, opponent_name=args.planner, seeds=args.eval_seeds, episode_steps=args.eval_steps)
        best_full_margin = max(best_full_margin, m_full["mean_score_margin"])
        rec = {
            "chunk": chunk, "scale": round(scale, 2),
            "margin_at_level": round(m_level["mean_score_margin"], 4),
            "margin_vs_full": round(m_full["mean_score_margin"], 4),
            "win_at_level": round(m_level["win_rate"], 3),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)

        # promote the curriculum if the policy dominates the current level
        if m_level["mean_score_margin"] >= args.promote_margin and scale < 1.0:
            scale = min(1.0, scale + args.scale_step)
            level_chunks = 0
        elif level_chunks >= args.max_chunks_per_level:
            # stuck at this level → stop (cliff is fundamental at this scale)
            print(json.dumps({"stop": "stuck", "scale": round(scale, 2), "chunks_at_level": level_chunks}), flush=True)
            break
        if scale >= 1.0 and m_full["mean_score_margin"] >= args.promote_margin:
            print(json.dumps({"success": "beat full planner", "margin": m_full["mean_score_margin"]}), flush=True)
            break

    report = {"history": history, "best_margin_vs_full": best_full_margin, "final_scale": round(scale, 2)}
    (out / "curriculum_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
