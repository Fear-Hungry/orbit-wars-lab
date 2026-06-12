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
# ruff: noqa: E402,I001

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.train.train_ppo import (
    Phase0TrainingConfig,
    _parse_opponents,
    build_phase5_4p_config,
    train_phase0,
    train_phase5_4p,
)
from scripts.benchmark_ppo_submission import benchmark_exported_checkpoint
from scripts.drl_promotion_gate import run_drl_promotion_gate


def _opponent_parts(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(name).split("+") if part.strip())


def _pairwise_rate(pairwise: dict[str, Any], name: str) -> float | None:
    metrics = pairwise.get(name)
    if not isinstance(metrics, dict):
        return None
    for key in ("decisive_win_rate", "win_rate"):
        if key in metrics:
            return float(metrics[key])
    return None


def _pfsp_reweighted_opponents(
    opponents: tuple[str, ...],
    pairwise: dict[str, Any],
    *,
    band: tuple[float, float] = (0.35, 0.65),
    max_repeats: int = 4,
) -> tuple[str, ...]:
    """PFSP-style deterministic reweighting from gate pairwise win rates.

    Opponents with at least one component near the learning frontier
    (win-rate inside ``band``) are repeated most. Composite 4p lineups keep their
    shape; only their sampling frequency changes.
    """
    lo, hi = float(band[0]), float(band[1])
    cap = max(1, int(max_repeats))
    weighted: list[str] = []
    for opponent in opponents:
        rates = [
            rate
            for part in _opponent_parts(opponent)
            if (rate := _pairwise_rate(pairwise, part)) is not None
        ]
        repeats = 1
        if rates:
            if any(lo <= rate <= hi for rate in rates):
                repeats = cap
            else:
                closeness = max(max(0.0, 1.0 - abs(rate - 0.5) / 0.5) for rate in rates)
                if closeness >= 0.5:
                    repeats = max(2, cap // 2)
        weighted.extend([opponent] * repeats)
    return tuple(weighted)


def _margin(
    checkpoint: Path,
    *,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
    include_4p: bool = False,
) -> dict[str, Any]:
    report = benchmark_exported_checkpoint(
        checkpoint,
        submission_out=checkpoint.with_suffix(".sub.py"),
        opponents=opponents,
        seeds=seeds,
        episode_steps=episode_steps,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=bool(include_4p),
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
    training_track: str = "phase0_2p",
    num_players: int | None = None,
    strict_drl_gate: bool = False,
    drl_profile: str = "quick",
    drl_seeds: int | None = None,
    drl_steps: int | None = None,
    drl_jobs: int = 1,
    pfsp: bool = False,
    pfsp_min_winrate: float = 0.35,
    pfsp_max_winrate: float = 0.65,
    pfsp_max_repeats: int = 4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = init_checkpoint
    best_margin = float("-inf")
    best_path: Path | None = None
    best_observed_score = float("-inf")
    best_observed_path: Path | None = None
    best_observed_verdict = "NOT_RUN"
    no_improve = 0
    history: list[dict[str, Any]] = []
    active_opponents = tuple(opponents)

    for chunk in range(chunks):
        ckpt = out_dir / f"chunk{chunk:02d}.pt"
        players = int(num_players if num_players is not None else (4 if training_track == "phase5_4p" else 2))
        cfg_kwargs = dict(
            seed=seed + chunk,
            policy_track=training_track,
            num_players=players,
            opponents=active_opponents,
            total_timesteps=chunk_timesteps,
            rollout_steps=rollout_steps,
            rollout_num_envs=rollout_num_envs,
            ent_coef=ent_coef,
            device=device,
            checkpoint_in=prev,
            checkpoint_out=str(ckpt),
        )
        cfg = (
            build_phase5_4p_config(**cfg_kwargs)
            if training_track == "phase5_4p"
            else Phase0TrainingConfig(**cfg_kwargs)
        )
        train_summary = train_phase5_4p(cfg) if training_track == "phase5_4p" else train_phase0(cfg)
        include_eval_4p = training_track == "phase5_4p"
        summary = _margin(
            ckpt,
            opponents=eval_opponents,
            seeds=eval_seeds,
            episode_steps=eval_episode_steps,
            include_4p=include_eval_4p,
        )
        margin = float(summary["mean_score_margin"])
        gate_report_path: Path | None = None
        gate_verdict = "NOT_RUN"
        gate_score = margin
        next_opponents = active_opponents
        if strict_drl_gate:
            gate_dir = out_dir / "drl_gate" / f"chunk{chunk:02d}"
            gate_report = run_drl_promotion_gate(
                checkpoint_patterns=[str(ckpt)],
                league_candidates=[],
                out_dir=gate_dir,
                profile=drl_profile,
                seeds=drl_seeds,
                seed_base=200_000 + 10_000 * chunk + seed,
                steps=drl_steps,
                jobs=drl_jobs,
                match_chunk_size=0,
                skip_run=False,
                required_2p_threshold=0.50,
                min_decisive_2p=None,
                min_producer_winrate=0.50,
                min_incumbent_winrate=0.50,
                min_floor_winrate=0.60,
                max_annihilation_rate_4p=0.35,
                weight_2p=0.46,
            )
            gate_report_path = gate_dir / "report.json"
            gate_report_path.parent.mkdir(parents=True, exist_ok=True)
            gate_report_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True), encoding="utf-8")
            gate_row = next(
                row for row in gate_report["ranking"] if row["candidate"] in gate_report["prepared_candidates"]
            )
            gate_verdict = str(gate_row["verdict"])
            gate_score = float(gate_row["overall_score"])
            if pfsp:
                candidate_info = gate_report["candidates"].get(gate_row["candidate"], {})
                pairwise = candidate_info.get("pairwise", {})
                if isinstance(pairwise, dict):
                    next_opponents = _pfsp_reweighted_opponents(
                        active_opponents,
                        pairwise,
                        band=(pfsp_min_winrate, pfsp_max_winrate),
                        max_repeats=pfsp_max_repeats,
                    )
        record = {
            "chunk": chunk,
            "cumulative_timesteps": (chunk + 1) * chunk_timesteps,
            "training_opponents": list(active_opponents),
            "margin": margin,
            "gate_score": gate_score,
            "gate_verdict": gate_verdict,
            "gate_report": str(gate_report_path) if gate_report_path is not None else None,
            "pfsp_enabled": bool(pfsp),
            "next_training_opponents": list(next_opponents),
            "eval_include_4p": bool(include_eval_4p),
            "eval_games": float(summary.get("games", 0.0)),
            "win_rate": float(summary["win_rate"]),
            "invalid_action_rate": float(summary["invalid_action_rate"]),
            "explained_variance": float(train_summary.get("last_explained_variance", 0.0)),
            "entropy": float(train_summary.get("last_entropy", 0.0)),
            "checkpoint": str(ckpt),
        }
        history.append(record)

        score_for_selection = gate_score if strict_drl_gate else margin
        improved_observed = score_for_selection > best_observed_score
        if improved_observed:
            best_observed_score = score_for_selection
            best_observed_path = ckpt
            best_observed_verdict = gate_verdict
            no_improve = 0
        else:
            no_improve += 1

        selectable = not strict_drl_gate or gate_verdict == "PASS_LOCAL"
        if selectable and (best_path is None or score_for_selection > best_margin):
            best_margin = score_for_selection
            best_path = ckpt

        print(json.dumps(record))
        if no_improve >= patience:
            if strict_drl_gate and best_path is None:
                print(
                    "early stop: strict DRL gate score did not improve for "
                    f"{patience} chunks (best_observed={best_observed_score:.4f})"
                )
            else:
                best_display = best_margin if best_path is not None else best_observed_score
                print(f"early stop: margin did not improve for {patience} chunks (best={best_display:.4f})")
            break
        prev = str(ckpt)
        active_opponents = next_opponents

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
        "strict_drl_gate": strict_drl_gate,
        "pfsp": bool(pfsp),
        "pfsp_min_winrate": float(pfsp_min_winrate),
        "pfsp_max_winrate": float(pfsp_max_winrate),
        "pfsp_max_repeats": int(pfsp_max_repeats),
        "training_track": training_track,
        "num_players": num_players,
        "drl_profile": drl_profile,
        "drl_seeds": drl_seeds,
        "drl_steps": drl_steps,
        "drl_jobs": drl_jobs,
        "best_observed_score": best_observed_score if best_observed_path is not None else None,
        "best_observed_checkpoint": str(best_observed_path) if best_observed_path is not None else None,
        "best_observed_verdict": best_observed_verdict if best_observed_path is not None else None,
        "best_margin": best_margin if best_path is not None else None,
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
    parser.add_argument("--training-track", choices=("phase0_2p", "phase5_4p"), default="phase0_2p")
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument(
        "--strict-drl-gate",
        action="store_true",
        help="run the PPO/BReP 2p+4p promotion gate after each chunk; best.pt is written only for PASS_LOCAL chunks",
    )
    parser.add_argument("--drl-profile", choices=("quick", "standard", "strong"), default="quick")
    parser.add_argument("--drl-seeds", type=int, default=None)
    parser.add_argument("--drl-steps", type=int, default=None)
    parser.add_argument("--drl-jobs", type=int, default=1)
    parser.add_argument(
        "--pfsp",
        action="store_true",
        help="after each strict gate, reweight the next chunk's opponent pool toward 35-65% pairwise winrates",
    )
    parser.add_argument("--pfsp-min-winrate", type=float, default=0.35)
    parser.add_argument("--pfsp-max-winrate", type=float, default=0.65)
    parser.add_argument("--pfsp-max-repeats", type=int, default=4)
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
        training_track=str(args.training_track),
        num_players=args.num_players,
        strict_drl_gate=bool(args.strict_drl_gate),
        drl_profile=str(args.drl_profile),
        drl_seeds=args.drl_seeds,
        drl_steps=args.drl_steps,
        drl_jobs=int(args.drl_jobs),
        pfsp=bool(args.pfsp),
        pfsp_min_winrate=float(args.pfsp_min_winrate),
        pfsp_max_winrate=float(args.pfsp_max_winrate),
        pfsp_max_repeats=int(args.pfsp_max_repeats),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
