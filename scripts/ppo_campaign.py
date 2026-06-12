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
import math
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


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return float(start + (end - start) * progress)


def _mixed_stage_timesteps(total: int, two_player_fraction: float) -> tuple[int, int]:
    total = int(total)
    if total < 2:
        raise ValueError("mixed_2p4p requires chunk_timesteps >= 2")
    frac = min(max(float(two_player_fraction), 0.05), 0.95)
    two_player = int(round(total * frac))
    two_player = min(max(1, two_player), total - 1)
    return two_player, total - two_player


def _training_stages(
    training_track: str,
    *,
    chunk: int,
    chunk_timesteps: int,
    mixed_2p_fraction: float,
    final_checkpoint: Path,
) -> list[dict[str, Any]]:
    if training_track == "phase0_2p":
        return [
            {
                "track": "phase0_2p",
                "num_players": 2,
                "timesteps": int(chunk_timesteps),
                "checkpoint": final_checkpoint,
            }
        ]
    if training_track == "phase5_4p":
        return [
            {
                "track": "phase5_4p",
                "num_players": 4,
                "timesteps": int(chunk_timesteps),
                "checkpoint": final_checkpoint,
            }
        ]
    if training_track == "mixed_2p4p":
        steps_2p, steps_4p = _mixed_stage_timesteps(chunk_timesteps, mixed_2p_fraction)
        return [
            {
                "track": "phase0_2p",
                "num_players": 2,
                "timesteps": steps_2p,
                "checkpoint": final_checkpoint.with_name(f"chunk{chunk:02d}_2p.pt"),
            },
            {
                "track": "phase5_4p",
                "num_players": 4,
                "timesteps": steps_4p,
                "checkpoint": final_checkpoint,
            },
        ]
    raise ValueError(f"unknown training_track: {training_track}")


def _build_training_config(track: str, cfg_kwargs: dict[str, Any]) -> Phase0TrainingConfig:
    if track == "phase5_4p":
        return build_phase5_4p_config(**cfg_kwargs)
    if track == "phase0_2p":
        return Phase0TrainingConfig(**cfg_kwargs)
    raise ValueError(f"unknown training stage track: {track}")


def _train_stage(track: str, cfg: Phase0TrainingConfig) -> dict[str, Any]:
    return train_phase5_4p(cfg) if track == "phase5_4p" else train_phase0(cfg)


def _combined_train_summary(stage_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not stage_records:
        return {}
    summaries = [record["summary"] for record in stage_records]
    final = dict(summaries[-1])
    observed = sum(float(summary.get("episodes_observed", 0.0)) for summary in summaries)
    completed = sum(float(summary.get("completed_episodes", 0.0)) for summary in summaries)
    final["episodes_observed"] = observed
    final["completed_episodes"] = completed
    if observed > 0.0:
        final["mean_return"] = (
            sum(
                float(summary.get("mean_return", 0.0))
                * float(summary.get("episodes_observed", 0.0))
                for summary in summaries
            )
            / observed
        )
        final["mean_early_survival_rate"] = (
            sum(
                float(summary.get("mean_early_survival_rate", 0.0))
                * float(summary.get("episodes_observed", 0.0))
                for summary in summaries
            )
            / observed
        )
    final["learner_seat_rotation"] = any(
        bool(summary.get("learner_seat_rotation", False)) for summary in summaries
    )
    final["training_stages"] = stage_records
    return final


def _margin(
    checkpoint: Path,
    *,
    opponents: list[str],
    seeds: list[int],
    episode_steps: int,
    include_4p: bool = False,
    jobs: int | None = None,
) -> dict[str, Any]:
    eval_jobs = min(8, len(seeds)) if jobs is None else max(1, int(jobs))
    report = benchmark_exported_checkpoint(
        checkpoint,
        submission_out=checkpoint.with_suffix(".sub.py"),
        opponents=opponents,
        seeds=seeds,
        episode_steps=episode_steps,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=bool(include_4p),
        jobs=eval_jobs,
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
    eval_jobs: int | None = None,
    train_episode_steps: int = 500,
    learning_rate: float = 2.5e-4,
    clip_coef: float = 0.2,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    device: str = "cpu",
    rollout_num_envs: int = 1,
    training_track: str = "phase0_2p",
    num_players: int | None = None,
    mixed_2p_fraction: float = 0.5,
    strict_drl_gate: bool = False,
    drl_profile: str = "quick",
    drl_seeds: int | None = None,
    drl_steps: int | None = None,
    drl_jobs: int = 1,
    pfsp: bool = False,
    pfsp_min_winrate: float = 0.35,
    pfsp_max_winrate: float = 0.65,
    pfsp_max_repeats: int = 4,
    kl_to_ref_coef: float = 0.0,
    ref_checkpoint: str | None = None,
    require_terminal_reward: bool = False,
    bc_anchor_coef: float = 0.0,
    bc_anchor_coef_end: float | None = None,
    bc_anchor_teacher: str | None = None,
    bc_anchor_max_quant_error: float = float("inf"),
    decoder_max_moves_per_turn: int | None = None,
    decoder_min_ships_to_launch: int | None = None,
    decoder_reserve_home_ships: int | None = None,
    inherit_checkpoint_decoder: bool = True,
    min_decoder_max_moves_per_turn: int = 1,
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
    anchor_start = float(bc_anchor_coef)
    anchor_final = float(bc_anchor_coef if bc_anchor_coef_end is None else bc_anchor_coef_end)
    resolved_eval_jobs = min(8, len(eval_seeds)) if eval_jobs is None else max(1, int(eval_jobs))

    for chunk in range(chunks):
        ckpt = out_dir / f"chunk{chunk:02d}.pt"
        stages = _training_stages(
            training_track,
            chunk=chunk,
            chunk_timesteps=chunk_timesteps,
            mixed_2p_fraction=mixed_2p_fraction,
            final_checkpoint=ckpt,
        )
        stage_records: list[dict[str, Any]] = []
        stage_prev = prev
        elapsed_chunk_steps = 0
        for stage_idx, stage in enumerate(stages):
            stage_start_progress = (chunk * chunk_timesteps + elapsed_chunk_steps) / max(
                float(chunks * chunk_timesteps), 1.0
            )
            stage_end_progress = (
                chunk * chunk_timesteps + elapsed_chunk_steps + int(stage["timesteps"])
            ) / max(float(chunks * chunk_timesteps), 1.0)
            stage_bc_anchor_coef = _linear_schedule(
                anchor_start, anchor_final, stage_start_progress
            )
            stage_bc_anchor_coef_end = (
                _linear_schedule(anchor_start, anchor_final, stage_end_progress)
                if bc_anchor_coef_end is not None
                else None
            )
            cfg_kwargs = dict(
                seed=seed + chunk * 100 + stage_idx,
                policy_track=stage["track"],
                num_players=int(stage["num_players"]),
                opponents=active_opponents,
                total_timesteps=int(stage["timesteps"]),
                episode_steps=int(train_episode_steps),
                rollout_steps=rollout_steps,
                rollout_num_envs=rollout_num_envs,
                ent_coef=ent_coef,
                learning_rate=float(learning_rate),
                clip_coef=float(clip_coef),
                update_epochs=int(update_epochs),
                minibatch_size=int(minibatch_size),
                device=device,
                checkpoint_in=stage_prev,
                checkpoint_out=str(stage["checkpoint"]),
                kl_to_ref_coef=float(kl_to_ref_coef),
                ref_checkpoint=(ref_checkpoint or init_checkpoint)
                if float(kl_to_ref_coef) > 0.0
                else None,
                bc_anchor_coef=stage_bc_anchor_coef,
                bc_anchor_coef_end=stage_bc_anchor_coef_end,
                bc_anchor_teacher=bc_anchor_teacher,
                bc_anchor_max_quant_error=float(bc_anchor_max_quant_error),
                inherit_checkpoint_decoder=bool(inherit_checkpoint_decoder),
            )
            if decoder_max_moves_per_turn is not None:
                cfg_kwargs["decoder_max_moves_per_turn"] = int(decoder_max_moves_per_turn)
            if decoder_min_ships_to_launch is not None:
                cfg_kwargs["decoder_min_ships_to_launch"] = int(decoder_min_ships_to_launch)
            if decoder_reserve_home_ships is not None:
                cfg_kwargs["decoder_reserve_home_ships"] = int(decoder_reserve_home_ships)
            cfg = _build_training_config(stage["track"], cfg_kwargs)
            stage_summary = _train_stage(stage["track"], cfg)
            stage_records.append(
                {
                    "track": stage["track"],
                    "num_players": int(stage["num_players"]),
                    "timesteps": int(stage["timesteps"]),
                    "checkpoint": str(stage["checkpoint"]),
                    "bc_anchor_start_coef": stage_bc_anchor_coef,
                    "bc_anchor_end_coef": (
                        stage_bc_anchor_coef
                        if stage_bc_anchor_coef_end is None
                        else stage_bc_anchor_coef_end
                    ),
                    "episodes_observed": float(stage_summary.get("episodes_observed", 0.0)),
                    "completed_episodes": float(stage_summary.get("completed_episodes", 0.0)),
                    "learner_seat_rotation": bool(
                        stage_summary.get("learner_seat_rotation", False)
                    ),
                    "summary": stage_summary,
                }
            )
            stage_prev = str(stage["checkpoint"])
            elapsed_chunk_steps += int(stage["timesteps"])
        train_summary = _combined_train_summary(stage_records)
        episodes_observed = float(train_summary.get("episodes_observed", 0.0))
        completed_episodes = float(train_summary.get("completed_episodes", 0.0))
        terminal_reward_observed = completed_episodes > 0.0
        decoder_summary = (
            train_summary.get("decoder") if isinstance(train_summary.get("decoder"), dict) else {}
        )
        effective_decoder_max_moves = int(
            decoder_summary.get("max_moves_per_turn", getattr(cfg, "decoder_max_moves_per_turn", 1))
        )
        decoder_capacity_floor = max(1, int(min_decoder_max_moves_per_turn))
        decoder_capacity_ok = effective_decoder_max_moves >= decoder_capacity_floor
        terminal_signal_ok = terminal_reward_observed or not require_terminal_reward
        train_signal_ok = terminal_signal_ok and decoder_capacity_ok
        if not terminal_signal_ok:
            train_signal_verdict = "REJECT_NO_TERMINAL_REWARD"
        elif not decoder_capacity_ok:
            train_signal_verdict = "REJECT_DECODER_CAPACITY"
        else:
            train_signal_verdict = "PASS_TRAIN_SIGNAL"
        include_eval_4p = training_track in {"phase5_4p", "mixed_2p4p"}
        summary = _margin(
            ckpt,
            opponents=eval_opponents,
            seeds=eval_seeds,
            episode_steps=eval_episode_steps,
            include_4p=include_eval_4p,
            jobs=resolved_eval_jobs,
        )
        margin = float(summary["mean_score_margin"])
        gate_report_path: Path | None = None
        gate_verdict = "NOT_RUN"
        gate_score = margin
        gate_score_2p: float | None = None
        gate_score_4p: float | None = None
        gate_pairwise: dict[str, Any] = {}
        next_opponents = active_opponents
        if strict_drl_gate and not train_signal_ok:
            gate_verdict = "SKIP_NO_TERMINAL_REWARD"
        elif strict_drl_gate:
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
            gate_report_path.write_text(
                json.dumps(gate_report, indent=2, sort_keys=True), encoding="utf-8"
            )
            gate_row = next(
                row
                for row in gate_report["ranking"]
                if row["candidate"] in gate_report["prepared_candidates"]
            )
            gate_verdict = str(gate_row["verdict"])
            gate_score = float(gate_row["overall_score"])
            gate_score_2p = (
                float(gate_row["score_2p"]) if gate_row.get("score_2p") is not None else None
            )
            gate_score_4p = (
                float(gate_row["score_4p"]) if gate_row.get("score_4p") is not None else None
            )
            candidates_info = gate_report.get("candidates", {})
            candidate_info = (
                candidates_info.get(gate_row["candidate"], {})
                if isinstance(candidates_info, dict)
                else {}
            )
            pairwise = candidate_info.get("pairwise", {})
            if isinstance(pairwise, dict):
                gate_pairwise = {
                    str(name): {
                        "win_rate": float(metrics.get("win_rate", 0.0)),
                        "decisive_win_rate": (
                            float(metrics["decisive_win_rate"])
                            if metrics.get("decisive_win_rate") is not None
                            else None
                        ),
                        "nonloss_rate": float(metrics.get("nonloss_rate", 0.0)),
                        "faults": dict(metrics.get("faults", {})),
                    }
                    for name, metrics in sorted(pairwise.items())
                    if isinstance(metrics, dict)
                }
            if pfsp:
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
            "train_episode_steps": int(train_episode_steps),
            "training_opponents": list(active_opponents),
            "margin": margin,
            "gate_score": gate_score,
            "gate_score_2p": gate_score_2p,
            "gate_score_4p": gate_score_4p,
            "gate_pairwise": gate_pairwise,
            "gate_verdict": gate_verdict,
            "gate_report": str(gate_report_path) if gate_report_path is not None else None,
            "pfsp_enabled": bool(pfsp),
            "next_training_opponents": list(next_opponents),
            "training_stages": [
                {key: value for key, value in stage.items() if key != "summary"}
                for stage in stage_records
            ],
            "eval_include_4p": bool(include_eval_4p),
            "eval_games": float(summary.get("games", 0.0)),
            "win_rate": float(summary["win_rate"]),
            "invalid_action_rate": float(summary["invalid_action_rate"]),
            "episodes_observed": episodes_observed,
            "completed_episodes": completed_episodes,
            "terminal_reward_observed": terminal_reward_observed,
            "require_terminal_reward": bool(require_terminal_reward),
            "train_signal_verdict": train_signal_verdict,
            "decoder_max_moves_per_turn": effective_decoder_max_moves,
            "min_decoder_max_moves_per_turn": decoder_capacity_floor,
            "decoder_capacity_ok": decoder_capacity_ok,
            "inherit_checkpoint_decoder": bool(inherit_checkpoint_decoder),
            "learner_seat_rotation": bool(train_summary.get("learner_seat_rotation", False)),
            "mean_train_return": float(train_summary.get("mean_return", 0.0)),
            "mean_early_survival_rate": float(train_summary.get("mean_early_survival_rate", 0.0)),
            "explained_variance": float(train_summary.get("last_explained_variance", 0.0)),
            "entropy": float(train_summary.get("last_entropy", 0.0)),
            "kl_to_ref": float(train_summary.get("last_kl_to_ref", 0.0)),
            "bc_anchor_loss": float(train_summary.get("last_bc_anchor_loss", 0.0)),
            "bc_anchor_chunk_start_coef": float(stage_records[0]["bc_anchor_start_coef"]),
            "bc_anchor_chunk_end_coef": float(stage_records[-1]["bc_anchor_end_coef"]),
            "bc_anchor_examples": float(train_summary.get("last_bc_anchor_examples", 0.0)),
            "bc_anchor_launch_rate": float(train_summary.get("last_bc_anchor_launch_rate", 0.0)),
            "bc_anchor_mean_quant_error": float(
                train_summary.get("last_bc_anchor_mean_quant_error", 0.0)
            ),
            "bc_anchor_effective_coef": float(
                train_summary.get("last_bc_anchor_effective_coef", bc_anchor_coef)
            ),
            "learning_rate": float(learning_rate),
            "clip_coef": float(clip_coef),
            "update_epochs": float(update_epochs),
            "minibatch_size": float(minibatch_size),
            "checkpoint": str(ckpt),
        }
        history.append(record)

        score_for_selection = gate_score if strict_drl_gate else margin
        improved_observed = train_signal_ok and score_for_selection > best_observed_score
        if improved_observed:
            best_observed_score = score_for_selection
            best_observed_path = ckpt
            best_observed_verdict = gate_verdict
            no_improve = 0
        else:
            no_improve += 1

        selectable = train_signal_ok and (not strict_drl_gate or gate_verdict == "PASS_LOCAL")
        if selectable and (best_path is None or score_for_selection > best_margin):
            best_margin = score_for_selection
            best_path = ckpt

        print(json.dumps(record))
        if no_improve >= patience:
            if strict_drl_gate and best_path is None:
                if best_observed_path is None:
                    print(
                        f"early stop: no train-signal-valid strict DRL chunk improved for {patience} chunks"
                    )
                else:
                    print(
                        "early stop: strict DRL gate score did not improve for "
                        f"{patience} chunks (best_observed={best_observed_score:.4f})"
                    )
            else:
                best_display = best_margin if best_path is not None else best_observed_score
                print(
                    f"early stop: margin did not improve for {patience} chunks (best={best_display:.4f})"
                )
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
        "eval_jobs": int(resolved_eval_jobs),
        "train_episode_steps": int(train_episode_steps),
        "ent_coef": ent_coef,
        "learning_rate": float(learning_rate),
        "clip_coef": float(clip_coef),
        "update_epochs": int(update_epochs),
        "minibatch_size": int(minibatch_size),
        "kl_to_ref_coef": float(kl_to_ref_coef),
        "ref_checkpoint": (ref_checkpoint or init_checkpoint)
        if float(kl_to_ref_coef) > 0.0
        else None,
        "bc_anchor_coef": float(bc_anchor_coef),
        "bc_anchor_coef_end": (
            float(bc_anchor_coef_end) if bc_anchor_coef_end is not None else float(bc_anchor_coef)
        ),
        "bc_anchor_teacher": (
            bc_anchor_teacher
            if max(
                float(bc_anchor_coef),
                float(bc_anchor_coef if bc_anchor_coef_end is None else bc_anchor_coef_end),
            )
            > 0.0
            else None
        ),
        "bc_anchor_max_quant_error": (
            float(bc_anchor_max_quant_error)
            if math.isfinite(float(bc_anchor_max_quant_error))
            else None
        ),
        "decoder_max_moves_per_turn": (
            int(decoder_max_moves_per_turn) if decoder_max_moves_per_turn is not None else None
        ),
        "decoder_min_ships_to_launch": (
            int(decoder_min_ships_to_launch) if decoder_min_ships_to_launch is not None else None
        ),
        "decoder_reserve_home_ships": (
            int(decoder_reserve_home_ships) if decoder_reserve_home_ships is not None else None
        ),
        "inherit_checkpoint_decoder": bool(inherit_checkpoint_decoder),
        "min_decoder_max_moves_per_turn": max(1, int(min_decoder_max_moves_per_turn)),
        "require_terminal_reward": bool(require_terminal_reward),
        "patience": patience,
        "strict_drl_gate": strict_drl_gate,
        "pfsp": bool(pfsp),
        "pfsp_min_winrate": float(pfsp_min_winrate),
        "pfsp_max_winrate": float(pfsp_max_winrate),
        "pfsp_max_repeats": int(pfsp_max_repeats),
        "training_track": training_track,
        "num_players": num_players,
        "mixed_2p_fraction": float(mixed_2p_fraction) if training_track == "mixed_2p4p" else None,
        "learner_seat_rotation": bool(
            any(bool(item.get("learner_seat_rotation", False)) for item in history)
        ),
        "drl_profile": drl_profile,
        "drl_seeds": drl_seeds,
        "drl_steps": drl_steps,
        "drl_jobs": drl_jobs,
        "best_observed_score": best_observed_score if best_observed_path is not None else None,
        "best_observed_checkpoint": str(best_observed_path)
        if best_observed_path is not None
        else None,
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
    parser.add_argument(
        "--train-episode-steps",
        type=int,
        default=500,
        help="training episode horizon; shorter values make PPO observe terminal win/loss during curriculum probes",
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-episode-steps", type=int, default=256)
    parser.add_argument(
        "--eval-jobs",
        type=int,
        default=None,
        help="worker count for the post-chunk export benchmark; use 1 for CUDA campaigns to avoid fork-after-CUDA hangs",
    )
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--kl-to-ref-coef", type=float, default=0.0)
    parser.add_argument(
        "--ref-checkpoint",
        default=None,
        help="reference checkpoint for KL anti-drift; defaults to --init when --kl-to-ref-coef > 0",
    )
    parser.add_argument("--bc-anchor-coef", type=float, default=0.0)
    parser.add_argument("--bc-anchor-coef-end", type=float, default=None)
    parser.add_argument("--bc-anchor-teacher", default=None)
    parser.add_argument("--bc-anchor-max-quant-error", type=float, default=float("inf"))
    parser.add_argument(
        "--decoder-max-moves-per-turn",
        type=int,
        default=None,
        help="override decoder max moves for the campaign; by default the checkpoint decoder may be inherited",
    )
    parser.add_argument("--decoder-min-ships-to-launch", type=int, default=None)
    parser.add_argument("--decoder-reserve-home-ships", type=int, default=None)
    parser.add_argument(
        "--no-inherit-checkpoint-decoder",
        action="store_false",
        dest="inherit_checkpoint_decoder",
        help="do not inherit decoder settings from --init/previous chunk; use explicit/default campaign decoder settings",
    )
    parser.add_argument(
        "--min-decoder-max-moves-per-turn",
        type=int,
        default=1,
        help="reject chunk selection when the effective decoder max_moves_per_turn is below this floor",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--require-terminal-reward",
        action="store_true",
        help="skip selection/gate for chunks that did not complete an episode and therefore saw no terminal win/loss",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-num-envs", type=int, default=1)
    parser.add_argument(
        "--training-track", choices=("phase0_2p", "phase5_4p", "mixed_2p4p"), default="phase0_2p"
    )
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument(
        "--mixed-2p-fraction",
        type=float,
        default=0.5,
        help="fraction of each mixed_2p4p chunk spent on 2p before the 4p substage",
    )
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
        help="after each strict gate, reweight the next chunk's opponent pool toward 35-65%% pairwise winrates",
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
        train_episode_steps=int(args.train_episode_steps),
        rollout_steps=args.rollout_steps,
        eval_seeds=list(range(args.eval_seeds)),
        eval_episode_steps=args.eval_episode_steps,
        eval_jobs=args.eval_jobs,
        ent_coef=args.ent_coef,
        learning_rate=float(args.learning_rate),
        clip_coef=float(args.clip_coef),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        kl_to_ref_coef=float(args.kl_to_ref_coef),
        ref_checkpoint=args.ref_checkpoint,
        bc_anchor_coef=float(args.bc_anchor_coef),
        bc_anchor_coef_end=args.bc_anchor_coef_end,
        bc_anchor_teacher=args.bc_anchor_teacher,
        bc_anchor_max_quant_error=float(args.bc_anchor_max_quant_error),
        decoder_max_moves_per_turn=args.decoder_max_moves_per_turn,
        decoder_min_ships_to_launch=args.decoder_min_ships_to_launch,
        decoder_reserve_home_ships=args.decoder_reserve_home_ships,
        inherit_checkpoint_decoder=bool(args.inherit_checkpoint_decoder),
        min_decoder_max_moves_per_turn=int(args.min_decoder_max_moves_per_turn),
        require_terminal_reward=bool(args.require_terminal_reward),
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        rollout_num_envs=args.rollout_num_envs,
        training_track=str(args.training_track),
        num_players=args.num_players,
        mixed_2p_fraction=float(args.mixed_2p_fraction),
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
