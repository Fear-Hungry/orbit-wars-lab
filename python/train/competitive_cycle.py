from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.agents.registry import HEURISTIC_NAMES
from python.league.evaluation import AgentSpec, member_from_evaluation_entry
from python.league.hall_of_fame import HallOfFame
from python.league.map_elites import MapElitesArchive
from python.league.pbt import Member
from python.train.evaluate_population import EvaluationConfig, evaluate_population
from python.train.train_league import LeagueIterationResult, run_league_iteration
from python.train.train_ppo import (
    Phase0TrainingConfig,
    build_phase5_4p_config,
    train_phase0,
    train_phase5_4p,
)


@dataclass(frozen=True)
class CompetitiveCycleConfig:
    iterations: int = 2
    training_track: str = "phase0_2p"
    total_timesteps_per_member: int = 16
    rollout_steps: int = 8
    update_epochs: int = 1
    minibatch_size: int = 8
    device: str = "cpu"


@dataclass
class CompetitiveCycleResult:
    iterations: int
    manifests: list[dict[str, list[AgentSpec]]]
    summaries: list[LeagueIterationResult]


def manifest_from_members(
    population: list[Member],
    hall_of_fame: HallOfFame,
    heuristics: list[str],
) -> dict[str, list[AgentSpec]]:
    return {
        "population": [
            AgentSpec(
                id=member.id,
                kind=member.kind,
                role="league",
                checkpoint=member.checkpoint,
                rating=member.rating,
                hparams=dict(member.hparams),
                decoder=dict(member.decoder),
            )
            for member in population
        ],
        "hall_of_fame": [
            AgentSpec(
                id=entry.member_id,
                kind="ppo" if entry.checkpoint else "heuristic",
                role="hall_of_fame",
                checkpoint=entry.checkpoint,
                rating=entry.rating,
                hparams={},
                decoder=dict(entry.decoder),
            )
            for entry in hall_of_fame.entries
        ],
        "heuristics": [
            AgentSpec(id=name, kind="heuristic", role="heuristic", policy=name)
            for name in heuristics
        ],
    }


def update_trainable_population(
    population: list[Member],
    cycle_cfg: CompetitiveCycleConfig,
    *,
    output_dir: str | Path,
) -> list[Member]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for offset, member in enumerate(population):
        cloned = copy.deepcopy(member)
        if cloned.kind != "ppo":
            updated.append(cloned)
            continue
        checkpoint_path = out_dir / f"{cloned.id}.pt"
        base_kwargs: dict[str, Any] = {
            "seed": offset,
            "total_timesteps": cycle_cfg.total_timesteps_per_member,
            "rollout_steps": cycle_cfg.rollout_steps,
            "update_epochs": cycle_cfg.update_epochs,
            "minibatch_size": cycle_cfg.minibatch_size,
            "checkpoint_out": str(checkpoint_path),
            "device": cycle_cfg.device,
            "learning_rate": float(cloned.hparams.get("learning_rate", 2.5e-4)),
            "gamma": float(cloned.hparams.get("gamma", 0.99)),
            "gae_lambda": float(cloned.hparams.get("gae_lambda", 0.95)),
            "clip_coef": float(cloned.hparams.get("clip_coef", 0.2)),
            "ent_coef": float(cloned.hparams.get("entropy_coef", 0.01)),
        }
        if cycle_cfg.training_track == "phase5_4p":
            train_phase5_4p(build_phase5_4p_config(**base_kwargs))
        else:
            train_phase0(Phase0TrainingConfig(policy_track="phase0_2p", **base_kwargs))
        cloned.checkpoint = str(checkpoint_path)
        updated.append(cloned)
    return updated


def run_competitive_cycle(
    manifest: dict[str, list[AgentSpec]],
    eval_cfg: EvaluationConfig,
    league_cfg: dict[str, Any],
    cycle_cfg: CompetitiveCycleConfig,
    *,
    output_dir: str | Path,
) -> CompetitiveCycleResult:
    hall_of_fame = HallOfFame(max_size=int(league_cfg["population"]["hall_of_fame_size"]))
    archive = MapElitesArchive()
    current_manifest = manifest
    manifests = [current_manifest]
    summaries: list[LeagueIterationResult] = []
    out_dir = Path(output_dir)

    for generation in range(cycle_cfg.iterations):
        report = evaluate_population(current_manifest, eval_cfg)
        evaluated_population = [
            member_from_evaluation_entry(record, league_cfg["fitness"])
            for record in report["members"]
        ]
        trained_population = update_trainable_population(
            evaluated_population,
            cycle_cfg,
            output_dir=out_dir / f"generation_{generation:03d}",
        )
        next_population, hall_of_fame, archive, summary = run_league_iteration(
            trained_population,
            league_cfg,
            generation=generation,
            hall_of_fame=hall_of_fame,
            archive=archive,
        )
        summaries.append(summary)
        current_manifest = manifest_from_members(
            next_population,
            hall_of_fame,
            list(league_cfg.get("heuristics", list(HEURISTIC_NAMES))),
        )
        manifests.append(current_manifest)

    return CompetitiveCycleResult(iterations=cycle_cfg.iterations, manifests=manifests, summaries=summaries)
