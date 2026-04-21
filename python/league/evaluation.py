from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from python.league.pbt import Member


@dataclass(frozen=True)
class AgentSpec:
    id: str
    kind: str
    role: str
    checkpoint: str | None = None
    policy: str | None = None
    rating: float = 1000.0
    hparams: dict[str, Any] = field(default_factory=dict)
    decoder: dict[str, Any] = field(default_factory=dict)


def default_hparams() -> dict[str, float]:
    return {
        "learning_rate": 2.5e-4,
        "entropy_coef": 0.01,
        "clip_coef": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }


def default_decoder() -> dict[str, Any]:
    return {
        "reserve_home_ships": 8,
        "min_ships_to_launch": 2,
        "max_moves_per_turn": 8,
        "fractions": [0.10, 0.25, 0.50, 0.75],
        "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
    }


def _load_structured_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw)


def _coerce_spec(record: Any, *, default_role: str) -> AgentSpec:
    if isinstance(record, str):
        return AgentSpec(id=record, kind="heuristic", role=default_role, policy=record)

    if not isinstance(record, dict):
        raise TypeError(f"agent spec must be a string or dict, got {type(record)!r}")

    kind = str(record.get("kind", "ppo"))
    spec = AgentSpec(
        id=str(record["id"]),
        kind=kind,
        role=str(record.get("role", default_role)),
        checkpoint=record.get("checkpoint"),
        policy=record.get("policy"),
        rating=float(record.get("rating", 1000.0)),
        hparams=dict(record.get("hparams", default_hparams() if kind == "ppo" else {})),
        decoder=dict(record.get("decoder", default_decoder())),
    )
    if spec.kind == "ppo" and spec.checkpoint is None:
        raise ValueError(f"ppo agent {spec.id!r} requires checkpoint")
    if spec.kind == "heuristic" and not spec.policy:
        raise ValueError(f"heuristic agent {spec.id!r} requires policy")
    return spec


def load_population_manifest(path: str | Path) -> dict[str, list[AgentSpec]]:
    payload = _load_structured_file(path)
    population = [_coerce_spec(record, default_role="league") for record in payload.get("population", [])]
    hall_of_fame = [_coerce_spec(record, default_role="hall_of_fame") for record in payload.get("hall_of_fame", [])]
    heuristics = [_coerce_spec(record, default_role="heuristic") for record in payload.get("heuristics", [])]
    if not population:
        raise ValueError("population manifest requires at least one population member")
    return {
        "population": population,
        "hall_of_fame": hall_of_fame,
        "heuristics": heuristics,
    }


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    payload = _load_structured_file(path)
    members = payload.get("members", [])
    if not members:
        raise ValueError("evaluation report requires non-empty `members`")
    return payload


def compute_fitness(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return (
        float(weights["win_rate"]) * float(metrics.get("win_rate_vs_league", 0.0))
        + float(weights["score_margin"]) * float(metrics.get("normalized_score_margin", 0.0))
        + float(weights["hof_win_rate"]) * float(metrics.get("win_rate_vs_hall_of_fame", 0.0))
        + float(weights["robustness"]) * float(metrics.get("robustness_across_seeds", 0.0))
        + float(weights["novelty"]) * float(metrics.get("novelty_bonus", 0.0))
        - float(weights["crash_penalty"]) * float(metrics.get("crash_rate", 0.0))
        - float(weights["invalid_action_penalty"]) * float(metrics.get("invalid_action_rate", 0.0))
    )


def member_from_evaluation_entry(record: dict[str, Any], fitness_weights: dict[str, float]) -> Member:
    metrics = dict(record.get("metrics", {}))
    return Member(
        id=str(record["id"]),
        kind=str(record.get("kind", "ppo")),
        fitness=compute_fitness(metrics, fitness_weights),
        rating=float(record.get("rating", 1000.0)),
        hparams=dict(record.get("hparams", default_hparams())),
        decoder=dict(record.get("decoder", default_decoder())),
        metrics=metrics,
        checkpoint=record.get("checkpoint"),
    )
