from __future__ import annotations

import copy
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Member:
    id: str
    kind: str = "ppo"
    fitness: float = 0.0
    rating: float = 1000.0
    hparams: dict[str, Any] = field(default_factory=dict)
    decoder: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: str | None = None
    parent_id: str | None = None


def exploit_explore(
    population: list[Member],
    elite_frac: float = 0.25,
    seed: int = 0,
    *,
    mutate_hparams_enabled: bool = True,
    mutate_decoder_enabled: bool = True,
) -> list[Member]:
    rng = random.Random(seed)
    ranked = sorted(population, key=lambda m: m.fitness, reverse=True)
    elite_n = max(1, int(len(ranked) * elite_frac))
    elites = ranked[:elite_n]
    new_pop = copy.deepcopy(elites)
    while len(new_pop) < len(population):
        parent = copy.deepcopy(rng.choice(elites))
        parent.parent_id = parent.id
        parent.id = f"{parent.id}_mut_{len(new_pop):03d}"
        if mutate_hparams_enabled:
            parent.hparams = mutate_hparams(parent.hparams, rng)
        if mutate_decoder_enabled:
            parent.decoder = mutate_decoder(parent.decoder, rng)
        parent.checkpoint = _clone_checkpoint(parent.checkpoint, parent.id)
        new_pop.append(parent)
    return new_pop


def _clone_checkpoint(checkpoint: str | None, member_id: str) -> str | None:
    if checkpoint is None:
        return None
    source = Path(checkpoint)
    suffix = "".join(source.suffixes) or ".pt"
    target = source.with_name(f"{member_id}{suffix}")
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return str(target)


def mutate_hparams(h: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out = dict(h)
    for key in ["learning_rate", "entropy_coef", "clip_coef", "gamma", "gae_lambda"]:
        if key not in out:
            continue
        factor = rng.choice([0.5, 0.8, 1.2, 1.5])
        out[key] *= factor
    return out


def mutate_decoder(decoder: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out = copy.deepcopy(decoder)
    if "reserve_home_ships" in out:
        out["reserve_home_ships"] = max(2, int(out["reserve_home_ships"] + rng.choice([-2, -1, 1, 2])))
    if "min_ships_to_launch" in out:
        out["min_ships_to_launch"] = max(1, int(out["min_ships_to_launch"] + rng.choice([-1, 1])))
    if "max_moves_per_turn" in out:
        out["max_moves_per_turn"] = max(1, int(out["max_moves_per_turn"] + rng.choice([-1, 1])))
    if "fractions" in out:
        factor = rng.choice([0.9, 1.1])
        out["fractions"] = [min(0.95, max(0.05, round(float(value) * factor, 4))) for value in out["fractions"]]
    if "angle_offsets" in out:
        factor = rng.choice([0.85, 1.15])
        out["angle_offsets"] = [round(float(value) * factor, 6) for value in out["angle_offsets"]]
    return out
