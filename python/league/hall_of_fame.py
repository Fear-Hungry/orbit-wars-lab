from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python.league.map_elites import Behavior


@dataclass
class HallOfFameEntry:
    member_id: str
    checkpoint: str | None
    fitness: float
    rating: float
    generation: int
    behavior: Any | None = None
    decoder: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class HallOfFame:
    max_size: int = 128
    entries: list[HallOfFameEntry] = field(default_factory=list)

    def add(self, entry: HallOfFameEntry):
        if all(existing.checkpoint != entry.checkpoint for existing in self.entries):
            self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size :]

    def sample(self, n: int) -> list[HallOfFameEntry]:
        ranked = sorted(self.entries, key=lambda entry: (entry.fitness, entry.rating, entry.generation), reverse=True)
        return ranked[:n]


def _behavior_to_payload(behavior: Any | None) -> dict[str, int] | None:
    if behavior is None:
        return None
    if isinstance(behavior, Behavior):
        return {
            "expansion_bin": behavior.expansion_bin,
            "aggression_bin": behavior.aggression_bin,
            "defense_bin": behavior.defense_bin,
            "fleet_size_bin": behavior.fleet_size_bin,
        }
    return None


def _behavior_from_payload(payload: dict[str, Any] | None) -> Behavior | None:
    if payload is None:
        return None
    return Behavior(
        expansion_bin=int(payload["expansion_bin"]),
        aggression_bin=int(payload["aggression_bin"]),
        defense_bin=int(payload["defense_bin"]),
        fleet_size_bin=int(payload["fleet_size_bin"]),
    )


def load_hall_of_fame(path: str | Path, *, max_size: int) -> HallOfFame:
    file_path = Path(path)
    hall_of_fame = HallOfFame(max_size=max_size)
    if not file_path.exists():
        return hall_of_fame
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    for record in payload.get("entries", []):
            hall_of_fame.add(
                HallOfFameEntry(
                    member_id=str(record["member_id"]),
                    checkpoint=record.get("checkpoint"),
                    fitness=float(record["fitness"]),
                    rating=float(record["rating"]),
                    generation=int(record["generation"]),
                    behavior=_behavior_from_payload(record.get("behavior")),
                    decoder=dict(record.get("decoder", {})),
                    metrics=dict(record.get("metrics", {})),
                )
            )
    return hall_of_fame


def save_hall_of_fame(hall_of_fame: HallOfFame, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [
            {
                "member_id": entry.member_id,
                "checkpoint": entry.checkpoint,
                "fitness": entry.fitness,
                "rating": entry.rating,
                "generation": entry.generation,
                "behavior": _behavior_to_payload(entry.behavior),
                "decoder": entry.decoder,
                "metrics": entry.metrics,
            }
            for entry in hall_of_fame.entries
        ]
    }
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
