from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Behavior:
    expansion_bin: int
    aggression_bin: int
    defense_bin: int
    fleet_size_bin: int


class MapElitesArchive:
    def __init__(self):
        self.cells: dict[Behavior, tuple[float, Any]] = {}

    def insert(self, behavior: Behavior, fitness: float, payload: Any) -> bool:
        current = self.cells.get(behavior)
        if current is None or fitness > current[0]:
            self.cells[behavior] = (fitness, payload)
            return True
        return False

    def elites(self, limit: int | None = None) -> list[Any]:
        ranked = sorted(self.cells.values(), key=lambda item: item[0], reverse=True)
        payloads = [payload for _, payload in ranked]
        if limit is None:
            return payloads
        return payloads[:limit]

    def cell_count(self) -> int:
        return len(self.cells)


def load_map_elites_archive(path: str | Path) -> MapElitesArchive:
    file_path = Path(path)
    archive = MapElitesArchive()
    if not file_path.exists():
        return archive
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        archive.insert(
            Behavior(
                expansion_bin=int(cell["behavior"]["expansion_bin"]),
                aggression_bin=int(cell["behavior"]["aggression_bin"]),
                defense_bin=int(cell["behavior"]["defense_bin"]),
                fleet_size_bin=int(cell["behavior"]["fleet_size_bin"]),
            ),
            float(cell["fitness"]),
            cell["payload"],
        )
    return archive


def save_map_elites_archive(archive: MapElitesArchive, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {
                "behavior": {
                    "expansion_bin": behavior.expansion_bin,
                    "aggression_bin": behavior.aggression_bin,
                    "defense_bin": behavior.defense_bin,
                    "fleet_size_bin": behavior.fleet_size_bin,
                },
                "fitness": fitness,
                "payload": stored_payload,
            }
            for behavior, (fitness, stored_payload) in archive.cells.items()
        ]
    }
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
