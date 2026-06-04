from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

Policy = Callable[[dict[str, Any], int], list[list[float]]]

HEURISTIC_NAMES: tuple[str, ...] = (
    "producer",
    "greedy",
    "defensive",
    "rush",
    "anti_meta",
    "weak_random",
)

_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_SETUP_COMMAND = "rtk .venv/bin/python -m scripts.prepare_producer_opponent"
PRODUCER_AGENT_PATH = _ROOT / "tests/opponents/producer/producer_agent.py"
_PRODUCER_AGENT: Callable[[dict[str, Any]], list[list[float]]] | None = None


def _load_producer_agent() -> Callable[[dict[str, Any]], list[list[float]]]:
    global _PRODUCER_AGENT
    if _PRODUCER_AGENT is not None:
        return _PRODUCER_AGENT
    if not PRODUCER_AGENT_PATH.exists():
        raise RuntimeError(
            f"Producer opponent fixture is missing. Restore tests/opponents/producer or run "
            f"`{PRODUCER_SETUP_COMMAND}`."
        )
    spec = importlib.util.spec_from_file_location("producer_agent", PRODUCER_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Producer opponent from {PRODUCER_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError(f"Producer opponent at {PRODUCER_AGENT_PATH} does not define agent(obs)")
    _PRODUCER_AGENT = agent
    return _PRODUCER_AGENT


def producer_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    from python.orbit_wars_gym.observation import to_official_observation

    agent = _load_producer_agent()
    moves = agent(to_official_observation(state, player=player))
    return list(moves) if isinstance(moves, list) else []


def get_heuristic_policies() -> dict[str, Policy]:
    from .heuristics import (
        anti_meta_agent,
        defensive_agent,
        greedy_agent,
        rush_agent,
        weak_random_agent,
    )

    return {
        "producer": producer_agent,
        "greedy": greedy_agent,
        "defensive": defensive_agent,
        "rush": rush_agent,
        "anti_meta": anti_meta_agent,
        "weak_random": weak_random_agent,
    }
