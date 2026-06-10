from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

Policy = Callable[[dict[str, Any], int], list[list[float]]]

HEURISTIC_NAMES: tuple[str, ...] = (
    "producer",
    "oep",
    "greedy",
    "defensive",
    "rush",
    "anti_meta",
    "weak_random",
)

# Opponents backed by a module-level singleton runtime whose per-game memory is
# reset only when the observation carries ``step == 0``. They are deterministic
# and isolated across *sequential* games (one game finishes before the next
# starts), but interleaving two live games through the same callable would let
# one game's plan memory leak into the other. Callers that fan out across
# concurrent games (e.g. batched rollout) must give each game its own runtime.
STATEFUL_SINGLETON_OPPONENTS: frozenset[str] = frozenset({"producer", "oep", "pgs"})

_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_SETUP_COMMAND = "rtk .venv/bin/python -m scripts.prepare_producer_opponent"
PRODUCER_AGENT_PATH = _ROOT / "bots/producer/agent.py"
_PRODUCER_MODULE: Any = None
_OEP_AGENT: Callable[[dict[str, Any]], list[list[float]]] | None = None


def _load_producer_module() -> Any:
    global _PRODUCER_MODULE
    if _PRODUCER_MODULE is not None:
        return _PRODUCER_MODULE
    if not PRODUCER_AGENT_PATH.exists():
        raise RuntimeError(
            f"Producer bot is missing. Restore bots/producer or run "
            f"`{PRODUCER_SETUP_COMMAND}`."
        )
    spec = importlib.util.spec_from_file_location("producer_agent", PRODUCER_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Producer opponent from {PRODUCER_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise RuntimeError(f"Producer bot at {PRODUCER_AGENT_PATH} does not define agent(obs)")
    _PRODUCER_MODULE = module
    return _PRODUCER_MODULE


def _load_producer_agent() -> Callable[[dict[str, Any]], list[list[float]]]:
    return _load_producer_module().agent


def producer_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    from python.orbit_wars_gym.observation import to_official_observation

    agent = _load_producer_agent()
    moves = agent(to_official_observation(state, player=player))
    return list(moves) if isinstance(moves, list) else []


def _load_oep_agent() -> Callable[[dict[str, Any]], list[list[float]]]:
    # Lazy import so merely importing the registry does not pull in the OEP
    # planner (and torch) when only the lightweight heuristics are needed.
    global _OEP_AGENT
    if _OEP_AGENT is not None:
        return _OEP_AGENT
    from bots.oep.agent import agent

    _OEP_AGENT = agent
    return _OEP_AGENT


def oep_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    from python.orbit_wars_gym.observation import to_official_observation

    agent = _load_oep_agent()
    moves = agent(to_official_observation(state, player=player))
    return list(moves) if isinstance(moves, list) else []


def pgs_agent(state: dict[str, Any], player: int) -> list[list[float]]:
    from bots.pgs.planner import agent as _pgs
    from python.orbit_wars_gym.observation import to_official_observation

    moves = _pgs(to_official_observation(state, player=player))
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
        "oep": oep_agent,
        "pgs": pgs_agent,
        "greedy": greedy_agent,
        "defensive": defensive_agent,
        "rush": rush_agent,
        "anti_meta": anti_meta_agent,
        "weak_random": weak_random_agent,
    }


def _make_isolated_policy(name: str) -> Policy:
    """Build ONE fresh opponent instance with its own per-game memory.

    Producer/OEP keep per-game memory in a module-level singleton, so concurrent
    games would cross-contaminate through the shared callable. Each isolated
    instance has its own runtime (and, for OEP, its own private Producer
    seed/opponent policies). Stateless heuristics need no isolation.
    """
    from python.orbit_wars_gym.observation import to_official_observation

    if name == "producer":
        bot = _load_producer_module().make_agent()
    elif name == "oep":
        from bots.oep.agent import make_agent

        bot = make_agent()
    elif name == "pgs":
        from bots.pgs.agent import make_agent

        bot = make_agent()
    else:
        raise ValueError(f"{name!r} is stateless; use get_heuristic_policies()[name] directly")

    def _policy(state: dict[str, Any], player: int) -> list[list[float]]:
        moves = bot(to_official_observation(state, player=player))
        return list(moves) if isinstance(moves, list) else []

    return _policy


_ISOLATED_POOL: dict[str, list[Policy]] = {}


def get_isolated_opponents(name: str, count: int) -> list[Policy]:
    """Return ``count`` opponent callables safe for concurrent (batched) games.

    For stateful singleton opponents (producer/oep) this returns ``count`` cached
    independent instances (each resets its own memory on ``step == 0``), so a
    vectorized rollout can give one per env without cross-contamination. For
    stateless heuristics the same shared callable is reused (it carries no state).
    The pool is cached and grown on demand, so building the (one-time) instances
    is not repeated across rollout segments.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if name not in STATEFUL_SINGLETON_OPPONENTS:
        shared = get_heuristic_policies()[name]
        return [shared] * count
    pool = _ISOLATED_POOL.setdefault(name, [])
    while len(pool) < count:
        pool.append(_make_isolated_policy(name))
    return pool[:count]
