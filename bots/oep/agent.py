from __future__ import annotations

from typing import Any

import bots.oep.planner as _planner
from bots.oep.planner import agent as _agent


def agent(obs: dict[str, Any]):
    return _agent(obs)


def notify_fallback_applied() -> None:
    _planner.reset_runtime()


def make_agent():
    """Isolated OEP agent (own runtime/memory) for batched/vectorized rollouts."""
    from bots.oep.planner import make_runtime

    runtime = make_runtime()
    return lambda obs: runtime.act(obs)
