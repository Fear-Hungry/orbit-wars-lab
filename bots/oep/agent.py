from __future__ import annotations

from typing import Any

from bots.oep.planner import agent as _agent


def agent(obs: dict[str, Any]):
    return _agent(obs)


def make_agent():
    """Isolated OEP agent (own runtime/memory) for batched/vectorized rollouts."""
    from bots.oep.planner import make_runtime

    runtime = make_runtime()
    return lambda obs: runtime.act(obs)
