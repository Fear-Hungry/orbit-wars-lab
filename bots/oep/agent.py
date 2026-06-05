from __future__ import annotations

from typing import Any

from bots.oep.planner import agent as _agent


def agent(obs: dict[str, Any]):
    return _agent(obs)
