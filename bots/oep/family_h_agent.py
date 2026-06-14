"""Submittable wrapper for a single Family-H generator (régua benchmarking).

Selects the family by the ``FAMILY_H_NAME`` env var so the canonical
``scripts.benchmark_submission`` régua can score any Family-H generator as a
full ``agent(obs)`` bot from game start, vs every existing opponent. This is the
honest "compare against all that exist" measurement — distinct from the mid-game
candidate_action oracle.
"""

from __future__ import annotations

import os
from typing import Any

from bots.oep.candidate_factory import _FAMILY_BUILDERS

_DEFAULT = "hyperheuristic"


def _select() -> str:
    name = os.environ.get("FAMILY_H_NAME", _DEFAULT)
    if name not in _FAMILY_BUILDERS:
        raise KeyError(f"unknown FAMILY_H_NAME={name!r}; known: {sorted(_FAMILY_BUILDERS)}")
    return name


_GENERATOR = _FAMILY_BUILDERS[_select()]()


def agent(obs: dict[str, Any]):
    return _GENERATOR(obs)


def make_agent():
    """Isolated instance (own generator) for batched/vectorized rollouts."""

    generator = _FAMILY_BUILDERS[_select()]()
    return lambda obs: generator(obs)
