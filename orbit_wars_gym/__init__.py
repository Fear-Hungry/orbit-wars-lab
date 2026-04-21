from __future__ import annotations

import sys
from importlib import import_module

_PACKAGE = import_module("python.orbit_wars_gym")

__all__ = getattr(_PACKAGE, "__all__", [])

for name in __all__:
    globals()[name] = getattr(_PACKAGE, name)

for submodule in (
    "action_decoder",
    "backend",
    "encoding",
    "entities",
    "gym_env",
    "observation",
    "parallel_env",
    "parity",
    "snapshots",
):
    sys.modules[f"{__name__}.{submodule}"] = import_module(f"python.orbit_wars_gym.{submodule}")
