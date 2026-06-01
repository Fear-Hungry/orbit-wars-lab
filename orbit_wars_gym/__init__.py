from __future__ import annotations

from importlib import import_module
from pathlib import Path

_PACKAGE = import_module("python.orbit_wars_gym")
_IMPLEMENTATION_PATH = Path(__file__).resolve().parents[1] / "python" / "orbit_wars_gym"

__all__ = getattr(_PACKAGE, "__all__", [])
__path__.append(str(_IMPLEMENTATION_PATH))


def __getattr__(name):
    if name in __all__:
        return getattr(_PACKAGE, name)
    raise AttributeError(name)
