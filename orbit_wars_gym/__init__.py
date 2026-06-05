"""Import shim — NOT an implementation.

This package only re-exports the real gym implementation that lives in
``python/orbit_wars_gym``. It exists so that code run from the repo root
(scripts, bots, agents) can write ``import orbit_wars_gym`` without putting
``python/`` on PYTHONPATH. There is no logic here; do not add any.

Roles, so the two ``orbit_wars_gym`` names aren't confusing:
  - ``python/orbit_wars_gym``  = the real gym, Rust-backed, TRAINING/eval only.
  - ``orbit_wars_gym`` (this)  = thin compatibility shim re-exporting it.
  - ``orbit_lite``             = the pure-Python engine that is actually SUBMITTED.

See docs/ARCHITECTURE.md and DECISIONS.md (D10/D11) for the full boundary.
"""

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
