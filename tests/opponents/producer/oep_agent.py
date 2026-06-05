from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_FIXTURE_ROOT = Path(__file__).resolve().parent
if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))

from producer_agent import _to_list_observation  # noqa: E402

_ROOT = _FIXTURE_ROOT / "producer"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location("_producer_oep", _ROOT / "oep.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load OEP agent from {_ROOT}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def agent(obs):
    return _MODULE.agent(_to_list_observation(obs))
