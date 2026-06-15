from __future__ import annotations

from typing import Any

from bots.pgs.planner import PGSConfig, PGSRuntime

# Operational (gate-approved) config — the ONLY config that ships in a
# submission. PGSConfig's dataclass defaults keep all scripts enabled as an
# ablation knob; the 2026-06-09 submission accidentally shipped those defaults
# (rejected offensive scripts included) instead of the gated hold-only.
# hold-only: frozen 96 +0.218 (id=122); +wave w60s150: frozen paired no-regress
# vs Producer/OEP (id=141).
SUBMISSION_CONFIG = PGSConfig(scripts="hold", wave_min_ships=60.0, wave_start_step=150)

_RUNTIME: PGSRuntime | None = None


def agent(obs: dict[str, Any]):
    global _RUNTIME
    if _RUNTIME is None or (isinstance(obs, dict) and int(obs.get("step", 0)) == 0):
        _RUNTIME = PGSRuntime(SUBMISSION_CONFIG)
    return _RUNTIME.act(obs)


def notify_fallback_applied() -> None:
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.notify_fallback_applied()
    _RUNTIME = None


def runtime_stats() -> dict[str, int]:
    if _RUNTIME is None:
        return {}
    return _RUNTIME.runtime_stats()


def make_agent():
    """Isolated PGS agent (own runtime, operational config) for batched rollouts."""
    runtime = PGSRuntime(SUBMISSION_CONFIG)
    return lambda obs: runtime.act(obs)
