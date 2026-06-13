from __future__ import annotations

from typing import Any

from bots.mahoraga_ppo import ppo_policy
from bots.mahoraga_ppo.selector import PPO, roles_from_runtime, select_policy
from bots.pgs.planner import PGSConfig, PGSRuntime

# Mahoraga base = the league's pgs_v3_adaptive_full2p config (adaptive profiles +
# rescue/punish/hammer missions over the operational holdwave floor). Keep in
# sync with scripts/league_agents.py and the imitation collector's
# _MAHORAGA_CONFIG — this is the bot the PPO specialist was distilled from.
MAHORAGA_CONFIG = PGSConfig(
    scripts="hold",
    wave_min_ships=60.0,
    wave_start_step=150,
    floor_in_4p=True,
    adaptive_mode=True,
    adaptive_reply_models=True,
    mission_mode=True,
    enabled_missions="rescue,punish,hammer",
    max_mission_candidates=8,
    max_selected_missions=1,
    hammer_top_targets=3,
    hammer_top_sources=4,
    deadline_ms=450.0,
    deadline_guard_ms=100.0,
    value_mode="scalar",
)


def _player_count(obs: dict[str, Any]) -> int:
    """Match size from the initial planet owners (official rows: [id, owner, ...])."""
    owners = {int(row[1]) for row in obs.get("initial_planets", []) if int(row[1]) >= 0}
    return len(owners) if len(owners) in (2, 4) else 2


class MahoragaPPORuntime:
    """Mahoraga every turn (profiler + warm fallback); PPO only when selected."""

    def __init__(self) -> None:
        self._mahoraga = PGSRuntime(MAHORAGA_CONFIG)
        self._stats = {"calls": 0, "ppo_turns": 0, "mahoraga_turns": 0, "ppo_errors": 0}

    def act(self, obs: dict[str, Any]) -> list[list[float]]:
        self._stats["calls"] += 1
        # Mahoraga MUST act every turn: the opponent profiler only updates inside
        # act(), and its plan is the always-valid fallback for this turn.
        base_moves = self._mahoraga.act(obs)

        choice = select_policy(
            num_players=_player_count(obs),
            roles=roles_from_runtime(self._mahoraga),
            min_confidence=float(MAHORAGA_CONFIG.profile_switch_confidence),
            ppo_ready=ppo_policy.ready(),
        )
        if choice == PPO:
            try:
                moves = ppo_policy.agent(obs)
                if isinstance(moves, list):
                    self._stats["ppo_turns"] += 1
                    return list(moves)
                self._stats["ppo_errors"] += 1
            except Exception:
                self._stats["ppo_errors"] += 1
        self._stats["mahoraga_turns"] += 1
        return base_moves if isinstance(base_moves, list) else []

    def notify_fallback_applied(self) -> None:
        self._mahoraga.notify_fallback_applied()

    def runtime_stats(self) -> dict[str, int]:
        stats = dict(self._mahoraga.runtime_stats())
        stats.update(self._stats)
        return stats

    def selected_policy_name(self, obs: dict[str, Any]) -> str:
        """Selector decision for the CURRENT profiles (diagnostics only)."""
        return select_policy(
            num_players=_player_count(obs),
            roles=roles_from_runtime(self._mahoraga),
            min_confidence=float(MAHORAGA_CONFIG.profile_switch_confidence),
            ppo_ready=ppo_policy.ready(),
        )


_RUNTIME: MahoragaPPORuntime | None = None


def agent(obs: dict[str, Any]):
    global _RUNTIME
    if _RUNTIME is None or (isinstance(obs, dict) and int(obs.get("step", 0)) == 0):
        _RUNTIME = MahoragaPPORuntime()
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
    """Isolated hybrid agent (own runtimes) for batched rollouts/league play."""
    runtime = MahoragaPPORuntime()
    return lambda obs: runtime.act(obs)
