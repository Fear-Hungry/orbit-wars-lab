"""Policy selector for the Mahoraga-PPO hybrid.

The Mahoraga runtime (PGS full2p) runs EVERY turn — its opponent profiler only
updates inside ``act`` — and is always the fallback. The selector hands the turn
to the PPO specialist only in the conservative window: 2-player games, every
opponent classified into a PPO-favourable role with enough confidence. Dangerous
or unresolved matchups (rusher/wave/sprayer/unknown, any 4p game) stay with
Mahoraga, so the hybrid can never do worse than the incumbent in those regimes.

Future specialists plug in as new policy ids without touching the agent wiring.
"""

from __future__ import annotations

from typing import Any

# Roles where handing the turn to the PPO specialist is allowed. Rusher, wave
# and sprayer are exactly the regimes that killed past submissions on the LB —
# those always stay with Mahoaraga's defensive machinery.
PPO_FAVOURABLE_ROLES = frozenset({"expander", "producer-like", "turtle"})

MAHORAGA = "mahoraga"
PPO = "ppo"


def roles_from_runtime(runtime: Any) -> dict[int, tuple[str, float]]:
    """Opponent ``{owner_id: (role, confidence)}`` from a PGSRuntime.

    Prefers the public ``opponent_roles()`` accessor; falls back to the
    profile dict so the hybrid also works against older planner builds.
    """
    getter = getattr(runtime, "opponent_roles", None)
    if callable(getter):
        return {int(k): (str(r), float(c)) for k, (r, c) in dict(getter()).items()}
    profiles = getattr(runtime, "_opp_profiles", None) or {}
    return {
        int(owner): (str(stat.role), float(stat.confidence))
        for owner, stat in profiles.items()
    }


def select_policy(
    *,
    num_players: int,
    roles: dict[int, tuple[str, float]],
    min_confidence: float,
    ppo_ready: bool,
) -> str:
    """Return the policy id ("mahoraga" | "ppo") for this turn."""
    if not ppo_ready:
        return MAHORAGA
    if int(num_players) != 2:
        return MAHORAGA
    if not roles:
        return MAHORAGA
    for role, confidence in roles.values():
        if role not in PPO_FAVOURABLE_ROLES:
            return MAHORAGA
        if float(confidence) < float(min_confidence):
            return MAHORAGA
    return PPO
