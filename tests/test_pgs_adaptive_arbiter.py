"""Unit coverage for adaptive PGS reply-model routing."""

from __future__ import annotations


def test_reply_models_follow_observed_profile_role() -> None:
    from bots.pgs.planner import OpponentOnlineStats, PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(adaptive_mode=True, adaptive_reply_models=True))
    runtime._opp_profiles[1] = OpponentOnlineStats(owner_id=1, role="rusher", confidence=0.8)

    assert runtime._reply_models_for_current_enemy(2, 1) == ["producer", "rush", "hold"]

    runtime._opp_profiles[1].role = "expander"
    assert runtime._reply_models_for_current_enemy(2, 1) == ["producer", "expand"]

    runtime._opp_profiles[1].role = "sprayer"
    assert runtime._reply_models_for_current_enemy(2, 1) == ["producer", "hold"]


def test_reply_models_default_outside_two_player_adaptation() -> None:
    from bots.pgs.planner import OpponentOnlineStats, PGSConfig, PGSRuntime

    runtime = PGSRuntime(
        PGSConfig(
            adaptive_mode=True,
            adaptive_reply_models=True,
            reply_models_default="producer,hold",
        )
    )
    runtime._opp_profiles[1] = OpponentOnlineStats(owner_id=1, role="rusher", confidence=0.8)

    assert runtime._reply_models_for_current_enemy(4, 1) == ["producer", "hold"]
