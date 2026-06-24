"""Smoke tests for PGS v3 league-only variants."""

from __future__ import annotations


def test_pgs_v3_league_variants_are_registered_and_constructible() -> None:
    from scripts.league_agents import FACTORIES

    expected = {
        "pgs_v3_profile_only",
        "pgs_v3_adaptive_arbiter",
        "pgs_v3_adaptive_defense",
        "pgs_v3_adaptive_full2p",
    }

    assert expected <= set(FACTORIES)
    for name in expected:
        agent = FACTORIES[name]()
        assert callable(agent)
