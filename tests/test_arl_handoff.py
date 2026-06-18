"""Pin the ARL Mode-3 promotion handoff helpers (scripts/research_loop/policy.py).

The handoff decides WHICH candidates are worth the expensive seat-rotated ruler and
builds the (governed, human-run) command. These are pure functions; the file-drop
registration into FACTORIES and the actual ruler run are exercised elsewhere.
"""
from __future__ import annotations

from scripts.research_loop.policy import (
    build_promotion_command,
    candidate_name,
    select_survivors,
)


def _it(run_id, decision, delta):
    return {"run_id": run_id, "decision": decision, "delta": delta}


def test_survivor_is_inconclusive_or_promoted_beating_parent():
    iters = [
        _it("r-i0", "promoted", 0.5),       # trusted promotion -> survivor
        _it("r-i1", "inconclusive", 0.3),   # downgraded promotion (delta>band) -> survivor
    ]
    surv = select_survivors(iters, noise_band=0.10)
    assert [s["run_id"] for s in surv] == ["r-i0", "r-i1"]


def test_rejected_and_technical_fail_never_survive():
    iters = [
        _it("r-i0", "rejected", -0.5),
        _it("r-i1", "technical_fail", None),
        _it("r-i2", "needs_more_seeds", 0.9),  # sample too small -> not trusted
    ]
    assert select_survivors(iters, noise_band=0.10) == []


def test_within_band_inconclusive_does_not_survive():
    # Beats parent but inside the noise band -> can't discriminate -> not a survivor.
    assert select_survivors([_it("r-i0", "inconclusive", 0.05)], noise_band=0.10) == []


def test_none_delta_does_not_crash_or_survive():
    assert select_survivors([_it("r-i0", "inconclusive", None)], noise_band=0.10) == []


def test_candidate_name_is_factories_safe():
    assert candidate_name("20260618-research-i0") == "arl_20260618_research_i0"
    assert candidate_name("a/b c") == "arl_a_b_c"


def test_build_promotion_command_for_survivors():
    cmd = build_promotion_command(["arl_x", "arl_y"], profile="strong")
    assert "scripts/league_submit_ruler.py" in cmd
    assert "--candidates arl_x arl_y" in cmd
    assert "--profile strong" in cmd


def test_build_promotion_command_empty_when_no_survivors():
    assert build_promotion_command([], profile="strong") == ""
    assert build_promotion_command([None, ""], profile="quick") == ""
