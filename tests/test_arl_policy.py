"""Pin the ARL metrics parser and keep/discard policy (scripts/research_loop/policy.py).

policy.py is the auditable core of the Auto-Research Loop: it turns raw eval
output into a decision. goal.md demands two invariants this suite locks down:

  * a TECHNICAL failure (timeout/invalid/bad-status/fallback/exception/p95) is
    ``technical_fail`` and NEVER competitive ``rejected`` — even when the
    candidate's fitness is far below the parent;
  * a candidate without a valid sample (or below the seed floor) is
    ``inconclusive``/``needs_more_seeds``, never a competitive verdict.

All of policy.py is pure (no torch / no Rust .so), so these run in milliseconds.
"""
from __future__ import annotations

import json

from scripts.research_loop.policy import (
    DECISIONS,
    Decision,
    has_faults,
    keep_or_discard,
    parse_metrics,
    status_for,
)

CLEAN = {"death_rate": 0.4, "mean_margin": 0.1, "mean_final_planets": 5.0,
         "timeouts": 0, "seeds": 24, "steps": 500, "pool": ["producer", "oep"],
         "per_opponent": {"producer": {"death_rate": 0.4}}}


# --------------------------------------------------------------------------- #
# parse_metrics
# --------------------------------------------------------------------------- #
def test_parse_clean_sample_is_valid_and_fault_free():
    p = parse_metrics(CLEAN)
    assert p.valid is True
    assert p.death_rate == 0.4 and p.mean_margin == 0.1
    assert p.n_seeds == 24 and p.steps == 500
    assert p.pool == ("producer", "oep")
    assert has_faults(p.faults) is False


def test_parse_n_seeds_override_wins_over_payload():
    p = parse_metrics(CLEAN, n_seeds=3)
    assert p.n_seeds == 3  # the runner knows the budget it asked for


def test_parse_missing_core_metric_is_invalid():
    p = parse_metrics({"mean_margin": 0.1, "seeds": 10})  # no death_rate
    assert p.valid is False
    assert "no valid sample" in p.note


def test_parse_nan_metric_is_invalid():
    p = parse_metrics({"death_rate": float("nan"), "mean_margin": 0.1, "seeds": 10})
    assert p.valid is False


def test_parse_timeout_is_a_fault():
    p = parse_metrics({**CLEAN, "timeouts": 3})
    assert p.faults["timeouts"] == 3
    assert has_faults(p.faults) is True


def test_parse_p95_over_budget_is_a_fault():
    p = parse_metrics({**CLEAN, "p95_ms": 1500.0}, act_timeout_ms=1000.0)
    assert p.faults["p95_over_budget"] == 1
    assert has_faults(p.faults) is True


def test_parse_fault_aliases_are_recognised():
    p = parse_metrics({**CLEAN, "fallback_errors": 2, "bad_statuses": 1})
    assert p.faults["fallbacks"] == 2
    assert p.faults["bad_status"] == 1
    assert has_faults(p.faults) is True


def test_parse_error_key_becomes_exception_fault():
    p = parse_metrics({"error": "RuntimeError: boom"})
    assert p.faults["exceptions"] >= 1
    assert p.valid is False
    assert has_faults(p.faults) is True


def test_parse_json_string_payload():
    p = parse_metrics(json.dumps(CLEAN))
    assert p.valid is True and p.death_rate == 0.4


def test_parse_unparseable_string_is_invalid_with_exception_fault():
    p = parse_metrics("not json {{{")
    assert p.valid is False
    assert p.faults["exceptions"] == 1


def test_parse_none_payload_is_invalid():
    p = parse_metrics(None)
    assert p.valid is False
    # None is "no eval ran" (dry-run), not a runtime crash -> not a fault.
    assert has_faults(p.faults) is False


# --------------------------------------------------------------------------- #
# keep_or_discard — every decision is reachable and exclusive
# --------------------------------------------------------------------------- #
def _decide(raw, *, fitness, parent_fitness, min_seeds=16, band=0.10, n_seeds=None):
    p = parse_metrics(raw, n_seeds=n_seeds)
    return keep_or_discard(p, fitness=fitness, parent_fitness=parent_fitness,
                           min_promotion_seeds=min_seeds, noise_band=band)


def test_promoted_when_delta_beats_band_with_enough_seeds():
    d = _decide(CLEAN, fitness=0.5, parent_fitness=0.2)
    assert d.decision == "promoted"
    assert d.competitive is True and d.status == "applied"


def test_rejected_when_delta_below_negative_band():
    d = _decide(CLEAN, fitness=-0.5, parent_fitness=0.2)
    assert d.decision == "rejected"
    assert d.status == "rejected"


def test_inconclusive_within_noise_band():
    d = _decide(CLEAN, fitness=0.22, parent_fitness=0.20, band=0.10)
    assert d.decision == "inconclusive"
    assert d.competitive is False


def test_inconclusive_when_no_parent_bar():
    d = _decide(CLEAN, fitness=0.5, parent_fitness=None)
    assert d.decision == "inconclusive"


def test_needs_more_seeds_when_below_floor_even_with_valid_sample():
    # This is what stops a SMOKE run (2 seeds) from ever promoting competitively.
    d = _decide({**CLEAN, "seeds": 2}, fitness=0.9, parent_fitness=0.1, min_seeds=16, n_seeds=2)
    assert d.decision == "needs_more_seeds"
    assert d.competitive is False


def test_needs_more_seeds_when_invalid_sample_and_below_floor():
    d = _decide({"seeds": 2}, fitness=None, parent_fitness=0.1, min_seeds=16, n_seeds=2)
    assert d.decision == "needs_more_seeds"


def test_inconclusive_when_invalid_sample_but_enough_seeds():
    d = _decide({"seeds": 24}, fitness=None, parent_fitness=0.1, min_seeds=16, n_seeds=24)
    assert d.decision == "inconclusive"


# --------------------------------------------------------------------------- #
# THE invariant: technical_fail dominates and is never competitive rejected
# --------------------------------------------------------------------------- #
def test_technical_fail_dominates_even_with_terrible_fitness():
    # Faulty run AND fitness far below parent: a naive policy would say "rejected".
    # The contract requires technical_fail (broken harness, not a weak candidate).
    d = _decide({**CLEAN, "timeouts": 5}, fitness=-2.0, parent_fitness=0.5)
    assert d.decision == "technical_fail"
    assert d.competitive is False
    assert d.status == "logged"  # NOT "rejected"


def test_technical_fail_from_exception_payload():
    d = _decide({"error": "Segfault"}, fitness=None, parent_fitness=0.5)
    assert d.decision == "technical_fail"


def test_technical_fail_status_never_maps_to_rejected():
    assert status_for("technical_fail") == "logged"
    assert status_for("promoted") == "applied"
    assert status_for("rejected") == "rejected"
    assert status_for("inconclusive") == "logged"
    assert status_for("needs_more_seeds") == "logged"


def test_every_decision_is_in_the_contract_vocabulary():
    seen = {
        _decide(CLEAN, fitness=0.5, parent_fitness=0.2).decision,
        _decide(CLEAN, fitness=-0.5, parent_fitness=0.2).decision,
        _decide(CLEAN, fitness=0.22, parent_fitness=0.20).decision,
        _decide({**CLEAN, "seeds": 2}, fitness=0.9, parent_fitness=0.1, n_seeds=2).decision,
        _decide({**CLEAN, "timeouts": 5}, fitness=0.5, parent_fitness=0.2).decision,
    }
    assert seen == set(DECISIONS)
    assert all(isinstance(d, str) for d in DECISIONS)


def test_decision_is_frozen_dataclass():
    d = Decision("promoted", "x", fitness=1.0, parent_fitness=0.0, delta=1.0)
    try:
        d.decision = "rejected"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Decision should be immutable")
