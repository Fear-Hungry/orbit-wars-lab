"""Regression: _greedy_select must cap per-source SPEND at source_spend_budget.

Locked-down bug (2026-06-11 /diagnose, DEBUG-sd01): funding was checked against
``source_budget`` (raw garrison) only, so two same-source waves could each pass
``can_fund`` and jointly draw 2x the safe_drain from a threatened planet —
exactly the losing regime (drain << ships under attack; measured live in
producer 4p+rusher and OEP vs rusher). The fix threads a second budget
(``source_spend_budget``) that caps the per-turn draw, while ``source_budget``
keeps its _plan_regroup contract (real physical leftover).
"""
from __future__ import annotations

import torch
from orbit_lite.planner_core import _greedy_select


def _run_select(sends, scores, *, source_budget, spend_budget, W=4):
    """Two-target setup: C candidates, all from planet 0, one per target."""
    device = torch.device("cpu")
    dtype = torch.float32
    C = len(sends)
    P = C + 1  # planet 0 = the single source; planets 1..C = targets
    L = 1
    kwargs = dict(
        P=P,
        W=W,
        device=device,
        dtype=dtype,
        score=torch.tensor(scores, dtype=dtype),
        cand_src=torch.zeros(C, L, dtype=torch.long),
        cand_send=torch.tensor(sends, dtype=dtype).view(C, L),
        cand_angle=torch.zeros(C, L, dtype=dtype),
        cand_eta=torch.ones(C, L, dtype=dtype),
        cand_active=torch.ones(C, L, dtype=torch.bool),
        cand_tgt_slot=torch.arange(1, C + 1, dtype=torch.long),
        cand_tgt_short=torch.arange(C, dtype=torch.long),
        cand_is_def=torch.zeros(C, dtype=torch.bool),
        source_budget=torch.tensor(source_budget, dtype=dtype),
        target_exists=torch.ones(C, dtype=torch.bool),
        roi_threshold=0.0,
    )
    if spend_budget is not None:
        kwargs["source_spend_budget"] = torch.tensor(spend_budget, dtype=dtype)
    entries, leftover = _greedy_select(**kwargs)
    sent = float(entries.ships[entries.valid].sum().item())
    waves = int(entries.valid.sum().item())
    return waves, sent, leftover


def test_spend_budget_caps_same_source_double_drain():
    # source has 100 ships but safe_drain=40; two 40-ship candidates from it.
    waves, sent, leftover = _run_select(
        sends=[40.0, 40.0],
        scores=[10.0, 9.0],
        source_budget=[100.0, 5.0, 5.0],
        spend_budget=[40.0, 0.0, 0.0],
    )
    assert waves == 1, "second same-source wave must fail can_fund on the spend budget"
    assert sent <= 40.0
    assert float(leftover[0].item()) == 60.0, "leftover stays PHYSICAL (regroup contract)"


def test_spend_budget_caps_oep_style_fractions():
    # OEP-style: full-drain (40) and half-drain (20) candidates of one source,
    # spend cap 40 -> only the full-drain wave fires (was 1.5x overdrain live).
    waves, sent, _ = _run_select(
        sends=[40.0, 20.0],
        scores=[10.0, 9.0],
        source_budget=[100.0, 5.0, 5.0],
        spend_budget=[40.0, 0.0, 0.0],
    )
    assert waves == 1
    assert sent <= 40.0


def test_default_none_reproduces_legacy_single_budget():
    # Compat lock: without source_spend_budget both waves fund against the raw
    # garrison (the historical behavior this fix makes opt-out).
    waves, sent, leftover = _run_select(
        sends=[40.0, 40.0],
        scores=[10.0, 9.0],
        source_budget=[100.0, 5.0, 5.0],
        spend_budget=None,
    )
    assert waves == 2
    assert sent == 80.0
    assert float(leftover[0].item()) == 20.0
