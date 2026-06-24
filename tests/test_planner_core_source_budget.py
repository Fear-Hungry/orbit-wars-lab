"""_greedy_select must respect the TACTICAL spend budget (safe_drain) cumulatively.

Regression for the confirmed overspend: each candidate is individually sized to
safe_drain, but two waves funded by the SAME source could sum past it (memory:
safe_drain_overspend_confirmed — producer financed 2+ waves from one threatened
source). With ``source_spend_budget`` the cumulative spend per source is capped
while the returned leftover still reflects the physical garrison.
"""
from __future__ import annotations

import torch

from orbit_lite.planner_core import _greedy_select


def _two_candidates_same_source(send: float = 40.0):
    """P=3: planet 0 is the source (100 ships); targets at slots 1 and 2."""
    device = torch.device("cpu")
    dtype = torch.float32
    C, L = 2, 1
    return dict(
        P=3,
        W=2,
        device=device,
        dtype=dtype,
        score=torch.tensor([10.0, 9.0]),
        cand_src=torch.zeros(C, L, dtype=torch.long),
        cand_send=torch.full((C, L), float(send), dtype=dtype),
        cand_angle=torch.zeros(C, L, dtype=dtype),
        cand_eta=torch.full((C, L), 3.0, dtype=dtype),
        cand_active=torch.ones(C, L, dtype=torch.bool),
        cand_tgt_slot=torch.tensor([1, 2], dtype=torch.long),
        cand_tgt_short=torch.tensor([0, 1], dtype=torch.long),
        cand_is_def=torch.tensor([False, False]),
        source_budget=torch.tensor([100.0, 0.0, 0.0], dtype=dtype),
        target_exists=torch.tensor([True, True]),
        roi_threshold=0.0,
    )


def _spent_by_source(entries, source: int) -> float:
    sel = entries.valid & (entries.source_slots == source)
    return float(entries.ships[sel].sum().item())


def test_without_spend_budget_overspends_physical_source():
    """Documents the legacy behavior: physical budget alone funds both waves."""
    entries, leftover = _greedy_select(**_two_candidates_same_source())
    assert _spent_by_source(entries, 0) == 80.0
    assert float(leftover[0].item()) == 20.0


def test_spend_budget_caps_cumulative_spend_per_source():
    kwargs = _two_candidates_same_source()
    spend = torch.tensor([40.0, 0.0, 0.0], dtype=kwargs["dtype"])
    entries, leftover = _greedy_select(**kwargs, source_spend_budget=spend)
    # at most ONE of the two 40-ship waves may fire from source 0
    assert _spent_by_source(entries, 0) <= 40.0
    assert int((entries.valid & (entries.source_slots == 0)).sum().item()) == 1
    # leftover still reflects the PHYSICAL garrison (used by _plan_regroup)
    assert float(leftover[0].item()) == 60.0


def test_spend_budget_zero_blocks_all_waves_from_source():
    kwargs = _two_candidates_same_source()
    spend = torch.zeros(3, dtype=kwargs["dtype"])
    entries, _ = _greedy_select(**kwargs, source_spend_budget=spend)
    assert _spent_by_source(entries, 0) == 0.0
