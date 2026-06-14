"""Regression: `_greedy_select` must not let one source out-drain its safe cap.

Bug "drenagem dupla" (todo.md, 2026-06-11): funding was checked against the raw
ship count (`source_budget`) while each candidate's send is already capped at
`safe_drain`. A threatened planet (safe_drain << ships) could therefore fire
several waves that, summed, exceeded the safe drain.

The fix adds a separate `source_spend_budget` (scattered `safe_drain`) that
gates and is debited per selection, while `source_budget` (raw ships) stays the
leftover returned for regroup.
"""

import pytest

torch = pytest.importorskip("torch")

from orbit_lite.planner_core import _greedy_select


def _run(*, source_budget, source_spend_budget, sends, scores, W=2):
    """Two candidates from the SAME source (planet 0) to two distinct targets
    (planets 1 and 2). L=1 (one contributor per candidate)."""
    device = torch.device("cpu")
    dtype = torch.float32
    P = 3
    C, L = 2, 1

    cand_src = torch.tensor([[0], [0]], dtype=torch.long, device=device)        # both from planet 0
    cand_send = torch.tensor(sends, dtype=dtype, device=device).view(C, L)
    cand_angle = torch.zeros(C, L, dtype=dtype, device=device)
    cand_eta = torch.ones(C, L, dtype=dtype, device=device)
    cand_active = torch.ones(C, L, dtype=torch.bool, device=device)
    cand_tgt_slot = torch.tensor([1, 2], dtype=torch.long, device=device)       # planet slots
    cand_tgt_short = torch.tensor([0, 1], dtype=torch.long, device=device)      # shortlist idx
    cand_is_def = torch.zeros(C, dtype=torch.bool, device=device)               # attacks, not reinforce
    score = torch.tensor(scores, dtype=dtype, device=device)
    target_exists = torch.ones(2, dtype=torch.bool, device=device)

    entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle,
        cand_eta=cand_eta, cand_active=cand_active, cand_tgt_slot=cand_tgt_slot,
        cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def,
        source_budget=source_budget.to(dtype),
        source_spend_budget=(None if source_spend_budget is None
                             else source_spend_budget.to(dtype)),
        target_exists=target_exists, roi_threshold=0.0,
    )
    fired = entries.valid & (entries.ships > 0)
    n_waves = int(fired.sum().item())
    total_from_src0 = float(entries.ships[fired].sum().item())
    return n_waves, total_from_src0, leftover


def test_spend_budget_caps_total_drain_per_source():
    """Case 1: 2 candidates, send=40 each, raw=100, spend(safe_drain)=40.
    Only one wave may fire; total from the source <= 40; leftover ships = 60."""
    n_waves, total, leftover = _run(
        source_budget=torch.tensor([100.0, 0.0, 0.0]),
        source_spend_budget=torch.tensor([40.0, 0.0, 0.0]),
        sends=[40.0, 40.0],
        scores=[10.0, 5.0],   # candidate A (target 1) wins the first pick
    )
    assert n_waves == 1
    assert total <= 40.0 + 1e-6
    assert leftover[0].item() == pytest.approx(60.0)   # real leftover = raw - spent


def test_spend_budget_oep_style_fractions():
    """Case 2 (OEP-style): sends 40 (frac 1.0) and 20 (frac 0.5), spend=40.
    The 40 fires; the 20 can no longer fund against the spent-down budget."""
    n_waves, total, _ = _run(
        source_budget=torch.tensor([100.0, 0.0, 0.0]),
        source_spend_budget=torch.tensor([40.0, 0.0, 0.0]),
        sends=[40.0, 20.0],
        scores=[10.0, 5.0],   # the 40-ship candidate is preferred
    )
    assert n_waves == 1
    assert total == pytest.approx(40.0)


def test_default_spend_budget_preserves_old_behavior():
    """Case 3: with no spend budget (None) the funding falls back to raw ships,
    reproducing the *old* double-wave behavior. Locks backward-compat so callers
    that don't pass the new arg are unaffected."""
    n_waves, total, leftover = _run(
        source_budget=torch.tensor([100.0, 0.0, 0.0]),
        source_spend_budget=None,
        sends=[40.0, 40.0],
        scores=[10.0, 5.0],
    )
    assert n_waves == 2
    assert total == pytest.approx(80.0)
    assert leftover[0].item() == pytest.approx(20.0)
