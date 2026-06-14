"""Property test for the per-source launch-budget invariant (G0.3).

Generalises the pontual regression in ``test_planner_core_source_budget.py`` to
the *full* planner driven by real games. The "drenagem dupla" fix added a
separate ``source_spend_budget`` (scattered ``safe_drain``) that gates and is
debited per selection in ``_greedy_select``. The invariant we lock here:

    For every ``_greedy_select`` call made while planning a turn, the ships it
    actually launches per source planet never exceed that source's offensive
    spend budget (its ``safe_drain`` share).

``_greedy_select`` is the single chokepoint through which every Producer / OEP
launch is decided, so wrapping it and driving thousands of real planning states
(2p and 4p, both Producer and OEP, fixed seeds) covers the whole pipeline. For
the Producer (one ``_greedy_select`` call per turn) the wrapped budget IS the
turn's ``safe_drain`` and the entries ARE the turn's launches, so the check is
literally "sum of the turn's sends per source <= safe_drain(source)". For OEP's
beam search the budget is threaded across the forced prefix, so the per-call
check holds transitively for the final plan.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")


def _install_invariant_probe(stats: dict) -> list:
    """Patch every loaded module exposing ``_greedy_select`` with a wrapper that
    audits the per-source send budget. Returns the patched modules so the caller
    can restore them."""
    targets = []
    for _name, mod in list(__import__("sys").modules.items()):
        fn = getattr(mod, "__dict__", {}).get("_greedy_select")
        if isinstance(fn, types.FunctionType):
            targets.append(mod)
    assert targets, "no module exposes _greedy_select; load the agents first"

    def make_wrapped(orig):
        def wrapped(**kw):
            spend_in = kw.get("source_spend_budget")
            used_fix_path = spend_in is not None
            if spend_in is None:
                spend_in = kw["source_budget"]
            spend_in = spend_in.clone()
            entries, leftover = orig(**kw)
            P = int(kw["P"])
            send = torch.where(
                entries.valid, entries.ships, torch.zeros_like(entries.ships)
            )
            src = entries.source_slots.clamp(0, P - 1).to(torch.long)
            per_src = torch.zeros(P, dtype=send.dtype)
            per_src.scatter_add_(0, src, send)
            # one offensive wave may exactly equal safe_drain; allow float slack.
            viol = per_src > spend_in + 1e-3
            stats["calls"] += 1
            if used_fix_path:
                stats["fix_path_calls"] += 1
            if bool(viol.any()):
                stats["violations"] += int(viol.sum())
                idx = int(torch.where(viol)[0][0])
                if len(stats["examples"]) < 8:
                    stats["examples"].append(
                        f"source slot={idx} launched={float(per_src[idx]):.1f} "
                        f"> safe_drain budget={float(spend_in[idx]):.1f}"
                    )
            return entries, leftover

        wrapped.__wrapped_orig__ = orig
        return wrapped

    for mod in targets:
        mod._greedy_select = make_wrapped(mod.__dict__["_greedy_select"])
    return targets


def _restore(targets) -> None:
    for mod in targets:
        orig = getattr(mod._greedy_select, "__wrapped_orig__", None)
        if orig is not None:
            mod._greedy_select = orig


@pytest.mark.parametrize(
    "lineup,seeds",
    [
        (["producer", "rusher"], [11, 12, 13]),
        (["producer", "producer", "producer", "rusher"], [21, 22, 23]),
        (["oep", "rusher"], [31, 32, 33]),
        (["oep", "oep", "oep", "rusher"], [41, 42, 43]),
    ],
)
def test_per_source_launch_never_exceeds_safe_drain(lineup, seeds):
    """For real games (fixed seeds), no source ever launches more ships in a
    turn than its offensive spend budget allows."""
    from scripts.league_agents import make
    from scripts.league_match import play_batch

    for n in set(lineup):
        make(n)  # pre-load modules (incl. bare importlib upstream)

    stats = {"calls": 0, "fix_path_calls": 0, "violations": 0, "examples": []}
    targets = _install_invariant_probe(stats)
    try:
        play_batch(lineup, seeds=seeds, steps=500, decision_ms={}, crashes={})
    finally:
        _restore(targets)

    assert stats["calls"] > 0, "planner never ran; test exercised nothing"
    # the fix path (source_spend_budget set) must actually be exercised, else we
    # are only validating the trivial raw-ship fallback.
    assert stats["fix_path_calls"] > 0, (
        f"fix path never taken (calls={stats['calls']}); "
        "source_spend_budget was always None"
    )
    assert stats["violations"] == 0, (
        f"{stats['violations']} per-source over-drain(s) in "
        f"{stats['calls']} planner calls ({lineup}): {stats['examples']}"
    )
