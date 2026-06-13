"""Family H — H10: OEP backbone + value-gated H overlay.

The pure-H families are weak as full policies (~-0.9 vs Producer): they only
attack. OEP, by contrast, ~ties Producer. So the only Family-H line with a real
shot at the régua keeps OEP as the BACKBONE and overrides it with the best H
candidate ONLY when the light forward model says the H move wins by a margin —
a conservative, realizable selector (no oracle leak; the value is a forward
prediction).

This directly tests whether a value-gated overlay can convert the oracle's
sparse marginal blood that the bucket hyper-heuristic (H7) could not.
"""

from __future__ import annotations

from typing import Any

from bots.oep.rhea import _value_of_moves, best_by_value

Obs = dict[str, Any]
Moves = list[list[float]]

#: Light-model territory gain the H override must beat OEP by (conservative —
#: avoids regressing the OEP backbone on noisy ties).
OVERLAY_MARGIN = 3.0


def make_oep_value_overlay() -> Any:
    """Build a fresh overlay generator with its own persistent OEP runtime."""

    from bots.oep.agent import make_agent as make_oep

    oep = make_oep()

    def plan(obs: Obs) -> Moves:
        oep_moves = oep(obs)
        if not isinstance(oep_moves, list):
            oep_moves = []
        h_moves = best_by_value(obs)
        me = int(obs.get("player", 0))
        v_oep = _value_of_moves(obs, oep_moves, me)
        v_h = _value_of_moves(obs, h_moves, me)
        return h_moves if v_h > v_oep + OVERLAY_MARGIN else oep_moves

    return plan
