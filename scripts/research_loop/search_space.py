"""Search space for the auto-research loop (FunSearch/AlphaEvolve-style MVP).

A *genome* is a flat ``dict[str, float|int|str]`` of ``PGSConfig`` knobs. The
loop mutates a genome, materialises it into a ``PGSConfig`` (via the h9 gate's
``run_config(..., pgs_config=genome)``), and scores it against a diverse
opponent pool. Only the knobs declared here are ever mutated — everything else
falls back to the ``BASELINE`` config.

DESIGN CHOICES (honest about what we deliberately do NOT search):
- We mutate REAL strategy knobs: WAVE discipline (wave_min_ships / wave_start_step
  / wave_max_delay), HOARD/value trade-offs (prod_weight, arbiter_margin,
  payback_max_turns, value_horizon), and the H9 THREAT-VALUE forward survival
  weights (threat_*_weight, exposed as a minimal hook in bots/pgs/threat.py).
- We do NOT mutate eval_function / heuristic weights: tuning those vs Producer is
  a PROVEN structural ceiling in this project (memory: family_h_eval_tuning_is_
  proven_ceiling, EXPERIMENTS 77-84). Searching there would just re-confirm a
  dead end.

The space is intentionally declared in ONE place (the ``SEARCH_SPACE`` dict) so
it is trivial to edit. Each entry is ``name -> Knob(lo, hi, kind)`` where ``kind``
is "float" or "int". String/bool knobs that toggle a regime (``scripts``,
``threat_value_4p``) live in ``BASELINE`` and are held fixed during the search —
the loop optimises *within* the H9 threat-value regime, not across regimes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    lo: float
    hi: float
    kind: str = "float"  # "float" | "int"

    def clamp(self, x: float) -> float | int:
        x = max(self.lo, min(self.hi, x))
        return int(round(x)) if self.kind == "int" else float(x)


# --- THE search space (edit here) -------------------------------------------
# Ranges are centred on the shipped H9 config and span a plausible neighbourhood.
SEARCH_SPACE: dict[str, Knob] = {
    # WAVE discipline (attack-wave merging): when/how big a coordinated wave must be.
    "wave_min_ships": Knob(20.0, 110.0, "float"),
    "wave_start_step": Knob(50, 250, "int"),
    "wave_max_delay": Knob(2, 16, "int"),
    # HOARD / value trade-off: how much territory (production) is worth vs ships,
    # how hard the arbiter vetoes deviations, capture payback patience, lookahead.
    "prod_weight": Knob(5.0, 30.0, "float"),
    "arbiter_margin": Knob(5.0, 60.0, "float"),
    "payback_max_turns": Knob(8.0, 40.0, "float"),
    "value_horizon": Knob(24, 64, "int"),
    # H9 THREAT-VALUE forward survival weights (the 4p survival scalar). Defaults
    # reproduce the shipped v2 scalar; these trade off "keep planets" vs "delay
    # first loss" vs "fear the worst enemy's incoming".
    "threat_planet_weight": Knob(40.0, 200.0, "float"),
    "threat_first_loss_weight": Knob(0.5, 8.0, "float"),
    "threat_incoming_weight": Knob(0.05, 2.0, "float"),
}

# Baseline = the ACTUAL shipped H9 4p config (scripts/h9_4p_gate.py `shipped` +
# threat_value_4p). Regime knobs (scripts, threat_value_4p) are fixed here so the
# search stays inside the H9 threat-value regime; the survival portfolio
# (reinforce/evac) is auto-enabled by threat_value_4p, so `scripts` includes them.
BASELINE: dict[str, object] = {
    "scripts": "hold,reinforce,evac",
    "wave_min_ships": 60.0,
    "wave_start_step": 150,
    "wave_max_delay": 8,
    "threat_value_4p": True,
    "prod_weight": 15.0,
    "arbiter_margin": 25.0,
    "payback_max_turns": 20.0,
    "value_horizon": 40,
    "threat_planet_weight": 100.0,
    "threat_first_loss_weight": 2.0,
    "threat_incoming_weight": 0.5,
}
