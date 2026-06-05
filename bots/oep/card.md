# OEP

Status: experimental, not promoted.

Baseline: Producer-corrected local opponent.

Best recorded local result:

- `2026-06-04`, OEP-1ply tournament vs Producer, 4 seeds:
  `win=0.375`, `margin=-0.25`, `mean_ms=177.34`, `crash/timeout/invalid=0`.

Design:

- Uses shared `orbit_lite`.
- Uses public policy injection for the seed and opponent model.
- Emits the OEP plan only when its full-plan fitness beats the seed policy fitness by `min_advantage`.

Next bottleneck: improve candidate diversity and/or fitness; do not promote this bot as-is.
