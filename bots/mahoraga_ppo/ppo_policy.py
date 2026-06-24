"""PPO specialist slot for the Mahoraga-PPO hybrid.

This module is a STUB until a PPO checkpoint passes the promotion criteria
(Fase 2b best.pt + export parity). The packager then replaces it with the
rendered pure-Python policy (scripts/export_submission.render_submission), which
must expose the same two symbols:

  - ``ready() -> bool``      whether a trained policy is bundled
  - ``agent(obs) -> moves``  Kaggle-format policy (only called when ready)

While not ready, the hybrid degrades to pure Mahoraga — by construction it can
never be weaker than the incumbent because of a missing checkpoint.
"""

from __future__ import annotations

from typing import Any


def ready() -> bool:
    return False


def agent(obs: dict[str, Any]) -> list[list[float]]:
    raise RuntimeError("ppo_policy stub: no trained policy bundled (ready() is False)")
