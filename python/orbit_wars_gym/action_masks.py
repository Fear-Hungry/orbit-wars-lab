"""Legal-action masks for the launch-gated action space (todo P1.5f).

The PPO action is ``[launch, source_rank, target_rank, frac_idx, offset_idx]``.
Most ranks are technically "legal" because the decoder wraps them modulo the
candidate count, but many are *redundant* (they decode to the same move) or
meaningless (launch when there is nothing to launch). Masking removes those so
the policy spends probability mass only on distinct, sensible actions.

Mask rules (from the design agreed in P1.5):
  - ``launch == 0`` (pass) is always valid.
  - ``launch == 1`` is invalid when the player has no launchable own planet.
  - ``source_rank`` is valid only up to the number of launchable own planets.
  - ``target_rank`` is valid up to ``planet_count - 1`` (every other planet).
  - ``frac_idx`` / ``offset_idx`` are always valid.

CRITICAL: the same mask used when sampling an action must be reused in the PPO
update, otherwise the importance ratio is computed under a different distribution
and is mathematically wrong. The masks are therefore stored in the rollout buffer
and replayed at update time.

To keep the buffer simple, masks are stored as one flat boolean row per step
(``launch | source | target`` concatenated, length :data:`MASK_DIM`).
``frac``/``offset`` are never masked, so they are omitted from the row.
"""

from __future__ import annotations

import numpy as np

from .entities import planet_owner, planet_ships

LAUNCH_N = 2
SOURCE_N = 16
TARGET_N = 32
MASK_DIM = LAUNCH_N + SOURCE_N + TARGET_N  # 50

_LAUNCH_SLICE = slice(0, LAUNCH_N)
_SOURCE_SLICE = slice(LAUNCH_N, LAUNCH_N + SOURCE_N)
_TARGET_SLICE = slice(LAUNCH_N + SOURCE_N, MASK_DIM)


def build_action_masks(
    state: dict,
    player: int,
    *,
    min_ships_to_launch: int = 2,
    source_n: int = SOURCE_N,
    target_n: int = TARGET_N,
) -> np.ndarray:
    """Return a flat ``(MASK_DIM,)`` boolean mask for one state/player.

    Heads that would otherwise have no valid entry (e.g. ``source`` when nothing
    is launchable) fall back to all-valid so the categorical never sees an
    all ``-inf`` row (which would produce NaNs). Those entries are ignored anyway
    because ``launch == 0`` is forced in that case.
    """
    planets = state.get("planets", [])
    n_launchable = sum(
        1 for p in planets if planet_owner(p) == player and planet_ships(p) >= min_ships_to_launch
    )
    planet_count = len(planets)

    mask = np.zeros(LAUNCH_N + source_n + target_n, dtype=bool)
    # launch: pass always valid; launch only when something is launchable.
    mask[0] = True
    mask[1] = n_launchable > 0

    valid_src = min(source_n, n_launchable)
    if valid_src > 0:
        mask[LAUNCH_N : LAUNCH_N + valid_src] = True
    else:
        mask[LAUNCH_N : LAUNCH_N + source_n] = True  # ignored (launch forced to 0)

    valid_tgt = min(target_n, max(planet_count - 1, 0))
    tgt_start = LAUNCH_N + source_n
    if valid_tgt > 0:
        mask[tgt_start : tgt_start + valid_tgt] = True
    else:
        mask[tgt_start : tgt_start + target_n] = True

    return mask


def split_masks(flat: "np.ndarray | object") -> dict:
    """Split a flat mask row/batch into the per-head dict the policy expects.

    Accepts a numpy array or a torch tensor with last dim :data:`MASK_DIM`;
    returns a dict with keys ``launch``/``source``/``target`` preserving type.
    """
    return {
        "launch": flat[..., _LAUNCH_SLICE],
        "source": flat[..., _SOURCE_SLICE],
        "target": flat[..., _TARGET_SLICE],
    }
