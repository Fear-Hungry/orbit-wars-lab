"""Invert expert continuous moves into the compact PPO discrete action space.

The PPO policy emits a single discrete action ``[source_rank, target_rank,
fraction_idx, offset_idx]`` per turn, which :func:`decode_discrete_action`
expands into a *whole turn* of official moves (one move per own planet, all
sharing the same target/fraction/offset indices). Experts (Producer/OEP) instead
emit arbitrary ``[from_planet_id, angle, ships]`` moves with per-source targets
and fractions.

There is therefore no exact inverse: a rich multi-target expert turn cannot be
represented by a single ``(s, t, f, o)`` tuple in general. We pick the tuple
whose decoded move-set best reproduces the expert's move-set under an explicit
distance metric, and report the residual as the *quantization error* so the
caller can audit how lossy the projection is.

Distance metric (``move_set_distance``):
  - Decoded and expert moves are indexed by source planet id.
  - For each expert move whose source also appears in the decoded set, the cost
    is ``W_ANGLE * angular_distance + W_SHIPS * |Δships| / max(source_ships, 1)``.
  - An expert move whose source is missing from the decoded set costs
    ``MISS_PENALTY``; a decoded move whose source is absent from the expert set
    costs ``EXTRA_PENALTY`` (extra activity is cheaper than missing a real move).

The weights are deliberately simple and exposed on :class:`InverseConfig` so the
metric can be retuned without touching the search. Angle distance is in radians
(``0..π``) and the ships term is a fraction (``0..1``); ``MISS_PENALTY`` is set
above a full-angle miss so any reproduced source is preferred over a dropped one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .action_decoder import DEFAULT_DECODER_CONFIG, DecoderConfig, decode_discrete_action
from .entities import planet_id, planet_owner, planet_ships

# Policy head sizes; must match python/agents/policy.py FlatActorCritic defaults.
SOURCE_N = 16
TARGET_N = 32
FRAC_N = 4
OFFSET_N = 5


@dataclass(frozen=True)
class InverseConfig:
    source_n: int = SOURCE_N
    target_n: int = TARGET_N
    frac_n: int = FRAC_N
    offset_n: int = OFFSET_N
    w_angle: float = 1.0
    w_ships: float = 1.0
    miss_penalty: float = math.pi + 1.0
    extra_penalty: float = 0.5


DEFAULT_INVERSE_CONFIG = InverseConfig()


@dataclass(frozen=True)
class InverseResult:
    action: tuple[int, int, int, int]
    quant_error: float
    matched_moves: int
    expert_moves: int
    is_no_op: bool

    @property
    def launch(self) -> int:
        """Binary launch gate: 0 for a pass (no-op), 1 for a real launch turn."""
        return 0 if self.is_no_op else 1

    @property
    def action5(self) -> tuple[int, int, int, int, int]:
        """Canonical PPO action ``[launch, source, target, frac, offset]``."""
        return (self.launch, *self.action)


def _angular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _moves_by_source(moves: list[list[float]]) -> dict[int, list[float]]:
    """Index a move list by source planet id, keeping the first move per source.

    The decoder emits at most one move per source per turn, and experts overwhelm-
    ingly do too; if an expert repeats a source we keep the first (highest-priority)
    entry to match the decoder's one-move-per-source structure.
    """
    indexed: dict[int, list[float]] = {}
    for move in moves:
        src = int(move[0])
        if src not in indexed:
            indexed[src] = move
    return indexed


def move_set_distance(
    decoded: list[list[float]],
    expert: list[list[float]],
    state: dict[str, Any],
    cfg: InverseConfig = DEFAULT_INVERSE_CONFIG,
) -> tuple[float, int]:
    """Return (cost, matched_move_count) between a decoded and expert move-set."""
    decoded_by_src = _moves_by_source(decoded)
    expert_by_src = _moves_by_source(expert)
    ships_by_src = {
        planet_id(p): max(1, planet_ships(p)) for p in state.get("planets", [])
    }

    cost = 0.0
    matched = 0
    for src, e_move in expert_by_src.items():
        d_move = decoded_by_src.get(src)
        if d_move is None:
            cost += cfg.miss_penalty
            continue
        matched += 1
        angle_err = _angular_distance(float(e_move[1]), float(d_move[1]))
        ships_ref = float(ships_by_src.get(src, 1))
        ships_err = abs(float(e_move[2]) - float(d_move[2])) / ships_ref
        cost += cfg.w_angle * angle_err + cfg.w_ships * ships_err

    for src in decoded_by_src:
        if src not in expert_by_src:
            cost += cfg.extra_penalty

    return cost, matched


def invert_moves(
    state: dict[str, Any],
    player: int,
    expert_moves: list[list[float]],
    *,
    decoder_cfg: DecoderConfig = DEFAULT_DECODER_CONFIG,
    cfg: InverseConfig = DEFAULT_INVERSE_CONFIG,
) -> InverseResult:
    """Project expert moves onto the closest PPO discrete action.

    Returns the best ``(source_rank, target_rank, fraction_idx, offset_idx)``
    tuple, its quantization error, and bookkeeping. When the expert produced no
    moves (or the state has no launchable own planet) the action space cannot
    express the intent faithfully, so we return ``(0, 0, 0, 0)`` flagged with
    ``is_no_op=True`` and zero error (the caller should treat these separately).
    """
    own = [
        p
        for p in state.get("planets", [])
        if planet_owner(p) == player and planet_ships(p) >= decoder_cfg.min_ships_to_launch
    ]
    if not expert_moves or not own:
        return InverseResult(
            action=(0, 0, 0, 0),
            quant_error=0.0,
            matched_moves=0,
            expert_moves=len(expert_moves),
            is_no_op=True,
        )

    planet_count = len(state.get("planets", []))
    source_cap = min(cfg.source_n, len(own))
    # ``target_rank`` wraps modulo the per-source candidate count, so ranks above
    # the number of other planets are redundant. Cap the search accordingly.
    target_cap = min(cfg.target_n, max(1, planet_count - 1))

    best: InverseResult | None = None
    for s in range(source_cap):
        for f in range(cfg.frac_n):
            for o in range(cfg.offset_n):
                for t in range(target_cap):
                    decoded = decode_discrete_action(state, player, [s, t, f, o], decoder_cfg)
                    error, matched = move_set_distance(decoded, expert_moves, state, cfg)
                    if best is None or error < best.quant_error:
                        best = InverseResult(
                            action=(s, t, f, o),
                            quant_error=error,
                            matched_moves=matched,
                            expert_moves=len(expert_moves),
                            is_no_op=False,
                        )
    assert best is not None  # source_cap >= 1 guarantees at least one candidate
    return best
