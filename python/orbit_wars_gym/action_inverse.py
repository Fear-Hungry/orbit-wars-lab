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

from .action_decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
    _angle,
    _predict_target_xy,
    _sun_safe_angle,
)
from .entities import planet_id, planet_owner, planet_production, planet_ships, planet_x, planet_y

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
    *,
    ships_by_src: dict[int, int] | None = None,
) -> tuple[float, int]:
    """Return (cost, matched_move_count) between a decoded and expert move-set.

    ``ships_by_src`` lets a grid-search caller precompute the planet-ships table
    once instead of rebuilding it for every candidate (hot path of invert_moves).
    """
    decoded_by_src = _moves_by_source(decoded)
    expert_by_src = _moves_by_source(expert)
    if ships_by_src is None:
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


class _TurnDecodeCache:
    """Per-(state, player) decode tables for the inverse grid search.

    ``invert_moves`` decodes thousands of candidate tuples against ONE fixed
    state, but ``decode_discrete_action`` re-runs the trigonometric target
    scoring from scratch on every call — that made dataset collection ~25min per
    seed. This cache precomputes, once per turn, the sorted own-planet list and
    each source's target keys in the decoder's EXACT float operation order (one
    key per repeat_penalty branch), so ``decode()`` reproduces
    ``decode_discrete_action`` byte-for-byte (asserted by
    tests/test_action_inverse_fast_decode.py) while only paying a small stable
    sort per source.
    """

    def __init__(self, state: dict[str, Any], player: int, cfg: DecoderConfig) -> None:
        self._state = state
        self._cfg = cfg
        planets = state.get("planets", [])
        own = [
            p
            for p in planets
            if planet_owner(p) == player and planet_ships(p) >= cfg.min_ships_to_launch
        ]
        own.sort(key=lambda p: (planet_ships(p), planet_production(p)), reverse=True)
        self.own = own

        # Per source: candidates in the original ``planets`` order (sort
        # stability!) with the decoder's exact key for repeat_penalty 0 and 3.
        self._per_source: list[tuple[Any, tuple[float, float], list[tuple[int, float, float, Any]]]] = []
        for src in own:
            source_xy = (planet_x(src), planet_y(src))
            max_launch_ships = max(1, int(planet_ships(src) * cfg.fractions[-1]))
            cands: list[tuple[int, float, float, Any]] = []
            for p in planets:
                if planet_id(p) == planet_id(src):
                    continue
                tx, ty = _predict_target_xy(state, source_xy, p, max_launch_ships)
                dist = math.hypot(tx - source_xy[0], ty - source_xy[1])
                owner = planet_owner(p)
                enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
                neutral_bonus = 4.0 if owner == -1 else 0.0
                # Mirror target_score's exact left-to-right float chain for both
                # repeat_penalty values (0.0 and 3.0).
                head = float(planet_production(p)) * 10.0 + enemy_bonus + neutral_bonus
                key0 = head - 0.0 - 0.15 * dist - 0.12 * float(planet_ships(p))
                key1 = head - 3.0 - 0.15 * dist - 0.12 * float(planet_ships(p))
                cands.append((planet_id(p), key0, key1, p))
            self._per_source.append((src, source_xy, cands))

    def decode(self, source_rank: int, target_rank: int, fraction_idx: int, offset_idx: int) -> list[list[float]]:
        cfg = self._cfg
        own = self.own
        if not own:
            return []
        rot = source_rank % len(own)
        order = list(range(rot, len(own))) + list(range(rot))

        moves: list[list[float]] = []
        used_targets: set[int] = set()
        for idx in order:
            if len(moves) >= cfg.max_moves_per_turn:
                break
            src, source_xy, cands = self._per_source[idx]
            if not cands:
                continue
            ranked = sorted(
                cands,
                key=lambda c: c[2] if c[0] in used_targets else c[1],
                reverse=True,
            )
            target_pid, _, _, target = ranked[target_rank % len(ranked)]

            frac = cfg.fractions[fraction_idx % len(cfg.fractions)]
            ships = int(max(0, math.floor(float(planet_ships(src)) * frac)))
            if ships <= 0:
                continue
            if planet_ships(src) - ships < cfg.reserve_home_ships and len(own) <= 2:
                ships = max(0, planet_ships(src) - cfg.reserve_home_ships)
            if ships <= 0:
                continue

            target_xy = _predict_target_xy(self._state, source_xy, target, ships)
            base = _angle(source_xy, target_xy)
            base = _sun_safe_angle(source_xy, target_xy, base)
            angle = base + cfg.angle_offsets[offset_idx % len(cfg.angle_offsets)]

            moves.append([planet_id(src), float(angle), int(ships)])
            used_targets.add(target_pid)

        return moves


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
    decode_cache = _TurnDecodeCache(state, player, decoder_cfg)
    own = decode_cache.own
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
    ships_by_src = {
        planet_id(p): max(1, planet_ships(p)) for p in state.get("planets", [])
    }

    best: InverseResult | None = None
    for s in range(source_cap):
        for f in range(cfg.frac_n):
            for o in range(cfg.offset_n):
                for t in range(target_cap):
                    decoded = decode_cache.decode(s, t, f, o)
                    error, matched = move_set_distance(
                        decoded, expert_moves, state, cfg, ships_by_src=ships_by_src
                    )
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
