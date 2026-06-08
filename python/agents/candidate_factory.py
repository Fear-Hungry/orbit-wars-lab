"""Frente B (redesenho RL) — candidate factory: the selector picks among expert plans.

Instead of emitting a raw ``MultiDiscrete([2,16,32,4,5])`` action (proven MISALIGNED
— see EXPERIMENTS.md 2026-06-08: campaign_h collapsed to margin -1.0 with EV 0.93
decoupled from winning), the redesigned policy chooses the INDEX of one candidate
move-set produced by a small panel of experts. This collapses the action space to a
handful of always-legal, high-quality options and makes a dense reward learnable —
the selector cannot emit an illegal/degenerate raw action, only "use this expert's
plan this turn (or pass)".

Candidates (fixed order, index 0 is always no-op): no_op, producer, oep, greedy,
defensive, rush. Stateful experts (producer/oep) use ISOLATED instances so a batched
rollout gives one runtime per env without cross-contamination (each resets on
``step == 0``); stateless heuristics are shared (they carry no state).
"""
from __future__ import annotations

from typing import Any, Callable

from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.rules import moves_are_legal

# Fixed candidate order. Index 0 is the no-op (pass) — the guaranteed-legal default
# of the action space. The rest are expert plan generators that exist in the registry
# (the user's spec listed "defend/reinforce"; "defensive" is the registry's defend,
# and there is no separate "reinforce" policy — kept out rather than faked).
CANDIDATE_NAMES: tuple[str, ...] = ("no_op", "producer", "oep", "greedy", "defensive", "rush")
NUM_CANDIDATES: int = len(CANDIDATE_NAMES)
_EXPERT_NAMES: tuple[str, ...] = CANDIDATE_NAMES[1:]


class CandidateFactory:
    """Builds the fixed candidate panel for ONE env (isolated expert runtimes).

    Construct one per env in a vectorized rollout: the stateful experts hold per-env
    memory that resets on ``step == 0``, so sharing a factory across envs would
    cross-contaminate plans. Stateless heuristics are shared internally and safe.
    """

    def __init__(self) -> None:
        self._experts: dict[str, Callable[..., Any]] = {
            name: make_isolated_opponent(name) for name in _EXPERT_NAMES
        }

    def candidates(self, state: dict[str, Any], player: int) -> list[dict[str, Any]]:
        """Return ``NUM_CANDIDATES`` entries ``{name, moves, legal}``.

        ``no_op`` is ``moves == []``. An expert that crashes or returns an illegal
        move-set collapses to a legal pass (``moves == []``, ``legal == False``) — no
        silent illegal action ever reaches the env (invariant: fail-safe to pass).
        """
        out: list[dict[str, Any]] = [{"name": "no_op", "moves": [], "legal": True}]
        for name in _EXPERT_NAMES:
            try:
                moves = self._experts[name](state, player)
                if not isinstance(moves, list):
                    moves = []
            except Exception:  # noqa: BLE001 - a crashing expert degrades to pass
                moves = []
            legal = bool(moves_are_legal(state, player, moves))
            out.append({"name": name, "moves": moves if legal else [], "legal": legal})
        return out

    def mask(self, candidates: list[dict[str, Any]]) -> list[bool]:
        """Selectability mask over the candidate panel (same length / order).

        v1: every candidate is selectable — ``no_op`` always, and an expert that
        produced an illegal/empty plan stays selectable but resolves to a pass (so the
        head never goes all-False → no NaN). Semantic refinements (dedup identical
        plans, drop expert==no_op) are a later step, not needed for a correct env.
        """
        return [True] * len(candidates)
