"""Frozen BReP checkpoint as a TRAINING OPPONENT — the league/self-play primitive.

Exp #1 (2026-06-10): training only vs Producer-style opponents overfits the gate
opponent — the PGS line beat Producer locally (+0.218) and then failed on the real
leaderboard (1001.7 vs 1228 ref), so the learner needs a DIVERSE pool (AlphaStar
league / PSRO style). This module lets a frozen champion checkpoint (brep_gpu/c05,
v1 n_edit=4) sit in the ``--opponents`` list as ``brep:<checkpoint path>``.

The opponent is a plain ``(state, player) -> moves`` callable like the heuristics:
its own ISOLATED Producer base planner (per instance, never shared — see the
cross-contamination diagnosis), the frozen net in eval mode (shared across envs;
weights are read-only), greedy edits, and the *v1* decode table recovered from the
packaged c05 submission. v1 REDUCE is ``ships // 2`` (NOT v2's round(x0.5)) — they
differ on odd ship counts, so the v2 table would misplay the v1 policy.

Python-side ``encode_state`` is byte-identical to the backend encoding (verified
maxdiff=0.0 during the submission parity probe), so the frozen net sees the same
observations it was trained on.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from python.agents.policy import ProducerResidualBranchActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.encoding import EncoderConfig, encode_state, observation_dim
from python.orbit_wars_gym.entities import planet_id, planet_ships

N_EDIT_V1 = 4
OPPONENT_PREFIX = "brep:"

_MODEL_CACHE: dict[str, ProducerResidualBranchActorCritic] = {}


def _apply_residual_edits_v1(
    state: dict[str, Any], base_moves: Sequence[Sequence[float]], edits: Sequence[int], k_max: int
) -> list[list[float]]:
    """v1 decode table: 0=KEEP, 1=CANCEL, 2=REDUCE(//2), 3=BOOST(min(avail-1, 2x))."""
    ships_by_id = {planet_id(p): planet_ships(p) for p in state.get("planets", [])}
    out: list[list[float]] = []
    for i, mv in enumerate(base_moves):
        ships = int(mv[2])
        if i >= k_max:
            out.append([mv[0], mv[1], float(ships)])
            continue
        e = int(edits[i])
        if e == 1:  # CANCEL
            continue
        if e == 2:  # REDUCE
            out.append([mv[0], mv[1], float(max(1, ships // 2))])
        elif e == 3:  # BOOST
            avail = int(ships_by_id.get(int(mv[0]), ships))
            boosted = min(max(1, avail - 1), ships * 2)
            out.append([mv[0], mv[1], float(boosted if boosted > 0 else ships)])
        else:  # KEEP (0) or unknown
            out.append([mv[0], mv[1], float(ships)])
    return out


def load_frozen_brep_v1(checkpoint_path: str) -> ProducerResidualBranchActorCritic:
    """Load (and cache) a v1 BReP checkpoint on CPU in eval mode."""
    model = _MODEL_CACHE.get(checkpoint_path)
    if model is not None:
        return model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt["config"]["policy_arch"]
    if arch != "producer_residual":
        raise ValueError(f"brep opponent checkpoint arch is {arch!r}, expected producer_residual")
    model = ProducerResidualBranchActorCritic(observation_dim(EncoderConfig()), n_edit=N_EDIT_V1)
    head_out = ckpt["model_state_dict"]["edit.weight"].shape[0]
    if head_out != model.k_max * model.n_edit:
        raise ValueError(
            f"brep opponent edit head is {head_out}, expected {model.k_max * model.n_edit} (v1 n_edit=4)"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _MODEL_CACHE[checkpoint_path] = model
    return model


class FrozenBRePV1Opponent:
    """``(state, player) -> moves`` callable wrapping a frozen v1 BReP checkpoint.

    One instance per env: the Producer base planner keeps per-game memory and must
    never be shared across concurrent games. The torch module IS shared (eval-mode,
    no state)."""

    def __init__(self, checkpoint_path: str):
        self._model = load_frozen_brep_v1(checkpoint_path)
        self._base = make_isolated_opponent("producer")
        self._k_max = int(self._model.k_max)

    def __call__(self, state: dict[str, Any], player: int) -> list[list[float]]:
        base = [list(m) for m in self._base(state, player)]
        if not base:
            return []
        obs = encode_state(state, player)
        with torch.no_grad():
            logits = self._model.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))["edit"][0]
        greedy = logits.argmax(-1).tolist()
        n_active = min(len(base), self._k_max)
        edits = [int(greedy[i]) if i < n_active else 0 for i in range(len(base))]
        return _apply_residual_edits_v1(state, base, edits, self._k_max)


def make_brep_opponents(opponent_name: str, num_envs: int) -> list[FrozenBRePV1Opponent]:
    """Resolve a ``brep:<path>`` opponent name into per-env isolated instances."""
    path = opponent_name[len(OPPONENT_PREFIX):]
    if not path:
        raise ValueError("brep opponent needs a checkpoint path: brep:<path>")
    return [FrozenBRePV1Opponent(path) for _ in range(num_envs)]
