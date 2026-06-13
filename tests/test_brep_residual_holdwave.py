"""BReP residual over the holdwave base: parity-floor invariant + arch sanity.

The non-regression guarantee of the whole approach rests on ONE invariant: with
edits all-KEEP, the residual output reproduces the base plan byte-for-byte. We
prove it over REAL rollout states (2p and 4p), both before and after the wave
start step (the holdwave emits ~nothing pre-150, then sparse big waves), plus the
CANCEL/SCALE edit semantics and the KEEP-init of the net.
"""

from __future__ import annotations

import numpy as np
import torch
from python.agents.policy import ProducerResidualBranchActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.train.train_ppo import _apply_residual_edits

K_MAX = 16


def _states_through_game(num_players: int, seed: int, steps: int) -> list[dict]:
    """Drive a real game with the holdwave on every seat; collect states."""
    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=steps, enable_comets=True, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    bases = [make_isolated_opponent("pgs") for _ in range(num_players)]
    out = [state]
    for _ in range(steps):
        flat = []
        for p in range(num_players):
            for m in bases[p](state, p):
                flat.append([0.0, float(p), float(m[0]), float(m[1]), float(m[2])])
        arr = np.asarray(flat, dtype=np.float64) if flat else np.zeros((0, 5), dtype=np.float64)
        outcomes, states = backend.step_flat_with_states(arr)
        state = states[0]
        out.append(state)
        if bool(outcomes[0].get("done", False)):
            break
    return out


def _holdwave_moves(state: dict, player: int) -> list[list[float]]:
    return [list(m) for m in make_isolated_opponent("pgs")(state, player)]


def test_keep_all_reproduces_holdwave_base_byte_for_byte() -> None:
    # Probe pre-wave (step<150) and post-wave (>=150) states, 2p and 4p.
    for num_players, seed in ((2, 0), (4, 1)):
        states = _states_through_game(num_players, seed, 200)
        probes = [states[40], states[min(160, len(states) - 1)], states[-1]]
        for state in probes:
            for player in range(num_players):
                base = _holdwave_moves(state, player)
                kept = _apply_residual_edits(state, base, [0] * len(base), K_MAX)
                assert len(kept) == len(base)
                for a, b in zip(base, kept, strict=True):
                    assert int(a[0]) == int(b[0])
                    assert abs(float(a[1]) - float(b[1])) < 1e-9
                    assert abs(float(a[2]) - float(b[2])) < 1e-9


def test_cancel_and_scale_edit_semantics() -> None:
    states = _states_through_game(2, 2, 200)
    # find a state where the holdwave actually launches something
    state = next(
        (s for s in states if _holdwave_moves(s, 0)),
        states[-1],
    )
    base = _holdwave_moves(state, 0)
    if not base:
        return  # holdwave held all game on this seed; nothing to edit
    n = min(len(base), K_MAX)
    # CANCEL all editable slots -> only the moves past k_max survive
    cancelled = _apply_residual_edits(state, base, [1] * len(base), K_MAX)
    assert len(cancelled) == max(0, len(base) - K_MAX)
    # every scaled move stays legal (>=1 ship)
    for code in (2, 3, 4, 5):
        scaled = _apply_residual_edits(state, base, [code] * len(base), K_MAX)
        for mv in scaled[:n]:
            assert float(mv[2]) >= 1.0


def test_net_keep_init_argmax_is_zero_everywhere() -> None:
    from python.orbit_wars_gym.encoding import observation_dim

    net = ProducerResidualBranchActorCritic(observation_dim())
    obs = torch.zeros(3, observation_dim())
    logits = net.forward(obs)["edit"]
    assert logits.shape == (3, K_MAX, ProducerResidualBranchActorCritic.N_EDIT)
    assert torch.all(logits.argmax(-1) == 0)  # KEEP-init: every slot prefers KEEP


def test_keep_init_mirrored_margin_is_exactly_zero() -> None:
    # The per-chunk gate must report TRUE parity (0.0) at KEEP-init, not seat luck.
    # Mirrored-seat play makes identical self-play cancel exactly — any nonzero here
    # would silently bias the whole training gate.
    from python.orbit_wars_gym.encoding import observation_dim
    from python.train.train_ppo import evaluate_residual_margin

    net = ProducerResidualBranchActorCritic(observation_dim())
    r = evaluate_residual_margin(
        net, base_agent="pgs_holdwave", opponent_name="pgs", seeds=6, episode_steps=160
    )
    assert abs(r["mean_score_margin"]) < 1e-9
    assert r["win_rate"] == 0.5
