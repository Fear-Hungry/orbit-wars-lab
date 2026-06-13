"""_TurnDecodeCache must reproduce decode_discrete_action EXACTLY.

The inverse grid search hot-path replaces the reference decoder with cached
target keys; any float-ordering or sort-stability divergence would silently
change every BC label, so equality here is byte-for-byte over real rollout
states (early + late game, 2p and 4p) and the full candidate tuple grid.
"""

from __future__ import annotations

from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG, decode_discrete_action
from python.orbit_wars_gym.action_inverse import DEFAULT_INVERSE_CONFIG, _TurnDecodeCache, invert_moves
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig


def _rollout_states(num_players: int, seed: int, steps: int) -> list[dict]:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=steps, enable_comets=True, act_timeout=1.0),
    )
    state = backend.reset(seed)[0]
    policies = get_isolated_opponents("producer", num_players)
    states = [state]
    for _ in range(steps):
        flat = []
        for player in range(num_players):
            for move in policies[player](state, player):
                flat.append([0.0, float(player), float(move[0]), float(move[1]), float(move[2])])
        import numpy as np

        arr = np.asarray(flat, dtype=np.float64) if flat else np.zeros((0, 5), dtype=np.float64)
        outcomes, new_states = backend.step_flat_with_states(arr)
        state = new_states[0]
        states.append(state)
        if bool(outcomes[0].get("done", False)):
            break
    return states


def test_fast_decode_matches_reference_on_rollout_states() -> None:
    cfg = DEFAULT_DECODER_CONFIG
    inv = DEFAULT_INVERSE_CONFIG
    for num_players, seed, steps in ((2, 0, 40), (4, 1, 24)):
        states = _rollout_states(num_players, seed, steps)
        # probe early/mid/late states; full tuple grid at each
        probes = [states[0], states[len(states) // 2], states[-1]]
        for state in probes:
            for player in range(num_players):
                cache = _TurnDecodeCache(state, player, cfg)
                planet_count = len(state.get("planets", []))
                for s in range(min(inv.source_n, max(1, len(cache.own)))):
                    for f in range(inv.frac_n):
                        for o in range(inv.offset_n):
                            for t in range(min(inv.target_n, max(1, planet_count - 1))):
                                fast = cache.decode(s, t, f, o)
                                ref = decode_discrete_action(state, player, [s, t, f, o], cfg)
                                assert fast == ref, (
                                    f"divergence at players={num_players} step-state "
                                    f"player={player} action={[s, t, f, o]}"
                                )


def test_invert_moves_still_round_trips_after_fast_path() -> None:
    # invert(decode(a)) must keep finding a zero-cost reproduction of decode(a).
    states = _rollout_states(2, 3, 12)
    state = states[-1]
    for action in ([0, 0, 2, 2], [1, 3, 1, 0], [0, 1, 3, 4]):
        moves = decode_discrete_action(state, 0, action, DEFAULT_DECODER_CONFIG)
        if not moves:
            continue
        result = invert_moves(state, 0, moves)
        assert result.quant_error == 0.0
