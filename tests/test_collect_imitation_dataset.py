from __future__ import annotations

import numpy as np
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG, decode_discrete_action
from python.orbit_wars_gym.action_inverse import invert_moves, move_set_distance
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from scripts.collect_imitation_dataset import (
    _EXPERT_IDS,
    DEFAULT_INVERSE_CONFIG,
    ELITE_EXPERT_POOL,
    STRONG_EXPERT_POOL,
    _content_hash,
    _dataset_report,
    _pack,
    collect_dataset,
    split_for_seed,
)
from scripts.collect_imitation_dataset import (
    DEFAULT_DECODER_CONFIG as COLLECT_DECODER_CFG,
)

_KW = dict(
    num_players=2,
    episode_steps=16,
    enable_comets=False,
    act_timeout=1.0,
    decoder_cfg=COLLECT_DECODER_CFG,
    inverse_cfg=DEFAULT_INVERSE_CONFIG,
)


def _initial_state(seed: int = 0, num_players: int = 2) -> dict:
    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=16, enable_comets=False, act_timeout=1.0),
    )
    return backend.reset(seed)[0]


def test_inverse_roundtrip_recovers_equivalent_action() -> None:
    state = _initial_state()
    for act in ([0, 0, 2, 2], [1, 3, 1, 0], [0, 1, 3, 4]):
        moves = decode_discrete_action(state, 0, act, DEFAULT_DECODER_CONFIG)
        result = invert_moves(state, 0, moves)
        recovered = decode_discrete_action(state, 0, list(result.action), DEFAULT_DECODER_CONFIG)
        err, _ = move_set_distance(recovered, moves, state)
        assert err == 0.0  # recovered tuple reproduces the original move-set exactly


def test_invert_empty_moves_is_flagged_no_op() -> None:
    state = _initial_state()
    result = invert_moves(state, 0, [])
    assert result.is_no_op
    assert result.action == (0, 0, 0, 0)
    assert result.quant_error == 0.0


def test_collect_is_deterministic_and_legal() -> None:
    a = collect_dataset("producer_only", seeds=[0, 1], **_KW)
    b = collect_dataset("producer_only", seeds=[0, 1], **_KW)
    packed_a, packed_b = _pack(a), _pack(b)
    assert _content_hash(packed_a) == _content_hash(packed_b)

    report = _dataset_report("producer_only", packed_a)
    assert report["legal_action_rate"] == 1.0
    # is_no_op must coincide exactly with empty expert turns (no silent drops).
    assert np.array_equal(packed_a["is_no_op"], packed_a["num_expert_moves"] == 0)


def test_split_assignment_is_by_seed() -> None:
    assert split_for_seed(0) == "train"
    assert split_for_seed(3) == "val"
    assert split_for_seed(4) == "test"
    # same seed always lands in the same split (no per-row leakage)
    assert {split_for_seed(s) for s in (5, 10, 15)} == {"train"}


def test_hard_states_only_records_disagreements() -> None:
    examples = collect_dataset("hard_states", seeds=[0], **_KW)
    packed = _pack(examples)
    if packed["is_hard"].size:
        assert bool(packed["is_hard"].all())


def test_league_strong_mix_collects_pgs_and_brep_examples() -> None:
    kw = {**_KW, "num_players": 4, "episode_steps": 2}
    examples = collect_dataset("league_strong_mix", seeds=[0], **kw)
    packed = _pack(examples)
    report = _dataset_report("league_strong_mix", packed)

    assert _content_hash(packed) == _content_hash(_pack(collect_dataset("league_strong_mix", seeds=[0], **kw)))
    assert packed["legal"].all()
    assert {"producer", "pgs_holdwave", "brep", "pgs_bigwave"}.issubset(report["by_expert"])
    assert {
        _EXPERT_IDS["producer"],
        _EXPERT_IDS["pgs_holdwave"],
        _EXPERT_IDS["brep"],
        _EXPERT_IDS["pgs_bigwave"],
    }.issubset(set(packed["expert_id"].tolist()))


def test_league_strong_mix_rotates_all_declared_experts_across_seeds() -> None:
    kw = {**_KW, "num_players": 4, "episode_steps": 1}
    examples = collect_dataset("league_strong_mix", seeds=list(range(len(STRONG_EXPERT_POOL))), **kw)
    expert_ids = set(_pack(examples)["expert_id"].tolist())

    assert {_EXPERT_IDS[name] for name in STRONG_EXPERT_POOL}.issubset(expert_ids)


def test_league_elite_mix_uses_teacher_pool_without_greedy_rush() -> None:
    kw = {**_KW, "num_players": 4, "episode_steps": 1}
    examples = collect_dataset("league_elite_mix", seeds=list(range(len(ELITE_EXPERT_POOL))), **kw)
    expert_ids = set(_pack(examples)["expert_id"].tolist())

    assert {_EXPERT_IDS[name] for name in ELITE_EXPERT_POOL}.issubset(expert_ids)
    assert _EXPERT_IDS["greedy"] not in expert_ids
    assert _EXPERT_IDS["rush"] not in expert_ids


def test_launch_oversample_repeats_non_empty_decisions() -> None:
    base = _pack(collect_dataset("league_strong_mix", seeds=[0], num_players=4, episode_steps=2, launch_oversample=1, **{
        k: v for k, v in _KW.items() if k not in {"num_players", "episode_steps"}
    }))
    over = _pack(collect_dataset("league_strong_mix", seeds=[0], num_players=4, episode_steps=2, launch_oversample=3, **{
        k: v for k, v in _KW.items() if k not in {"num_players", "episode_steps"}
    }))

    assert over["obs"].shape[0] >= base["obs"].shape[0]
    assert int((over["action"][:, 0] == 1).sum()) >= int((base["action"][:, 0] == 1).sum())


def test_launch_oversample_does_not_repeat_validation_or_test_decisions() -> None:
    common = {
        k: v for k, v in _KW.items() if k not in {"num_players", "episode_steps"}
    }
    base = _pack(
        collect_dataset(
            "league_strong_mix",
            seeds=[3, 4],
            num_players=4,
            episode_steps=3,
            launch_oversample=1,
            **common,
        )
    )
    over = _pack(
        collect_dataset(
            "league_strong_mix",
            seeds=[3, 4],
            num_players=4,
            episode_steps=3,
            launch_oversample=4,
            **common,
        )
    )

    assert _content_hash(over) == _content_hash(base)
