from __future__ import annotations

import numpy as np
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG, decode_discrete_action
from python.orbit_wars_gym.action_inverse import invert_moves, move_set_distance
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from scripts import collect_imitation_dataset as cid
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


def _stub_external_brep(monkeypatch) -> None:
    original = cid._expert_policies

    def policies():
        patched = original()
        patched["brep"] = patched["greedy"]
        return patched

    monkeypatch.setattr(cid, "_expert_policies", policies)


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


def test_pgs_only_is_deterministic_legal_and_labelled() -> None:
    a = collect_dataset("pgs_only", seeds=[0, 1], **_KW)
    b = collect_dataset("pgs_only", seeds=[0, 1], **_KW)
    packed_a, packed_b = _pack(a), _pack(b)
    assert _content_hash(packed_a) == _content_hash(packed_b)

    report = _dataset_report("pgs_only", packed_a)
    assert report["legal_action_rate"] == 1.0
    assert report["num_examples"] > 0
    # every example carries the pgs expert id (2)
    assert np.array_equal(packed_a["expert_id"], np.full_like(packed_a["expert_id"], 2))


def test_mahoraga_only_is_deterministic_and_labelled() -> None:
    a = collect_dataset("mahoraga_only", seeds=[0], **_KW)
    b = collect_dataset("mahoraga_only", seeds=[0], **_KW)
    packed_a, packed_b = _pack(a), _pack(b)
    assert _content_hash(packed_a) == _content_hash(packed_b)
    assert _dataset_report("mahoraga_only", packed_a)["legal_action_rate"] == 1.0
    assert np.array_equal(packed_a["expert_id"], np.full_like(packed_a["expert_id"], 3))


def test_hard_states_pgs_labels_come_from_the_pair() -> None:
    # At 16 steps PGS ≈ Producer floor, so disagreements may be empty; the
    # contract under test: any recorded example is hard and labelled with one
    # of the pair's expert ids (producer=0, pgs=2).
    examples = collect_dataset("hard_states_pgs", seeds=[0], **_KW)
    packed = _pack(examples)
    if packed["is_hard"].size:
        assert bool(packed["is_hard"].all())
        assert set(np.unique(packed["expert_id"]).tolist()) <= {0, 2}
