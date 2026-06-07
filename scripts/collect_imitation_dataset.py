"""Collect a Producer/OEP imitation dataset for behavioral cloning (todo P1).

Runs deterministic local games driven by expert policies, records each expert
decision as a supervised example, and projects the expert's continuous moves onto
the compact PPO discrete action space via :mod:`action_inverse`. The projection is
lossy by construction, so every example carries its quantization error and the
report summarises the residual distribution.

Datasets produced (``--datasets``):
  - ``producer_only``     both sides driven by Producer; label = producer.
  - ``oep_only``          both sides driven by OEP; label = oep.
  - ``producer_oep_mix``  players alternate Producer/OEP; label = mover's expert.
  - ``hard_states``       Producer vs OEP; at states where the two experts'
                          projected actions disagree, emit one example per expert
                          (both labels) so contested decisions are oversampled.

Splits are assigned by seed (``seed % 5`` -> train/val/test), never by row, to
avoid leaking states from the same game across splits.

Storage: one ``<name>.npz`` per dataset with fixed-size arrays plus a CSR-style
ragged encoding of raw moves, and a ``<name>.meta.json`` sidecar with the expert
content hashes and the decoder/inverse/encoder configs. A combined
``dataset_report.json`` aggregates distributions and quantization statistics.

Determinism: experts are deterministic and the backend is seeded, so re-running
with the same arguments reproduces byte-identical arrays (asserted by
``--self-check`` and by tests/test_collect_imitation_dataset.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from python.agents.registry import get_heuristic_policies
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG, DecoderConfig
from python.orbit_wars_gym.action_inverse import (
    DEFAULT_INVERSE_CONFIG,
    InverseConfig,
    invert_moves,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state
from python.orbit_wars_gym.rules import moves_are_legal
from python.orbit_wars_gym.symmetry import reflect_state_x, rotate_state_180

Policy = Callable[[dict[str, Any], int], list[list[float]]]

_ROOT = Path(__file__).resolve().parents[1]
_EXPERT_FILES = {
    "producer": ["bots/producer/agent.py", "bots/producer/_upstream.py"],
    "oep": ["bots/oep/agent.py", "bots/oep/planner.py"],
}
_EXPERT_IDS = {"producer": 0, "oep": 1}
_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
DATASETS = ("producer_only", "oep_only", "producer_oep_mix", "hard_states")


def expert_content_hash(name: str) -> str:
    digest = hashlib.sha256()
    for rel in _EXPERT_FILES[name]:
        digest.update((_ROOT / rel).read_bytes())
    return digest.hexdigest()


def split_for_seed(seed: int) -> str:
    bucket = int(seed) % 5
    if bucket <= 2:
        return "train"
    if bucket == 3:
        return "val"
    return "test"


def _player_experts(dataset: str, num_players: int) -> dict[int, str]:
    """Map each player index to the expert that drives it for a dataset."""
    if dataset in ("producer_only", "oep_only"):
        name = "producer" if dataset == "producer_only" else "oep"
        return {p: name for p in range(num_players)}
    # mix + hard_states: alternate Producer (even players) / OEP (odd players).
    return {p: ("producer" if p % 2 == 0 else "oep") for p in range(num_players)}


def _flat_rows(player: int, moves: list[list[float]]) -> list[list[float]]:
    return [[0.0, float(player), float(m[0]), float(m[1]), float(m[2])] for m in moves]


def _example(
    *,
    state: dict[str, Any],
    player: int,
    step: int,
    expert: str,
    moves: list[list[float]],
    seed: int,
    is_hard: bool,
    decoder_cfg: DecoderConfig,
    inverse_cfg: InverseConfig,
    augment: bool = False,
) -> list[dict[str, Any]]:
    """Return the example row(s) for one expert decision.

    Returns a single base row, plus (when ``augment`` and the example lands in the
    train split) symmetry-augmented rows. Board symmetries (180° rotation, vertical
    reflection) leave the discrete action indices invariant (P1.5), so each
    augmented row reuses the SAME action label with a re-encoded observation — this
    teaches spatial invariance without relabelling. Augmentation is restricted to
    the train split so val/test stay clean generalization probes; perspective is
    already symmetric (map-bias audit gap 0), so only spatial symmetries are added.
    """
    result = invert_moves(state, player, moves, decoder_cfg=decoder_cfg, cfg=inverse_cfg)
    split_id = _SPLIT_IDS[split_for_seed(seed)]
    action5 = np.asarray(result.action5, dtype=np.int64)
    legal = bool(moves_are_legal(state, player, moves))

    def _row(enc_state: dict[str, Any], raw_moves: list[list[float]], is_aug: bool) -> dict[str, Any]:
        return {
            "obs": encode_state(enc_state, player, DEFAULT_ENCODER_CONFIG).astype(np.float32),
            "player": int(player),
            "step": int(step),
            "expert_id": _EXPERT_IDS[expert],
            "seed": int(seed),
            "split_id": split_id,
            "action": action5,
            "quant_error": float(result.quant_error),
            "matched_moves": int(result.matched_moves),
            "num_expert_moves": int(len(moves)),
            "is_no_op": bool(result.is_no_op),
            "is_hard": bool(is_hard),
            "legal": legal,
            "is_aug": bool(is_aug),
            "raw_moves": np.asarray(raw_moves, dtype=np.float32).reshape(-1, 3),
        }

    rows = [_row(state, moves, False)]
    if augment and split_id == _SPLIT_IDS["train"]:
        for transform, angle_fn in (
            (rotate_state_180, lambda a: a + math.pi),
            (reflect_state_x, lambda a: math.pi - a),
        ):
            t_moves = [
                [m[0], math.atan2(math.sin(angle_fn(float(m[1]))), math.cos(angle_fn(float(m[1])))), m[2]]
                for m in moves
            ]
            rows.append(_row(transform(state), t_moves, True))
    return rows


def collect_dataset(
    dataset: str,
    *,
    seeds: list[int],
    num_players: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    decoder_cfg: DecoderConfig,
    inverse_cfg: InverseConfig,
    augment: bool = False,
) -> list[dict[str, Any]]:
    policies = get_heuristic_policies()
    player_experts = _player_experts(dataset, num_players)
    examples: list[dict[str, Any]] = []

    for seed in seeds:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=num_players,
            seed=int(seed),
            config=RustConfig(
                episode_steps=episode_steps,
                enable_comets=enable_comets,
                act_timeout=act_timeout,
            ),
        )
        state = backend.reset(int(seed))[0]
        for step in range(episode_steps):
            flat_rows: list[list[float]] = []
            moves_by_player: dict[int, list[float]] = {}
            for player in range(num_players):
                expert = player_experts[player]
                moves = list(policies[expert](state, player))
                moves_by_player[player] = moves
                flat_rows.extend(_flat_rows(player, moves))

                if dataset != "hard_states":
                    examples.extend(
                        _example(
                            state=state, player=player, step=step, expert=expert,
                            moves=moves, seed=int(seed), is_hard=False,
                            decoder_cfg=decoder_cfg, inverse_cfg=inverse_cfg, augment=augment,
                        )
                    )

            if dataset == "hard_states":
                # For every player perspective, ask BOTH experts; record where the
                # projected actions disagree (contested decisions).
                for player in range(num_players):
                    prod_moves = list(policies["producer"](state, player))
                    oep_moves = list(policies["oep"](state, player))
                    prod_act = invert_moves(
                        state, player, prod_moves, decoder_cfg=decoder_cfg, cfg=inverse_cfg
                    ).action
                    oep_act = invert_moves(
                        state, player, oep_moves, decoder_cfg=decoder_cfg, cfg=inverse_cfg
                    ).action
                    if prod_act == oep_act:
                        continue
                    examples.extend(
                        _example(
                            state=state, player=player, step=step, expert="producer",
                            moves=prod_moves, seed=int(seed), is_hard=True,
                            decoder_cfg=decoder_cfg, inverse_cfg=inverse_cfg, augment=augment,
                        )
                    )
                    examples.extend(
                        _example(
                            state=state, player=player, step=step, expert="oep",
                            moves=oep_moves, seed=int(seed), is_hard=True,
                            decoder_cfg=decoder_cfg, inverse_cfg=inverse_cfg, augment=augment,
                        )
                    )

            flat = (
                np.asarray(flat_rows, dtype=np.float64)
                if flat_rows
                else np.zeros((0, 5), dtype=np.float64)
            )
            outcomes, states = backend.step_flat_with_states(flat)
            state = states[0]
            if bool(outcomes[0].get("done", False)):
                break

    return examples


def _pack(examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not examples:
        return {
            "obs": np.zeros((0, 0), dtype=np.float32),
            "player": np.zeros((0,), dtype=np.int64),
            "step": np.zeros((0,), dtype=np.int64),
            "expert_id": np.zeros((0,), dtype=np.int64),
            "seed": np.zeros((0,), dtype=np.int64),
            "split_id": np.zeros((0,), dtype=np.int64),
            "action": np.zeros((0, 5), dtype=np.int64),
            "quant_error": np.zeros((0,), dtype=np.float32),
            "matched_moves": np.zeros((0,), dtype=np.int64),
            "num_expert_moves": np.zeros((0,), dtype=np.int64),
            "is_no_op": np.zeros((0,), dtype=np.bool_),
            "is_hard": np.zeros((0,), dtype=np.bool_),
            "is_aug": np.zeros((0,), dtype=np.bool_),
            "legal": np.zeros((0,), dtype=np.bool_),
            "raw_moves_flat": np.zeros((0, 3), dtype=np.float32),
            "raw_moves_offsets": np.zeros((1,), dtype=np.int64),
        }

    raw_flat = [ex["raw_moves"] for ex in examples]
    offsets = np.zeros(len(examples) + 1, dtype=np.int64)
    for i, rm in enumerate(raw_flat):
        offsets[i + 1] = offsets[i] + rm.shape[0]
    raw_moves_flat = (
        np.concatenate(raw_flat, axis=0)
        if any(rm.shape[0] for rm in raw_flat)
        else np.zeros((0, 3), dtype=np.float32)
    )
    return {
        "obs": np.stack([ex["obs"] for ex in examples]),
        "player": np.asarray([ex["player"] for ex in examples], dtype=np.int64),
        "step": np.asarray([ex["step"] for ex in examples], dtype=np.int64),
        "expert_id": np.asarray([ex["expert_id"] for ex in examples], dtype=np.int64),
        "seed": np.asarray([ex["seed"] for ex in examples], dtype=np.int64),
        "split_id": np.asarray([ex["split_id"] for ex in examples], dtype=np.int64),
        "action": np.stack([ex["action"] for ex in examples]),
        "quant_error": np.asarray([ex["quant_error"] for ex in examples], dtype=np.float32),
        "matched_moves": np.asarray([ex["matched_moves"] for ex in examples], dtype=np.int64),
        "num_expert_moves": np.asarray([ex["num_expert_moves"] for ex in examples], dtype=np.int64),
        "is_no_op": np.asarray([ex["is_no_op"] for ex in examples], dtype=np.bool_),
        "is_hard": np.asarray([ex["is_hard"] for ex in examples], dtype=np.bool_),
        "is_aug": np.asarray([ex["is_aug"] for ex in examples], dtype=np.bool_),
        "legal": np.asarray([ex["legal"] for ex in examples], dtype=np.bool_),
        "raw_moves_flat": raw_moves_flat.astype(np.float32),
        "raw_moves_offsets": offsets,
    }


def _max_bucket_share(values: np.ndarray, num_buckets: int) -> float:
    if values.size == 0:
        return 0.0
    counts = np.bincount(values.astype(np.int64), minlength=num_buckets)
    return float(counts.max() / values.size)


def _dataset_report(name: str, packed: dict[str, np.ndarray]) -> dict[str, Any]:
    n = int(packed["player"].shape[0])
    qe = packed["quant_error"]
    action = packed["action"]
    is_no_op = packed["is_no_op"]
    legal = packed["legal"]
    active = ~is_no_op  # action-head collapse is only meaningful for real launches
    # action layout is [launch, source, target, frac, offset]; launch == col 0.

    def _share(col: int, num_buckets: int) -> float:
        return _max_bucket_share(action[active, col], num_buckets) if active.any() else 0.0

    report: dict[str, Any] = {
        "num_examples": n,
        "num_aug": int(packed["is_aug"].sum()) if n else 0,
        "num_no_op": int(is_no_op.sum()),
        "no_op_rate": float(is_no_op.mean()) if n else 0.0,
        "launch_rate": float((action[:, 0] == 1).mean()) if n else 0.0,
        "legal_action_rate": float(legal.mean()) if n else 1.0,
        "hard_rate": float(packed["is_hard"].mean()) if n else 0.0,
        "by_expert": {
            ename: int((packed["expert_id"] == eid).sum())
            for ename, eid in _EXPERT_IDS.items()
        },
        "by_split": {
            sname: int((packed["split_id"] == sid).sum())
            for sname, sid in _SPLIT_IDS.items()
        },
        "max_bucket_share": {
            "source": _share(1, 16),
            "target": _share(2, 32),
            "frac": _share(3, 4),
            "offset": _share(4, 5),
        },
        "quant_error": {
            "mean": float(qe[active].mean()) if active.any() else 0.0,
            "p50": float(np.percentile(qe[active], 50)) if active.any() else 0.0,
            "p95": float(np.percentile(qe[active], 95)) if active.any() else 0.0,
            "max": float(qe[active].max()) if active.any() else 0.0,
        },
        "step_histogram": np.bincount(
            (packed["step"] // 16).astype(np.int64)
        ).tolist()
        if n
        else [],
    }
    return report


def _content_hash(packed: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(packed):
        arr = np.ascontiguousarray(packed[key])
        digest.update(key.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def run(
    *,
    datasets: list[str],
    seeds: list[int],
    num_players: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    out_dir: Path,
    decoder_cfg: DecoderConfig,
    inverse_cfg: InverseConfig,
    augment: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_hashes = {name: expert_content_hash(name) for name in _EXPERT_FILES}
    report: dict[str, Any] = {
        "seeds": seeds,
        "num_players": num_players,
        "episode_steps": episode_steps,
        "enable_comets": enable_comets,
        "augment": bool(augment),
        "expert_hashes": expert_hashes,
        "decoder_config": asdict(decoder_cfg),
        "inverse_config": asdict(inverse_cfg),
        "encoder_config": asdict(DEFAULT_ENCODER_CONFIG),
        "datasets": {},
    }
    for name in datasets:
        examples = collect_dataset(
            name,
            seeds=seeds,
            num_players=num_players,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
            decoder_cfg=decoder_cfg,
            inverse_cfg=inverse_cfg,
            augment=augment,
        )
        packed = _pack(examples)
        np.savez(out_dir / f"{name}.npz", **packed)
        meta = {
            "dataset": name,
            "content_hash": _content_hash(packed),
            "expert_hashes": expert_hashes,
            "decoder_config": asdict(decoder_cfg),
            "inverse_config": asdict(inverse_cfg),
            "encoder_config": asdict(DEFAULT_ENCODER_CONFIG),
            "seeds": seeds,
        }
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        ds_report = _dataset_report(name, packed)
        ds_report["content_hash"] = meta["content_hash"]
        report["datasets"][name] = ds_report

    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_seeds(raw: str) -> list[int]:
    if "-" in raw and "," not in raw:
        lo, hi = raw.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="0-7", help="e.g. '0-7' or '0,1,2'")
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--enable-comets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--out-dir", default="artifacts/imitation")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="add 180°-rotation/reflection symmetry copies to the train split (same action label).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="re-run each dataset and assert identical content hash (determinism).",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; valid: {DATASETS}")
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)

    report = run(
        datasets=datasets,
        seeds=seeds,
        num_players=args.num_players,
        episode_steps=args.episode_steps,
        enable_comets=args.enable_comets,
        act_timeout=args.act_timeout,
        out_dir=out_dir,
        decoder_cfg=DEFAULT_DECODER_CONFIG,
        inverse_cfg=DEFAULT_INVERSE_CONFIG,
        augment=args.augment,
    )

    if args.self_check:
        report2 = run(
            datasets=datasets,
            seeds=seeds,
            num_players=args.num_players,
            episode_steps=args.episode_steps,
            enable_comets=args.enable_comets,
            act_timeout=args.act_timeout,
            out_dir=out_dir / "_selfcheck",
            decoder_cfg=DEFAULT_DECODER_CONFIG,
            inverse_cfg=DEFAULT_INVERSE_CONFIG,
            augment=args.augment,
        )
        for name in datasets:
            h1 = report["datasets"][name]["content_hash"]
            h2 = report2["datasets"][name]["content_hash"]
            assert h1 == h2, f"determinism check failed for {name}: {h1} != {h2}"
        print("self-check OK: content hashes reproducible")

    print(json.dumps(report["datasets"], indent=2))


if __name__ == "__main__":
    main()
