"""Collect a Producer/OEP imitation dataset for behavioral cloning (todo P1).

Runs deterministic local games driven by expert policies, records each expert
decision as a supervised example, and projects the expert's continuous moves onto
the compact PPO discrete action space via :mod:`action_inverse`. The projection is
lossy by construction, so every example carries its quantization error and the
report summarises the residual distribution.

Datasets produced (``--datasets``):
  - ``producer_only``     both sides driven by Producer; label = producer.
  - ``oep_only``          both sides driven by OEP; label = oep.
  - ``pgs_only``          both sides driven by PGS holdwave (operational config).
  - ``mahoraga_only``     both sides driven by the Mahoraga full2p config
                          (adaptive profiles + rescue/punish/hammer missions).
  - ``producer_oep_mix``  players alternate Producer/OEP; label = mover's expert.
  - ``producer_pgs_mix``  players alternate Producer/PGS.
  - ``oep_pgs_mix``       players alternate OEP/PGS.
  - ``hard_states``       Producer vs OEP; at states where the two experts'
                          projected actions disagree, emit one example per expert
                          (both labels) so contested decisions are oversampled.
  - ``hard_states_pgs``   same disagreement scheme for Producer vs PGS.

Stateful experts (producer/oep/pgs/mahoraga) get ONE isolated instance per player
seat so per-game memory (e.g. PGS opponent profiles) never leaks across seats; the
instance index == player index is also used for the hard-states probes, so each
probe sees a consistent per-seat history.

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
# ruff: noqa: E402,I001

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from python.agents.registry import (
    STATEFUL_SINGLETON_OPPONENTS,
    get_heuristic_policies,
    get_isolated_opponents,
)
from python.orbit_wars_gym.action_decoder import DEFAULT_DECODER_CONFIG, DecoderConfig
from python.orbit_wars_gym.action_inverse import (
    DEFAULT_INVERSE_CONFIG,
    InverseConfig,
    invert_moves,
)
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal
from python.orbit_wars_gym.symmetry import reflect_state_x, rotate_state_180

Policy = Callable[[dict[str, Any], int], list[list[float]]]
_ROOT = Path(__file__).resolve().parents[1]
_EXPERT_FILES = {
    "producer": ["bots/producer/agent.py", "bots/producer/_upstream.py"],
    "oep": ["bots/oep/agent.py", "bots/oep/planner.py"],
    # PGS rides on the Producer floor, so its behaviour hash covers both.
    "pgs": [
        "bots/pgs/agent.py",
        "bots/pgs/planner.py",
        "bots/producer/agent.py",
        "bots/producer/_upstream.py",
    ],
    "pgs_holdwave": [
        "bots/pgs/agent.py",
        "bots/pgs/planner.py",
        "bots/producer/agent.py",
        "bots/producer/_upstream.py",
    ],
    "pgs_bigwave": [
        "bots/pgs/agent.py",
        "bots/pgs/planner.py",
        "bots/producer/agent.py",
        "bots/producer/_upstream.py",
    ],
    # Mahoraga = PGSRuntime with _MAHORAGA_CONFIG (below); no agent.py of its own.
    "mahoraga": [
        "bots/pgs/planner.py",
        "bots/producer/agent.py",
        "bots/producer/_upstream.py",
    ],
}
_EXTERNAL_EXPERT_FILES = {
    "brep": Path.home() / "projects/Kaggle/orbit-wars-lab-B/artifacts/submission_brep.tar.gz",
}
_EXPERT_IDS = {
    "producer": 0,
    "oep": 1,
    "pgs": 2,
    "mahoraga": 3,
    "pgs_holdwave": 4,
    "pgs_bigwave": 5,
    "brep": 6,
}
_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
STRONG_EXPERT_POOL = ("producer", "pgs_holdwave", "brep", "pgs_bigwave", "oep", "rush", "greedy")
ELITE_EXPERT_POOL = ("producer", "pgs_holdwave", "brep", "pgs_bigwave", "oep")
DATASETS = (
    "producer_only",
    "oep_only",
    "pgs_only",
    "mahoraga_only",
    "producer_oep_mix",
    "producer_pgs_mix",
    "oep_pgs_mix",
    "hard_states",
    "hard_states_pgs",
    "league_strong_mix",
    "league_elite_mix",
)

# Single-expert datasets -> expert; pair datasets -> (even seats, odd seats).
_SINGLE = {
    "producer_only": "producer",
    "oep_only": "oep",
    "pgs_only": "pgs",
    "mahoraga_only": "mahoraga",
}
_PAIRS = {
    "producer_oep_mix": ("producer", "oep"),
    "producer_pgs_mix": ("producer", "pgs"),
    "oep_pgs_mix": ("oep", "pgs"),
    "hard_states": ("producer", "oep"),
    "hard_states_pgs": ("producer", "pgs"),
}

# Same config as the league's pgs_v3_adaptive_full2p (scripts/league_agents.py):
# operational holdwave profile + adaptive opponent profiles + rescue/punish/hammer
# missions — the live-code equivalent of the shipped Mahoraga submission.
_MAHORAGA_CONFIG: dict[str, Any] = {
    "scripts": "hold",
    "wave_min_ships": 60.0,
    "wave_start_step": 150,
    "floor_in_4p": True,
    "adaptive_mode": True,
    "adaptive_reply_models": True,
    "mission_mode": True,
    "enabled_missions": "rescue,punish,hammer",
    "max_mission_candidates": 8,
    "max_selected_missions": 1,
    "hammer_top_targets": 3,
    "hammer_top_sources": 4,
    "deadline_ms": 450.0,
    "deadline_guard_ms": 100.0,
    "value_mode": "scalar",
}


def _make_mahoraga_policy() -> Policy:
    """One fresh Mahoraga (PGS full2p) instance with per-game memory.

    Mirrors the reset-on-step-0 contract of the registry's isolated opponents so
    sequential games never share opponent profiles or mission state.
    """
    from bots.pgs.planner import PGSConfig, PGSRuntime
    from python.orbit_wars_gym.observation import to_official_observation

    cfg = PGSConfig(**_MAHORAGA_CONFIG)
    box: list[PGSRuntime] = [PGSRuntime(cfg)]

    def _policy(state: dict[str, Any], player: int) -> list[list[float]]:
        if int(state.get("step", 0)) == 0:
            box[0] = PGSRuntime(cfg)
        moves = box[0].act(to_official_observation(state, player=player))
        return list(moves) if isinstance(moves, list) else []

    return _policy


def _policy_table(experts: set[str], num_players: int) -> dict[str, list[Policy]]:
    """Per-(expert, seat) policy table: ``table[name][player]``.

    Stateful experts get one isolated instance per seat (sharing one singleton
    across seats would cross-contaminate per-game memory — e.g. PGS opponent
    profiles would see every fleet twice). Stateless heuristics share a callable.
    """
    table: dict[str, list[Policy]] = {}
    for name in experts:
        if name == "mahoraga":
            table[name] = [_make_mahoraga_policy() for _ in range(num_players)]
        elif name in STATEFUL_SINGLETON_OPPONENTS:
            table[name] = list(get_isolated_opponents(name, num_players))
        else:
            table[name] = [get_heuristic_policies()[name]] * num_players
    return table


def expert_content_hash(name: str) -> str:
    digest = hashlib.sha256()
    if name in _EXPERT_FILES:
        for rel in _EXPERT_FILES[name]:
            digest.update((_ROOT / rel).read_bytes())
    elif name in _EXTERNAL_EXPERT_FILES:
        path = _EXTERNAL_EXPERT_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"expert artifact for {name!r} is missing: {path}")
        digest.update(path.read_bytes())
    else:
        raise KeyError(name)
    return digest.hexdigest()


def split_for_seed(seed: int) -> str:
    bucket = int(seed) % 5
    if bucket <= 2:
        return "train"
    if bucket == 3:
        return "val"
    return "test"


def _player_experts(dataset: str, num_players: int, *, seed: int = 0) -> dict[int, str]:
    """Map each player index to the expert that drives it for a dataset."""
    if dataset in _SINGLE:
        return {p: _SINGLE[dataset] for p in range(num_players)}
    if dataset in ("league_strong_mix", "league_elite_mix"):
        pool = STRONG_EXPERT_POOL if dataset == "league_strong_mix" else ELITE_EXPERT_POOL
        offset = int(seed) % len(pool)
        return {p: pool[(offset + p) % len(pool)] for p in range(num_players)}
    even, odd = _PAIRS[dataset]
    return {p: (even if p % 2 == 0 else odd) for p in range(num_players)}


def _expert_policies() -> dict[str, Policy]:
    policies = dict(get_heuristic_policies())

    def league_policy(name: str) -> Policy:
        bot = None

        def policy(state: dict[str, Any], player: int) -> list[list[float]]:
            nonlocal bot
            if bot is None:
                from scripts.league_agents import make

                bot = make(name)
            moves = bot(to_official_observation(state, player=player))
            return list(moves) if isinstance(moves, list) else []

        return policy

    for name in ("pgs_holdwave", "pgs_bigwave", "brep"):
        policies[name] = league_policy(name)
    return policies


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

    def _row(
        enc_state: dict[str, Any], raw_moves: list[list[float]], is_aug: bool
    ) -> dict[str, Any]:
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
                [
                    m[0],
                    math.atan2(math.sin(angle_fn(float(m[1]))), math.cos(angle_fn(float(m[1])))),
                    m[2],
                ]
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
    launch_oversample: int = 1,
) -> list[dict[str, Any]]:
    player_experts = _player_experts(dataset, num_players)
    is_hard_dataset = dataset.startswith("hard_states")
    probe_pair = _PAIRS[dataset] if is_hard_dataset else None
    needed = set(player_experts.values()) | (set(probe_pair) if probe_pair else set())
    policies = _policy_table(needed, num_players)
    examples: list[dict[str, Any]] = []

    for seed in seeds:
        player_experts = _player_experts(dataset, num_players, seed=int(seed))
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
                moves = list(policies[expert][player](state, player))
                moves_by_player[player] = moves
                flat_rows.extend(_flat_rows(player, moves))

                if not is_hard_dataset:
                    rows = _example(
                        state=state, player=player, step=step, expert=expert,
                        moves=moves, seed=int(seed), is_hard=False,
                        decoder_cfg=decoder_cfg, inverse_cfg=inverse_cfg, augment=augment,
                    )
                    train_split = split_for_seed(int(seed)) == "train"
                    repeats = max(1, int(launch_oversample)) if moves and train_split else 1
                    for _ in range(repeats):
                        examples.extend(rows)

            if is_hard_dataset:
                # For every player perspective, ask BOTH experts of the pair;
                # record where the projected actions disagree (contested
                # decisions). The driving expert's moves are reused (no second
                # act() on the same state — stateful experts would double-step
                # their per-game memory); the other expert probes with its own
                # per-seat instance.
                assert probe_pair is not None
                for player in range(num_players):
                    driver = player_experts[player]
                    moves_by_expert: dict[str, list[list[float]]] = {
                        driver: moves_by_player[player]
                    }
                    for name in probe_pair:
                        if name not in moves_by_expert:
                            moves_by_expert[name] = list(policies[name][player](state, player))
                    acts = {
                        name: invert_moves(
                            state, player, mv, decoder_cfg=decoder_cfg, cfg=inverse_cfg
                        ).action
                        for name, mv in moves_by_expert.items()
                    }
                    first, second = probe_pair
                    if acts[first] == acts[second]:
                        continue
                    for name in probe_pair:
                        examples.extend(
                            _example(
                                state=state, player=player, step=step, expert=name,
                                moves=moves_by_expert[name], seed=int(seed), is_hard=True,
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
            if n and int((packed["expert_id"] == eid).sum()) > 0
        },
        "by_split": {
            sname: int((packed["split_id"] == sid).sum()) for sname, sid in _SPLIT_IDS.items()
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
        "step_histogram": np.bincount((packed["step"] // 16).astype(np.int64)).tolist()
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
    launch_oversample: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    used_experts = set()
    for dataset in datasets:
        if dataset == "league_strong_mix":
            used_experts.update(STRONG_EXPERT_POOL)
        elif dataset == "league_elite_mix":
            used_experts.update(ELITE_EXPERT_POOL)
        else:
            used_experts.update(_player_experts(dataset, num_players).values())
    used_experts = sorted(used_experts)
    if "hard_states" in datasets:
        used_experts = sorted(set(used_experts) | {"producer", "oep"})
    expert_hashes = {name: expert_content_hash(name) for name in used_experts}
    report: dict[str, Any] = {
        "seeds": seeds,
        "num_players": num_players,
        "episode_steps": episode_steps,
        "enable_comets": enable_comets,
        "augment": bool(augment),
        "launch_oversample": int(launch_oversample),
        "expert_hashes": expert_hashes,
        "mahoraga_config": dict(_MAHORAGA_CONFIG),
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
            launch_oversample=launch_oversample,
        )
        packed = _pack(examples)
        np.savez(out_dir / f"{name}.npz", **packed)
        meta = {
            "dataset": name,
            "content_hash": _content_hash(packed),
            "expert_hashes": expert_hashes,
            "mahoraga_config": dict(_MAHORAGA_CONFIG),
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
        "--decoder-max-moves-per-turn", type=int, default=DEFAULT_DECODER_CONFIG.max_moves_per_turn
    )
    parser.add_argument(
        "--decoder-min-ships-to-launch",
        type=int,
        default=DEFAULT_DECODER_CONFIG.min_ships_to_launch,
    )
    parser.add_argument(
        "--decoder-reserve-home-ships", type=int, default=DEFAULT_DECODER_CONFIG.reserve_home_ships
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="add 180°-rotation/reflection symmetry copies to the train split (same action label).",
    )
    parser.add_argument(
        "--launch-oversample",
        type=int,
        default=1,
        help="repeat non-empty train-split expert decisions this many times to counter pass-heavy datasets",
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
    decoder_cfg = DecoderConfig(
        fractions=DEFAULT_DECODER_CONFIG.fractions,
        angle_offsets=DEFAULT_DECODER_CONFIG.angle_offsets,
        max_moves_per_turn=int(args.decoder_max_moves_per_turn),
        min_ships_to_launch=int(args.decoder_min_ships_to_launch),
        reserve_home_ships=int(args.decoder_reserve_home_ships),
    )

    report = run(
        datasets=datasets,
        seeds=seeds,
        num_players=args.num_players,
        episode_steps=args.episode_steps,
        enable_comets=args.enable_comets,
        act_timeout=args.act_timeout,
        out_dir=out_dir,
        decoder_cfg=decoder_cfg,
        inverse_cfg=DEFAULT_INVERSE_CONFIG,
        augment=args.augment,
        launch_oversample=max(1, int(args.launch_oversample)),
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
            decoder_cfg=decoder_cfg,
            inverse_cfg=DEFAULT_INVERSE_CONFIG,
            augment=args.augment,
            launch_oversample=max(1, int(args.launch_oversample)),
        )
        for name in datasets:
            h1 = report["datasets"][name]["content_hash"]
            h2 = report2["datasets"][name]["content_hash"]
            assert h1 == h2, f"determinism check failed for {name}: {h1} != {h2}"
        print("self-check OK: content hashes reproducible")

    print(json.dumps(report["datasets"], indent=2))


if __name__ == "__main__":
    main()
