"""Check PyTorch PPO checkpoint vs exported pure-Python submission parity."""

from __future__ import annotations
# ruff: noqa: E402,I001

import argparse
import contextlib
import io
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_wars_gym.action_decoder import DecoderConfig, decode_discrete_action
from orbit_wars_gym.action_masks import build_action_masks, split_masks
from orbit_wars_gym.backend import RustBatchBackend, RustConfig
from orbit_wars_gym.encoding import encode_state, observation_dim
from python.agents.registry import get_heuristic_policies
from python.orbit_wars_gym.observation import to_official_observation
from python.train.train_ppo import _build_policy
from scripts.export_submission import _decoder_payload, render_submission


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"ppo_export_parity_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load exported submission from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_submission_tarball(submission_path: Path, tarball_path: Path) -> None:
    source = submission_path.read_text(encoding="utf-8")
    data = source.encode("utf-8")
    info = tarfile.TarInfo("main.py")
    info.size = len(data)
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))


@contextlib.contextmanager
def _loaded_tarball_module(tarball_path: Path):
    with tempfile.TemporaryDirectory(prefix="ppo_export_parity_") as tmp_name:
        tmp = Path(tmp_name)
        with tarfile.open(tarball_path) as tar:
            names = {member.name for member in tar.getmembers()}
            if "main.py" not in names:
                raise ValueError(f"PPO submission tarball is missing main.py: {tarball_path}")
            tar.extractall(tmp, filter="data")
        inserted = str(tmp) not in sys.path
        if inserted:
            sys.path.insert(0, str(tmp))
        try:
            yield _load_module(tmp / "main.py")
        finally:
            if inserted:
                sys.path.remove(str(tmp))


def _checkpoint_arch(checkpoint: dict[str, Any]) -> str:
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
    return str(summary.get("arch", "flat"))


def _load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid PPO checkpoint: {checkpoint_path}")
    arch = _checkpoint_arch(checkpoint)
    model = _build_policy(arch, observation_dim())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _decoder_config(payload: dict[str, Any]) -> DecoderConfig:
    decoder = dict(payload)
    return DecoderConfig(
        fractions=tuple(float(value) for value in decoder.get("fractions", [0.10, 0.25, 0.50, 0.75])),
        angle_offsets=tuple(
            float(value)
            for value in decoder.get("angle_offsets", [-0.261799, -0.130899, 0.0, 0.130899, 0.261799])
        ),
        max_moves_per_turn=int(decoder.get("max_moves_per_turn", 8)),
        min_ships_to_launch=int(decoder.get("min_ships_to_launch", 2)),
        reserve_home_ships=int(decoder.get("reserve_home_ships", 8)),
    )


def _local_action(model, obs: dict[str, Any], player: int, decoder_cfg: DecoderConfig) -> list[int]:
    encoded = torch.as_tensor(encode_state(obs, player), dtype=torch.float32).unsqueeze(0)
    mask_np = build_action_masks(
        obs,
        player,
        min_ships_to_launch=int(decoder_cfg.min_ships_to_launch),
    )
    masks = {key: torch.as_tensor(value).unsqueeze(0) for key, value in split_masks(mask_np).items()}
    with torch.no_grad():
        out = model(encoded)

    def masked_argmax(key: str) -> int:
        logits = out[key].clone()
        if key in masks:
            logits = logits.masked_fill(~masks[key].bool(), float("-inf"))
        return int(torch.argmax(logits, dim=-1).item())

    return [
        masked_argmax("launch"),
        masked_argmax("source"),
        masked_argmax("target"),
        int(torch.argmax(out["frac"], dim=-1).item()),
        int(torch.argmax(out["offset"], dim=-1).item()),
    ]


def _moves_close(left: list[list[float]], right: list[list[float]], *, atol: float = 1e-6) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if len(a) != len(b):
            return False
        if int(a[0]) != int(b[0]) or int(a[2]) != int(b[2]):
            return False
        if abs(float(a[1]) - float(b[1])) > atol:
            return False
    return True


_DEGRADATION_STATS = {
    "fallbacks",
    "timeouts",
    "timeout_thread_blocks",
    "fallback_errors",
    "illegal_moves",
    "policy_illegal_moves",
    "invalid_actions",
}


def _submission_stats_snapshot(module: Any) -> dict[str, int]:
    stats = getattr(module, "SUBMISSION_STATS", None)
    if not isinstance(stats, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in stats.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _degradation_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    bad: dict[str, int] = {}
    for key, value in after.items():
        if key in _DEGRADATION_STATS or key.endswith("_fallbacks"):
            delta = int(value) - int(before.get(key, 0))
            if delta > 0:
                bad[key] = delta
    return bad


def _snapshot_states(*, num_players: int, seeds: list[int], steps: int) -> list[dict[str, Any]]:
    policies = get_heuristic_policies()
    driver = policies["greedy"]
    states: list[dict[str, Any]] = []
    sample_steps = {0, max(0, steps // 2), max(0, steps)}
    for seed in seeds:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=num_players,
            seed=int(seed),
            config=RustConfig(episode_steps=max(1, int(steps)), enable_comets=True),
        )
        state = backend.reset(int(seed))[0]
        for step in range(max(0, int(steps)) + 1):
            if step in sample_steps:
                states.append(state)
            if step >= int(steps):
                break
            actions = [driver(state, player) for player in range(num_players)]
            outcomes, next_states = backend.step_with_states([actions])
            state = next_states[0]
            if outcomes[0].get("done"):
                break
    return states


def check_checkpoint_export_parity(
    checkpoint_path: Path,
    *,
    submission_path: Path,
    tarball_path: Path | None = None,
    seeds: list[int],
    steps: int,
    player_counts: tuple[int, ...] = (2, 4),
) -> dict[str, Any]:
    checkpoint, model = _load_model(checkpoint_path)
    decoder_cfg = _decoder_config(_decoder_payload(checkpoint))
    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(render_submission(template, checkpoint=str(checkpoint_path)), encoding="utf-8")
    module = _load_module(submission_path)
    tarball_path = tarball_path or submission_path.with_suffix(".tar.gz")
    _write_submission_tarball(submission_path, tarball_path)

    mismatches: list[dict[str, Any]] = []
    checked = 0
    with _loaded_tarball_module(tarball_path) as tarball_module:
        for num_players in player_counts:
            for state in _snapshot_states(num_players=int(num_players), seeds=seeds, steps=steps):
                for player in range(int(num_players)):
                    obs = to_official_observation(state, player)
                    local_action = _local_action(model, obs, player, decoder_cfg)
                    exported_action = [int(value) for value in module._neural_action(obs, player)]
                    local_moves = decode_discrete_action(obs, player, local_action, decoder_cfg)
                    exported_moves = module._neural_decode(obs, player, exported_action)
                    tarball_stats_before = _submission_stats_snapshot(tarball_module)
                    tarball_moves = [list(move) for move in tarball_module.agent(obs)]
                    tarball_bad_stats = _degradation_delta(
                        tarball_stats_before,
                        _submission_stats_snapshot(tarball_module),
                    )
                    checked += 1
                    if (
                        local_action != exported_action
                        or not _moves_close(local_moves, exported_moves)
                        or not _moves_close(local_moves, tarball_moves)
                        or tarball_bad_stats
                    ):
                        mismatches.append(
                            {
                                "num_players": int(num_players),
                                "player": int(player),
                                "step": int(obs.get("step", 0)),
                                "local_action": local_action,
                                "exported_action": exported_action,
                                "local_moves": local_moves,
                                "exported_moves": exported_moves,
                                "tarball_moves": tarball_moves,
                                "tarball_degradation_stats": tarball_bad_stats,
                            }
                        )
                        break
                if mismatches:
                    break
            if mismatches:
                break

    return {
        "checkpoint": str(checkpoint_path),
        "submission": str(submission_path),
        "tarball": str(tarball_path),
        "seeds": [int(seed) for seed in seeds],
        "steps": int(steps),
        "player_counts": [int(value) for value in player_counts],
        "checked_observations": checked,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission-out", default="artifacts/ppo/export_parity/submission.py")
    parser.add_argument("--tarball-out", default=None)
    parser.add_argument("--out", default="artifacts/ppo/export_parity/report.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--players", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args(argv)

    report = check_checkpoint_export_parity(
        Path(args.checkpoint),
        submission_path=Path(args.submission_out),
        tarball_path=Path(args.tarball_out) if args.tarball_out else None,
        seeds=list(range(max(1, int(args.seeds)))),
        steps=int(args.steps),
        player_counts=tuple(int(value) for value in args.players),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": report["passed"],
                "checked": report["checked_observations"],
                "tarball": report["tarball"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("PPO export parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
