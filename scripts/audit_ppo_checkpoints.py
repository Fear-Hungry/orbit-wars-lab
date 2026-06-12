"""Audit existing PPO/BC checkpoints before spending new training compute."""

from __future__ import annotations
# ruff: noqa: E402,I001

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.drl_promotion_gate import run_drl_promotion_gate
from scripts.ppo_export_parity import check_checkpoint_export_parity
from scripts.select_ppo_checkpoint import _expand_checkpoints, checkpoint_candidate_id

_VERDICT_PRIORITY = {"PASS_LOCAL": 2, "INCONCLUSIVE": 1, "REJECT_LOCAL": 0, "NOT_RUN": -1}


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "checkpoint": str(path),
            "load_error": repr(exc),
            "valid_checkpoint": False,
        }
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return {
            "checkpoint": str(path),
            "valid_checkpoint": False,
            "load_error": "missing model_state_dict",
        }
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    return {
        "checkpoint": str(path),
        "valid_checkpoint": True,
        "arch": str(summary.get("arch", "flat")),
        "policy_track": config.get("policy_track"),
        "num_players": config.get("num_players"),
        "total_timesteps": config.get("total_timesteps"),
        "opponents": list(config.get("opponents", [])) if isinstance(config.get("opponents"), (list, tuple)) else config.get("opponents"),
        "has_optimizer_state": "optimizer_state_dict" in checkpoint,
    }


def _candidate_row_for_checkpoint(gate_report: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    candidate_id = f"ppo_{checkpoint_candidate_id(checkpoint)}"
    for row in gate_report.get("ranking", []):
        if row.get("candidate") == candidate_id:
            return dict(row)
    return {
        "candidate": candidate_id,
        "verdict": "NOT_RUN",
        "overall_score": 0.0,
        "score_2p": 0.0,
        "score_4p": 0.0,
    }


def audit_ppo_checkpoints(
    checkpoint_patterns: list[str],
    *,
    out_dir: Path,
    max_checkpoints: int | None,
    parity_seeds: int,
    parity_steps: int,
    run_gate: bool,
    gate_profile: str,
    gate_seeds: int | None,
    gate_steps: int | None,
    gate_jobs: int,
) -> dict[str, Any]:
    checkpoints = _expand_checkpoints(checkpoint_patterns)
    if max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, int(max_checkpoints))]
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for idx, checkpoint in enumerate(checkpoints):
        cid = checkpoint_candidate_id(checkpoint)
        metadata = _checkpoint_metadata(checkpoint)
        record: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "candidate_id": f"ppo_{cid}",
            "metadata": metadata,
            "export_parity": None,
            "gate_report": None,
            "gate_row": {
                "candidate": f"ppo_{cid}",
                "verdict": "NOT_RUN",
                "overall_score": 0.0,
                "score_2p": 0.0,
                "score_4p": 0.0,
            },
        }
        if metadata.get("valid_checkpoint"):
            parity_path = out_dir / "export_parity" / f"{cid}.json"
            try:
                parity = check_checkpoint_export_parity(
                    checkpoint,
                    submission_path=out_dir / "submissions" / f"ppo_{cid}.py",
                    seeds=list(range(max(1, int(parity_seeds)))),
                    steps=int(parity_steps),
                    player_counts=(2, 4),
                )
            except Exception as exc:
                parity = {
                    "checkpoint": str(checkpoint),
                    "passed": False,
                    "checked_observations": 0,
                    "mismatches": [],
                    "error": repr(exc),
                }
            parity_path.parent.mkdir(parents=True, exist_ok=True)
            parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8")
            record["export_parity"] = {
                "path": str(parity_path),
                "passed": bool(parity.get("passed")),
                "checked_observations": int(parity.get("checked_observations", 0)),
                "error": parity.get("error"),
                "mismatches": parity.get("mismatches", [])[:1],
            }
            if run_gate and parity.get("passed"):
                gate_dir = out_dir / "gates" / cid
                gate_report = run_drl_promotion_gate(
                    checkpoint_patterns=[str(checkpoint)],
                    league_candidates=[],
                    out_dir=gate_dir,
                    profile=gate_profile,
                    seeds=gate_seeds,
                    seed_base=300_000 + 10_000 * idx,
                    steps=gate_steps,
                    jobs=gate_jobs,
                    match_chunk_size=0,
                    skip_run=False,
                    required_2p_threshold=0.50,
                    min_decisive_2p=None,
                    min_producer_winrate=0.50,
                    min_incumbent_winrate=0.50,
                    min_floor_winrate=0.60,
                    max_annihilation_rate_4p=0.35,
                    weight_2p=0.46,
                    skip_parity=True,
                )
                gate_report_path = gate_dir / "report.json"
                gate_report_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True), encoding="utf-8")
                record["gate_report"] = str(gate_report_path)
                record["gate_row"] = _candidate_row_for_checkpoint(gate_report, checkpoint)
        candidates.append(record)

    candidates.sort(
        key=lambda item: (
            _VERDICT_PRIORITY.get(str(item["gate_row"].get("verdict")), -1),
            bool((item.get("export_parity") or {}).get("passed")),
            float(item["gate_row"].get("overall_score", 0.0)),
            float(item["gate_row"].get("score_4p", 0.0)),
            float(item["gate_row"].get("score_2p", 0.0)),
        ),
        reverse=True,
    )
    return {
        "checkpoint_patterns": checkpoint_patterns,
        "checkpoint_count": len(checkpoints),
        "run_gate": run_gate,
        "parity_seeds": parity_seeds,
        "parity_steps": parity_steps,
        "gate_profile": gate_profile,
        "gate_seeds": gate_seeds,
        "gate_steps": gate_steps,
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        default=["artifacts/ppo/**/*.pt", "artifacts/bc/*.pt"],
        help="checkpoint path or glob; defaults to PPO and BC artifact directories",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ppo/audit"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--parity-seeds", type=int, default=2)
    parser.add_argument("--parity-steps", type=int, default=16)
    parser.add_argument("--run-gate", action="store_true")
    parser.add_argument("--gate-profile", choices=("quick", "standard", "strong"), default="quick")
    parser.add_argument("--gate-seeds", type=int, default=None)
    parser.add_argument("--gate-steps", type=int, default=None)
    parser.add_argument("--gate-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    report = audit_ppo_checkpoints(
        list(args.checkpoints),
        out_dir=args.out_dir,
        max_checkpoints=args.max_checkpoints,
        parity_seeds=int(args.parity_seeds),
        parity_steps=int(args.parity_steps),
        run_gate=bool(args.run_gate),
        gate_profile=str(args.gate_profile),
        gate_seeds=args.gate_seeds,
        gate_steps=args.gate_steps,
        gate_jobs=int(args.gate_jobs),
    )
    out = args.out or (args.out_dir / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "checkpoint_count": report["checkpoint_count"],
                "top": [
                    {
                        "checkpoint": item["checkpoint"],
                        "parity": (item.get("export_parity") or {}).get("passed"),
                        "verdict": item["gate_row"]["verdict"],
                        "score": item["gate_row"]["overall_score"],
                    }
                    for item in report["candidates"][:5]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
