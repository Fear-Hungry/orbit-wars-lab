"""Chunked campaign for the Frente B candidate_selector + dense_potential policy.

Each chunk runs, in SEPARATE processes:
  - train: ``python -m python.train.train_ppo --policy-arch candidate_selector
    --reward-mode dense_potential ...`` warm-started from the previous checkpoint,
  - eval: ``python -m scripts.eval_candidate_selector`` (in-process greedy eval; the
    candidate_selector is not exportable, so benchmark_ppo_submission can't be used).

Trains against producer,oep,greedy — the two strong archetypes (Producer economy + OEP
search, distinct styles) plus a cheap one — so the policy must beat BOTH, not overfit to
one (the goal's anti-overfitting requirement). The best checkpoint is kept by the WORST
of {margin vs producer, margin vs oep} (``min``), so a champion has to be good against
both. Held-out defensive/rush are evaluated too (generalization sanity), not used for
keep-best. Solves the single-run gaps: per-chunk checkpoints (no progress lost on a
crash) and a visible margin trajectory. Early-stops when min-margin stops improving.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.run_campaign import _wait_for_memory  # reuse the shared-box mem guard

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=str(_ROOT), stdout=fh, stderr=subprocess.STDOUT, check=True)


def _train_chunk(prev: str, ckpt: Path, args: argparse.Namespace, chunk: int, log: Path) -> None:
    cmd = [
        sys.executable, "-u", "-m", "python.train.train_ppo",
        "--policy-arch", "candidate_selector",
        "--reward-mode", "dense_potential",
        "--opponents", args.opponents,
        "--total-timesteps", str(args.chunk_timesteps),
        "--rollout-steps", str(args.rollout_steps),
        "--ent-coef", str(args.ent_coef),
        "--device", args.device,
        "--seed", str(args.seed + chunk),
        "--checkpoint-out", str(ckpt),
    ]
    if prev:
        cmd += ["--checkpoint-in", prev]
    _run(cmd, log)


def _eval_chunk(ckpt: Path, args: argparse.Namespace, log: Path, out_json: Path) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "scripts.eval_candidate_seats",  # SEAT-NEUTRAL (honest); not the biased seat-fixed eval
        "--checkpoint", str(ckpt),
        "--opponents", *args.eval_opponents.split(","),
        "--seeds", str(args.eval_seeds),
        "--episode-steps", str(args.eval_episode_steps),
        "--json",
    ]
    result = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=True)
    log.write_text(result.stdout + "\n---STDERR---\n" + result.stderr, encoding="utf-8")
    report = json.loads(result.stdout)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    margins = {r["opponent"]: float(r["mean_score_margin"]) for r in report["per_opponent"]}
    wins = {r["opponent"]: float(r.get("win_rate", 0.0)) for r in report["per_opponent"]}
    return {"margins": margins, "wins": wins}


def _keep_best_score(margins: dict[str, float]) -> float:
    """Champion criterion: must be good against BOTH strong archetypes -> the worst of
    {producer, oep}. Falls back to producer alone if oep was not evaluated."""
    if "producer" in margins and "oep" in margins:
        return min(margins["producer"], margins["oep"])
    return margins.get("producer", float("-inf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="", help="warm-start checkpoint (e.g. the 50k/150k candidate ckpt)")
    parser.add_argument("--out-dir", default="artifacts/ppo/frente_b/campaign")
    parser.add_argument("--opponents", default="producer,oep,greedy", help="TRAIN opponents")
    parser.add_argument("--eval-opponents", default="producer,oep,defensive,rush")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="entropy bonus; raise (e.g. 0.03) to sustain exploration of beneficial "
                             "deviations from the producer-parity baseline (B4 collapsed to entropy 0.13)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--eval-episode-steps", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "campaign_log.jsonl"

    prev = args.init
    best_score = float("-inf")
    no_improve = 0
    history: list[dict[str, Any]] = []

    for chunk in range(args.chunks):
        ckpt = out_dir / f"c{chunk:02d}.pt"
        _wait_for_memory(args.min_free_gb, chunk)
        try:
            _train_chunk(prev, ckpt, args, chunk, out_dir / f"train_c{chunk:02d}.log")
            ev = _eval_chunk(ckpt, args, out_dir / f"eval_c{chunk:02d}.log", out_dir / f"eval_c{chunk:02d}.json")
        except subprocess.CalledProcessError as exc:
            (out_dir / "errors.jsonl").open("a").write(json.dumps({"chunk": chunk, "error": str(exc)}) + "\n")
            print(f"chunk {chunk} FAILED: {exc}", flush=True)
            break

        score = _keep_best_score(ev["margins"])
        record = {
            "chunk": chunk,
            "cumulative_timesteps": (chunk + 1) * args.chunk_timesteps,
            "keep_best_score": score,
            "margins": ev["margins"],
            "wins": ev["wins"],
            "checkpoint": str(ckpt),
        }
        history.append(record)
        with log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        improved = score > best_score
        if improved:
            best_score = score
            shutil.copyfile(ckpt, out_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"chunk {chunk}: ts={record['cumulative_timesteps']} "
            f"min(prod,oep)={score:.4f} margins={ev['margins']} best={best_score:.4f} no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"early stop: no improvement for {args.patience} chunks", flush=True)
            break
        # continue from the best so a regressed chunk doesn't propagate
        prev = str(out_dir / "best.pt") if (out_dir / "best.pt").exists() else str(ckpt)

    (out_dir / "campaign_report.json").write_text(
        json.dumps({"best_score": best_score, "history": history}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
