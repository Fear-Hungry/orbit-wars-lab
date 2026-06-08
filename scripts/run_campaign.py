"""Unattended eval-gated PPO campaign via isolated SUBPROCESSES (todo P3).

Each chunk runs training and evaluation in SEPARATE processes:

  - train: ``python -m python.train.train_ppo ... --device cuda`` writes a
    checkpoint, then the process exits (releasing the CUDA context).
  - eval:  ``python -m scripts.benchmark_ppo_submission ...`` in a FRESH process
    (no CUDA) benchmarks the checkpoint vs the ruler with ``--jobs`` parallelism.

This avoids the fork-after-CUDA deadlock that hangs an in-process orchestrator
(ProcessPoolExecutor fork after torch initialised CUDA). The best checkpoint by
paired margin is kept; the run early-stops when the margin stops improving, so a
long campaign can run overnight without supervision and without the blind-scaling
regression (extending a single run was measured to *worsen* the margin).

Progress is appended to ``<out_dir>/campaign_log.jsonl`` and summarised in
``<out_dir>/campaign_report.json``. Promotion to a Kaggle submission stays a human
96-seed decision; this only finds/keeps the best local checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def _mem_available_gb() -> float:
    """Host MemAvailable in GiB; inf if unreadable (never block on a parse miss)."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return float("inf")


def _wait_for_memory(min_free_gb: float, chunk: int) -> None:
    """Block until MemAvailable clears the floor, so a shared-box overcommit
    parks the chunk instead of tripping the global OOM-killer (diagnosed
    2026-06-07: train_ppo itself peaks ~1.6GB; the OOM was external pressure)."""
    avail = _mem_available_gb()
    while avail < min_free_gb:
        print(f"low mem ({avail:.1f}GB < {min_free_gb}GB) — "
              f"aguardando 60s antes do chunk {chunk}", flush=True)
        time.sleep(60)
        avail = _mem_available_gb()


def _accrue_no_improve(margin: float, best_margin: float, no_improve: int, floor_margin: float) -> int:
    """Updated early-stop counter. Reset to 0 on improvement; increment ONLY once the
    best has escaped the loss floor — a tie at the floor is 'not winning yet', not a
    converged plateau, so counting it would early-stop before cumulative training can
    climb out (diagnosed 2026-06-07). Call with best_margin BEFORE any improvement update."""
    if margin > best_margin:
        return 0
    if best_margin > floor_margin:
        return no_improve + 1
    return no_improve


def _reset_to_best(margin: float, best_margin: float, regress_reset: float, has_best: bool) -> bool:
    """Whether the next chunk restarts from best.pt (anti-drift on a REAL regression)
    instead of continuing cumulatively from the last chunk. A tie at the floor must NOT
    reset, else eval-gating freezes cold-start progress (diagnosed 2026-06-07)."""
    return has_best and margin < best_margin - regress_reset


def _run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(_ROOT), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")


def _train_chunk(prev: str, ckpt: Path, args: argparse.Namespace, chunk: int, log: Path) -> None:
    cmd = [
        sys.executable, "-m", "python.train.train_ppo",
        "--opponents", args.opponents,
        "--checkpoint-in", prev,
        "--checkpoint-out", str(ckpt),
        "--device", args.device,
        "--rollout-num-envs", str(args.rollout_num_envs),
        "--rollout-steps", str(args.rollout_steps),
        "--ent-coef", str(args.ent_coef),
        "--total-timesteps", str(args.chunk_timesteps),
        "--seed", str(args.seed + chunk),
    ]
    _run(cmd, log)


def _eval_chunk(ckpt: Path, args: argparse.Namespace, log: Path, out_json: Path) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "scripts.benchmark_ppo_submission",
        "--checkpoint", str(ckpt),
        "--submission-out", str(ckpt.with_suffix(".sub.py")),
        "--out", str(out_json),
        "--opponents", *args.eval_opponents.split(","),
        "--seeds", str(args.eval_seeds),
        "--episode-steps", str(args.eval_episode_steps),
        "--skip-4p",
        "--jobs", str(args.eval_jobs),
    ]
    _run(cmd, log)
    return json.loads(out_json.read_text(encoding="utf-8"))["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, help="initial checkpoint (e.g. a BC checkpoint)")
    parser.add_argument("--out-dir", default="artifacts/ppo/campaign")
    parser.add_argument(
        "--opponents",
        default="producer,producer,producer,producer,producer,producer,producer,greedy,greedy,oep",
    )
    parser.add_argument("--eval-opponents", default="producer,oep")
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--chunk-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollout-num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--eval-episode-steps", type=int, default=256)
    # Sequential eval: the ProcessPool (fork OR spawn) deadlocks in this WSL env
    # after torch is loaded; the eval is cheap enough sequentially (~5 min/16 seeds).
    parser.add_argument("--eval-jobs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=4.0,
                        help="park the chunk while MemAvailable is below this (anti-OOM on a shared box)")
    parser.add_argument("--regress-reset", type=float, default=0.05,
                        help="reset training to best.pt only if a chunk's margin drops more than this below best (else continue cumulatively from the last chunk)")
    parser.add_argument("--floor-margin", type=float, default=-0.99,
                        help="don't accrue early-stop patience while best margin is still at/below this loss floor (cold-start ties are not a converged plateau)")
    args = parser.parse_args()

    out_dir = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "campaign_log.jsonl"

    prev = args.init
    best_margin = float("-inf")
    best_path: Path | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    consecutive_failures = 0
    for chunk in range(args.chunks):
        ckpt = out_dir / f"c{chunk:02d}.pt"
        train_log = out_dir / f"train_c{chunk:02d}.log"
        _wait_for_memory(args.min_free_gb, chunk)
        try:
            _train_chunk(prev, ckpt, args, chunk, train_log)
            summary = _eval_chunk(ckpt, args, out_dir / f"eval_c{chunk:02d}.log", out_dir / f"eval_c{chunk:02d}.json")
        except Exception as exc:  # noqa: BLE001 - keep the overnight run alive
            consecutive_failures += 1
            tail = ""
            try:
                tail = train_log.read_text(encoding="utf-8")[-1500:]
            except OSError:
                pass
            err = {"chunk": chunk, "error": str(exc), "train_log_tail": tail,
                   "consecutive_failures": consecutive_failures}
            with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(err) + "\n")
            print(f"chunk {chunk} FAILED ({consecutive_failures} em seguida): {exc}", flush=True)
            if consecutive_failures >= 3:
                print("abortando: 3 falhas de chunk seguidas", flush=True)
                break
            continue  # retry next chunk from the last good checkpoint (prev unchanged)
        consecutive_failures = 0
        margin = float(summary["mean_score_margin"])
        record = {
            "chunk": chunk,
            "cumulative_timesteps": (chunk + 1) * args.chunk_timesteps,
            "margin": margin,
            "win_rate": float(summary["win_rate"]),
            "invalid_action_rate": float(summary["invalid_action_rate"]),
            "crash_rate": float(summary["crash_rate"]),
            "checkpoint": str(ckpt),
        }
        history.append(record)
        with log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        improved = margin > best_margin
        no_improve = _accrue_no_improve(margin, best_margin, no_improve, args.floor_margin)
        if improved:
            best_margin = margin
            best_path = ckpt
            shutil.copyfile(ckpt, out_dir / "best.pt")
        print(
            f"chunk {chunk}: ts={record['cumulative_timesteps']} margin={margin:.4f} "
            f"win={record['win_rate']:.3f} best={best_margin:.4f} no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"early stop: no margin improvement for {args.patience} chunks", flush=True)
            break
        # Cumulative training: continue from the LAST chunk so timesteps accumulate
        # (cold-start needs millions of continuous ts to escape the loss floor; the
        # old "restart from best" froze progress when every chunk tied at -1.0).
        # Anti-drift guard: fall back to best ONLY on a real regression (margin drops
        # below best by more than --regress-reset), not on a tie at the floor.
        if _reset_to_best(margin, best_margin, args.regress_reset, best_path is not None):
            prev = str(out_dir / "best.pt")
            print(f"regression {margin:.4f} < best {best_margin:.4f} - {args.regress_reset} "
                  f"-> reset to best", flush=True)
        else:
            prev = str(ckpt)

    report = {
        "init": args.init,
        "opponents": args.opponents,
        "eval_opponents": args.eval_opponents,
        "ent_coef": args.ent_coef,
        "rollout_num_envs": args.rollout_num_envs,
        "device": args.device,
        "best_margin": best_margin,
        "best_checkpoint": str(out_dir / "best.pt") if best_path is not None else None,
        "best_source": str(best_path) if best_path is not None else None,
        "history": history,
    }
    (out_dir / "campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
