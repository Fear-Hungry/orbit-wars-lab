from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


def _parse_ints(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    games = 0
    decision_turns = 0.0
    policy_seconds = 0.0
    for item in report.get("formats", []):
        if item.get("format") == "2p":
            for opponent in item.get("opponents", []):
                records = opponent.get("records", [])
                games += len(records)
                decision_turns += sum(
                    float(record.get("decision_turns", 0.0)) for record in records
                )
                policy_seconds += sum(
                    float(record.get("elapsed_seconds", 0.0)) for record in records
                )
        else:
            records = item.get("records", [])
            games += len(records)
            decision_turns += sum(float(record.get("decision_turns", 0.0)) for record in records)
            policy_seconds += sum(float(record.get("elapsed_seconds", 0.0)) for record in records)
    return {
        "games": games,
        "decision_turns": decision_turns,
        "policy_seconds": policy_seconds,
        "mean_policy_ms": 1000.0 * policy_seconds / max(decision_turns, 1.0),
    }


def _run_case(args: argparse.Namespace, seeds: int, jobs: int, out_dir: Path) -> dict[str, Any]:
    report_path = out_dir / f"throughput_seeds{seeds}_jobs{jobs}.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.benchmark_submission",
        "--submission",
        str(args.submission),
        "--opponents",
        *args.opponents,
        "--seeds",
        str(seeds),
        "--episode-steps",
        str(args.episode_steps),
        "--jobs",
        str(jobs),
        "--out",
        str(report_path),
    ]
    if args.skip_4p:
        cmd.append("--skip-4p")
    if args.skip_2p:
        cmd.append("--skip-2p")

    start = perf_counter()
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    wall_seconds = perf_counter() - start
    result: dict[str, Any] = {
        "seeds": seeds,
        "jobs": jobs,
        "episode_steps": int(args.episode_steps),
        "returncode": int(completed.returncode),
        "wall_seconds": wall_seconds,
        "report": str(report_path),
    }
    if completed.returncode == 0:
        result.update(_summarize_report(report_path))
        result["games_per_second"] = result["games"] / max(wall_seconds, 1e-9)
        result["turns_per_second"] = result["decision_turns"] / max(wall_seconds, 1e-9)
    else:
        result["stderr_tail"] = completed.stderr[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure benchmark_submission wall-clock scaling.")
    parser.add_argument("--submission", default="artifacts/submission.py")
    parser.add_argument(
        "--opponents", nargs="+", default=["artifacts/submission_v_old.py", "greedy", "rush"]
    )
    parser.add_argument("--seeds", nargs="+", default=["4", "8", "16"])
    parser.add_argument("--jobs", nargs="+", default=["1", "4", "8"])
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--skip-2p", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/throughput"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/throughput/summary.json"))
    args = parser.parse_args()

    seeds_values = _parse_ints(args.seeds)
    jobs_values = _parse_ints(args.jobs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for seeds in seeds_values:
        for jobs in jobs_values:
            result = _run_case(args, seeds, jobs, args.out_dir)
            cases.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "submission": str(args.submission),
        "opponents": list(args.opponents),
        "skip_2p": bool(args.skip_2p),
        "skip_4p": bool(args.skip_4p),
        "cases": cases,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
