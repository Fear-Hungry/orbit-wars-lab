from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from python.agents.registry import HEURISTIC_NAMES

# Imports the lab cannot run without. orbit_wars_rs is the Rust backend; without
# it every real measurement silently falls back or crashes. kaggle_environments
# and duckdb back the official engine and the experiment tracker.
REQUIRED_IMPORTS = ("orbit_wars_rs", "kaggle_environments", "duckdb")
# Toolchain needed to (re)build the Rust backend.
REQUIRED_TOOLS = ("maturin", "cargo", "rustc")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_SUBMISSION = ARTIFACTS / "submission.py"
DEFAULT_BENCHMARK = ARTIFACTS / "submission_benchmark.json"
DEFAULT_EVALUATION = ARTIFACTS / "evaluation_report.json"
DEFAULT_HOF_STATE = ARTIFACTS / "hall_of_fame.json"
DEFAULT_MAP_ELITES_STATE = ARTIFACTS / "map_elites.json"
DEFAULT_BENCHMARK_OPPONENTS = ["producer"]
DEFAULT_BENCHMARK_SEEDS = 16
DEFAULT_JOBS = max(1, os.cpu_count() or 1)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _run(command: Sequence[str], *, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _export_command(args: argparse.Namespace) -> list[str]:
    command = _python_module(
        "scripts.export_submission",
        "--out",
        _display_path(args.out),
    )
    if args.checkpoint:
        command.extend(["--checkpoint", _display_path(args.checkpoint)])
    return command


def _benchmark_command(args: argparse.Namespace) -> list[str]:
    command = _python_module(
        "scripts.benchmark_submission",
        "--submission",
        _display_path(args.submission),
        "--seeds",
        str(args.seeds),
        "--episode-steps",
        str(args.episode_steps),
        "--act-timeout",
        str(args.act_timeout),
        "--jobs",
        str(args.jobs),
        "--out",
        _display_path(args.out),
    )
    if args.opponents:
        command.extend(["--opponents", *args.opponents])
    if args.skip_2p:
        command.append("--skip-2p")
    if args.skip_4p:
        command.append("--skip-4p")
    # Only ever set when an explicit diagnostic opt-in (quick --dirty-4p) wants the
    # seat-0-pinned 4p path; never on a promotion route.
    if getattr(args, "ack_seat_biased", False):
        command.append("--ack-seat-biased")
    if args.disable_comets:
        command.append("--disable-comets")
    return command


def _check_import(module: str) -> tuple[bool, str]:
    """Actually import the module so a broken .so / missing dep is caught,
    not just an absent spec."""
    try:
        importlib.import_module(module)
        return True, "import ok"
    except BaseException as exc:  # noqa: BLE001 - report any failure verbatim
        return False, f"{type(exc).__name__}: {exc}".splitlines()[0][:200]


def _find_tool(name: str) -> str | None:
    """Locate a CLI on PATH, falling back to the venv bin dir next to the running
    interpreter (where `maturin` lives when the venv is not activated)."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return None


def _check_tool(name: str) -> tuple[bool, str]:
    path = _find_tool(name)
    if not path:
        return False, "not found on PATH or venv bin"
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}"
    version = (proc.stdout or proc.stderr).strip().splitlines()
    return True, version[0] if version else path


def _cmd_doctor(_args: argparse.Namespace) -> int:
    files = {
        "root": ROOT.exists(),
        "submission_template": (ROOT / "python/submission/submission_template.py").exists(),
        "quick_eval_config": (ROOT / "configs/eval_quick.yaml").exists(),
        "final_candidate_pool": (ROOT / "configs/final_candidate_pool.yaml").exists(),
        "benchmark_script": (ROOT / "scripts/benchmark_submission.py").exists(),
        "rust_binding_crate": (ROOT / "crates/orbit_wars_py/Cargo.toml").exists(),
    }
    # Importing kaggle_environments emits INFO logs to stdout; keep them out of
    # the doctor's own JSON report by routing import-time output to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        imports = {name: _check_import(name) for name in REQUIRED_IMPORTS}
    tools = {name: _check_tool(name) for name in REQUIRED_TOOLS}

    report = {
        "root": str(ROOT),
        "python": sys.executable,
        "files": files,
        "imports": {name: {"ok": ok, "detail": detail} for name, (ok, detail) in imports.items()},
        "tools": {name: {"ok": ok, "detail": detail} for name, (ok, detail) in tools.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    ok = (
        all(files.values())
        and all(ok for ok, _ in imports.values())
        and all(ok for ok, _ in tools.values())
    )
    if not ok:
        failed = (
            [n for n, v in files.items() if not v]
            + [n for n, (o, _) in imports.items() if not o]
            + [n for n, (o, _) in tools.items() if not o]
        )
        print(f"doctor: FAIL ({', '.join(failed)})", file=sys.stderr)
    return 0 if ok else 1


def _cmd_heuristics(_args: argparse.Namespace) -> int:
    print("\n".join(HEURISTIC_NAMES))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    _repo_path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return _run(_export_command(args), dry_run=args.dry_run)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    _repo_path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return _run(_benchmark_command(args), dry_run=args.dry_run)


def _cmd_quick(args: argparse.Namespace) -> int:
    # quick is a crash/obvious-failure smoke check, NOT a promotion gate. It runs
    # seat-rotated 2p only by default: the benchmark's 4p path pins the submission
    # to seat 0 (no rotation), so 4p win/margin there can be a seat-0 artifact.
    # The real, seat-rotated promotion ruler is scripts/league_submit_ruler.py.
    # --dirty-4p opts into the seat-0-pinned 4p path as a DIAGNOSTIC ONLY.
    if args.dirty_4p:
        sys.stderr.write(
            "[lab quick] --dirty-4p: running the SEAT-0-PINNED 4p path as a diagnostic "
            "ONLY; this is NOT a promotion gate. Promote via "
            "scripts/league_submit_ruler.py (seat-rotated).\n"
        )
    export_args = argparse.Namespace(
        out=args.submission,
        checkpoint=args.checkpoint,
    )
    benchmark_args = argparse.Namespace(
        submission=args.submission,
        seeds=args.seeds,
        episode_steps=args.episode_steps,
        act_timeout=args.act_timeout,
        jobs=args.jobs,
        out=args.out,
        opponents=args.opponents,
        skip_2p=False,
        skip_4p=not args.dirty_4p,
        ack_seat_biased=args.dirty_4p,
        disable_comets=args.disable_comets,
    )
    _repo_path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _run(_export_command(export_args), dry_run=args.dry_run)
    return _run(_benchmark_command(benchmark_args), dry_run=args.dry_run)


def _cmd_eval(args: argparse.Namespace) -> int:
    _repo_path(args.output).parent.mkdir(parents=True, exist_ok=True)
    command = _python_module(
        "python.train.evaluate_population",
        "--manifest",
        _display_path(args.manifest),
        "--config",
        _display_path(args.config),
        "--output",
        _display_path(args.output),
    )
    if args.hall_of_fame_state:
        command.extend(["--hall-of-fame-state", _display_path(args.hall_of_fame_state)])
    return _run(command, dry_run=args.dry_run)


def _cmd_league(args: argparse.Namespace) -> int:
    _repo_path(args.hall_of_fame_state).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(args.map_elites_state).parent.mkdir(parents=True, exist_ok=True)
    command = _python_module(
        "python.train.train_league",
        "--config",
        _display_path(args.config),
        "--evaluation-report",
        _display_path(args.evaluation_report),
        "--hall-of-fame-state",
        _display_path(args.hall_of_fame_state),
        "--map-elites-state",
        _display_path(args.map_elites_state),
        "--generation",
        str(args.generation),
    )
    return _run(command, dry_run=args.dry_run)


def _cmd_test(args: argparse.Namespace) -> int:
    commands = {
        "smoke": _python_module("scripts.smoke_test"),
        "unit": [sys.executable, "-m", "pytest", "-q"],
        "submission": [sys.executable, "-m", "pytest", "-q", "tests/test_submission_pipeline.py"],
        "parity": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_official_spec.py",
            "tests/test_parity_tolerances.py",
        ],
    }
    selected = (
        ["smoke", "submission"]
        if args.group == "fast"
        else ([args.group] if args.group != "all" else list(commands))
    )
    for group in selected:
        _run(commands[group], dry_run=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbit Wars lab workflow entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the files needed by the local lab workflow")
    doctor.set_defaults(func=_cmd_doctor)

    heuristics = sub.add_parser("heuristics", help="list built-in heuristic policy names")
    heuristics.set_defaults(func=_cmd_heuristics)

    export = sub.add_parser("export", help="render a Kaggle submission artifact")
    export.add_argument("--checkpoint")
    export.add_argument("--out", default=str(DEFAULT_SUBMISSION))
    export.add_argument("--dry-run", action="store_true")
    export.set_defaults(func=_cmd_export)

    bench = sub.add_parser("bench-submission", help="benchmark a submission against Producer")
    bench.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    bench.add_argument("--seeds", type=int, default=DEFAULT_BENCHMARK_SEEDS)
    bench.add_argument("--episode-steps", type=int, default=500)
    bench.add_argument("--act-timeout", type=float, default=1.0)
    bench.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    bench.add_argument("--opponents", nargs="+", default=DEFAULT_BENCHMARK_OPPONENTS)
    bench.add_argument("--skip-2p", action="store_true")
    bench.add_argument("--skip-4p", action="store_true")
    bench.add_argument("--disable-comets", action="store_true")
    bench.add_argument("--out", default=str(DEFAULT_BENCHMARK))
    bench.add_argument("--dry-run", action="store_true")
    bench.set_defaults(func=_cmd_benchmark)

    quick = sub.add_parser(
        "quick",
        help="export and run a short seat-rotated 2p smoke benchmark; "
        "crash/failure check only, NOT a promotion gate (use scripts/league_submit_ruler.py)",
    )
    quick.add_argument("--checkpoint")
    quick.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    quick.add_argument("--seeds", type=int, default=2)
    quick.add_argument("--episode-steps", type=int, default=128)
    quick.add_argument("--act-timeout", type=float, default=1.0)
    quick.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    quick.add_argument("--opponents", nargs="+", default=DEFAULT_BENCHMARK_OPPONENTS)
    quick.add_argument(
        "--dirty-4p",
        action="store_true",
        help="also run the seat-0-pinned 4p path as a DIAGNOSTIC ONLY (not a gate)",
    )
    quick.add_argument("--disable-comets", action="store_true")
    quick.add_argument("--out", default=str(DEFAULT_BENCHMARK))
    quick.add_argument("--dry-run", action="store_true")
    quick.set_defaults(func=_cmd_quick)

    eval_cmd = sub.add_parser("eval", help="evaluate a manifest with the quick or final config")
    eval_cmd.add_argument("--manifest", default="configs/final_candidate_pool.yaml")
    eval_cmd.add_argument("--config", default="configs/eval_quick.yaml")
    eval_cmd.add_argument("--output", default=str(DEFAULT_EVALUATION))
    eval_cmd.add_argument("--hall-of-fame-state")
    eval_cmd.add_argument("--dry-run", action="store_true")
    eval_cmd.set_defaults(func=_cmd_eval)

    league = sub.add_parser("league", help="run one league/PBT iteration from an evaluation report")
    league.add_argument("--config", default="configs/league.yaml")
    league.add_argument("--evaluation-report", default=str(DEFAULT_EVALUATION))
    league.add_argument("--hall-of-fame-state", default=str(DEFAULT_HOF_STATE))
    league.add_argument("--map-elites-state", default=str(DEFAULT_MAP_ELITES_STATE))
    league.add_argument("--generation", type=int, default=0)
    league.add_argument("--dry-run", action="store_true")
    league.set_defaults(func=_cmd_league)

    test = sub.add_parser("test", help="run common validation groups")
    test.add_argument(
        "--group", choices=("fast", "smoke", "unit", "submission", "parity", "all"), default="fast"
    )
    test.add_argument("--dry-run", action="store_true")
    test.set_defaults(func=_cmd_test)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
