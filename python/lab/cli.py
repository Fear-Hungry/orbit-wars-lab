from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from python.agents.registry import HEURISTIC_NAMES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_SUBMISSION = ARTIFACTS / "submission.py"
DEFAULT_BENCHMARK = ARTIFACTS / "submission_benchmark.json"
DEFAULT_EVALUATION = ARTIFACTS / "evaluation_report.json"
DEFAULT_HOF_STATE = ARTIFACTS / "hall_of_fame.json"
DEFAULT_MAP_ELITES_STATE = ARTIFACTS / "map_elites.json"


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
        "--out",
        _display_path(args.out),
    )
    if args.opponents:
        command.extend(["--opponents", *args.opponents])
    if args.skip_2p:
        command.append("--skip-2p")
    if args.skip_4p:
        command.append("--skip-4p")
    if args.disable_comets:
        command.append("--disable-comets")
    return command


def _cmd_doctor(_args: argparse.Namespace) -> int:
    checks = {
        "root": ROOT.exists(),
        "submission_template": (ROOT / "python/submission/submission_template.py").exists(),
        "quick_eval_config": (ROOT / "configs/eval_quick.yaml").exists(),
        "final_candidate_pool": (ROOT / "configs/final_candidate_pool.yaml").exists(),
        "benchmark_script": (ROOT / "scripts/benchmark_submission.py").exists(),
        "rust_binding_crate": (ROOT / "crates/orbit_wars_py/Cargo.toml").exists(),
    }
    print(json.dumps({"root": str(ROOT), "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


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
    export_args = argparse.Namespace(
        out=args.submission,
        checkpoint=args.checkpoint,
    )
    benchmark_args = argparse.Namespace(
        submission=args.submission,
        seeds=args.seeds,
        episode_steps=args.episode_steps,
        act_timeout=args.act_timeout,
        out=args.out,
        opponents=args.opponents,
        skip_2p=False,
        skip_4p=False,
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

    bench = sub.add_parser(
        "bench-submission", help="benchmark a submission against local heuristics"
    )
    bench.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    bench.add_argument("--seeds", type=int, default=3)
    bench.add_argument("--episode-steps", type=int, default=500)
    bench.add_argument("--act-timeout", type=float, default=1.0)
    bench.add_argument("--opponents", nargs="+", default=list(HEURISTIC_NAMES))
    bench.add_argument("--skip-2p", action="store_true")
    bench.add_argument("--skip-4p", action="store_true")
    bench.add_argument("--disable-comets", action="store_true")
    bench.add_argument("--out", default=str(DEFAULT_BENCHMARK))
    bench.add_argument("--dry-run", action="store_true")
    bench.set_defaults(func=_cmd_benchmark)

    quick = sub.add_parser("quick", help="export and run a short submission benchmark")
    quick.add_argument("--checkpoint")
    quick.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    quick.add_argument("--seeds", type=int, default=2)
    quick.add_argument("--episode-steps", type=int, default=128)
    quick.add_argument("--act-timeout", type=float, default=1.0)
    quick.add_argument("--opponents", nargs="+", default=list(HEURISTIC_NAMES))
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
