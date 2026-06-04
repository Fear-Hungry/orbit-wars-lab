from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/opponents/producer"
DEFAULT_WORK_DIR = ROOT / "artifacts/producer_sweeps"

_FIELD_RE = re.compile(
    r"^(?P<indent>\s{4})(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<annotation>\s*:[^=]+?)(?P<spacing>\s*=\s*)"
    r"(?P<value>.*?)(?P<comment>\s*(?:#.*)?)$"
)


def _producer_main_path(fixture_dir: Path) -> Path:
    return fixture_dir / "producer/main.py"


def _agent_path(fixture_dir: Path) -> Path:
    return fixture_dir / "producer_agent.py"


def _ensure_fixture(fixture_dir: Path) -> None:
    missing = [
        str(path)
        for path in (_agent_path(fixture_dir), _producer_main_path(fixture_dir))
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Producer fixture is incomplete; missing: {', '.join(missing)}")


def _parse_literal(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." not in raw and "e" not in lowered:
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def _python_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(str(value))


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.replace(".", "p").replace("-", "m")


def _dataclass_fields(main_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_config = False
    for line in main_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("class ProducerLiteConfig:"):
            in_config = True
            continue
        if in_config and line.startswith("def "):
            break
        if not in_config:
            continue
        match = _FIELD_RE.match(line)
        if match:
            fields[match.group("name")] = match.group("value").strip()
    return fields


def _apply_override(main_path: Path, field: str, value: Any) -> None:
    lines = main_path.read_text(encoding="utf-8").splitlines()
    changed = False
    in_config = False
    replacement = _python_literal(value)
    next_lines: list[str] = []
    for line in lines:
        if line.startswith("class ProducerLiteConfig:"):
            in_config = True
        elif in_config and line.startswith("def "):
            in_config = False

        match = _FIELD_RE.match(line) if in_config else None
        if match and match.group("name") == field:
            line = (
                f"{match.group('indent')}{field}{match.group('annotation')}"
                f"{match.group('spacing')}{replacement}{match.group('comment')}"
            )
            changed = True
        next_lines.append(line)
    if not changed:
        valid = ", ".join(sorted(_dataclass_fields(main_path)))
        raise SystemExit(f"Unknown ProducerLiteConfig field '{field}'. Valid fields: {valid}")
    main_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def _copy_candidate(
    *,
    fixture_dir: Path,
    work_dir: Path,
    field: str,
    value: Any,
    force: bool,
) -> Path:
    candidate_dir = work_dir / f"{field}_{_safe_name(str(value))}"
    if candidate_dir.exists():
        if not force:
            raise SystemExit(
                f"Candidate already exists: {candidate_dir} (use --force to overwrite)"
            )
        shutil.rmtree(candidate_dir)
    shutil.copytree(
        fixture_dir,
        candidate_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _apply_override(_producer_main_path(candidate_dir), field, value)
    metadata = {
        "source_fixture": str(fixture_dir),
        "field": field,
        "value": value,
        "agent": str(_agent_path(candidate_dir)),
    }
    (candidate_dir / "sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate_dir


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _benchmark_cmd(
    *,
    submission: Path,
    seeds: int,
    episode_steps: int,
    jobs: int,
    out: Path,
    act_timeout: float,
    opponent: str,
    skip_4p: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.benchmark_submission",
        "--submission",
        str(submission),
        "--opponents",
        opponent,
        "--seeds",
        str(seeds),
        "--episode-steps",
        str(episode_steps),
        "--jobs",
        str(jobs),
        "--act-timeout",
        str(act_timeout),
        "--out",
        str(out),
    ]
    if skip_4p:
        cmd.append("--skip-4p")
    return cmd


def _compare_cmd(
    *,
    baseline: Path,
    candidate: Path,
    out: Path,
    alpha: float,
    min_games: int,
    min_effect: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.compare_benchmark_significance",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--alpha",
        str(alpha),
        "--min-games",
        str(min_games),
        "--min-effect",
        str(min_effect),
        "--out",
        str(out),
    ]


def _load_summary(benchmark_path: Path, compare_path: Path | None) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    opponent_summary = benchmark["formats"][0]["opponents"][0]["summary"]
    row: dict[str, Any] = {
        "win_rate": opponent_summary["win_rate"],
        "mean_score_margin": opponent_summary["mean_score_margin"],
        "crash_rate": opponent_summary["crash_rate"],
        "timeout_rate": opponent_summary["timeout_rate"],
        "invalid_action_rate": opponent_summary["invalid_action_rate"],
        "mean_decision_ms": opponent_summary["mean_decision_ms"],
    }
    if compare_path is not None and compare_path.exists():
        compare = json.loads(compare_path.read_text(encoding="utf-8"))
        comparisons = compare.get("comparisons", [])
        if comparisons:
            item = comparisons[0]
            row.update(
                {
                    "verdict": item["verdict"],
                    "margin_delta": item["margin_delta"],
                    "margin_p_value": item["margin_p_value"],
                    "paired_margin_delta": item["paired_margin_delta"],
                    "paired_margin_p_value": item["paired_margin_p_value"],
                }
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and benchmark one-at-a-time ProducerLiteConfig knob variants."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="One ProducerLiteConfig override. Repeat for multiple one-knob candidates.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--opponent", default="producer")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-games", type=int, default=128)
    parser.add_argument("--min-effect", type=float, default=0.05)
    parser.add_argument("--skip-4p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-baseline", action="store_true")
    args = parser.parse_args()

    fixture_dir = args.fixture.resolve()
    work_dir = args.work_dir.resolve()
    _ensure_fixture(fixture_dir)
    if not args.overrides:
        fields = ", ".join(sorted(_dataclass_fields(_producer_main_path(fixture_dir))))
        raise SystemExit(f"Provide at least one --set FIELD=VALUE. Valid fields: {fields}")

    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = work_dir / f"baseline_producer_{args.seeds}seed_{args.episode_steps}steps.json"
    if args.run and (args.refresh_baseline or not baseline_path.exists()):
        _run(
            _benchmark_cmd(
                submission=_agent_path(fixture_dir),
                seeds=int(args.seeds),
                episode_steps=int(args.episode_steps),
                jobs=int(args.jobs),
                out=baseline_path,
                act_timeout=float(args.act_timeout),
                opponent=str(args.opponent),
                skip_4p=bool(args.skip_4p),
            )
        )

    rows = []
    for override in args.overrides:
        if "=" not in override:
            raise SystemExit(f"Invalid --set value '{override}'. Expected FIELD=VALUE.")
        field, raw_value = override.split("=", 1)
        field = field.strip()
        value = _parse_literal(raw_value)
        candidate_dir = _copy_candidate(
            fixture_dir=fixture_dir,
            work_dir=work_dir,
            field=field,
            value=value,
            force=bool(args.force),
        )
        benchmark_path = candidate_dir / "benchmark_vs_producer.json"
        compare_path = candidate_dir / "significance_vs_producer.json"
        if args.run:
            _run(
                _benchmark_cmd(
                    submission=_agent_path(candidate_dir),
                    seeds=int(args.seeds),
                    episode_steps=int(args.episode_steps),
                    jobs=int(args.jobs),
                    out=benchmark_path,
                    act_timeout=float(args.act_timeout),
                    opponent=str(args.opponent),
                    skip_4p=bool(args.skip_4p),
                )
            )
            if args.compare:
                _run(
                    _compare_cmd(
                        baseline=baseline_path,
                        candidate=benchmark_path,
                        out=compare_path,
                        alpha=float(args.alpha),
                        min_games=int(args.min_games),
                        min_effect=float(args.min_effect),
                    )
                )
        row = {
            "field": field,
            "value": value,
            "candidate_dir": str(candidate_dir),
            "agent": str(_agent_path(candidate_dir)),
        }
        if benchmark_path.exists():
            row.update(_load_summary(benchmark_path, compare_path if args.compare else None))
        rows.append(row)

    summary_path = work_dir / "latest_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
