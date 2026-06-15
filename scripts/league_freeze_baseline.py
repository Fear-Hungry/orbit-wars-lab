"""Freeze a submit-ruler run as an immutable selector baseline.

Before any scoring-rule change, the current ruler output is frozen so a future
"the selector improved" claim can be tested against the SAME games instead of a
moved ruler (etapa 1 of the selector plan). The baseline is self-contained:
task_results.json + every per-task game JSON + a manifest binding them to the
code hash, panel, seeds and the LB anchors of the day. report.json is copied
when present but is NOT required — it is derivative (re-scorable from the games
under any ruler version).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.league_agents import GATE_REFERENCE, INCUMBENT, LB_ANCHORS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "artifacts" / "league" / "baselines"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def build_manifest(
    task_results_path: Path,
    *,
    label: str,
    report_path: Path | None = None,
    game_files: list[Path] | None = None,
) -> dict:
    task_results = json.loads(task_results_path.read_text())
    candidates = sorted({r.get("candidate") for r in task_results if r.get("candidate")})
    settings = {
        key: sorted({r.get(key) for r in task_results if r.get(key) is not None})
        for key in ("seed_base", "seeds", "steps", "chunk_size", "mode")
    }
    manifest = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "source_task_results": str(task_results_path),
        "task_results_sha1": _sha1(task_results_path),
        "task_count": len(task_results),
        "candidates": candidates,
        "settings": settings,
        "incumbent": INCUMBENT,
        "gate_reference": GATE_REFERENCE,
        "lb_anchors": dict(LB_ANCHORS),
    }
    if report_path is not None and report_path.exists():
        manifest["report_sha1"] = _sha1(report_path)
        report = json.loads(report_path.read_text())
        manifest["report_settings"] = report.get("settings")
        manifest["report_references"] = report.get("references")
    if game_files:
        manifest["game_files"] = {p.name: _sha1(p) for p in game_files}
    return manifest


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def freeze(
    task_results_path: Path,
    *,
    label: str,
    out_dir: Path,
    report_path: Path | None = None,
    force: bool = False,
) -> Path:
    dest = out_dir / label
    if dest.exists() and not force:
        raise SystemExit(
            f"baseline {label} already exists at {dest}; refusing to overwrite "
            "(an immutable baseline that silently changes is worse than none — use --force)"
        )
    task_results = json.loads(task_results_path.read_text())
    game_files: list[Path] = []
    missing: list[str] = []
    for result in task_results:
        out = result.get("out")
        if not out:
            continue
        # "out" is repo-root-relative in the producing checkout; the reliable
        # anchor is the basename next to task_results.json itself.
        src = next(
            (p for p in (Path(out), task_results_path.parent / Path(out).name) if p.exists()),
            None,
        )
        if src is not None:
            game_files.append(src)
        else:
            missing.append(str(out))
    if missing:
        raise SystemExit(
            f"{len(missing)} game JSONs referenced by task_results are missing "
            f"(first: {missing[0]}); a baseline with holes is not a baseline"
        )

    dest.mkdir(parents=True, exist_ok=True)
    games_dir = dest / "games"
    games_dir.mkdir(exist_ok=True)
    copied: list[Path] = []
    for src in game_files:
        target = games_dir / src.name
        shutil.copyfile(src, target)
        copied.append(target)
    shutil.copyfile(task_results_path, dest / "task_results.json")
    if report_path is not None and report_path.exists():
        shutil.copyfile(report_path, dest / "report.json")

    manifest = build_manifest(
        task_results_path, label=label, report_path=report_path, game_files=copied
    )
    _atomic_write_json(dest / "manifest.json", manifest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--label", default="selector_baseline_2026_06_12")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    dest = freeze(
        args.task_results,
        label=args.label,
        out_dir=args.out_dir,
        report_path=args.report,
        force=args.force,
    )
    manifest = json.loads((dest / "manifest.json").read_text())
    print(json.dumps({
        "frozen": str(dest),
        "git_head": manifest["git_head"],
        "git_dirty": manifest["git_dirty"],
        "tasks": manifest["task_count"],
        "candidates": manifest["candidates"],
        "games": len(manifest.get("game_files", {})),
    }, indent=2))


if __name__ == "__main__":
    main()
