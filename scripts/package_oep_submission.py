"""Package the OEP agent as a Kaggle submission tar.gz.

The OEP planner is torch + orbit_lite based (same dependency profile as the
Producer submission, which runs on Kaggle), plus `bots.producer._upstream`.
The tarball bundles the `bots` and `orbit_lite` packages so the package-style
imports resolve when Kaggle puts the tarball dir on sys.path, and a thin
`main.py` bakes the best selection threshold (OEP_MIN_ADVANTAGE) via env since
Kaggle does not pass env vars.

The robustness wrapper mirrors the PGS submission: OEP runs under a time budget
on a worker thread; on overrun or exception the main thread returns the
Producer plan — always a valid, on-time move. Every fallback is INSTRUMENTED in
``SUBMISSION_STATS`` (calls/fallbacks/timeouts/fallback_errors) so the local
gate fails loud instead of degrading silently (docs/SUBMISSION.md). NOTE:
``agent`` is defined LAST in main.py (Kaggle picks the last callable in the
namespace).
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

MAIN_TEMPLATE = '''import os
os.environ.setdefault("OEP_MIN_ADVANTAGE", "{min_advantage}")
import threading
import time
from bots.oep.agent import agent as _oep
from bots.producer.agent import agent as _producer

# Total OEP wrapper budget. A dedicated Producer shadow is called every turn
# first, so fallback is stateful/warm instead of a cold mid-game Producer plan.
_BUDGET_S = {budget_s}
# Stop launching OEP after the first overrun. Timed-out daemon threads cannot
# be killed and may keep mutating the module runtime, so overlapping attempts
# are worse than a visible Producer fallback.
_MAX_CONSEC_TIMEOUTS = 1

SUBMISSION_STATS = {{
    "calls": 0,
    "fallbacks": 0,
    "timeouts": 0,
    "fallback_errors": 0,
}}

_consec_timeouts = [0]


def _submission_stats_increment(name, amount=1):
    SUBMISSION_STATS[name] = int(SUBMISSION_STATS.get(name, 0)) + int(amount)


def agent(obs):
    _submission_stats_increment("calls")
    fallback_error = False
    t0 = time.perf_counter()
    try:
        fallback = _producer(obs)
    except Exception:
        fallback = []
        fallback_error = True
    if _consec_timeouts[0] >= _MAX_CONSEC_TIMEOUTS:
        _submission_stats_increment("fallbacks")
        if fallback_error:
            _submission_stats_increment("fallback_errors")
        return fallback
    box = {{}}

    def _run():
        try:
            box["r"] = _oep(obs)
        except Exception:
            box["err"] = True

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(max(0.0, _BUDGET_S - (time.perf_counter() - t0)))
    if box.get("r") is not None:
        _consec_timeouts[0] = 0
        return box["r"]
    _submission_stats_increment("fallbacks")
    if th.is_alive():
        _consec_timeouts[0] += 1
        _submission_stats_increment("timeouts")
    else:
        _consec_timeouts[0] = 0
        _submission_stats_increment("fallback_errors")
    if fallback_error:
        _submission_stats_increment("fallback_errors")
    return fallback
'''

BOT_FILES = [
    "bots/__init__.py",
    "bots/oep/__init__.py",
    "bots/oep/agent.py",
    "bots/oep/planner.py",
    "bots/producer/__init__.py",
    "bots/producer/agent.py",
    "bots/producer/_upstream.py",
]


def _add_text(tar: tarfile.TarFile, arcname: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_tree(tar: tarfile.TarFile, source: Path, prefix: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            tar.add(
                path,
                arcname=str(Path(prefix) / path.relative_to(source)),
                recursive=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the OEP agent for Kaggle.")
    parser.add_argument("--min-advantage", type=float, default=15.0)
    parser.add_argument("--budget-s", type=float, default=0.9, help="OEP total wrapper budget before Producer fallback")
    parser.add_argument("--orbit-lite-dir", type=Path, default=Path("orbit_lite"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_oep.tar.gz"))
    args = parser.parse_args()

    missing = [f for f in BOT_FILES if not Path(f).exists()]
    if missing or not args.orbit_lite_dir.is_dir():
        raise FileNotFoundError(f"missing OEP packaging inputs: {missing} / {args.orbit_lite_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    min_adv = int(args.min_advantage) if float(args.min_advantage).is_integer() else args.min_advantage
    with tarfile.open(args.out, "w:gz") as tar:
        _add_text(tar, "main.py", MAIN_TEMPLATE.format(min_advantage=min_adv, budget_s=args.budget_s))
        for f in BOT_FILES:
            tar.add(f, arcname=f, recursive=False)
        _add_tree(tar, args.orbit_lite_dir, "orbit_lite")
    print({"wrote": str(args.out), "min_advantage": min_adv, "files": len(BOT_FILES) + 1})


if __name__ == "__main__":
    main()
