"""Package the OEP agent as a Kaggle submission tar.gz.

The OEP planner is torch + orbit_lite based (same dependency profile as the
Producer submission, which runs on Kaggle), plus `bots.producer._upstream`.
The tarball bundles the `bots` and `orbit_lite` packages so the package-style
imports resolve when Kaggle puts the tarball dir on sys.path, and a thin
`main.py` bakes the best selection threshold (OEP_MIN_ADVANTAGE) via env since
Kaggle does not pass env vars.
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

MAIN_TEMPLATE = '''import os
os.environ.setdefault("OEP_MIN_ADVANTAGE", "{min_advantage}")
import threading
from bots.oep.agent import agent as _oep
from bots.producer.agent import agent as _producer

# Robustness wrapper for the SUBMISSION (not the runtime — keeps the repo's
# no-silent-fallback invariant intact). The OEP lookahead (~370ms locally) can
# spike past actTimeout on slower Kaggle hardware, which ERRORs the whole
# episode. Run it under a time budget on a worker thread; if it overruns or
# raises, the main thread returns the Producer plan — always a valid, on-time
# move — so the agent never times out / crashes the validation episode.
_BUDGET_S = {budget_s}


def agent(obs):
    box = {{}}

    def _run():
        try:
            box["r"] = _oep(obs)
        except Exception:
            pass

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(_BUDGET_S)
    if box.get("r") is not None:
        return box["r"]
    return _producer(obs)
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
    parser.add_argument("--budget-s", type=float, default=0.6, help="OEP time budget before Producer fallback")
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
