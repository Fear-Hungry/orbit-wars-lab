"""Package the PGS agent (bots/pgs) as a Kaggle submission tar.gz.

PGS is torch + orbit_lite based (same dependency profile as the Producer
submission) plus ``bots.producer._upstream`` (its base-plan generator). The
tarball bundles the ``bots`` and ``orbit_lite`` packages and a thin ``main.py``.

The robustness wrapper mirrors the OEP submission: PGS runs under a time budget
on a worker thread; on overrun or exception the main thread returns the
Producer plan — always a valid, on-time move. NOTE: ``agent`` is defined LAST
in main.py (Kaggle picks the last callable in the namespace).
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

MAIN_TEMPLATE = '''import threading
from bots.pgs.agent import agent as _pgs
from bots.producer.agent import agent as _producer

# PGS time budget before falling back to the Producer plan (p95 ~84ms local;
# generous slack for slower Kaggle CPUs while staying inside actTimeout=1s).
_BUDGET_S = {budget_s}


def agent(obs):
    box = {{}}

    def _run():
        try:
            box["r"] = _pgs(obs)
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
    "bots/pgs/__init__.py",
    "bots/pgs/agent.py",
    "bots/pgs/planner.py",
    "bots/pgs/_helpers.py",
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
    parser = argparse.ArgumentParser(description="Package the PGS agent for Kaggle.")
    parser.add_argument("--budget-s", type=float, default=0.7, help="PGS time budget before Producer fallback")
    parser.add_argument("--orbit-lite-dir", type=Path, default=Path("orbit_lite"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_pgs.tar.gz"))
    parser.add_argument(
        "--pgs-config", default=None,
        help='override the shipped SUBMISSION_CONFIG kwargs, e.g. \'scripts="hold"\' '
             "(default: ship bots/pgs/agent.py as-is)")
    args = parser.parse_args()

    missing = [f for f in BOT_FILES if not Path(f).exists()]
    if missing or not args.orbit_lite_dir.is_dir():
        raise FileNotFoundError(f"missing PGS packaging inputs: {missing} / {args.orbit_lite_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        _add_text(tar, "main.py", MAIN_TEMPLATE.format(budget_s=args.budget_s))
        for f in BOT_FILES:
            if args.pgs_config and f == "bots/pgs/agent.py":
                src = Path(f).read_text()
                patched = []
                hit = False
                for line in src.splitlines():
                    if line.startswith("SUBMISSION_CONFIG = "):
                        patched.append(f"SUBMISSION_CONFIG = PGSConfig({args.pgs_config})")
                        hit = True
                    else:
                        patched.append(line)
                if not hit:
                    raise RuntimeError("SUBMISSION_CONFIG line not found in bots/pgs/agent.py")
                _add_text(tar, f, "\n".join(patched) + "\n")
            else:
                tar.add(f, arcname=f, recursive=False)
        _add_tree(tar, args.orbit_lite_dir, "orbit_lite")
    print({"wrote": str(args.out), "files": len(BOT_FILES) + 1, "budget_s": args.budget_s,
           "pgs_config": args.pgs_config or "as-is"})


if __name__ == "__main__":
    main()
