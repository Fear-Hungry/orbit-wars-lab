"""Package the PGS agent (bots/pgs) as a Kaggle submission tar.gz.

PGS is torch + orbit_lite based (same dependency profile as the Producer
submission) plus ``bots.producer._upstream`` (its base-plan generator). The
tarball bundles the ``bots`` and ``orbit_lite`` packages and a thin ``main.py``.

The robustness wrapper mirrors the OEP submission: PGS runs under a time budget
on a worker thread; on overrun or exception the main thread returns the
Producer plan — always a valid, on-time move. Every fallback is INSTRUMENTED in
``SUBMISSION_STATS`` (calls/fallbacks/timeouts/fallback_errors), mirroring the
BReP template, so the local gate fails loud instead of degrading silently
(docs/SUBMISSION.md). NOTE: ``agent`` is defined LAST in main.py (Kaggle picks
the last callable in the namespace).
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

MAIN_TEMPLATE = '''import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

import threading
import time
import bots.pgs.agent as _pgs_agent
from bots.producer.agent import agent as _producer

# Total PGS wrapper budget. A dedicated Producer shadow is called every turn
# first, so fallback is stateful/warm instead of a cold mid-game Producer plan.
_BUDGET_S = {budget_s}
# Timed-out daemon threads cannot be killed and may keep mutating the module
# runtime. Do not launch another PGS while the old overrun is alive, but resume
# PGS after it finishes instead of degrading the rest of the game to Producer.
_active_timeout_thread = [None]

SUBMISSION_STATS = {{
    "calls": 0,
    "fallbacks": 0,
    "timeouts": 0,
    "timeout_thread_blocks": 0,
    "fallback_errors": 0,
}}


def _submission_stats_increment(name, amount=1):
    SUBMISSION_STATS[name] = int(SUBMISSION_STATS.get(name, 0)) + int(amount)


def _timeout_thread_still_alive():
    th = _active_timeout_thread[0]
    if th is None:
        return False
    if th.is_alive():
        return True
    _active_timeout_thread[0] = None
    return False


def _notify_fallback_applied():
    try:
        _pgs_agent.notify_fallback_applied()
    except Exception:
        pass


def agent(obs):
    _submission_stats_increment("calls")
    fallback_error = False
    t0 = time.perf_counter()
    try:
        fallback = _producer(obs)
    except Exception:
        fallback = []
        fallback_error = True
    if _timeout_thread_still_alive():
        _submission_stats_increment("fallbacks")
        _submission_stats_increment("timeout_thread_blocks")
        if fallback_error:
            _submission_stats_increment("fallback_errors")
        _notify_fallback_applied()
        return fallback
    box = {{}}

    def _run():
        try:
            box["r"] = _pgs_agent.agent(obs)
        except Exception:
            box["err"] = True

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(max(0.0, _BUDGET_S - (time.perf_counter() - t0)))
    if box.get("r") is not None:
        return box["r"]
    _submission_stats_increment("fallbacks")
    if th.is_alive():
        _active_timeout_thread[0] = th
        _submission_stats_increment("timeouts")
    else:
        _submission_stats_increment("fallback_errors")
    if fallback_error:
        _submission_stats_increment("fallback_errors")
    _notify_fallback_applied()
    return fallback
'''

BOT_FILES = [
    "bots/__init__.py",
    "bots/pgs/__init__.py",
    "bots/pgs/agent.py",
    "bots/pgs/planner.py",
    "bots/pgs/_helpers.py",
    "bots/pgs/threat.py",
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
    parser.add_argument("--budget-s", type=float, default=0.9, help="PGS total wrapper budget before Producer fallback")
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
    if args.pgs_config and "value_net_path" in args.pgs_config:
        raise SystemExit(
            "--pgs-config value_net_path is not submission-safe yet: the tarball "
            "does not bundle python.agents.value_net or the checkpoint. Package "
            "that path explicitly before enabling it."
        )

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
