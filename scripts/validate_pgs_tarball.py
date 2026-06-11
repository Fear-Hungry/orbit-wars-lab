"""Validate the packaged PGS tarball the way Kaggle will run it.

Extracts the tar.gz to a temp dir and, in a SUBPROCESS rooted there (so imports
resolve from the tarball, not the repo), loads main.py exactly like Kaggle
(exec + LAST callable in the namespace) and runs official episodes vs the
bundled Producer. Fails loud on agent ERROR/TIMEOUT/INVALID and on any
instrumented fallback (SUBMISSION_STATS.fallbacks > 0 — docs/SUBMISSION.md
forbids silent degradation to the Producer); reports decision-time stats and
the final rewards.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RUNNER = r'''
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
cfg = json.loads(sys.argv[1])


def _last_agent_from_source(path, module_name):
    src = Path(path).read_text()
    ns = {"__name__": module_name}
    exec(compile(src, str(path), "exec"), ns)
    last_callable = None
    for name, value in ns.items():
        if callable(value) and not name.startswith("__"):
            last_callable = (name, value)
    assert last_callable is not None, f"no callable in {path} namespace"
    agent_name, loaded_agent = last_callable
    assert agent_name == "agent", (
        f"LAST callable is {agent_name!r}, not agent (Kaggle picks the last!)"
    )
    return loaded_agent, ns


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load module {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_main_counter = itertools.count()


def _fresh_main_agent():
    loaded_agent, _ = _last_agent_from_source(
        "main.py", f"_validator_submission_main_{next(_main_counter)}"
    )
    return loaded_agent


agent, ns = _last_agent_from_source("main.py", "submission_main")


def _producer_factory_from_tarball():
    if Path("bots/producer/agent.py").exists():
        from bots.producer.agent import make_agent as make_producer

        return make_producer
    if Path("_producer_agent.py").exists():
        counter = itertools.count()

        def make_flat_producer():
            mod = _load_module(
                "_producer_agent.py",
                f"_validator_flat_producer_{next(counter)}",
            )
            make_agent = getattr(mod, "make_agent", None)
            if callable(make_agent):
                return make_agent()
            producer_agent = getattr(mod, "agent", None)
            assert callable(producer_agent), "_producer_agent.py has no agent/make_agent"
            return producer_agent

        return make_flat_producer
    if Path("_upstream.py").exists():
        # Flat Producer tarball: use fresh main.py copies as mirror opponents.
        return _fresh_main_agent
    raise AssertionError(
        "tarball has no bundled Producer opponent "
        "(expected bots/producer/agent.py, _producer_agent.py, or _upstream.py)"
    )


make_producer = _producer_factory_from_tarball()
if cfg.get("check_pgs_planner", True):
    import bots.pgs.planner as pgs_planner

    assert not hasattr(
        pgs_planner, "agent"
    ), "bundled bots/pgs/planner.py exposes rejected all-scripts agent(); regenerate the tarball"
    assert not hasattr(
        pgs_planner, "_RUNTIME"
    ), "bundled bots/pgs/planner.py exposes module runtime; regenerate the tarball"

from kaggle_environments import make

stats = ns.get("SUBMISSION_STATS")
if cfg.get("require_submission_stats", True):
    assert isinstance(stats, dict), "main.py must define SUBMISSION_STATS (instrumented fallback)"
elif not isinstance(stats, dict):
    stats = {}

episodes = []
all_times = []
for players in cfg["players"]:
    seat_values = list(range(players)) if cfg["seats"] == "all" else [
        int(s) for s in cfg["seats"] if int(s) < players
    ]
    for seat in seat_values:
        times = []

        def timed(obs):
            t0 = time.perf_counter()
            r = agent(obs)
            times.append((time.perf_counter() - t0) * 1000.0)
            return r

        lineup = [make_producer() for _ in range(players)]
        lineup[seat] = timed
        env = make("orbit_wars", debug=True)
        env.run(lineup)
        times.sort()
        all_times.extend(times)
        statuses = [s.status for s in env.state]
        rewards = [s.reward for s in env.state]
        episodes.append({
            "players": players,
            "seat": seat,
            "statuses": statuses,
            "rewards": rewards,
            "steps": len(env.steps),
            "decision_ms_p95": times[max(0, int(len(times)*0.95)-1)] if times else None,
            "decision_ms_max": times[-1] if times else None,
            "submission_status": statuses[seat],
        })

all_times.sort()
out = {
    "episodes": episodes,
    "decision_ms_p95": all_times[max(0, int(len(all_times)*0.95)-1)] if all_times else None,
    "decision_ms_max": all_times[-1] if all_times else None,
    "submission_stats": stats,
}
print(json.dumps(out))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", type=Path, default=Path("artifacts/submission_pgs.tar.gz"))
    ap.add_argument("--players", type=int, nargs="+", default=[2, 4])
    ap.add_argument(
        "--seats",
        default="all",
        help='seat list such as "0,1", or "all" for every seat in every player count',
    )
    ap.add_argument("--label", default="PGS")
    ap.add_argument(
        "--skip-pgs-planner-check",
        action="store_true",
        help="validate a non-PGS tarball with the same official runner",
    )
    ap.add_argument(
        "--allow-missing-submission-stats",
        action="store_true",
        help="allow pure baseline tarballs without SUBMISSION_STATS instrumentation",
    )
    args = ap.parse_args()
    seat_cfg: str | list[int]
    if args.seats.strip().lower() == "all":
        seat_cfg = "all"
    else:
        seat_cfg = [int(part.strip()) for part in args.seats.split(",") if part.strip()]
    runner_cfg = {
        "players": [int(p) for p in args.players],
        "seats": seat_cfg,
        "check_pgs_planner": not args.skip_pgs_planner_check,
        "require_submission_stats": not args.allow_missing_submission_stats,
    }

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(args.tarball) as tar:
            tar.extractall(tmp, filter="data")
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER, json.dumps(runner_cfg)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
            raise SystemExit(f"validation subprocess failed (rc={proc.returncode})")
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        print(json.dumps(result, indent=2))
        bad = [
            ep for ep in result["episodes"]
            if ep["submission_status"] not in {"DONE"}
        ]
        if bad:
            raise SystemExit(f"{args.label} failed official episode(s): {bad!r}")
        fallbacks = int(result["submission_stats"].get("fallbacks", 0))
        if fallbacks > 0:
            raise SystemExit(
                f"{args.label} fell back to the Producer on {fallbacks} step(s) "
                f"({result['submission_stats']!r}) — silent degradation is forbidden "
                f"(docs/SUBMISSION.md)"
            )
        print("VALIDATION OK")


if __name__ == "__main__":
    main()
