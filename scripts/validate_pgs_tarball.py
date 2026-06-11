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

RUNNER = '''
import json, sys, time
sys.path.insert(0, ".")
cfg = json.loads(sys.argv[1])
src = open("main.py").read()
ns = {}
exec(compile(src, "main.py", "exec"), ns)
last_callable = None
for name, value in ns.items():
    if callable(value) and not name.startswith("__"):
        last_callable = (name, value)
assert last_callable is not None, "no callable in main.py namespace"
agent_name, agent = last_callable
assert agent_name == "agent", f"LAST callable is {agent_name!r}, not agent (Kaggle picks the last!)"

from bots.producer.agent import make_agent as make_producer

from kaggle_environments import make
stats = ns.get("SUBMISSION_STATS")
assert isinstance(stats, dict), "main.py must define SUBMISSION_STATS (instrumented fallback)"
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
    args = ap.parse_args()
    seat_cfg: str | list[int]
    if args.seats.strip().lower() == "all":
        seat_cfg = "all"
    else:
        seat_cfg = [int(part.strip()) for part in args.seats.split(",") if part.strip()]
    runner_cfg = {"players": [int(p) for p in args.players], "seats": seat_cfg}

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
