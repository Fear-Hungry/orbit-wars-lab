"""Autopilot — runs a route until exhausted, six-hats steers the transitions.

The MOTOR the user asked for: it advances the research autonomously, and at every
checkpoint a Six Thinking Hats judgment decides the direction. Mechanics
(deterministic) live here; judgment (model reasoning) lives in the /loop tick that
calls `tick` then `verdict`.

State machine (one route at a time, from route_ladder.py):

    idle ──tick──▶ launch route action in BACKGROUND ──▶ action_running
    action_running ──tick(done)──▶ build CHECKPOINT ──▶ awaiting_verdict
    awaiting_verdict ──verdict(succeed|pivot|continue)──▶ idle (next route / retry)
                                                       └▶ terminal (STOP_*)

The checkpoint carries the FACTS (metrics, success_when met?, mechanically
exhausted?, progress vs last attempt). The six-hats reads it and decides — the
engine never auto-pivots on judgment, only surfaces the facts. "until exhausted" =
attempts hit max with no success, OR the six-hats calls it.

INVARIANT: no Kaggle submission anywhere (search routes pass --seats but never submit).

Usage (driven by the /loop tick):
    .venv/bin/python -m scripts.research_loop.autopilot init        # once
    .venv/bin/python -m scripts.research_loop.autopilot tick        # advance one step
    .venv/bin/python -m scripts.research_loop.autopilot verdict --decision pivot --note "..."
    .venv/bin/python -m scripts.research_loop.autopilot status
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scripts.research_loop import route_ladder as L

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "research_loop"
STATE = ART / "autopilot_state.json"
LOGDIR = ART / "autopilot_logs"


def _now() -> float:
    return time.time()


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {
        "status": "running",          # running | win | exhausted | plateau | stopped
        "route": L.ENTRY,
        "attempt": 0,                 # attempts spent on the current route
        "phase": "idle",              # idle | action_running | awaiting_verdict
        "action": None,               # live background action descriptor
        "last_checkpoint": None,
        "history": [],                # one entry per completed checkpoint+verdict
    }


def save_state(s: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def _proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch(route_name: str, route: dict, attempt: int) -> dict:
    """Start the route's action as a detached background process."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    idx = max(0, attempt - 1)  # attempt is 1-based; schedule[0] for the first try
    if route["kind"] == "calibrate":
        seeds = route["seeds_schedule"][min(idx, len(route["seeds_schedule"]) - 1)]
        cmd = route["cmd"].format(seeds=seeds)
        fill = {"seeds": seeds}
    else:  # search
        hours = route["hours_schedule"][min(idx, len(route["hours_schedule"]) - 1)]
        cmd = route["cmd"].format(hours=hours)
        fill = {"hours": hours}
    artifact = REPO / route["artifact"]
    mtime_before = artifact.stat().st_mtime if artifact.exists() else 0.0
    log = LOGDIR / f"{route_name}_a{attempt}.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, shell=True, cwd=str(REPO), stdout=fh, stderr=fh,
                                start_new_session=True)
    return {
        "route": route_name, "attempt": attempt, "cmd": cmd, "fill": fill,
        "pid": proc.pid, "log": str(log), "artifact": str(artifact),
        "mtime_before": mtime_before, "started": _now(),
    }


def _action_done(action: dict) -> bool:
    if _proc_alive(action["pid"]):
        return False
    art = Path(action["artifact"])
    # finished process AND a fresh artifact = a real result (not a crash before writing)
    return art.exists() and art.stat().st_mtime > action["mtime_before"]


def _parse_artifact(route: dict) -> dict | None:
    art = REPO / route["artifact"]
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _checkpoint(s: dict, route_name: str, route: dict, blocked: str | None = None) -> dict:
    ckpt: dict = {
        "route": route_name, "kind": route["kind"], "attempt": s["attempt"],
        "max_attempts": route["max_attempts"], "ts": _now(),
    }
    if blocked:
        ckpt.update(status="blocked", blocked_reason=blocked, success=False,
                    mechanically_exhausted=True)
        return ckpt

    art = _parse_artifact(route)
    if art is None:
        ckpt.update(status="error", error="artifact missing or unparseable; check the route log",
                    success=False, mechanically_exhausted=s["attempt"] >= route["max_attempts"])
        return ckpt

    success = bool(route["success_when"](art))
    ckpt["status"] = "measured"
    ckpt["success"] = success
    ckpt["mechanically_exhausted"] = (s["attempt"] >= route["max_attempts"]) and not success

    if route["kind"] == "calibrate":
        ckpt["metrics"] = {
            "rho": art.get("rho"), "competitive_tied": art.get("competitive_tied"),
            "n_distinct_fitness": art.get("n_distinct_fitness"), "n": art.get("n"),
            "seats": art.get("seats"), "verdict": art.get("verdict"),
            "anchors": [{"name": r["name"], "lb": r["lb"], "fitness": r["fitness"]}
                        for r in art.get("rows", [])],
        }
    else:  # search
        ckpt["metrics"] = {
            "champion_fitness": art.get("champion_fitness"),
            "start_fitness": art.get("start_fitness"),
            "beats_holdwave_anchor": art.get("beats_holdwave_anchor"),
            "confirm": art.get("confirm"), "candidates": art.get("candidates_evaluated"),
            "trusted": art.get("trusted"),
        }

    # progress vs the previous checkpoint of the SAME route (a stalling signal)
    prev = next((h["checkpoint"] for h in reversed(s["history"])
                 if h["checkpoint"].get("route") == route_name
                 and h["checkpoint"].get("status") == "measured"), None)
    if prev and route["kind"] == "calibrate":
        d_rho = (ckpt["metrics"]["rho"] or 0) - (prev["metrics"]["rho"] or 0)
        tie_broke = bool(prev["metrics"]["competitive_tied"]) and not bool(ckpt["metrics"]["competitive_tied"])
        ckpt["progress"] = {"d_rho": round(d_rho, 4), "tie_broke": tie_broke}
    return ckpt


def cmd_tick(args) -> int:
    s = load_state()
    if s["status"] != "running":
        print(f"TERMINAL status={s['status']}: {L.TERMINALS.get(s.get('terminal_route',''), '')}")
        return 0
    route_name = s["route"]
    route = L.ROUTES[route_name]

    if s["phase"] == "idle":
        s["attempt"] += 1
        if route.get("blocked_reason") or route.get("cmd") is None:
            ckpt = _checkpoint(s, route_name, route, blocked=route.get("blocked_reason")
                               or "route action not wired (cmd is None)")
            s["last_checkpoint"] = ckpt
            s["phase"] = "awaiting_verdict"
            save_state(s)
            print("CHECKPOINT_READY")
            print(json.dumps(ckpt, indent=2))
            return 0
        action = _launch(route_name, route, s["attempt"])
        s["action"] = action
        s["phase"] = "action_running"
        save_state(s)
        print(f"LAUNCHED route={route_name} attempt={s['attempt']}/{route['max_attempts']} "
              f"pid={action['pid']} cmd={action['cmd']}")
        print(f"log={action['log']}")
        return 0

    if s["phase"] == "action_running":
        action = s["action"]
        if _action_done(action):
            ckpt = _checkpoint(s, route_name, route)
            s["last_checkpoint"] = ckpt
            s["phase"] = "awaiting_verdict"
            s["action"] = None
            save_state(s)
            print("CHECKPOINT_READY")
            print(json.dumps(ckpt, indent=2))
        else:
            elapsed = int(_now() - action["started"])
            print(f"RUNNING route={route_name} attempt={s['attempt']} pid={action['pid']} "
                  f"elapsed={elapsed}s (artifact not fresh yet)")
        return 0

    if s["phase"] == "awaiting_verdict":
        print("AWAITING_VERDICT — six-hats must judge the last checkpoint, then call `verdict`:")
        print(json.dumps(s["last_checkpoint"], indent=2))
        return 0
    return 0


def cmd_verdict(args) -> int:
    s = load_state()
    if s["phase"] != "awaiting_verdict":
        print(f"ERROR: phase is {s['phase']}, not awaiting_verdict. Run `tick` first.")
        return 1
    route_name = s["route"]
    route = L.ROUTES[route_name]
    decision = args.decision  # succeed | pivot | continue
    ckpt = s["last_checkpoint"]

    s["history"].append({
        "route": route_name, "attempt": s["attempt"], "decision": decision,
        "note": args.note or "", "ts": _now(),
        "checkpoint": ckpt,
    })

    nxt = None
    if decision == "succeed":
        nxt = route["on_success"]
        s["route"] = nxt
        s["attempt"] = 0
        s["phase"] = "idle"
    elif decision == "pivot":
        nxt = route["on_exhaust"]
        s["route"] = nxt
        s["attempt"] = 0
        s["phase"] = "idle"
    elif decision == "continue":
        s["phase"] = "idle"  # retry same route; attempt already counts up
    else:
        print(f"ERROR: unknown decision '{decision}' (use succeed|pivot|continue)")
        return 1

    if nxt and L.is_terminal(nxt):
        s["status"] = {"STOP_WIN": "win", "STOP_EXHAUSTED": "exhausted",
                       "STOP_PLATEAU": "plateau"}.get(nxt, "stopped")
        s["terminal_route"] = nxt
        s["phase"] = "done"

    save_state(s)
    print(f"VERDICT={decision} note={args.note!r}")
    if s["status"] != "running":
        print(f"TERMINAL: status={s['status']} — {L.TERMINALS.get(nxt, '')}")
    else:
        print(f"NEXT route={s['route']} phase={s['phase']} attempt={s['attempt']}")
    return 0


def cmd_status(args) -> int:
    s = load_state()
    print(f"status={s['status']} route={s['route']} phase={s['phase']} "
          f"attempt={s['attempt']}")
    if s.get("action"):
        a = s["action"]
        print(f"  action pid={a['pid']} alive={_proc_alive(a['pid'])} "
              f"elapsed={int(_now() - a['started'])}s log={a['log']}")
    if s.get("last_checkpoint"):
        print("  last_checkpoint:")
        print(json.dumps(s["last_checkpoint"], indent=2))
    print(f"  history: {len(s['history'])} checkpoints")
    for h in s["history"]:
        ck = h["checkpoint"]
        print(f"    - {h['route']} a{h['attempt']}: {ck.get('status')} "
              f"success={ck.get('success')} → {h['decision']} ({h['note'][:60]})")
    return 0


def cmd_init(args) -> int:
    if STATE.exists() and not args.force:
        print(f"state already exists at {STATE} (use --force to reset)")
        return cmd_status(args)
    if args.force and STATE.exists():
        STATE.unlink()
    s = load_state()
    save_state(s)
    print(f"initialized autopilot at {STATE}; entry route = {s['route']}")
    return 0


def cmd_stop(args) -> int:
    s = load_state()
    if s.get("action") and _proc_alive(s["action"]["pid"]):
        try:
            os.killpg(os.getpgid(s["action"]["pid"]), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    s["status"] = "stopped"
    s["phase"] = "idle"
    s["action"] = None
    save_state(s)
    print("autopilot stopped; any running action killed.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Autopilot: run a route until exhausted; six-hats steers.")
    sub = ap.add_subparsers(dest="sub", required=True)
    sub.add_parser("tick")
    pv = sub.add_parser("verdict")
    pv.add_argument("--decision", required=True, choices=("succeed", "pivot", "continue"))
    pv.add_argument("--note", default="")
    sub.add_parser("status")
    pi = sub.add_parser("init")
    pi.add_argument("--force", action="store_true")
    sub.add_parser("stop")
    args = ap.parse_args(argv)
    return {"tick": cmd_tick, "verdict": cmd_verdict, "status": cmd_status,
            "init": cmd_init, "stop": cmd_stop}[args.sub](args)


if __name__ == "__main__":
    raise SystemExit(main())
