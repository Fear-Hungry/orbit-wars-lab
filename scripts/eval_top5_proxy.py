"""T0 top-5 proxy ruler runner (G1.1).

Measures one or more candidate agents against the SAME fixed public proxy pool
declared in ``configs/eval_top5_proxy.yaml`` — 2p and 4p reported separately,
fixed seeds, thread-pinned. Default run evaluates the in-repo baseline pair
(Producer + OEP) and writes ``artifacts/top5_proxy/baseline_producer_oep.json``.

A future candidate is measured against the identical pool with::

    .venv/bin/python -m scripts.eval_top5_proxy \
        --candidate path/to/agent.py --label my_candidate --out /tmp/cand.json

so "beats Producer but loses to the pool" is directly visible against the same
ruler. Beating only the Producer is not a top-5 signal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _summ(s: dict) -> dict:
    keep = (
        "games", "win_rate", "mean_score_margin", "crash_rate", "timeout_rate",
        "invalid_action_rate", "fallback_rate", "fallback_error_rate",
    )
    return {k: s.get(k) for k in keep}


def _bench(entry: str, opponents: list[str], cfg: dict, out_json: Path, extra: list[str]) -> dict:
    """One benchmark_submission subprocess. Thread-pinned to the Kaggle CPU model
    so jobs==cores and contention can't manufacture timeouts."""
    ev = cfg["eval"]
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    )
    cmd = [
        sys.executable, "-m", "scripts.benchmark_submission",
        "--submission", entry,
        "--opponents", *opponents,
        "--seeds", str(int(ev["seeds"])),
        "--episode-steps", str(int(ev["episode_steps"])),
        "--jobs", str(int(ev.get("jobs", 5))),
        "--allow-technical-failures",
        "--out", str(out_json), *extra,
    ]
    if not ev.get("enable_comets", True):
        cmd.append("--disable-comets")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if not out_json.exists():
        raise RuntimeError(
            f"benchmark_submission produced no output for {entry} (rc={proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return json.loads(out_json.read_text())


def _run_candidate(entry: str, pool: list[dict], cfg: dict, tmp: Path, cid: str) -> dict:
    """2p is run once PER pool agent (every external is `<dir>/agent.py`, so the
    benchmark's path.stem label collapses them all to "agent" if batched) — we
    key by our own config id instead. 4p is run once over the whole pool (the
    pool fills the extra seats)."""
    result: dict = {"entry": entry, "2p": {}, "4p": {}}
    for p in pool:
        rep = _bench(entry, [p["entry"]], cfg, tmp / f"{cid}__2p__{p['id']}.json", ["--skip-4p"])
        s = rep["formats"][0]["opponents"][0]["summary"]
        result["2p"][p["id"]] = {**_summ(s), "declared_lb": p.get("declared_lb")}
    # This ruler is an explicit robustness VETO/diagnostic (see README + memory
    # top5_proxy_ruler_t0), NOT a promotion gate, so the seat-biased 4p path of
    # benchmark_submission is acknowledged here. (Promotion uses the seat-rotated
    # scripts/league_submit_ruler.py instead.)
    rep4 = _bench(entry, [p["entry"] for p in pool], cfg,
                  tmp / f"{cid}__4p.json", ["--skip-2p", "--ack-seat-biased"])
    fmt4 = next(f for f in rep4["formats"] if f["format"] == "4p")
    result["4p"] = {"pool": [p["id"] for p in pool], "summary": _summ(fmt4["summary"])}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval_top5_proxy.yaml")
    ap.add_argument("--candidate", help="entry .py of an extra candidate to measure")
    ap.add_argument("--label", help="label for --candidate")
    ap.add_argument("--out", help="override output json path")
    ap.add_argument("--only", nargs="*", help="subset of baseline candidate ids to run")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    pool = cfg["pool"]

    candidates = []
    if args.candidate:
        candidates.append({"id": args.label or Path(args.candidate).stem, "entry": args.candidate})
    else:
        for c in cfg["candidates"]:
            if args.only and c["id"] not in args.only:
                continue
            candidates.append(c)

    out_path = Path(args.out) if args.out else (ROOT / cfg["baseline"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "config": args.config,
        "seeds": int(cfg["eval"]["seeds"]),
        "episode_steps": int(cfg["eval"]["episode_steps"]),
        "pool": [{"id": p["id"], "declared_lb": p.get("declared_lb")} for p in cfg["pool"]],
        "candidates": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        for c in candidates:
            print(f"[top5_proxy] evaluating {c['id']} vs {len(pool)} pool agents ...", flush=True)
            results["candidates"][c["id"]] = _run_candidate(c["entry"], pool, cfg, Path(tmp), c["id"])
            print(f"[top5_proxy] done {c['id']}", flush=True)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"[top5_proxy] wrote {out_path}")


if __name__ == "__main__":
    main()
