"""Replay mining CLI for G1.2 — find WHERE our real losses come from.

Subcommands
-----------
  collect   ListEpisodes for the target Producer/OEP submissions, write an
            index of every episode (outcome / format / opponent) and download
            the LOSS replays, organised as
                replays/<submission_ref>/<outcome>/<opponent>/<episode_id>.json
  taxonomy  Parse + classify every downloaded loss replay into
                artifacts/replay_mining/loss_taxonomy.csv
            copy 5-10 exemplars per class under exemplars/<class>/ and print the
            top-3 loss patterns.

Run with the project venv (network + repo invariants):
    .venv/bin/python scripts/replay_mining/mine.py collect
    .venv/bin/python scripts/replay_mining/mine.py taxonomy
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay_mining import episode_service as es
from replay_mining.parse import CLASS_LABELS, classify_loss, parse_replay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "replay_mining"
REPLAYS = OUT / "replays"
EXEMPLARS = OUT / "exemplars"
INDEX = OUT / "episode_index.json"
TAXONOMY = OUT / "loss_taxonomy.csv"
SUMMARY = OUT / "loss_taxonomy_summary.md"

# Producer / OEP submissions that actually played the real leaderboard field.
TARGETS = {
    "producer_53366194": 53366194,
    "oep_53433131": 53433131,
    "oep_53582886": 53582886,
}


def _outcome(reward) -> str:
    if reward is None:
        return "unknown"
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "tie"


def _opponent_key(rec: dict) -> str:
    if rec["n_players"] == 2 and rec["opponents"]:
        return str(rec["opponents"][0]["submission_id"])
    return "4p"


def cmd_collect(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"targets": {}, "episodes": []}
    download_jobs: list[tuple[str, dict, Path]] = []

    for ref, sid in TARGETS.items():
        print(f"[list] {ref} (submission {sid}) ...", flush=True)
        eps = es.list_episodes(sid)
        recs = []
        for ep in eps:
            rec = es.episode_outcome(ep, sid)
            if rec is None:
                continue
            rec["submission_ref"] = ref
            rec["submission_id"] = sid
            rec["outcome"] = _outcome(rec["our_reward"])
            recs.append(rec)
            index["episodes"].append(rec)
            if rec["outcome"] == "loss":
                dest = REPLAYS / ref / "loss" / _opponent_key(rec) / f"{rec['episode_id']}.json"
                download_jobs.append((ref, rec, dest))
        counts = Counter((r["outcome"], r["n_players"]) for r in recs)
        index["targets"][ref] = {
            "submission_id": sid,
            "n_episodes": len(recs),
            "losses": sum(1 for r in recs if r["outcome"] == "loss"),
            "by_outcome_format": {f"{o}_{p}p": c for (o, p), c in sorted(counts.items())},
        }
        print(f"       {len(recs)} eps | {index['targets'][ref]['by_outcome_format']}", flush=True)

    INDEX.write_text(json.dumps(index, indent=1))
    print(f"[index] {INDEX} ({len(index['episodes'])} episodes)")

    if args.limit:
        download_jobs = download_jobs[: args.limit]
    todo = [j for j in download_jobs if not (j[2].exists() and j[2].stat().st_size > 0)]
    print(f"[download] {len(todo)} loss replays to fetch ({len(download_jobs)} total, rest cached)")

    done = 0
    failed = []

    def _fetch(job):
        ref, rec, dest = job
        es.download_replay(rec["episode_id"], dest)
        return rec["episode_id"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_fetch, j): j for j in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
                if done % 25 == 0:
                    print(f"       {done}/{len(todo)}", flush=True)
            except Exception as exc:  # pragma: no cover - network
                failed.append((futs[fut][1]["episode_id"], str(exc)))
    print(f"[download] done={done} failed={len(failed)}")
    for eid, err in failed[:10]:
        print(f"   FAIL {eid}: {err}")


def cmd_taxonomy(args: argparse.Namespace) -> None:
    replay_files = sorted(REPLAYS.glob("*/loss/*/*.json"))
    if not replay_files:
        sys.exit("no loss replays found — run `collect` first")
    print(f"[parse] {len(replay_files)} loss replays")

    rows = []
    by_class: dict[str, list[tuple[dict, Path]]] = defaultdict(list)
    parse_fail = 0
    for path in replay_files:
        ref = path.parents[2].name
        try:
            replay = json.loads(path.read_text())
            feat = parse_replay(replay)
        except Exception as exc:
            parse_fail += 1
            print(f"   parse FAIL {path.name}: {exc}")
            continue
        cls = classify_loss(feat)
        row = {"submission_ref": ref, **feat,
               "primary_class": cls["primary_class"],
               "flags": "|".join(cls["flags"])}
        rows.append(row)
        by_class[cls["primary_class"]].append((row, path))

    # CSV.
    fields = list(rows[0].keys())
    with TAXONOMY.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {TAXONOMY} ({len(rows)} losses, {parse_fail} parse failures)")

    # Exemplars: copy up to `n_examples` per class (prefer strongest signal).
    if EXEMPLARS.exists():
        shutil.rmtree(EXEMPLARS)
    for cls, items in by_class.items():
        # strongest = biggest collapse / clearest signal first
        items_sorted = sorted(items, key=lambda x: -(x[0]["max_ship_drop"] or 0))
        for row, path in items_sorted[: args.n_examples]:
            dest = EXEMPLARS / cls / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest)

    # Summary.
    counts = Counter(r["primary_class"] for r in rows)
    fmt_counts = Counter((r["primary_class"], r["format"]) for r in rows)
    total = len(rows)
    top = counts.most_common()
    lines = ["# Loss taxonomy — real leaderboard losses (G1.2)", ""]
    lines.append(f"Source: {total} loss replays from Producer/OEP submissions "
                 f"({', '.join(TARGETS)}).")
    lines.append("Score proxy = total controlled ships; classifier rule-based "
                 "(evidence strength PARTIAL).")
    lines.append("")
    lines.append("| rank | class | losses | share | 2p | 4p |")
    lines.append("|------|-------|--------|-------|----|----|")
    for i, (cls, n) in enumerate(top, 1):
        n2 = fmt_counts.get((cls, "2p"), 0)
        n4 = fmt_counts.get((cls, "4p"), 0)
        lines.append(f"| {i} | {CLASS_LABELS.get(cls, cls)} | {n} | "
                     f"{n / total:.1%} | {n2} | {n4} |")
    lines.append("")
    lines.append("## Top-3 patterns")
    for i, (cls, n) in enumerate(top[:3], 1):
        ex = [r["episode_id"] for r, _ in sorted(by_class[cls],
              key=lambda x: -(x[0]["max_ship_drop"] or 0))[:args.n_examples]]
        lines.append(f"{i}. **{CLASS_LABELS.get(cls, cls)}** — {n} losses "
                     f"({n / total:.1%}). Exemplar episodes: {ex}")
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[summary] {SUMMARY}")
    print(f"[exemplars] {EXEMPLARS} (<= {args.n_examples} per class)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="list episodes + download loss replays")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--limit", type=int, default=0, help="cap total downloads (0 = all losses)")
    c.set_defaults(func=cmd_collect)
    t = sub.add_parser("taxonomy", help="parse + classify + emit loss_taxonomy.csv")
    t.add_argument("--n-examples", type=int, default=8)
    t.set_defaults(func=cmd_taxonomy)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
