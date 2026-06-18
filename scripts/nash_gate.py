"""Nash gate — the operational promotion gate defined by goal.md (2026-06-18).

Decides whether a candidate (default: H11 = `pgs_valuenet_attn`, the AttnValueNet
learned evaluator inside the PGS search) earns entry as a learned value head, by
its ONLINE advantage against the MAXENT-NASH MIXTURE of a fixed strong pool — not
the redundancy-biased mean-vs-pool that the local league was falsified on
(memory: local_league_is_submission_gate; Balduzzi et al. 2018, scripts/nash_eval.py).

Pool minimo (goal.md): incumbent pgs_holdwave + producer + oep + brep +
pgs_bigwave + greedy + rush.

Primary metric (goal.md "Metrica primaria"):
  - ref-vs-ref pairwise win-rate matrix W (seat-rotated 2p, the trusted ruler);
  - p* = maxent_nash(W);
  - adv_j = winrate(cand, ref_j) - winrate(ref_j, cand);
  - competitive signal approved iff sum_j adv_j * p*_j > 0  (= nash_advantage_vs_pool).

PASS (goal.md "Criterio de sucesso", the gate-side subset) requires ALL:
  - nash_advantage_vs_pool > 0.0 with a valid sample;
  - no_faults: crashes=timeouts=invalid_moves=0 AND bad_status=0;
  - all subprocesses rc==0 (no silent fallback) and all games audited (instrumented);
  - 4p not failed: decisive_win_rate_4p >= 0.25 OR measurable annihilation
    improvement vs pgs_holdwave in the same 4p slice.
Else INCONCLUSIVE (no valid decisive sample) or REJECT_LOCAL.

The unit-test, checkpoint, and DB-registration criteria of goal.md are handled
outside this script (pytest + python.lab.experiments add).

Run:
  .venv/bin/python -m scripts.nash_gate --seeds 12 --jobs 6
  .venv/bin/python -m scripts.nash_gate --candidate pgs_valuenet_attn --seeds 24 --jobs 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts.league_agents import FACTORIES, LB_ANCHORS
from scripts.league_submit_ruler import (
    MatchTask,
    _candidate_p95,
    _games_for_seat,
    _load_games,
    _safe_label,
    _score_games,
    _seed_slice_base,
    run_tasks,
)
from scripts.nash_eval import maxent_nash, nash_advantage_vs_pool, winrate_to_advantage
from scripts.nash_lb_calibrate import build_winrate_matrix

ROOT = Path(__file__).resolve().parent.parent

POOL = ["pgs_holdwave", "producer", "oep", "brep", "pgs_bigwave", "greedy", "rush"]
INCUMBENT = "pgs_holdwave"
# 4p slices: each puts the candidate against the incumbent + two strong peers so
# decisive_win_rate_4p and the "same slice vs incumbent" comparison are both read
# off the SAME games. Lineups are 4 distinct FACTORIES agents; seats rotate by seed.
FOUR_P_LINEUPS = [
    ("pgs_holdwave", "producer", "oep"),
    ("pgs_holdwave", "brep", "pgs_bigwave"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _candidate_checkpoint(candidate: str) -> dict:
    """Best-effort path+hash of the artefact the candidate loads (goal.md requires
    'hash ou caminho do candidato avaliado'). For pgs_valuenet* this is the value
    net checkpoint; otherwise just the FACTORIES name."""
    info = {"agent": candidate, "value_net_path": None, "value_net_sha256": None}
    if "valuenet_attn" in candidate:
        p = ROOT / "artifacts/h7/value_net_attn.pt"
    elif "valuenet" in candidate:
        p = ROOT / "artifacts/h7/value_net.pt"
    else:
        return info
    if p.exists():
        info["value_net_path"] = str(p.relative_to(ROOT))
        info["value_net_sha256"] = _sha256(p)
    return info


def build_candidate_tasks(candidate: str, *, seeds: int, seed_base: int, steps: int,
                          out_dir: Path) -> list[MatchTask]:
    tasks = []
    for ref in POOL:
        label = _safe_label([candidate, "2p", ref])
        tasks.append(MatchTask(
            label=label, mode="2p", candidate=candidate, names=(candidate, ref),
            seeds=seeds, seed_base=_seed_slice_base(seed_base, "2p_cand", ref),
            steps=steps, out=out_dir / f"{label}.json"))
    for i, lineup in enumerate(FOUR_P_LINEUPS):
        names = (candidate, *lineup)
        label = _safe_label([candidate, "4p", f"line{i}"])
        tasks.append(MatchTask(
            label=label, mode="4p", candidate=candidate, names=names,
            seeds=seeds, seed_base=seed_base + 7000 + 100 * i,
            steps=steps, out=out_dir / f"{label}.json"))
    return tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="pgs_valuenet_attn", help="FACTORIES agent under test")
    ap.add_argument("--seeds", type=int, default=12, help="seeds per pair (4p slices need %4==0)")
    ap.add_argument("--steps", type=int, default=500, help="episode horizon (500 = official)")
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=6, help="<= physical cores (threads pinned)")
    ap.add_argument("--stamp", default=None, help="report filename stamp (default: epoch seconds)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "h11")
    ap.add_argument("--ref-matrix-from", type=Path, default=None,
                    help="reuse the ref-vs-ref matrix from a prior gate report JSON "
                         "(candidate-independent; same pool/seeds/seed-base required). "
                         "Skips the 21 ref-vs-ref matches — only the candidate is re-run.")
    args = ap.parse_args(argv)

    candidate = args.candidate
    if candidate not in FACTORIES:
        raise SystemExit(f"unknown candidate {candidate!r}")
    missing = [n for n in POOL if n not in FACTORIES]
    if missing:
        raise SystemExit(f"pool entries not in FACTORIES: {missing}")
    stamp = args.stamp or str(int(time.time()))
    work = args.out_dir / f"gate_{candidate}_{stamp}_matches"
    work.mkdir(parents=True, exist_ok=True)

    print(f"=== Nash gate: {candidate} vs maxent-Nash mixture of the fixed pool ===")
    print(f"pool={POOL}\nincumbent={INCUMBENT} seeds={args.seeds} steps={args.steps} jobs={args.jobs}\n")
    t0 = time.perf_counter()

    # 1) ref-vs-ref matrix -> Nash mixture p* over the pool. Candidate-independent,
    # so a prior report's matrix can be reused verbatim when pool/seeds/seed-base match.
    if args.ref_matrix_from is not None:
        prior = json.loads(args.ref_matrix_from.read_text())
        if prior.get("pool") != POOL or prior.get("seeds") != args.seeds:
            raise SystemExit(
                f"--ref-matrix-from mismatch: prior pool/seeds "
                f"{prior.get('pool')}/{prior.get('seeds')} != {POOL}/{args.seeds}")
        print(f"--- [1/2] reusing ref-vs-ref matrix from {args.ref_matrix_from} ---", flush=True)
        W_ref = np.array(prior["ref_vs_ref_matrix"], dtype=float)
        ref_diag = prior.get("ref_vs_ref_pairs", {})
    else:
        print("--- [1/2] ref-vs-ref pool matrix (seat-rotated 2p) ---", flush=True)
        W_ref, ref_diag = build_winrate_matrix(
            POOL, seeds=args.seeds, seed_base=args.seed_base, steps=args.steps,
            jobs=args.jobs, out_dir=work / "ref_vs_ref", progress=True)
    p_star = maxent_nash(winrate_to_advantage(W_ref))

    # 2) candidate vs every ref (2p) + candidate in 4p slices
    print("\n--- [2/2] candidate matches (2p vs each ref + 4p slices) ---", flush=True)
    tasks = build_candidate_tasks(candidate, seeds=args.seeds, seed_base=args.seed_base,
                                  steps=args.steps, out_dir=work / "candidate")
    results = run_tasks(tasks, args.jobs, progress=True)
    by_label = {r["label"]: r for r in results}

    failed_runs = [r for r in results if r["returncode"] != 0]

    # candidate 2p pairwise + the antisymmetric advantage vector
    cand_vs_pool, pool_vs_cand, pair_2p = [], [], {}
    all_cand_games = []
    for ref in POOL:
        r = by_label[_safe_label([candidate, "2p", ref])]
        games = _load_games(Path(r["out"])) if r["returncode"] == 0 else []
        all_cand_games.extend(games)
        s = _score_games(games, ref_name := candidate)
        wins, losses, app = s["wins"], s["losses"], s["appearances"]
        cvp = wins / app if app else 0.0
        pvc = losses / app if app else 0.0
        cand_vs_pool.append(cvp)
        pool_vs_cand.append(pvc)
        # per-seat decisive WR vs incumbent (false-gain guard)
        seat_wr = {}
        if ref == INCUMBENT:
            for si in (0, 1):
                ss = _score_games(_games_for_seat(games, candidate, si), candidate)
                seat_wr[f"seat{si}"] = ss["decisive_win_rate"]
        pair_2p[ref] = {
            "win_rate": round(cvp, 4), "ref_win_rate": round(pvc, 4),
            "decisive_win_rate": s["decisive_win_rate"], "decisive": s["decisive"],
            "appearances": app, "annihilation_rate": round(s["annihilation_rate"], 4),
            "adv": round(cvp - pvc, 4), "seat_decisive_wr": seat_wr,
            "rc": r["returncode"],
        }

    # candidate 4p (aggregate over slices) + incumbent in the SAME 4p games
    four_games = []
    for i in range(len(FOUR_P_LINEUPS)):
        r = by_label[_safe_label([candidate, "4p", f"line{i}"])]
        if r["returncode"] == 0:
            four_games.extend(_load_games(Path(r["out"])))
    all_cand_games.extend(four_games)
    cand_4p = _score_games(four_games, candidate)
    inc_4p = _score_games(four_games, INCUMBENT)  # same slice, for the OR-clause

    # overall faults / status / instrumentation across ALL candidate games
    overall = _score_games(all_cand_games, candidate)
    faults = overall["faults"]
    bad_status = overall["bad_status"]
    audited_ok = overall["audited"] == overall["appearances"]
    p95 = _candidate_p95(all_cand_games, candidate)

    # Nash advantage vs the pool mixture
    nash_adv = nash_advantage_vs_pool(
        np.array(cand_vs_pool), np.array(pool_vs_cand), W_ref)
    contrib = [round(float((cand_vs_pool[j] - pool_vs_cand[j]) * p_star[j]), 4)
               for j in range(len(POOL))]

    # ---- goal.md verdict ----
    valid_sample = all(pair_2p[r]["decisive"] > 0 for r in POOL) and cand_4p["decisive"] > 0
    competitive = nash_adv > 0.0
    technical_clean = (not any(faults.values())) and bad_status == 0 and not failed_runs and audited_ok
    fp_wr = cand_4p["decisive_win_rate"]
    fp_pass_mainclause = fp_wr is not None and fp_wr >= 0.25
    fp_annih_improves = (cand_4p["appearances"] > 0 and inc_4p["appearances"] > 0
                         and cand_4p["annihilation_rate"] < inc_4p["annihilation_rate"])
    four_p_ok = fp_pass_mainclause or fp_annih_improves

    if not technical_clean:
        verdict, decision = "REJECT_LOCAL", "rejeitar"
        why = "technical gate failed (faults/bad_status/fallback/instrumentation)"
    elif not valid_sample:
        verdict, decision = "INCONCLUSIVE", "aumentar seeds"
        why = "no valid decisive sample on some pair/4p slice"
    elif not competitive:
        verdict, decision = "REJECT_LOCAL", "rejeitar"
        why = f"nash_advantage_vs_pool={nash_adv:+.4f} <= 0 (no edge vs the Nash mixture)"
    elif not four_p_ok:
        verdict, decision = "REJECT_LOCAL", "rejeitar"
        why = f"4p fails: decisive_wr_4p={fp_wr} < 0.25 and no annihilation improvement vs incumbent"
    else:
        verdict, decision = "PASS_LOCAL", "promover"
        why = (f"nash_advantage_vs_pool={nash_adv:+.4f}>0, clean technical, "
               f"4p decisive_wr={fp_wr} (>=0.25={fp_pass_mainclause}, annih_improve={fp_annih_improves})")

    # ---- report ----
    print("\n=== ref-vs-ref Nash mixture p* ===")
    for n, w in sorted(zip(POOL, p_star), key=lambda kv: -kv[1]):
        print(f"  {n:14s} {w:6.3f}")
    print("\n=== candidate 2p vs pool (adv = cand_wr - ref_wr; contrib = adv * p*) ===")
    print(f"{'ref':14s} {'p*':>6s} {'cand_wr':>8s} {'ref_wr':>7s} {'adv':>7s} {'contrib':>8s} {'dec_wr':>7s}")
    for j, ref in enumerate(POOL):
        d = pair_2p[ref]
        dw = "n/a" if d["decisive_win_rate"] is None else f"{d['decisive_win_rate']:.2f}"
        print(f"{ref:14s} {p_star[j]:6.3f} {d['win_rate']:8.2f} {d['ref_win_rate']:7.2f} "
              f"{d['adv']:+7.2f} {contrib[j]:+8.3f} {dw:>7s}")
    print(f"\nnash_advantage_vs_pool = {nash_adv:+.4f}")
    print(f"4p: decisive_wr={fp_wr} annihilation={cand_4p['annihilation_rate']:.3f} "
          f"(incumbent same-slice annih={inc_4p['annihilation_rate']:.3f}) appearances={cand_4p['appearances']}")
    print(f"faults={faults} bad_status={bad_status} audited_ok={audited_ok} "
          f"failed_runs={len(failed_runs)} p95_ms={p95}")
    print(f"\nVERDICT: {verdict}  (decision={decision})\n  {why}")
    print(f"(elapsed {time.perf_counter() - t0:.0f}s)")

    report = {
        "candidate": candidate,
        "candidate_checkpoint": _candidate_checkpoint(candidate),
        "pool": POOL, "incumbent": INCUMBENT,
        "seeds": args.seeds, "steps": args.steps, "stamp": stamp,
        "ref_vs_ref_matrix": W_ref.tolist(),
        "nash_mixture": {n: round(float(w), 4) for n, w in zip(POOL, p_star)},
        "nash_advantage_vs_pool": float(nash_adv),
        "nash_contributions": {n: contrib[j] for j, n in enumerate(POOL)},
        "split_2p": pair_2p,
        "split_4p": {
            "candidate": {"decisive_win_rate": cand_4p["decisive_win_rate"],
                          "decisive": cand_4p["decisive"], "wins": cand_4p["wins"],
                          "appearances": cand_4p["appearances"],
                          "annihilation_rate": round(cand_4p["annihilation_rate"], 4)},
            "incumbent_same_slice": {"annihilation_rate": round(inc_4p["annihilation_rate"], 4),
                                     "decisive_win_rate": inc_4p["decisive_win_rate"]},
            "lineups": [list((candidate, *lu)) for lu in FOUR_P_LINEUPS],
        },
        "technical": {"faults": faults, "bad_status": bad_status,
                      "audited_ok": audited_ok, "failed_runs": len(failed_runs),
                      "p95_ms": p95},
        "ref_vs_ref_pairs": ref_diag,
        "valid_sample": valid_sample,
        "verdict": verdict, "decision": decision, "why": why,
    }
    out = args.out_dir / f"nash_gate_{candidate}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
