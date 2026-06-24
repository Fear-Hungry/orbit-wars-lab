"""Strict local ruler: VETO + feature extraction for the submission selector.

This is deliberately different from the continuous league. The continuous league
is exploratory and population-biased by construction; this script runs a paired,
balanced schedule against FIXED references and emits an explicit local verdict.

CONTRACT (selector v1): the ruler NEVER recommends a submission. Its report
carries `local_veto_passes` (who survived the hard gates), split scores
(`score_2p_fixed` / `score_2p_peer` / `score_4p_fixed`), normalized advantages
and risk features. `selection_status` stays "VETO_ONLY" and
`promotion_order_valid` stays false in this report — choosing a candidate is
the job of scripts/league_submission_selector.py, and only with a valid
scripts/league_selector_calibration.py artifact (the league was falsified as a
promotion gate on 2026-06-10: local Spearman vs LB = 0.0).

Funnel: quick (smoke) -> standard (dev filter) -> strong (serious veto, top
2-3) -> selector (submission choice, holdout seed split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_agents import CRITICAL_BUCKETS, FACTORIES, INCUMBENT, bucket_of  # noqa: E402
from scripts.league_report import decisive_winner  # noqa: E402

DEFAULT_REFERENCES = [
    "producer",
    "oep",
    "pgs_holdwave",
    "pgs_wave_s100",
    "pgs_bigwave",
    "pgs_allscripts",
    "ext_lb1050",
    "ext_hellburner",
]

DEFAULT_4P_TEMPLATES = [
    ("producer", "oep", "pgs_holdwave"),
    ("producer", "pgs_bigwave", "pgs_allscripts"),
    ("oep", "pgs_wave_s100", "pgs_bigwave"),
    ("pgs_holdwave", "pgs_allscripts", "ext_hellburner"),
]

# 4p lineups are completed ONLY from this fixed list — never from peer
# candidates. A peer filler makes the candidate's 4p score depend on who else
# is being evaluated in the same command (panel dependence).
DEFAULT_4P_FILLERS = ["producer", "oep", "pgs_bigwave", "ext_hellburner"]

PROFILE_DEFAULTS = {
    "quick": {"seeds": 4, "steps": 250, "min_decisive_2p": 4},
    "standard": {"seeds": 8, "steps": 500, "min_decisive_2p": 12},
    "strong": {"seeds": 24, "steps": 500, "min_decisive_2p": 40},
    # submission choice: only top 2-3 candidates, holdout seed split
    "selector": {"seeds": 48, "steps": 500, "min_decisive_2p": 100},
}

# Disjoint seed universes. dev burns freely during development; validation is
# the veto/calibration split; selector is the holdout used ONLY for the final
# submission decision — a bot changed after seeing selector results must wait
# for a fresh holdout split.
SEED_SPLITS = {"dev": 70_000, "validation": 170_000, "selector": 270_000}

FIELD_2P_WEIGHT = 0.46  # DB id=168: Kaggle env sample was 46% 2p / 54% 4p.
_VERDICT_PRIORITY = {"PASS_LOCAL": 2, "INCONCLUSIVE": 1, "REJECT_LOCAL": 0}

# Latency ladder (decision_ms_p95). >hard never ships; warn feeds risk_penalty.
LATENCY_WARN_MS = 500.0
LATENCY_HARD_MS = 800.0

# Stricter 4p levels applied by the SELECTOR when choosing a submission (the
# ruler's own hard gates use the pass-to-selector levels in the CLI defaults).
CHOICE_GATES_4P = {
    "min_4p_winrate": 0.28,
    "max_4p_annihilation": 0.20,
    "min_worst_template_4p_winrate": 0.18,
}


@dataclass(frozen=True)
class MatchTask:
    label: str
    mode: str
    candidate: str
    names: tuple[str, ...]
    seeds: int
    seed_base: int
    steps: int
    out: Path
    chunk_size: int = 0
    # fixed_2p / fixed_4p enter the canonical submission score; peer_2p is
    # candidate-vs-candidate H2H, diagnostic only.
    role: str = "fixed_2p"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _known(names: Iterable[str]) -> list[str]:
    return [name for name in names if name in FACTORIES]


def _safe_label(parts: Iterable[str]) -> str:
    return "__".join(part.replace("/", "_") for part in parts)


def _seed_slice_base(seed_base: int, mode: str, label: str) -> int:
    digest = hashlib.sha1(f"{mode}:{label}".encode()).hexdigest()
    return seed_base + 100 * (int(digest[:8], 16) % 100_000)


def _complete_4p_lineup(candidate: str, template: tuple[str, ...], references: list[str]) -> tuple[str, ...] | None:
    names = [candidate]
    for name in template:
        if name != candidate and name not in names and name in FACTORIES:
            names.append(name)
    for name in references:
        if len(names) >= 4:
            break
        if name != candidate and name not in names and name in FACTORIES:
            names.append(name)
    return tuple(names[:4]) if len(names) == 4 else None


def build_tasks(
    candidates: list[str],
    *,
    incumbent: str,
    references: list[str],
    four_player_templates: list[tuple[str, ...]],
    seeds: int,
    seed_base: int,
    steps: int,
    out_dir: Path,
    match_chunk_size: int = 0,
    peer_h2h: bool = True,
    fixed_4p_references: list[str] | None = None,
) -> list[MatchTask]:
    """Panel-independent schedule: a candidate's fixed-reference tasks (2p and
    4p) are a pure function of (candidate, fixed panel, seeds) — adding or
    removing other candidates from the command MUST NOT change them. Peers meet
    only in dedicated `peer_2p` diagnostic tasks."""
    tasks: list[MatchTask] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    fillers = [n for n in (fixed_4p_references or DEFAULT_4P_FILLERS) if n in FACTORIES]
    fixed_panel_names = {incumbent, *references}
    for candidate in candidates:
        if candidate not in FACTORIES:
            raise ValueError(f"unknown candidate: {candidate}")
        fixed_panel = []
        for name in [incumbent, *references]:
            if name != candidate and name in FACTORIES and name not in fixed_panel:
                fixed_panel.append(name)
        for ref in fixed_panel:
            names = (candidate, ref)
            key = ("2p", names)
            if key in seen:
                continue
            seen.add(key)
            # Shared, stable seeds per reference across candidates and across
            # panel compositions: candidate A vs producer and candidate B vs
            # producer must see the same map slice even if a later run adds a
            # new peer candidate to the command.
            base = _seed_slice_base(seed_base, "2p", ref)
            label = _safe_label([candidate, "2p", ref])
            tasks.append(MatchTask(
                label=label,
                mode="2p",
                candidate=candidate,
                names=names,
                seeds=seeds,
                seed_base=base,
                steps=steps,
                out=out_dir / f"{label}.json",
                chunk_size=max(0, int(match_chunk_size)),
                role="fixed_2p",
            ))
        for tpl_idx, template in enumerate(four_player_templates):
            names = _complete_4p_lineup(candidate, template, fillers)
            if names is None:
                continue
            key = ("4p", names)
            if key in seen:
                continue
            seen.add(key)
            # Shared 4p seeds per template across candidates for the same reason
            # as the 2p tasks above. league_match rotates seats by seed index, so
            # requiring seeds % 4 == 0 gives balanced seats per 4p template.
            base = seed_base + 5_000 + 100 * tpl_idx
            label = _safe_label([candidate, "4p", f"line{tpl_idx}"])
            tasks.append(MatchTask(
                label=label,
                mode="4p",
                candidate=candidate,
                names=names,
                seeds=seeds,
                seed_base=base,
                steps=steps,
                out=out_dir / f"{label}.json",
                chunk_size=max(0, int(match_chunk_size)),
                role="fixed_4p",
            ))
    if peer_h2h:
        uniq = list(dict.fromkeys(candidates))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                lo, hi = sorted((a, b))
                if lo in fixed_panel_names or hi in fixed_panel_names:
                    continue  # the matchup already exists as a fixed_2p task
                key = ("2p", (lo, hi))
                if key in seen:
                    continue
                seen.add(key)
                base = _seed_slice_base(seed_base, "2p_peer", f"{lo}__{hi}")
                label = _safe_label([lo, "2p_peer", hi])
                tasks.append(MatchTask(
                    label=label,
                    mode="2p",
                    candidate=lo,
                    names=(lo, hi),
                    seeds=seeds,
                    seed_base=base,
                    steps=steps,
                    out=out_dir / f"{label}.json",
                    chunk_size=max(0, int(match_chunk_size)),
                    role="peer_2p",
                ))
    return tasks


def _run_task(task: MatchTask) -> dict[str, Any]:
    task.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/league_match.py",
        "--agents",
        ",".join(task.names),
        "--seeds",
        str(task.seeds),
        "--seed-base",
        str(task.seed_base),
        "--steps",
        str(task.steps),
        "--out",
        str(task.out),
    ]
    if task.chunk_size > 0:
        cmd += ["--chunk-size", str(task.chunk_size)]
    start = time.perf_counter()
    # Pin BLAS/OMP threads to 1 (matching scripts.eval_top5_proxy) so jobs==cores
    # and CPU oversubscription can't manufacture false per-move timeouts. The
    # tarball league wrapper treats ANY fallback/timeout delta as a crash, so an
    # unpinned, contended run inflates the fault count and masks the real H2H.
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    return {
        "label": task.label,
        "mode": task.mode,
        "candidate": task.candidate,
        "names": list(task.names),
        "role": task.role,
        "seeds": task.seeds,
        "seed_base": task.seed_base,
        "steps": task.steps,
        "chunk_size": task.chunk_size,
        "out": str(task.out),
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - start,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


def _write_task_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def run_tasks(
    tasks: list[MatchTask],
    jobs: int,
    *,
    progress: bool = False,
    task_results_out: Path | None = None,
) -> list[dict[str, Any]]:
    def _note(done: int, total: int, result: dict[str, Any]) -> None:
        if progress:
            print(
                f"[{done:02d}/{total:02d}] {result['label']} rc={result['returncode']} "
                f"{result['seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    if jobs <= 1:
        out = []
        total = len(tasks)
        for task in tasks:
            result = _run_task(task)
            out.append(result)
            _note(len(out), total, result)
            if task_results_out is not None:
                _write_task_results(task_results_out, out)
        return out
    out = []
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            out.append(result)
            _note(len(out), total, result)
            if task_results_out is not None:
                _write_task_results(task_results_out, out)
    return out


def _task_has_strict_metadata(result: dict[str, Any]) -> bool:
    return all(key in result for key in ("mode", "names", "seeds", "seed_base", "steps"))


def _expected_seat_counter(result: dict[str, Any]) -> Counter[tuple[int, tuple[str, ...]]]:
    names = tuple(str(name) for name in result["names"])
    seeds = [int(result["seed_base"]) + i for i in range(int(result["seeds"]))]
    counter: Counter[tuple[int, tuple[str, ...]]] = Counter()
    if str(result["mode"]) == "2p":
        for seed in seeds:
            counter[(seed, names)] += 1
            counter[(seed, tuple(reversed(names)))] += 1
    elif str(result["mode"]) == "4p":
        for seed in seeds:
            for r in range(4):
                counter[(seed, names[r:] + names[:r])] += 1
    return counter


def _validate_task_payload(path: Path, payload: dict[str, Any], games: list[dict[str, Any]],
                           result: dict[str, Any]) -> None:
    errors: list[str] = []
    mode = str(result["mode"])
    names = tuple(str(name) for name in result["names"])
    expected_count = int(result["seeds"]) * (2 if mode == "2p" else 4)

    if payload.get("mode") != mode:
        errors.append(f"mode {payload.get('mode')!r} != {mode!r}")
    if tuple(payload.get("agents") or ()) != names:
        errors.append(f"agents {payload.get('agents')!r} != {list(names)!r}")
    if payload.get("seed_base") != int(result["seed_base"]):
        errors.append(f"seed_base {payload.get('seed_base')!r} != {int(result['seed_base'])!r}")
    if payload.get("seed_count") != int(result["seeds"]):
        errors.append(f"seed_count {payload.get('seed_count')!r} != {int(result['seeds'])!r}")
    if payload.get("steps") != int(result["steps"]):
        errors.append(f"steps {payload.get('steps')!r} != {int(result['steps'])!r}")
    if len(games) != expected_count:
        errors.append(f"games {len(games)} != expected {expected_count}")

    actual: Counter[tuple[int, tuple[str, ...]]] = Counter()
    for idx, game in enumerate(games):
        seats = tuple(str(name) for name in (game.get("seats") or ()))
        if len(seats) != len(names):
            errors.append(f"game[{idx}] has invalid seats {list(seats)!r}")
        try:
            seed = int(game["seed"])
            actual[(seed, seats)] += 1
        except (KeyError, TypeError, ValueError):
            errors.append(f"game[{idx}] has invalid seed {game.get('seed')!r}")
        if "faults" not in game or not isinstance(game.get("faults"), dict):
            errors.append(f"game[{idx}] missing audited faults dict")
        status = game.get("agent_status")
        if (
            not isinstance(status, list)
            or len(status) != len(seats)
            or any(item not in {"DONE", "ERROR", "TIMEOUT"} for item in status)
        ):
            errors.append(f"game[{idx}] has invalid agent_status {status!r}")

    expected = _expected_seat_counter(result)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f"missing seed/seat entries {list(missing.elements())[:4]!r}")
        if extra:
            errors.append(f"unexpected seed/seat entries {list(extra.elements())[:4]!r}")

    if errors:
        label = result.get("label", path.name)
        sample = "; ".join(errors[:8])
        if len(errors) > 8:
            sample += f"; ... +{len(errors) - 8} more"
        raise ValueError(f"{path} does not match task {label}: {sample}")


def _load_games(path: Path, result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    games, _meta = _load_task_payload(path, result)
    return games


def _load_task_payload(
    path: Path, result: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Games plus the per-agent latency metadata the payload already carries
    (old payloads predate decision_ms_*; meta flags them as unaudited)."""
    payload = json.loads(path.read_text())
    games = payload["games"]
    if result is not None and _task_has_strict_metadata(result):
        _validate_task_payload(path, payload, games, result)
    # league_match writes the per-agent p95 decision latency at payload level
    # (decision_ms_p95). Attach it to each game so the candidate's worst-case
    # latency can be gated alongside win-rate (Kaggle enforces a 1s actTimeout;
    # a fast local win that p95-blows the budget is a false promotion).
    p95 = payload.get("decision_ms_p95") or {}
    for game in games:
        game["mode"] = payload["mode"]
    meta = {
        "decision_ms_p95": payload.get("decision_ms_p95"),
        "decision_ms_max": payload.get("decision_ms_max"),
        "latency_audited": "decision_ms_p95" in payload,
    }
    return games, meta


def _result_role(result: dict[str, Any]) -> str:
    """Old task_results predate roles; everything they ran was fixed-panel."""
    role = result.get("role")
    if role:
        return str(role)
    return "fixed_2p" if str(result.get("mode")) == "2p" else "fixed_4p"


def _games_for_seat(games: list[dict[str, Any]], name: str, seat_index: int) -> list[dict[str, Any]]:
    """The subset of games where `name` occupies a specific seat. With balanced
    rotation each 2p seat gets ~half the games, so a candidate that only wins from
    one seat (a false gain) splits cleanly here."""
    return [
        game for game in games
        if name in game["seats"] and game["seats"].index(name) == seat_index
    ]


def _candidate_p95(games: list[dict[str, Any]], name: str) -> float | None:
    """Worst (max) p95 decision latency the candidate hit across all its payloads,
    or None when no payload carried latency instrumentation."""
    values = [
        float(game["payload_p95"][name])
        for game in games
        if isinstance(game.get("payload_p95"), dict) and name in game["payload_p95"]
    ]
    return max(values) if values else None


def _fault_totals(games: list[dict[str, Any]], name: str) -> dict[str, int]:
    totals = {"crashes": 0, "timeouts": 0, "invalid_moves": 0}
    for game in games:
        faults = game.get("faults") or {}
        entry = faults.get(name) or {}
        for key in totals:
            totals[key] += int(entry.get(key, 0))
    return totals


def _score_games(games: list[dict[str, Any]], name: str) -> dict[str, Any]:
    appearances = wins = losses = ties = annihilations = audited = bad_status = 0
    for game in games:
        if name not in game["seats"]:
            continue
        appearances += 1
        if "faults" in game:
            audited += 1
        idx = game["seats"].index(name)
        status = (game.get("agent_status") or ["DONE"] * len(game["seats"]))[idx]
        if status != "DONE":
            bad_status += 1
        if game.get("died_at", [None] * len(game["seats"]))[idx] is not None:
            annihilations += 1
        winner = decisive_winner(game)
        if winner == name:
            wins += 1
        elif winner is None:
            ties += 1
        else:
            losses += 1
    decisive = wins + losses
    return {
        "appearances": appearances,
        "audited": audited,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "decisive": decisive,
        "win_rate": wins / appearances if appearances else 0.0,
        "decisive_win_rate": wins / decisive if decisive else None,
        "nonloss_rate": (wins + ties) / appearances if appearances else 0.0,
        "annihilation_rate": annihilations / appearances if appearances else 0.0,
        "bad_status": bad_status,
        "faults": _fault_totals(games, name),
    }


def _empty_pair_summary() -> dict[str, Any]:
    return {
        "appearances": 0,
        "audited": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "decisive": 0,
        "win_rate": 0.0,
        "decisive_win_rate": None,
        "nonloss_rate": 0.0,
        "annihilation_rate": 0.0,
        "bad_status": 0,
        "faults": {"crashes": 0, "timeouts": 0, "invalid_moves": 0},
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# -- normalization (etapa 5) --------------------------------------------------
# Raw 2p and 4p win rates live on different scales (even-game baseline 0.50 vs
# 0.25); mixing them unnormalized measures scale soup, not field advantage.
def adv_2p(win_rate: float) -> float:
    return 2.0 * float(win_rate) - 1.0


def adv_4p(win_rate: float) -> float:
    return (float(win_rate) - 0.25) / 0.75


def field_advantage(adv2: float, adv4: float, weight_2p: float = FIELD_2P_WEIGHT) -> float:
    """Normalized field mix. A FEATURE for the calibrated selector, not a
    decision value by itself."""
    return float(weight_2p) * float(adv2) + (1.0 - float(weight_2p)) * float(adv4)


def risk_penalty(summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Strategic/operational risk discount on the selector score. Technical
    faults are NOT here — they hard-fail the verdict outright."""
    components: dict[str, float] = {}
    latency = summary.get("latency_p95_max")
    if latency is not None and LATENCY_WARN_MS < float(latency) <= LATENCY_HARD_MS:
        components["high_latency"] = 0.05
    ann = summary.get("four_player", {}).get("annihilation_rate", 0.0)
    if CHOICE_GATES_4P["max_4p_annihilation"] < float(ann):
        components["high_4p_annihilation"] = 0.05
    worst_bucket = summary.get("worst_bucket_score")
    if worst_bucket is not None and float(worst_bucket) < 0.35:
        components["weak_critical_bucket"] = 0.05
    templates = summary.get("four_player_templates") or {}
    rates = [t["win_rate"] for t in templates.values() if t.get("appearances")]
    if len(rates) >= 2 and (max(rates) - min(rates)) > 0.25:
        components["template_dependence"] = 0.03
    return sum(components.values()), components


def summarize_candidate(
    candidate: str,
    task_results: list[dict[str, Any]],
    *,
    incumbent: str,
    min_decisive_2p: int,
    min_producer_winrate: float,
    min_incumbent_winrate: float,
    min_floor_winrate: float,
    max_annihilation_rate_4p: float,
    min_4p_winrate: float = 0.25,
    min_worst_template_4p_winrate: float = 0.18,
    required_2p_winrates: dict[str, float] | None = None,
    weight_2p: float,
    min_incumbent_seat_winrate: float | None = None,
    min_4p_decisive_winrate: float = 0.25,
    max_p95_ms: float = 900.0,
) -> dict[str, Any]:
    pairwise_fixed: dict[str, dict[str, Any]] = {}
    pairwise_peer: dict[str, dict[str, Any]] = {}
    four_player_games: list[dict[str, Any]] = []
    four_player_templates: dict[str, dict[str, Any]] = {}
    all_games: list[dict[str, Any]] = []
    latency_p95: list[float] = []
    latency_audited = True

    def _mine(result: dict[str, Any]) -> bool:
        if _result_role(result) == "peer_2p":
            return candidate in result.get("names", [])
        return result.get("candidate") == candidate

    failed_runs = [r for r in task_results if _mine(r) and r["returncode"] != 0]
    for result in task_results:
        if not _mine(result) or result["returncode"] != 0:
            continue
        games, meta = _load_task_payload(Path(result["out"]), result)
        all_games.extend(games)  # faults/status anywhere are disqualifying
        if meta["latency_audited"]:
            value = (meta["decision_ms_p95"] or {}).get(candidate)
            if value is not None:
                latency_p95.append(float(value))
        else:
            latency_audited = False
        role = _result_role(result)
        opponent = next((n for n in result["names"] if n != candidate), None)
        if role == "peer_2p":
            pairwise_peer[opponent] = _score_games(games, candidate)
        elif result["mode"] == "2p":
            pairwise_fixed[opponent] = _score_games(games, candidate)
        else:
            four_player_games.extend(games)
            four_player_templates[result.get("label", opponent or "4p")] = _score_games(
                games, candidate
            )

    overall = _score_games(all_games, candidate)
    four_player = _score_games(four_player_games, candidate) if four_player_games else _empty_pair_summary()
    # canonical scores come from the FIXED panel only; peers are diagnostic
    score_2p_fixed = _mean([s["win_rate"] for s in pairwise_fixed.values()])
    score_2p_peer = (
        _mean([s["win_rate"] for s in pairwise_peer.values()]) if pairwise_peer else None
    )
    score_4p_fixed = four_player["win_rate"]
    adv2 = adv_2p(score_2p_fixed)
    adv4 = adv_4p(score_4p_fixed)

    # style buckets over the fixed panel (a good mean can hide a style collapse)
    buckets: dict[str, dict[str, Any]] = {}
    for opponent, summary in pairwise_fixed.items():
        bucket = bucket_of(opponent)
        if bucket is None:
            continue
        entry = buckets.setdefault(bucket, {"opponents": [], "win_rates": []})
        entry["opponents"].append(opponent)
        entry["win_rates"].append(summary["win_rate"])
    for bucket, entry in buckets.items():
        entry["mean_win_rate"] = _mean(entry["win_rates"])
        entry["critical"] = bucket in CRITICAL_BUCKETS
        del entry["win_rates"]
    critical_scores = [
        entry["mean_win_rate"] for bucket, entry in buckets.items() if entry["critical"]
    ]
    worst_bucket_score = min(critical_scores) if critical_scores else None
    latency_p95_max = max(latency_p95) if latency_p95 else None

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: dict[str, Any] | None = None,
                  *, severity: str = "fail") -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "details": details or {},
        })

    add_check("subprocesses_completed", not failed_runs, {"failed": failed_runs})
    add_check("all_games_audited", overall["audited"] == overall["appearances"], {
        "audited": overall["audited"],
        "appearances": overall["appearances"],
    })
    add_check("no_faults", not any(overall["faults"].values()), {"faults": overall["faults"]})
    add_check("all_status_done", overall["bad_status"] == 0, {"bad_status": overall["bad_status"]})

    for opponent, summary in sorted(pairwise_fixed.items()):
        add_check(
            f"coverage_2p_vs_{opponent}",
            summary["decisive"] >= min_decisive_2p,
            {"decisive": summary["decisive"], "required": min_decisive_2p},
            severity="inconclusive",
        )

    producer = pairwise_fixed.get("producer")
    if producer is None and candidate != "producer":
        add_check(
            "beats_or_ties_producer_floor",
            False,
            {"missing_required_opponent": "producer"},
        )
    elif producer is not None and candidate != "producer":
        wr = producer["decisive_win_rate"]
        add_check(
            "beats_or_ties_producer_floor",
            wr is not None and wr >= min_producer_winrate,
            {"decisive_win_rate": wr, "required": min_producer_winrate},
        )

    inc = pairwise_fixed.get(incumbent)
    if candidate == incumbent:
        add_check("incumbent_h2h", True, {"candidate_is_incumbent": True}, severity="info")
    elif inc is None:
        add_check(
            "beats_or_ties_incumbent_h2h",
            False,
            {"missing_required_opponent": incumbent},
        )
    else:
        wr = inc["decisive_win_rate"]
        add_check(
            "beats_or_ties_incumbent_h2h",
            wr is not None and wr >= min_incumbent_winrate,
            {"decisive_win_rate": wr, "required": min_incumbent_winrate},
        )

    floor = pairwise_fixed.get("pgs_allscripts")
    if floor is None and candidate != "pgs_allscripts":
        add_check(
            "clears_rejected_floor",
            False,
            {"missing_required_opponent": "pgs_allscripts"},
        )
    elif floor is not None and candidate != "pgs_allscripts":
        wr = floor["decisive_win_rate"]
        add_check(
            "clears_rejected_floor",
            wr is not None and wr >= min_floor_winrate,
            {"decisive_win_rate": wr, "required": min_floor_winrate},
        )

    for opponent, required in sorted((required_2p_winrates or {}).items()):
        if opponent == candidate:
            continue
        summary = pairwise_fixed.get(opponent)
        check_name = f"required_2p_vs_{opponent}"
        if summary is None:
            add_check(
                check_name,
                False,
                {"missing_required_opponent": opponent, "required": float(required)},
            )
            continue
        wr = summary["decisive_win_rate"]
        add_check(
            check_name,
            wr is not None and wr >= float(required),
            {"decisive_win_rate": wr, "required": float(required)},
        )

    # 4p gates: the field is majority 4p (FIELD_2P_WEIGHT), so 4p must be a
    # real gate, not just an annihilation cap; per-template floor catches a bot
    # that looks fine on the aggregate by feasting on a single lineup.
    has_4p = four_player["appearances"] > 0
    add_check(
        "survives_4p",
        not has_4p or four_player["annihilation_rate"] <= max_annihilation_rate_4p,
        {
            "annihilation_rate": four_player["annihilation_rate"],
            "required_max": max_annihilation_rate_4p,
            "appearances": four_player["appearances"],
        },
    )
    add_check(
        "min_4p_winrate",
        not has_4p or four_player["win_rate"] >= min_4p_winrate,
        {"win_rate": four_player["win_rate"], "required": min_4p_winrate,
         "appearances": four_player["appearances"]},
    )
    template_rates = {
        label: t["win_rate"] for label, t in four_player_templates.items() if t["appearances"]
    }
    add_check(
        "min_worst_template_4p_winrate",
        not template_rates or min(template_rates.values()) >= min_worst_template_4p_winrate,
        {"per_template": template_rates, "required": min_worst_template_4p_winrate},
    )

    add_check(
        "no_critical_bucket_total_failure",
        worst_bucket_score is None or worst_bucket_score > 0.0,
        {"worst_bucket_score": worst_bucket_score,
         "buckets": {b: e["mean_win_rate"] for b, e in buckets.items() if e["critical"]}},
    )

    if latency_p95_max is None:
        add_check(
            "latency_audited",
            latency_audited,
            {"latency_p95_max": None,
             "note": "payload predates decision_ms_* instrumentation"},
            severity="inconclusive",
        )
    else:
        add_check(
            "latency_within_budget",
            latency_p95_max <= LATENCY_HARD_MS,
            {"latency_p95_max": latency_p95_max, "hard_ms": LATENCY_HARD_MS,
             "warn_ms": LATENCY_WARN_MS,
             "warning": LATENCY_WARN_MS < latency_p95_max <= LATENCY_HARD_MS},
        )

    # 4p aggregate >= fair share. In a 4-player FFA the fair decisive share is
    # ~0.25; a candidate pulling below that is net-negative in the 4p regime that
    # is the majority of the field. Survives_4p (annihilation) guards being wiped
    # out; this guards being a passenger. Inconclusive when no 4p decisive game.
    fp_wr = four_player["decisive_win_rate"]
    if four_player["appearances"] == 0:
        add_check("four_player_fair_share", True,
                  {"appearances": 0}, severity="info")
    elif four_player["decisive"] == 0:
        add_check("four_player_fair_share", False,
                  {"decisive": 0, "required": min_4p_decisive_winrate},
                  severity="inconclusive")
    else:
        add_check(
            "four_player_fair_share",
            fp_wr is not None and fp_wr >= min_4p_decisive_winrate,
            {"decisive_win_rate": fp_wr, "decisive": four_player["decisive"],
             "required": min_4p_decisive_winrate},
        )

    # p95 decision latency under the Kaggle actTimeout budget. No p95 in the
    # payload (older runs / unit fixtures) => not judged, reported as info.
    p95 = _candidate_p95(all_games, candidate)
    if p95 is None:
        add_check("p95_within_limit", True, {"p95_ms": None, "limit_ms": max_p95_ms},
                  severity="info")
    else:
        add_check("p95_within_limit", p95 <= max_p95_ms,
                  {"p95_ms": p95, "limit_ms": max_p95_ms})

    hard_failures = [c for c in checks if not c["passed"] and c["severity"] == "fail"]
    inconclusive = [c for c in checks if not c["passed"] and c["severity"] == "inconclusive"]
    if hard_failures:
        verdict = "REJECT_LOCAL"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS_LOCAL"

    summary = {
        "candidate": candidate,
        "verdict": verdict,
        "score_2p_fixed": score_2p_fixed,
        "score_2p_peer": score_2p_peer,
        "score_4p_fixed": score_4p_fixed,
        "adv_2p_fixed": adv2,
        "adv_4p_fixed": adv4,
        "field_advantage": field_advantage(adv2, adv4, weight_2p),
        "overall": overall,
        "four_player": four_player,
        "four_player_templates": four_player_templates,
        "pairwise_fixed": pairwise_fixed,
        "pairwise_peer": pairwise_peer,
        "buckets": buckets,
        "worst_bucket_score": worst_bucket_score,
        "latency_p95_max": latency_p95_max,
        "latency_audited": latency_audited,
        "checks": checks,
    }
    penalty, components = risk_penalty(summary)
    summary["risk_penalty"] = penalty
    summary["risk_components"] = components
    return summary


def build_report(
    candidates: list[str],
    task_results: list[dict[str, Any]],
    *,
    incumbent: str,
    min_decisive_2p: int,
    min_producer_winrate: float,
    min_incumbent_winrate: float,
    min_floor_winrate: float,
    max_annihilation_rate_4p: float,
    min_4p_winrate: float = 0.25,
    min_worst_template_4p_winrate: float = 0.18,
    required_2p_winrates: dict[str, float] | None = None,
    weight_2p: float,
    min_incumbent_seat_winrate: float | None = None,
    min_4p_decisive_winrate: float = 0.25,
    max_p95_ms: float = 900.0,
) -> dict[str, Any]:
    summaries = [
        summarize_candidate(
            candidate,
            task_results,
            incumbent=incumbent,
            min_decisive_2p=min_decisive_2p,
            min_producer_winrate=min_producer_winrate,
            min_incumbent_seat_winrate=min_incumbent_seat_winrate,
            min_4p_decisive_winrate=min_4p_decisive_winrate,
            max_p95_ms=max_p95_ms,
            min_incumbent_winrate=min_incumbent_winrate,
            min_floor_winrate=min_floor_winrate,
            max_annihilation_rate_4p=max_annihilation_rate_4p,
            min_4p_winrate=min_4p_winrate,
            min_worst_template_4p_winrate=min_worst_template_4p_winrate,
            required_2p_winrates=required_2p_winrates,
            weight_2p=weight_2p,
        )
        for candidate in candidates
    ]
    # Display order only. The ruler does NOT recommend: choosing is the
    # calibrated selector's job (league_submission_selector.py); until a valid
    # calibration exists the local order has no proven LB meaning (2026-06-10
    # falsification: local Spearman vs LB = 0.0).
    ranking = sorted(
        summaries,
        key=lambda s: (
            _VERDICT_PRIORITY.get(str(s["verdict"]), -1),
            s["field_advantage"],
            s["adv_4p_fixed"],
            s["adv_2p_fixed"],
        ),
        reverse=True,
    )
    return {
        "local_veto_passes": [s["candidate"] for s in ranking if s["verdict"] == "PASS_LOCAL"],
        "selector_candidate": None,
        "selection_status": "VETO_ONLY",
        "promotion_order_valid": False,
        "ranking": [
            {
                "candidate": s["candidate"],
                "verdict": s["verdict"],
                "field_advantage": s["field_advantage"],
                "score_2p_fixed": s["score_2p_fixed"],
                "score_2p_peer": s["score_2p_peer"],
                "score_4p_fixed": s["score_4p_fixed"],
                "worst_bucket_score": s["worst_bucket_score"],
                "risk_penalty": s["risk_penalty"],
            }
            for s in ranking
        ],
        "candidates": {s["candidate"]: s for s in summaries},
    }


def _parse_4p_templates(values: list[str] | None) -> list[tuple[str, ...]]:
    if not values:
        return list(DEFAULT_4P_TEMPLATES)
    out = []
    for value in values:
        names = tuple(_split_csv(value))
        if len(names) != 3:
            raise ValueError(f"4p lineup template needs exactly 3 opponents: {value!r}")
        out.append(names)
    return out


def _parse_required_2p_winrates(values: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values or []:
        raw = value.strip()
        if not raw:
            continue
        if "=" in raw:
            name, threshold = raw.split("=", 1)
        else:
            name, threshold = raw, "0.50"
        name = name.strip()
        if not name:
            raise ValueError(f"empty required 2p opponent in {value!r}")
        out[name] = float(threshold)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="strong")
    parser.add_argument("--incumbent", default=INCUMBENT)
    parser.add_argument("--references", default=",".join(DEFAULT_REFERENCES))
    parser.add_argument("--four-player-lineup", action="append",
                        help="three comma-separated opponents; candidate is inserted as seat 0")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--seed-base", type=int, default=None,
        help="explicit seed base (mutually exclusive with --seed-split)",
    )
    parser.add_argument(
        "--seed-split", choices=sorted(SEED_SPLITS), default=None,
        help="named seed universe: dev (free), validation (veto/calibration), "
             "selector (submission-decision holdout; never used in development)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--match-chunk-size",
        type=int,
        default=0,
        help="pass --chunk-size to league_match so long H2H tasks write partial JSONs",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/league/submit_ruler"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--task-results-out",
        type=Path,
        default=None,
        help="write completed match metadata incrementally while the ruler is still running",
    )
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-decisive-2p", type=int, default=None)
    parser.add_argument("--min-producer-winrate", type=float, default=0.50)
    parser.add_argument("--min-incumbent-winrate", type=float, default=0.50)
    parser.add_argument("--min-floor-winrate", type=float, default=0.60)
    parser.add_argument("--max-annihilation-rate-4p", type=float, default=0.30)
    parser.add_argument("--min-4p-winrate", type=float, default=0.25)
    parser.add_argument("--min-worst-template-4p-winrate", type=float, default=0.18)
    parser.add_argument(
        "--no-peer-h2h", action="store_true",
        help="skip candidate-vs-candidate diagnostic tasks (e.g. anchor calibration)",
    )
    parser.add_argument(
        "--min-incumbent-seat-winrate", type=float, default=None,
        help="per-seat decisive winrate floor vs the incumbent (default: "
        "--min-incumbent-winrate); guards seat-split false gains")
    parser.add_argument(
        "--min-4p-decisive-winrate", type=float, default=0.25,
        help="4p aggregate decisive-winrate fair-share floor (FFA fair share ~0.25)")
    parser.add_argument(
        "--max-p95-ms", type=float, default=900.0,
        help="max candidate p95 decision latency (ms) under the 1s Kaggle actTimeout")
    parser.add_argument(
        "--required-2p-winrate",
        action="append",
        help="extra required 2p decisive winrate, as opponent=threshold or opponent for 0.50",
    )
    parser.add_argument("--weight-2p", type=float, default=FIELD_2P_WEIGHT)
    args = parser.parse_args(argv)
    profile = PROFILE_DEFAULTS[args.profile]
    if args.seeds is None:
        args.seeds = int(profile["seeds"])
    if args.steps is None:
        args.steps = int(profile["steps"])
    if args.min_decisive_2p is None:
        args.min_decisive_2p = int(profile["min_decisive_2p"])
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.seeds % 4 != 0:
        raise SystemExit("--seeds must be a multiple of 4 so 4p seat rotation is balanced")
    if args.seed_base is not None and args.seed_split is not None:
        raise SystemExit("--seed-base and --seed-split are mutually exclusive")
    seed_split = args.seed_split
    if args.seed_base is None:
        seed_split = seed_split or "dev"
        args.seed_base = SEED_SPLITS[seed_split]

    candidates = list(dict.fromkeys(args.candidates))
    requested_references = _split_csv(args.references)
    unknown_references = [name for name in requested_references if name not in FACTORIES]
    if unknown_references:
        raise SystemExit(f"unknown references: {', '.join(unknown_references)}")
    references = list(dict.fromkeys(requested_references))
    if args.incumbent not in FACTORIES:
        raise SystemExit(f"unknown incumbent: {args.incumbent}")
    templates = _parse_4p_templates(args.four_player_lineup)
    required_2p_winrates = _parse_required_2p_winrates(args.required_2p_winrate)
    unknown_required = [name for name in required_2p_winrates if name not in FACTORIES]
    if unknown_required:
        raise SystemExit(f"unknown required 2p opponents: {', '.join(unknown_required)}")
    out_dir = args.out_dir
    tasks = build_tasks(
        candidates,
        incumbent=args.incumbent,
        references=references,
        four_player_templates=templates,
        seeds=args.seeds,
        seed_base=args.seed_base,
        steps=args.steps,
        out_dir=out_dir,
        match_chunk_size=max(0, int(args.match_chunk_size)),
        peer_h2h=not args.no_peer_h2h,
    )
    if not tasks:
        raise SystemExit("no runnable tasks")
    if args.skip_run:
        task_results = [
            {
                "label": task.label,
                "mode": task.mode,
                "candidate": task.candidate,
                "names": list(task.names),
                "role": task.role,
                "seeds": task.seeds,
                "seed_base": task.seed_base,
                "steps": task.steps,
                "chunk_size": task.chunk_size,
                "out": str(task.out),
                "returncode": 0,
                "seconds": 0.0,
                "stdout": "",
                "stderr": "",
            }
            for task in tasks
        ]
    else:
        task_results_out = args.task_results_out
        if task_results_out is None:
            task_results_out = out_dir / "task_results.json"
        task_results = run_tasks(
            tasks,
            args.jobs,
            progress=not args.quiet,
            task_results_out=task_results_out,
        )
    report = build_report(
        candidates,
        task_results,
        incumbent=args.incumbent,
        min_decisive_2p=args.min_decisive_2p,
        min_producer_winrate=args.min_producer_winrate,
        min_incumbent_winrate=args.min_incumbent_winrate,
        min_floor_winrate=args.min_floor_winrate,
        max_annihilation_rate_4p=args.max_annihilation_rate_4p,
        min_4p_winrate=args.min_4p_winrate,
        min_worst_template_4p_winrate=args.min_worst_template_4p_winrate,
        required_2p_winrates=required_2p_winrates,
        weight_2p=args.weight_2p,
        min_incumbent_seat_winrate=args.min_incumbent_seat_winrate,
        min_4p_decisive_winrate=args.min_4p_decisive_winrate,
        max_p95_ms=args.max_p95_ms,
    )
    report.update({
        "incumbent": args.incumbent,
        "references": references,
        "tasks": task_results,
        "settings": {
            "seeds": args.seeds,
            "seed_base": args.seed_base,
            "seed_split": seed_split,
            "steps": args.steps,
            "profile": args.profile,
            "min_decisive_2p": args.min_decisive_2p,
            "min_producer_winrate": args.min_producer_winrate,
            "min_incumbent_winrate": args.min_incumbent_winrate,
            "min_floor_winrate": args.min_floor_winrate,
            "max_annihilation_rate_4p": args.max_annihilation_rate_4p,
            "min_4p_winrate": args.min_4p_winrate,
            "min_worst_template_4p_winrate": args.min_worst_template_4p_winrate,
            "required_2p_winrates": required_2p_winrates,
            "weight_2p": args.weight_2p,
            "min_incumbent_seat_winrate": args.min_incumbent_seat_winrate,
            "min_4p_decisive_winrate": args.min_4p_decisive_winrate,
            "max_p95_ms": args.max_p95_ms,
        },
    })
    out = args.out or (out_dir / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "out": str(out),
        "local_veto_passes": report["local_veto_passes"],
        "selection_status": report["selection_status"],
        "ranking": report["ranking"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
