"""Lever A — offline metaheuristic tuning of the H7 ``eval_function`` weights.

Optimises the weight genome of ``bots.oep.family_h.make_eval_policy`` directly
against a strong opponent (default: Producer), using the SAME paired metric the
promotion gate uses (``normalized_margin``, both seats played for symmetry). This
is the realisable form of the "learned value function" the PPO line called for —
learned by evolution over hand-designed features, no RL.

Optimiser: a dependency-free separable ``(μ/μ_w, λ)`` evolution strategy with
per-coordinate relative step sizes and a simple cumulative step-size adaptation.
Chosen over an external CMA-ES so the repo stays dependency-clean; the genome is
low-dim (12) and separable enough that diagonal covariance suffices.

Anti-overfit (Cawley & Talbot 2010, which already bit H7): the best genome by
TRAIN fitness is re-scored on a DISJOINT holdout seed set before any claim. The
report records both so the deflation is visible.

Reference: Gaina, Devlin, Lucas, Perez-Liebana 2020 (arXiv:2003.12331).

Run::

    rtk uv run --extra dev --extra train python -m scripts.tune_eval_weights \
        --opponent producer --generations 20 --pop 10 \
        --train-seeds 0-5 --holdout-seeds 100-115 --episode-steps 96 \
        --out artifacts/gates/family_h/a2_tuned.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from time import perf_counter

from bots.oep.family_h import EVAL_DEFAULT_WEIGHTS, EVAL_WEIGHT_NAMES, make_eval_policy
from python.orbit_wars_gym.rules import normalized_margin
from scripts.benchmark_submission import _resolve_opponent, _run_match, _submission_runtime

# Per-process opponent cache so warm workers don't re-resolve every game.
_WORKER_OPP: dict[str, object] = {}


def _game_worker(task: tuple) -> tuple[int, float, dict[str, float]]:
    """Play ONE game (one candidate, one seed, one seat). Returns (cand_idx, margin, bad).

    Top-level + self-contained so it pickles cleanly across processes; the policy
    is rebuilt from its weight vector inside the worker (torch objects never cross
    the process boundary).
    """

    cand_idx, weights, seed, seat, opponent, steps, act_timeout = task
    opp = _WORKER_OPP.get(opponent)
    if opp is None:
        _, opp = _resolve_opponent(opponent)
        _WORKER_OPP[opponent] = opp
    pol = _submission_runtime(make_eval_policy(weights))
    players = [pol, opp] if seat == 0 else [opp, pol]
    scores, stats = _run_match(
        players, seed=seed, episode_steps=steps, enable_comets=True, act_timeout=act_timeout
    )
    s = stats[seat]
    bad = {k: float(s.get(k, 0.0)) for k in ("crashes", "timeouts", "invalid_actions")}
    return cand_idx, normalized_margin(scores, seat), bad

# Per-coordinate lower bounds. Every term is a non-negative weight except the
# selection threshold ``min_score`` (col 11), which may go negative.
_LOWER = [0.0] * len(EVAL_WEIGHT_NAMES)
_LOWER[EVAL_WEIGHT_NAMES.index("min_score")] = -math.inf


def _clip(vec: list[float]) -> list[float]:
    return [max(lo, v) for v, lo in zip(vec, _LOWER, strict=True)]


def _parse_seeds(spec: str) -> list[int]:
    """Accept ``0-5`` ranges and ``1,3,7`` lists (combinable: ``0-3,10,12``)."""

    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make_eval_many(pool, opponent: str, episode_steps: int, act_timeout: float):
    """Return ``eval_many(list_of_weight_vectors, seeds) -> list[(margin, bad)]``.

    All (candidate × seed × seat) games of the call are submitted to the persistent
    pool at once, so warm workers stay saturated. Each seed is played in both seats
    for symmetry — the same bias cancellation the benchmark applies.
    """

    def eval_many(vectors: list[list[float]], seeds: list[int]) -> list[tuple[float, dict]]:
        tasks = []
        for ci, w in enumerate(vectors):
            for seed in seeds:
                for seat in (0, 1):
                    tasks.append((ci, list(w), seed, seat, opponent, episode_steps, act_timeout))
        margins: list[list[float]] = [[] for _ in vectors]
        bads: list[dict[str, float]] = [
            {"crashes": 0.0, "timeouts": 0.0, "invalid_actions": 0.0} for _ in vectors
        ]
        runner = pool.map(_game_worker, tasks) if pool is not None else map(_game_worker, tasks)
        for ci, margin, bad in runner:
            margins[ci].append(margin)
            for k in bad:
                bads[ci][k] += bad[k]
        return [(fmean(m) if m else -1.0, b) for m, b in zip(margins, bads, strict=True)]

    return eval_many


def separable_es(
    eval_many,
    x0: list[float],
    *,
    train_seeds: list[int],
    generations: int,
    pop: int,
    sigma0: float,
    rng: random.Random,
    log,
):
    """(μ/μ_w, λ)-ES with diagonal scaling. Returns (best_x, best_train, history)."""

    dim = len(x0)
    mu = max(1, pop // 2)
    # Log-weighted recombination weights (CMA-style), normalised.
    raw = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    wsum = sum(raw)
    weights = [r / wsum for r in raw]
    mueff = 1.0 / sum(w * w for w in weights)
    # Per-coordinate step scale: relative to |default| so a 60-weight and a
    # 0.25-weight both get proportional exploration.
    scale = [max(0.5, abs(v)) for v in x0]
    cs = (mueff + 2.0) / (dim + mueff + 5.0)  # step-size adaptation rate
    ps = [0.0] * dim
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    mean = list(x0)
    sigma = sigma0
    best_x, best_f = list(x0), -math.inf
    history = []

    for gen in range(generations):
        samples = []
        for _ in range(pop):
            z = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            x = _clip([mean[i] + sigma * scale[i] * z[i] for i in range(dim)])
            samples.append((x, z))
        results = eval_many([x for x, _ in samples], train_seeds)
        scored = [
            (results[i][0], samples[i][0], samples[i][1], results[i][1])
            for i in range(pop)
        ]
        scored.sort(key=lambda t: -t[0])

        # Recombination of the μ best (in z-space for ps, x-space for mean).
        new_mean = list(mean)
        zmean = [0.0] * dim
        for i in range(dim):
            new_mean[i] = sum(weights[k] * scored[k][1][i] for k in range(mu))
            zmean[i] = sum(weights[k] * scored[k][2][i] for k in range(mu))
        mean = _clip(new_mean)
        # Cumulative step-size adaptation.
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * zmean[i]
        ps_norm = math.sqrt(sum(p * p for p in ps))
        sigma *= math.exp((cs / 2.0) * (ps_norm / chiN - 1.0))
        sigma = min(max(sigma, 1e-3), 1.0)

        gen_best = scored[0]
        if gen_best[0] > best_f:
            best_f, best_x = gen_best[0], list(gen_best[1])
        history.append(
            {
                "gen": gen,
                "gen_best_margin": round(gen_best[0], 5),
                "gen_mean_margin": round(fmean(s[0] for s in scored), 5),
                "best_so_far": round(best_f, 5),
                "sigma": round(sigma, 4),
                "bad_total": {k: gen_best[3][k] for k in gen_best[3]},
            }
        )
        log(
            f"gen {gen:2d}  best={gen_best[0]:+.4f}  mean={fmean(s[0] for s in scored):+.4f}"
            f"  best_so_far={best_f:+.4f}  sigma={sigma:.3f}"
        )
    return best_x, best_f, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--train-seeds", default="0-5")
    ap.add_argument("--holdout-seeds", default="100-115")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--episode-steps", type=int, default=96)
    ap.add_argument("--act-timeout", type=float, default=1.0)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="artifacts/gates/family_h/a2_tuned.json")
    args = ap.parse_args()

    train_seeds = _parse_seeds(args.train_seeds)
    holdout_seeds = _parse_seeds(args.holdout_seeds)
    assert not (set(train_seeds) & set(holdout_seeds)), "train/holdout seeds must be disjoint"
    rng = random.Random(args.rng_seed)

    t0 = perf_counter()
    print(
        f"[tune] opponent={args.opponent} train={len(train_seeds)}seeds "
        f"holdout={len(holdout_seeds)}seeds pop={args.pop} gen={args.generations} "
        f"steps={args.episode_steps} jobs={args.jobs}"
    )
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        eval_many = make_eval_many(pool, args.opponent, args.episode_steps, args.act_timeout)

        default_train = eval_many([list(EVAL_DEFAULT_WEIGHTS)], train_seeds)[0][0]
        print(f"[tune] default train margin = {default_train:+.4f}")

        best_x, best_train, history = separable_es(
            eval_many,
            list(EVAL_DEFAULT_WEIGHTS),
            train_seeds=train_seeds,
            generations=args.generations,
            pop=args.pop,
            sigma0=args.sigma0,
            rng=rng,
            log=print,
        )

        # Anti-overfit: re-score best + default on the DISJOINT holdout.
        (best_holdout, best_bad), (default_holdout, _d) = eval_many(
            [best_x, list(EVAL_DEFAULT_WEIGHTS)], holdout_seeds
        )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    elapsed = perf_counter() - t0

    weights_named = dict(zip(EVAL_WEIGHT_NAMES, [round(v, 4) for v in best_x], strict=True))
    report = {
        "opponent": args.opponent,
        "episode_steps": args.episode_steps,
        "train_seeds": train_seeds,
        "holdout_seeds": holdout_seeds,
        "pop": args.pop,
        "generations": args.generations,
        "default_weights": list(EVAL_DEFAULT_WEIGHTS),
        "best_weights": best_x,
        "best_weights_named": weights_named,
        "default_train_margin": round(default_train, 5),
        "best_train_margin": round(best_train, 5),
        "default_holdout_margin": round(default_holdout, 5),
        "best_holdout_margin": round(best_holdout, 5),
        "holdout_deflation": round(best_train - best_holdout, 5),
        "holdout_improvement_over_default": round(best_holdout - default_holdout, 5),
        "holdout_bad_totals": best_bad,
        "history": history,
        "elapsed_seconds": round(elapsed, 1),
    }
    report["content_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()[:16]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"\n[tune] DONE in {elapsed:.0f}s\n"
        f"  default:  train {default_train:+.4f}  holdout {default_holdout:+.4f}\n"
        f"  tuned:    train {best_train:+.4f}  holdout {best_holdout:+.4f}\n"
        f"  holdout improvement over default: {best_holdout - default_holdout:+.4f}\n"
        f"  deflation (train-holdout):        {best_train - best_holdout:+.4f}\n"
        f"  best weights: {weights_named}\n"
        f"  -> {out}"
    )


if __name__ == "__main__":
    main()
