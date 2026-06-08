"""Dense per-opponent evaluation for the curriculum ladder (user spec 2026-06-07).

The aggregate ``mean_score_margin`` saturates at -1.0 against a hard opponent and
gives no gradient (diagnosed 2026-06-07, EXPERIMENTS.md). This eval instead steps
each game and logs PER-STEP state, reporting per opponent: win_rate,
mean_score_margin, survival_step, first_collapse_step, production_share_auc,
ship_share_auc, AUC_margin, and crash/timeout/invalid rates.

Promotion rule (do NOT use the aggregate): keep the checkpoint with the best
``auc_margin`` on the HARDEST opponent that is NOT yet saturated — i.e. the hardest
rung whose auc_margin still moves off the floor (``--promote-floor``). Use
``--select`` to print which opponent that is and the AUC to beat.

Usage::

    uv run --extra dev python -m scripts.dense_eval \
        --checkpoint artifacts/ppo/campaign/c01.pt \
        --opponents greedy producer_h50 producer --seeds 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import planet_owner, planet_production, planet_ships
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin
from scripts.benchmark_submission import _load_submission_agent, _resolve_opponent

try:  # fleets contribute to ship_share when the helpers exist
    from python.orbit_wars_gym.entities import fleet_owner, fleet_ships

    _HAS_FLEETS = True
except Exception:  # noqa: BLE001
    _HAS_FLEETS = False

_ROOT = Path(__file__).resolve().parents[1]
COLLAPSE_PROD_SHARE = 0.25  # first_collapse = first step prod_share < this OR owned<=1


def _shares(state: dict[str, Any], player: int) -> tuple[int, float, float]:
    """Own SHARE of CONTESTED resources = own / (own + enemy), excluding neutrals.
    Starts ~0.5 (even) and moves toward 1.0 (winning) or 0.0 (losing) — a clean
    'who is ahead' signal, unlike own/total-including-neutrals which is diluted."""
    own_prod = opp_prod = own_ships = opp_ships = own_planets = 0
    for p in state.get("planets", []):
        owner = planet_owner(p)
        if owner == player:
            own_prod += planet_production(p)
            own_ships += planet_ships(p)
            own_planets += 1
        elif owner >= 0:  # another player, not a neutral (owner == -1)
            opp_prod += planet_production(p)
            opp_ships += planet_ships(p)
    if _HAS_FLEETS:
        for f in state.get("fleets", []):
            owner = fleet_owner(f)
            if owner == player:
                own_ships += fleet_ships(f)
            elif owner >= 0:
                opp_ships += fleet_ships(f)
    prod_share = own_prod / (own_prod + opp_prod) if (own_prod + opp_prod) > 0 else 0.5
    ship_share = own_ships / (own_ships + opp_ships) if (own_ships + opp_ships) > 0 else 0.5
    return own_planets, prod_share, ship_share


def _run_game(
    *,
    submission_agent: Any,
    opponent_policy: Any,
    seed: int,
    submission_first: bool,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> dict[str, float]:
    players = (
        [submission_agent, opponent_policy] if submission_first else [opponent_policy, submission_agent]
    )
    submission_idx = 0 if submission_first else 1
    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(
            episode_steps=int(episode_steps),
            enable_comets=bool(enable_comets),
            act_timeout=float(act_timeout),
        ),
    )
    state = backend.reset(seed)[0]
    outcome: dict[str, Any] = {"scores": [0.0, 0.0], "done": False}
    crashes = timeouts = invalid = 0.0
    margins: list[float] = []
    prod_shares: list[float] = []
    ship_shares: list[float] = []
    survival_step = 0
    first_collapse: int | None = None
    steps = 0

    while True:
        actions: list[list[list[float]]] = [[], []]
        for player_idx, policy in enumerate(players):
            obs = to_official_observation(state, player=player_idx)
            try:
                start = perf_counter()
                moves = policy(obs) if player_idx == submission_idx else policy(state, player_idx)
                elapsed = perf_counter() - start
                if not isinstance(moves, list):
                    moves = []
            except Exception:  # noqa: BLE001
                if player_idx == submission_idx:
                    crashes += 1.0
                moves = []
                elapsed = 0.0
            if player_idx == submission_idx:
                if elapsed > float(act_timeout):
                    timeouts += 1.0
                    moves = []
                if not moves_are_legal(state, player_idx, moves):
                    invalid += 1.0
                    moves = []
            elif not moves_are_legal(state, player_idx, moves):
                moves = []
            actions[player_idx] = moves

        outcomes, states = backend.step_with_states([actions])
        outcome = outcomes[0]
        state = states[0]
        steps += 1
        own_planets, prod_share, ship_share = _shares(state, submission_idx)
        margins.append(normalized_margin([float(x) for x in outcome["scores"]], submission_idx))
        prod_shares.append(prod_share)
        ship_shares.append(ship_share)
        if own_planets > 0 or ship_share > 0.0:
            survival_step = steps
        if first_collapse is None and (prod_share < COLLAPSE_PROD_SHARE or own_planets == 0):
            first_collapse = steps
        if outcome["done"]:
            break

    scores = [float(s) for s in outcome["scores"]]
    others = max(s for i, s in enumerate(scores) if i != submission_idx)
    return {
        "normalized_margin": normalized_margin(scores, submission_idx),
        "win": 1.0 if scores[submission_idx] > others else 0.0,
        "survival_step": float(survival_step),
        "first_collapse_step": float(first_collapse if first_collapse is not None else steps),
        "production_share_auc": fmean(prod_shares) if prod_shares else 0.0,
        "ship_share_auc": fmean(ship_shares) if ship_shares else 0.0,
        "auc_margin": fmean(margins) if margins else -1.0,
        "crashes": crashes,
        "timeouts": timeouts,
        "invalid_actions": invalid,
    }


def _eval_opponent(
    submission_path: Path, opponent: str, seeds: int, episode_steps: int,
    enable_comets: bool, act_timeout: float,
) -> dict[str, float]:
    submission_agent = _load_submission_agent(submission_path)
    _, opponent_policy = _resolve_opponent(opponent)
    games: list[dict[str, float]] = []
    for seed in range(max(1, seeds)):
        for first in (True, False):  # both seats for symmetry
            games.append(_run_game(
                submission_agent=submission_agent, opponent_policy=opponent_policy,
                seed=seed, submission_first=first, episode_steps=episode_steps,
                enable_comets=enable_comets, act_timeout=act_timeout,
            ))
    keys = ("normalized_margin", "win", "survival_step", "first_collapse_step",
            "production_share_auc", "ship_share_auc", "auc_margin",
            "crashes", "timeouts", "invalid_actions")
    summary = {k: fmean(g[k] for g in games) for k in keys}
    summary["games"] = float(len(games))
    summary["win_rate"] = summary.pop("win")
    summary["mean_score_margin"] = summary.pop("normalized_margin")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="PPO .pt to export+eval")
    parser.add_argument("--submission", type=Path, help="already-exported .sub.py to eval")
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--enable-comets", action="store_true")
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--promote-floor", type=float, default=-0.97,
                        help="an opponent counts as SATURATED if its auc_margin <= this")
    args = parser.parse_args()

    if args.submission:
        submission_path = args.submission
    elif args.checkpoint:
        from scripts.benchmark_ppo_submission import _write_exported_submission

        submission_path = args.checkpoint.with_suffix(".dense.sub.py")
        _write_exported_submission(args.checkpoint, submission_path)
    else:
        parser.error("need --checkpoint or --submission")

    per_opponent = {
        opp: _eval_opponent(submission_path, opp, args.seeds, args.episode_steps,
                            args.enable_comets, args.act_timeout)
        for opp in args.opponents
    }
    # Promotion ruler: hardest opponent (lowest auc_margin) that is NOT saturated.
    unsaturated = {o: s for o, s in per_opponent.items() if s["auc_margin"] > args.promote_floor}
    ruler = min(unsaturated, key=lambda o: unsaturated[o]["auc_margin"]) if unsaturated else None
    report = {
        "submission": str(submission_path),
        "promotion_ruler": ruler,
        "promotion_auc_margin": per_opponent[ruler]["auc_margin"] if ruler else None,
        "per_opponent": per_opponent,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
