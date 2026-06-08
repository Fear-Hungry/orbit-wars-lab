"""H1 oracle — is the candidate GENERATOR the bottleneck, or the SELECTOR?

Family H exists because the *selector* line (E1/E2/C1/E3) saturated while the
*generator* stayed a near-greedy monoculture. Before spending compute on RHEA
(H6) or an offline hyper-heuristic (H7), this oracle answers the gating question
with a clean counterfactual.

Method (per seed, vs Producer):

1. Run a base game (submission side = ``--base-family``, opponent = Producer) and
   snapshot the decision state every ``--snapshot-stride`` steps (up to
   ``--max-snapshots``).
2. From each snapshot, **fork once per candidate family**: that family plays the
   submission side to the end, opponent = Producer. Record the terminal
   ``normalized_margin``.

Per snapshot::

    oracle_best  = max over families of forked margin   # perfect per-state pick
    selected     = forked margin of the base family      # the real selector proxy

Aggregated over snapshots/seeds. Decision rule (the Family-H gate), with the
known OEP-best floor ``baseline = -0.045`` and a folga ``--eps``::

    oracle_best <= baseline + eps                 -> GENERATOR_BOTTLENECK
    oracle_best  > baseline + eps AND
        selected <= baseline + eps                -> SELECTOR_BOTTLENECK
    else                                          -> SIGNAL_AND_SELECTOR_OK

GENERATOR_BOTTLENECK means new candidate families are the priority; without
signal here, a fancier selector (H7) would only choose lice more cleverly.

Note (approximation): a family used as a *continuation* policy from a forked
mid-game state starts with fresh memory (Producer/OEP are largely reactive to the
current obs, so this is acceptable for a diagnostic, not for promotion).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

from bots.oep.candidate_factory import _FAMILY_BUILDERS, available_families
from bots.oep.hyperheuristic import context_bucket
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin

State = dict[str, Any]
Policy = Callable[[State, int], list[list[float]]]

#: OEP-best floor — the bar a new generator must clear to be worth pursuing.
DEFAULT_BASELINE = -0.045


def _family_policy(family: str) -> tuple[Policy, dict[str, float]]:
    """Wrap an ``obs -> moves`` family as a ``(state, idx) -> moves`` policy.

    Returns the policy plus a mutable stats dict counting illegal/crash events so
    the gate can assert ``crash/timeout/invalid = 0`` for every family.
    """

    gen = _FAMILY_BUILDERS[family]()
    stats = {"crashes": 0.0, "invalid": 0.0}

    def policy(state: State, idx: int) -> list[list[float]]:
        try:
            obs = to_official_observation(state, player=int(idx))
            moves = gen(obs)
        except Exception:
            stats["crashes"] += 1.0
            return []
        if not isinstance(moves, list) or not moves_are_legal(state, int(idx), moves):
            stats["invalid"] += 1.0
            return []
        return moves

    return policy, stats


def _rollout(
    start_state: State,
    *,
    submission_policy: Policy,
    opponent_policy: Policy,
    submission_idx: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
) -> list[float]:
    """Play one forked game from ``start_state`` to terminal; return scores."""

    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=0,
        config=RustConfig(
            episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout
        ),
    )
    state = backend.reset_from_states([start_state])[0]
    players: list[Policy] = [opponent_policy, opponent_policy]
    players[submission_idx] = submission_policy
    opp_idx = 1 - submission_idx
    players[opp_idx] = opponent_policy
    while True:
        actions: list[list[list[float]]] = [[], []]
        for idx, policy in enumerate(players):
            actions[idx] = policy(state, idx)
        outcomes, states = backend.step_with_states([actions])
        state = states[0]
        if outcomes[0]["done"]:
            return [float(score) for score in outcomes[0]["scores"]]


def collect_candidate_action_records(
    *,
    seeds: list[int],
    families: list[str],
    base_family: str = "oep",
    incumbent: str = "producer",
    episode_steps: int = 96,
    snapshot_stride: int = 32,
    max_snapshots: int = 3,
    enable_comets: bool = True,
    act_timeout: float = 1.0,
    feature_fn: Callable[[State], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Per-snapshot candidate_action records for the H7a/H7b holdout selector.

    Each record = ``{seed, bucket, features, margins: {family: ca_margin}}``.
    ``features`` (from ``feature_fn``, default = just the context bucket) holds
    the runtime-available context so feature engineering can iterate OFFLINE on
    the stored records without re-running the expensive rollouts (no post-action
    leak). Returned with a legality tally so the gate can assert crash/invalid==0.
    """

    if feature_fn is None:
        def feature_fn(obs: State) -> dict[str, Any]:  # noqa: ANN001
            return {"bucket": context_bucket(obs)}

    submission_idx = 0
    records: list[dict[str, Any]] = []
    legality = {"crashes": 0.0, "invalid": 0.0}

    def _bump(*stats: dict[str, float]) -> None:
        for s in stats:
            legality["crashes"] += s["crashes"]
            legality["invalid"] += s["invalid"]

    for seed in seeds:
        base_policy, base_stats = _family_policy(base_family)
        opp0, opp0_stats = _family_policy("producer")
        snapshots = _base_game_snapshots(
            seed=int(seed),
            base_policy=base_policy,
            opponent_policy=opp0,
            submission_idx=submission_idx,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
            snapshot_stride=snapshot_stride,
            max_snapshots=max_snapshots,
        )
        _bump(base_stats, opp0_stats)
        for snap in snapshots:
            obs = to_official_observation(snap, player=submission_idx)
            features = feature_fn(obs)
            margins: dict[str, float] = {}
            for family in families:
                cand_gen, cand_stats = _family_policy(family)
                candidate_moves = cand_gen(snap, submission_idx)
                inc_policy, inc_stats = _family_policy(incumbent)
                opp_ca, opp_ca_stats = _family_policy("producer")
                scores = _rollout(
                    snap,
                    submission_policy=_grafted_policy(candidate_moves, inc_policy),
                    opponent_policy=opp_ca,
                    submission_idx=submission_idx,
                    episode_steps=episode_steps,
                    enable_comets=enable_comets,
                    act_timeout=act_timeout,
                )
                margins[family] = normalized_margin(scores, submission_idx)
                _bump(cand_stats, inc_stats, opp_ca_stats)
            records.append(
                {
                    "seed": int(seed),
                    "bucket": features.get("bucket", context_bucket(obs)),
                    "features": features,
                    "margins": margins,
                }
            )
    return records, legality


def _base_game_snapshots(
    *,
    seed: int,
    base_policy: Policy,
    opponent_policy: Policy,
    submission_idx: int,
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float,
    snapshot_stride: int,
    max_snapshots: int,
) -> list[State]:
    """Run the base game and capture decision states every ``snapshot_stride``."""

    backend = RustBatchBackend(
        num_envs=1,
        num_players=2,
        seed=int(seed),
        config=RustConfig(
            episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout
        ),
    )
    state = backend.reset(int(seed))[0]
    players: list[Policy] = [opponent_policy, opponent_policy]
    players[submission_idx] = base_policy
    snapshots: list[State] = []
    step = 0
    while True:
        if step % snapshot_stride == 0 and len(snapshots) < max_snapshots:
            snapshots.append(backend.states()[0])
        actions: list[list[list[float]]] = [[], []]
        for idx, policy in enumerate(players):
            actions[idx] = policy(state, idx)
        outcomes, states = backend.step_with_states([actions])
        state = states[0]
        step += 1
        if outcomes[0]["done"]:
            break
    return snapshots


def _grafted_policy(candidate_moves: list[list[float]], incumbent: Policy) -> Policy:
    """Play ``candidate_moves`` once (the grafted candidate action), then defer to
    the strong incumbent for the rest of the game."""

    box = {"first": True}

    def policy(state: State, idx: int) -> list[list[float]]:
        if box["first"]:
            box["first"] = False
            return candidate_moves
        return incumbent(state, idx)

    return policy


def _graded_verdict(margin: float, baseline: float, eps: float) -> str:
    """Graded gate (not binary): see module docstring / the agreed bands."""

    if margin <= baseline:  # <= -0.045: below OEP-best
        return "HARD_GENERATOR_BOTTLENECK"
    if margin <= baseline + eps:  # (-0.045, -0.025]: weak blood
        return "WEAK_BLOOD"
    if margin < 0.0:  # (-0.025, 0): useful signal, below Producer floor
        return "USEFUL_SIGNAL"
    return "CROSSES_FLOOR"  # >= 0: merits a 96-seed gate vs Producer


def run_oracle(
    *,
    seeds: list[int],
    families: list[str],
    base_family: str,
    incumbent: str = "producer",
    episode_steps: int = 96,
    snapshot_stride: int = 32,
    max_snapshots: int = 6,
    enable_comets: bool = True,
    act_timeout: float = 1.0,
    baseline: float = DEFAULT_BASELINE,
    eps: float = 0.02,
) -> dict[str, Any]:
    submission_idx = 0
    floor_families = {"producer", "oep"}
    new_families = [f for f in families if f not in floor_families]

    # candidate_action: graft ONE candidate move, then continue with the strong
    # incumbent (both sides). Isolates the MARGINAL quality of the candidate's
    # action — the correct H1 question. family_continuation: the family plays the
    # whole game as a standalone policy (contrast; a much harder bar).
    ca_margins: dict[str, list[float]] = {f: [] for f in families}
    fc_margins: dict[str, list[float]] = {f: [] for f in new_families}
    oracle_all_per_snapshot: list[float] = []
    oracle_new_per_snapshot: list[float] = []
    selected_per_snapshot: list[float] = []
    legality = {"crashes": 0.0, "invalid": 0.0}

    def _bump(*stats: dict[str, float]) -> None:
        for s in stats:
            legality["crashes"] += s["crashes"]
            legality["invalid"] += s["invalid"]

    for seed in seeds:
        base_policy, base_stats = _family_policy(base_family)
        opp_for_base, opp_stats0 = _family_policy("producer")
        snapshots = _base_game_snapshots(
            seed=int(seed),
            base_policy=base_policy,
            opponent_policy=opp_for_base,
            submission_idx=submission_idx,
            episode_steps=episode_steps,
            enable_comets=enable_comets,
            act_timeout=act_timeout,
            snapshot_stride=snapshot_stride,
            max_snapshots=max_snapshots,
        )
        _bump(base_stats, opp_stats0)

        for snap in snapshots:
            ca: dict[str, float] = {}
            for family in families:
                # The candidate's move at THIS state.
                cand_gen, cand_stats = _family_policy(family)
                candidate_moves = cand_gen(snap, submission_idx)
                inc_policy, inc_stats = _family_policy(incumbent)
                opp_ca, opp_ca_stats = _family_policy("producer")
                scores = _rollout(
                    snap,
                    submission_policy=_grafted_policy(candidate_moves, inc_policy),
                    opponent_policy=opp_ca,
                    submission_idx=submission_idx,
                    episode_steps=episode_steps,
                    enable_comets=enable_comets,
                    act_timeout=act_timeout,
                )
                margin = normalized_margin(scores, submission_idx)
                ca[family] = margin
                ca_margins[family].append(margin)
                _bump(cand_stats, inc_stats, opp_ca_stats)

                if family in fc_margins:  # family as a standalone policy (contrast)
                    sub_fc, sub_fc_stats = _family_policy(family)
                    opp_fc, opp_fc_stats = _family_policy("producer")
                    fc_scores = _rollout(
                        snap,
                        submission_policy=sub_fc,
                        opponent_policy=opp_fc,
                        submission_idx=submission_idx,
                        episode_steps=episode_steps,
                        enable_comets=enable_comets,
                        act_timeout=act_timeout,
                    )
                    fc_margins[family].append(normalized_margin(fc_scores, submission_idx))
                    _bump(sub_fc_stats, opp_fc_stats)

            oracle_all_per_snapshot.append(max(ca.values()))
            selected_per_snapshot.append(ca[base_family])
            if new_families:
                oracle_new_per_snapshot.append(max(ca[f] for f in new_families))

    oracle_all = fmean(oracle_all_per_snapshot) if oracle_all_per_snapshot else 0.0
    oracle_new_family = fmean(oracle_new_per_snapshot) if oracle_new_per_snapshot else 0.0
    selected = fmean(selected_per_snapshot) if selected_per_snapshot else 0.0
    verdict = _graded_verdict(oracle_new_family, baseline, eps)

    return {
        "seeds": list(seeds),
        "families": list(families),
        "base_family": base_family,
        "incumbent": incumbent,
        "episode_steps": episode_steps,
        "snapshot_stride": snapshot_stride,
        "max_snapshots": max_snapshots,
        "enable_comets": enable_comets,
        "baseline": baseline,
        "eps": eps,
        "snapshots_evaluated": len(oracle_all_per_snapshot),
        "oracle_all_margin": oracle_all,
        "oracle_new_family_margin": oracle_new_family,
        "selected_margin": selected,
        "new_families": new_families,
        "candidate_action_margin": {
            f: (fmean(v) if v else 0.0) for f, v in ca_margins.items()
        },
        "family_continuation_margin": {
            f: (fmean(v) if v else 0.0) for f, v in fc_margins.items()
        },
        "crashes": legality["crashes"],
        "invalid_actions": legality["invalid"],
        "verdict": verdict,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1 candidate oracle (generator-vs-selector)")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--families", type=str, default=",".join(available_families()))
    parser.add_argument("--base-family", type=str, default="oep")
    parser.add_argument("--incumbent", type=str, default="producer")
    parser.add_argument("--episode-steps", type=int, default=96)
    parser.add_argument("--snapshot-stride", type=int, default=32)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--no-comets", action="store_true")
    parser.add_argument("--act-timeout", type=float, default=1.0)
    parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    summary = run_oracle(
        seeds=list(range(int(args.seeds))),
        families=families,
        base_family=str(args.base_family),
        incumbent=str(args.incumbent),
        episode_steps=int(args.episode_steps),
        snapshot_stride=int(args.snapshot_stride),
        max_snapshots=int(args.max_snapshots),
        enable_comets=not bool(args.no_comets),
        act_timeout=float(args.act_timeout),
        baseline=float(args.baseline),
        eps=float(args.eps),
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
