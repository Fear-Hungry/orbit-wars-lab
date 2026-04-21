from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from python.league.evaluation import AgentSpec, load_population_manifest
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.train.evaluate_population import _moves_are_legal, _normalized_margin, _policy_runtime
from scripts.export_submission import render_submission

FINAL_EXPORT_TOP_K = 2


@dataclass(frozen=True)
class FinalSelectionConfig:
    retained_seeds: list[int]
    games_per_pair: int
    include_2p: bool
    include_4p: bool
    episode_steps: int = 500
    enable_comets: bool = True
    bad_replay_count: int = 8
    export_top_k: int = FINAL_EXPORT_TOP_K
    candidate_ids: tuple[str, ...] = ()


def load_final_selection_config(path: str | Path) -> FinalSelectionConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = payload["final_selection"]
    retained_seeds = [int(seed) for seed in cfg.get("retained_seeds", [])]
    if not retained_seeds:
        raise ValueError("final selection config requires non-empty `retained_seeds`")
    export_top_k = int(cfg.get("export_top_k", FINAL_EXPORT_TOP_K))
    if export_top_k != FINAL_EXPORT_TOP_K:
        raise ValueError(f"final selection requires export_top_k == {FINAL_EXPORT_TOP_K}")
    return FinalSelectionConfig(
        retained_seeds=retained_seeds,
        games_per_pair=int(cfg.get("games_per_pair", 2)),
        include_2p=bool(cfg.get("include_2p", True)),
        include_4p=bool(cfg.get("include_4p", True)),
        episode_steps=int(cfg.get("episode_steps", 500)),
        enable_comets=bool(cfg.get("enable_comets", True)),
        bad_replay_count=int(cfg.get("bad_replay_count", 8)),
        export_top_k=export_top_k,
        candidate_ids=tuple(str(candidate_id) for candidate_id in cfg.get("candidate_ids", [])),
    )


def _candidate_pool(manifest: dict[str, list[AgentSpec]], candidate_ids: tuple[str, ...]) -> list[AgentSpec]:
    pool: dict[str, AgentSpec] = {}
    for group in manifest.values():
        for spec in group:
            pool.setdefault(spec.id, spec)
    if candidate_ids:
        return [pool[candidate_id] for candidate_id in candidate_ids]
    return list(pool.values())


def _run_match_with_trace(players: list[AgentSpec], seed: int, cfg: FinalSelectionConfig) -> tuple[list[float], list[dict[str, Any]]]:
    runtime = {spec.id: _policy_runtime(spec) for spec in players}
    backend = RustBatchBackend(
        num_envs=1,
        num_players=len(players),
        seed=seed,
        config=RustConfig(episode_steps=cfg.episode_steps, enable_comets=cfg.enable_comets),
    )
    state = backend.reset(seed)[0]
    trace: list[dict[str, Any]] = []

    while True:
        actions = [[] for _ in players]
        for idx, spec in enumerate(players):
            try:
                moves = runtime[spec.id](state, idx)
                if not isinstance(moves, list) or not _moves_are_legal(state, idx, moves):
                    moves = []
            except Exception:
                moves = []
            actions[idx] = moves

        outcome = backend.step([actions])[0]
        state = backend.states()[0]
        trace.append(
            {
                "turn": len(trace),
                "scores": [float(score) for score in outcome["scores"]],
                "move_counts": [len(player_moves) for player_moves in actions],
                "launched_ships": [sum(int(move[2]) for move in player_moves) for player_moves in actions],
            }
        )
        if outcome["done"]:
            break

    return [float(score) for score in outcome["scores"]], trace


def _worst_decile(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    cutoff = max(1, math.ceil(len(ordered) * 0.1))
    return float(sum(ordered[:cutoff]) / cutoff)


def _selection_score(win_rate: float, mean_margin: float, worst_decile_margin: float) -> float:
    return 2.0 * win_rate + 0.8 * mean_margin + 0.6 * worst_decile_margin


def _analyze_bad_replay(record: dict[str, Any]) -> str:
    trace = record["trace"]
    own_idx = int(record["player_index"])
    own_scores = [float(turn["scores"][own_idx]) for turn in trace]
    move_counts = [int(turn["move_counts"][own_idx]) for turn in trace]
    launched_ships = [int(turn["launched_ships"][own_idx]) for turn in trace]
    turns = max(len(trace), 1)
    peak_score = max(own_scores, default=0.0)
    final_score = own_scores[-1] if own_scores else 0.0
    idle_turns = sum(1 for value in move_counts if value == 0)
    total_launched = sum(launched_ships)
    score_drop = peak_score - final_score

    if final_score <= 0.0:
        return f"colapso total apos pico={peak_score:.1f}; zerou o score no fim"
    if score_drop > max(10.0, 0.35 * max(peak_score, 1.0)):
        return f"perdeu controle apos pico={peak_score:.1f}; queda final={score_drop:.1f}"
    if idle_turns > turns // 2:
        return f"ritmo baixo em {idle_turns}/{turns} turnos; lancou {total_launched} naves"
    own_score = float(record["scores"][record["player_index"]])
    other_scores = [float(score) for idx, score in enumerate(record["scores"]) if idx != record["player_index"]]
    mean_other = fmean(other_scores) if other_scores else 0.0
    if mean_other > 0.0 and own_score < 0.5 * mean_other:
        return f"deficit severo contra o lobby; score_final={own_score:.1f} vs media={mean_other:.1f}"
    return f"pressao constante no pior decil; margem={record['normalized_margin']:.3f}, lancou={total_launched} naves"


def run_final_selection(
    manifest: dict[str, list[AgentSpec]],
    cfg: FinalSelectionConfig,
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if cfg.export_top_k != FINAL_EXPORT_TOP_K:
        raise ValueError(f"final selection requires export_top_k == {FINAL_EXPORT_TOP_K}")
    candidates = _candidate_pool(manifest, cfg.candidate_ids)
    if len(candidates) < 2:
        raise ValueError("final selection requires at least two candidates")

    metrics: dict[str, dict[str, Any]] = {
        spec.id: {"wins": 0.0, "games": 0, "margins": [], "records": []}
        for spec in candidates
    }
    global_records: list[dict[str, Any]] = []
    match_counts = {"2p_matches": 0, "4p_matches": 0}

    for seed in cfg.retained_seeds:
        if cfg.include_2p:
            for repeat in range(cfg.games_per_pair):
                for left, right in combinations(candidates, 2):
                    players = [left, right]
                    match_seed = seed * 10_000 + repeat * 101 + match_counts["2p_matches"]
                    scores, trace = _run_match_with_trace(players, match_seed, cfg)
                    match_counts["2p_matches"] += 1
                    max_score = max(scores)
                    winners = [idx for idx, score in enumerate(scores) if score == max_score]
                    for idx, spec in enumerate(players):
                        margin = _normalized_margin(scores, idx)
                        record = {
                            "mode": "2p",
                            "seed": match_seed,
                            "players": [player.id for player in players],
                            "player_index": idx,
                            "candidate_id": spec.id,
                            "scores": scores,
                            "normalized_margin": margin,
                            "trace": trace,
                        }
                        metrics[spec.id]["games"] += 1
                        metrics[spec.id]["wins"] += 1.0 / len(winners) if idx in winners else 0.0
                        metrics[spec.id]["margins"].append(margin)
                        metrics[spec.id]["records"].append(record)
                        global_records.append(record)

        if cfg.include_4p and len(candidates) >= 4:
            for repeat in range(cfg.games_per_pair):
                for players_tuple in combinations(candidates, 4):
                    players = list(players_tuple)
                    match_seed = seed * 100_000 + repeat * 101 + match_counts["4p_matches"]
                    scores, trace = _run_match_with_trace(players, match_seed, cfg)
                    match_counts["4p_matches"] += 1
                    max_score = max(scores)
                    winners = [idx for idx, score in enumerate(scores) if score == max_score]
                    for idx, spec in enumerate(players):
                        margin = _normalized_margin(scores, idx)
                        record = {
                            "mode": "4p",
                            "seed": match_seed,
                            "players": [player.id for player in players],
                            "player_index": idx,
                            "candidate_id": spec.id,
                            "scores": scores,
                            "normalized_margin": margin,
                            "trace": trace,
                        }
                        metrics[spec.id]["games"] += 1
                        metrics[spec.id]["wins"] += 1.0 / len(winners) if idx in winners else 0.0
                        metrics[spec.id]["margins"].append(margin)
                        metrics[spec.id]["records"].append(record)
                        global_records.append(record)

    ranking = []
    for spec in candidates:
        candidate_metrics = metrics[spec.id]
        games = max(int(candidate_metrics["games"]), 1)
        mean_margin = fmean(candidate_metrics["margins"]) if candidate_metrics["margins"] else 0.0
        worst_decile_margin = _worst_decile(candidate_metrics["margins"])
        win_rate = float(candidate_metrics["wins"]) / games
        ranking.append(
            {
                "candidate_id": spec.id,
                "kind": spec.kind,
                "checkpoint": spec.checkpoint,
                "win_rate": win_rate,
                "mean_score_margin": mean_margin,
                "worst_decile_score_margin": worst_decile_margin,
                "games": games,
                "selection_score": _selection_score(win_rate, mean_margin, worst_decile_margin),
            }
        )
    ranking.sort(
        key=lambda item: (
            item["selection_score"],
            item["worst_decile_score_margin"],
            item["mean_score_margin"],
            item["win_rate"],
        ),
        reverse=True,
    )

    worst_records = sorted(global_records, key=lambda record: record["normalized_margin"])[: cfg.bad_replay_count]
    bad_replays = [
        {
            "candidate_id": record["candidate_id"],
            "mode": record["mode"],
            "seed": record["seed"],
            "players": record["players"],
            "scores": record["scores"],
            "normalized_margin": record["normalized_margin"],
            "analysis": _analyze_bad_replay(record),
            "trace": record["trace"],
        }
        for record in worst_records
    ]

    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    exported = []
    for rank, candidate in enumerate(ranking[:FINAL_EXPORT_TOP_K], start=1):
        rendered = render_submission(template, candidate["checkpoint"])
        rendered = (
            f"# candidate_rank: {rank}\n"
            f"# candidate_id: {candidate['candidate_id']}\n"
            f"# selection_score: {candidate['selection_score']:.6f}\n"
            f"{rendered}"
        )
        export_path = out_path / f"candidate_{rank}_submission.py"
        export_path.write_text(rendered, encoding="utf-8")
        exported.append({"candidate_id": candidate["candidate_id"], "path": str(export_path)})

    report = {
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(candidates),
            "retained_seeds": cfg.retained_seeds,
            "match_counts": match_counts,
        },
        "ranking": ranking,
        "bad_replays": bad_replays,
        "exports": exported,
    }
    (out_path / "final_selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_path / "bad_replays.json").write_text(json.dumps(bad_replays, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/final_selection.yaml")
    parser.add_argument("--out-dir", default="artifacts/final_selection")
    args = parser.parse_args()

    manifest = load_population_manifest(args.manifest)
    cfg = load_final_selection_config(args.config)
    report = run_final_selection(manifest, cfg, out_dir=args.out_dir)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
