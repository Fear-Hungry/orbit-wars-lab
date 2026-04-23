from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import yaml
from kaggle_environments import make
from python.league.evaluation import AgentSpec, load_population_manifest
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.train.evaluate_population import _moves_are_legal, _normalized_margin, _policy_runtime
from python.train.final_selection import (
    FinalSelectionConfig,
    load_final_selection_config,
    run_final_selection,
)


@dataclass(frozen=True)
class ObjectiveValidationConfig:
    max_crash_rate: float = 0.0
    max_timeout_rate: float = 0.0
    max_invalid_action_rate: float = 0.0
    min_selection_win_rate: float = 0.25
    min_mean_score_margin: float = 0.0
    min_worst_decile_score_margin: float = -1.0
    max_selection_holdout_win_rate_gap: float = 1.0
    max_selection_holdout_mean_score_margin_gap: float = 2.0
    max_selection_holdout_worst_decile_score_margin_gap: float = 2.0
    holdout_seeds: list[int] = field(default_factory=list)
    require_holdout_hall_of_fame: bool = True
    min_holdout_win_rate: float = 0.5
    min_holdout_mean_score_margin: float = 0.0
    min_holdout_worst_decile_score_margin: float = -1.0
    min_holdout_per_opponent_win_rate: float = 0.0
    min_holdout_per_opponent_worst_decile_score_margin: float = -1.0
    max_holdout_position_win_rate_gap: float = 1.0
    max_holdout_position_mean_score_margin_gap: float = 2.0
    max_holdout_position_worst_decile_score_margin_gap: float = 2.0
    require_2p_self_play: bool = True
    require_4p_self_play: bool = True


def _parse_seed_list(raw: Any, *, field_name: str) -> list[int]:
    if raw is None:
        return []
    seeds = list(range(int(raw))) if isinstance(raw, int) else [int(seed) for seed in raw]
    if not seeds:
        raise ValueError(f"objective validation config requires non-empty `{field_name}` when provided")
    return seeds


def _same_agent_identity(left: AgentSpec, right: AgentSpec) -> bool:
    if left.id == right.id:
        return True
    if left.kind == right.kind == "heuristic" and left.policy and right.policy:
        return left.policy == right.policy
    if left.kind == right.kind == "ppo" and left.checkpoint and right.checkpoint:
        return left.checkpoint == right.checkpoint
    return False


def load_objective_validation_config(path: str | Path) -> ObjectiveValidationConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = payload["objective_validation"]
    return ObjectiveValidationConfig(
        max_crash_rate=float(cfg.get("max_crash_rate", 0.0)),
        max_timeout_rate=float(cfg.get("max_timeout_rate", 0.0)),
        max_invalid_action_rate=float(cfg.get("max_invalid_action_rate", 0.0)),
        min_selection_win_rate=float(cfg.get("min_selection_win_rate", 0.25)),
        min_mean_score_margin=float(cfg.get("min_mean_score_margin", 0.0)),
        min_worst_decile_score_margin=float(cfg.get("min_worst_decile_score_margin", -1.0)),
        max_selection_holdout_win_rate_gap=float(cfg.get("max_selection_holdout_win_rate_gap", 1.0)),
        max_selection_holdout_mean_score_margin_gap=float(
            cfg.get("max_selection_holdout_mean_score_margin_gap", 2.0)
        ),
        max_selection_holdout_worst_decile_score_margin_gap=float(
            cfg.get("max_selection_holdout_worst_decile_score_margin_gap", 2.0)
        ),
        holdout_seeds=_parse_seed_list(cfg.get("holdout_seeds"), field_name="holdout_seeds"),
        require_holdout_hall_of_fame=bool(cfg.get("require_holdout_hall_of_fame", True)),
        min_holdout_win_rate=float(cfg.get("min_holdout_win_rate", 0.5)),
        min_holdout_mean_score_margin=float(cfg.get("min_holdout_mean_score_margin", 0.0)),
        min_holdout_worst_decile_score_margin=float(cfg.get("min_holdout_worst_decile_score_margin", -1.0)),
        min_holdout_per_opponent_win_rate=float(cfg.get("min_holdout_per_opponent_win_rate", 0.0)),
        min_holdout_per_opponent_worst_decile_score_margin=float(
            cfg.get("min_holdout_per_opponent_worst_decile_score_margin", -1.0)
        ),
        max_holdout_position_win_rate_gap=float(cfg.get("max_holdout_position_win_rate_gap", 1.0)),
        max_holdout_position_mean_score_margin_gap=float(cfg.get("max_holdout_position_mean_score_margin_gap", 2.0)),
        max_holdout_position_worst_decile_score_margin_gap=float(
            cfg.get("max_holdout_position_worst_decile_score_margin_gap", 2.0)
        ),
        require_2p_self_play=bool(cfg.get("require_2p_self_play", True)),
        require_4p_self_play=bool(cfg.get("require_4p_self_play", True)),
    )


def _load_agent_callable(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load submission module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _self_play_validation(path: Path, num_players: int) -> dict[str, Any]:
    agent = _load_agent_callable(path)
    env = make("orbit_wars", configuration={"agents": num_players}, debug=True)
    env.run([agent] * num_players)
    return {
        "players": num_players,
        "all_done": all(state.status == "DONE" for state in env.state),
    }


def _export_runtime_validation(
    path: Path,
    *,
    num_players: int,
    seeds: list[int],
    episode_steps: int,
    enable_comets: bool,
    act_timeout: float = 1.0,
) -> dict[str, Any]:
    agent = _load_agent_callable(path)
    crashes = 0
    timeouts = 0
    invalid_actions = 0
    decision_turns = 0
    games = 0
    all_done = True

    for seed in seeds:
        backend = RustBatchBackend(
            num_envs=1,
            num_players=num_players,
            seed=seed,
            config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets, act_timeout=act_timeout),
        )
        state = backend.reset(seed)[0]
        games += 1
        while True:
            actions = [[] for _ in range(num_players)]
            for player in range(num_players):
                decision_turns += 1
                obs = to_official_observation(state, player=player)
                try:
                    start = perf_counter()
                    moves = agent(obs)
                    elapsed = perf_counter() - start
                except Exception as exc:
                    crashes += 1
                    raise RuntimeError(
                        f"export runtime validation crashed: path={path} seed={seed} "
                        f"players={num_players} player={player} decision_turn={decision_turns}"
                    ) from exc
                if elapsed > act_timeout:
                    timeouts += 1
                    raise TimeoutError(
                        f"export runtime validation timed out: path={path} seed={seed} "
                        f"players={num_players} player={player} elapsed={elapsed:.6f}s "
                        f"limit={act_timeout:.6f}s decision_turn={decision_turns}"
                    )
                if not isinstance(moves, list) or not _moves_are_legal(state, player, moves):
                    invalid_actions += 1
                    raise ValueError(
                        f"export runtime validation returned invalid moves: path={path} seed={seed} "
                        f"players={num_players} player={player} decision_turn={decision_turns} "
                        f"moves={moves!r}"
                    )
                actions[player] = moves

            outcome = backend.step([actions])[0]
            state = backend.states()[0]
            if outcome["done"]:
                break
        all_done = all_done and bool(outcome["done"])

    total_turns = max(decision_turns, 1)
    return {
        "players": num_players,
        "seeds": list(seeds),
        "games": games,
        "decision_turns": decision_turns,
        "crash_rate": crashes / total_turns,
        "timeout_rate": timeouts / total_turns,
        "invalid_action_rate": invalid_actions / total_turns,
        "all_done": all_done,
    }


def _worst_decile(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    cutoff = max(1, len(ordered) // 10)
    return float(sum(ordered[:cutoff]) / cutoff)


def _run_export_match(
    export_path: Path,
    opponents: list[AgentSpec],
    *,
    seed: int,
    episode_steps: int,
    enable_comets: bool,
    export_position: int = 0,
) -> list[float]:
    export_agent = _load_agent_callable(export_path)
    num_players = 1 + len(opponents)
    if export_position < 0 or export_position >= num_players:
        raise ValueError(f"invalid export_position={export_position} for num_players={num_players}")
    player_specs: list[AgentSpec | None] = []
    opponents_iter = iter(opponents)
    for idx in range(num_players):
        if idx == export_position:
            player_specs.append(None)
        else:
            player_specs.append(next(opponents_iter))
    opponent_runtime = {idx: _policy_runtime(spec) for idx, spec in enumerate(player_specs) if spec is not None}
    backend = RustBatchBackend(
        num_envs=1,
        num_players=num_players,
        seed=seed,
        config=RustConfig(episode_steps=episode_steps, enable_comets=enable_comets),
    )
    state = backend.reset(seed)[0]
    while True:
        actions = [[] for _ in range(num_players)]
        for idx in range(num_players):
            if idx == export_position:
                obs = to_official_observation(state, player=idx)
                try:
                    moves = export_agent(obs)
                except Exception as exc:
                    raise RuntimeError(
                        f"export holdout match crashed: path={export_path} seed={seed} "
                        f"export_position={export_position} player={idx}"
                    ) from exc
                if not isinstance(moves, list) or not _moves_are_legal(state, idx, moves):
                    raise ValueError(
                        f"export holdout match returned invalid moves: path={export_path} "
                        f"seed={seed} export_position={export_position} player={idx} moves={moves!r}"
                    )
                actions[idx] = moves
                continue
            runtime = opponent_runtime[idx]
            try:
                reply = runtime(state, idx)
            except Exception as exc:
                opponent = player_specs[idx]
                raise RuntimeError(
                    f"holdout opponent policy crashed: opponent={getattr(opponent, 'id', None)!r} "
                    f"policy={getattr(opponent, 'policy', None)!r} seed={seed} player={idx} "
                    f"export_position={export_position}"
                ) from exc
            if not isinstance(reply, list) or not _moves_are_legal(state, idx, reply):
                opponent = player_specs[idx]
                raise ValueError(
                    f"holdout opponent policy returned invalid moves: "
                    f"opponent={getattr(opponent, 'id', None)!r} "
                    f"policy={getattr(opponent, 'policy', None)!r} seed={seed} player={idx} "
                    f"export_position={export_position} moves={reply!r}"
                )
            actions[idx] = reply
        outcome = backend.step([actions])[0]
        state = backend.states()[0]
        if outcome["done"]:
            return [float(score) for score in outcome["scores"]]


def _resolve_holdout_seeds(
    selection_cfg: FinalSelectionConfig,
    validation_cfg: ObjectiveValidationConfig,
) -> tuple[list[int], str]:
    if validation_cfg.holdout_seeds:
        return list(validation_cfg.holdout_seeds), "objective_validation.holdout_seeds"
    return list(selection_cfg.retained_seeds), "final_selection.retained_seeds"


def _record_win_rate(record: dict[str, Any]) -> float:
    scores = [float(score) for score in record["scores"]]
    export_position = int(record.get("export_position", 0))
    best_score = max(scores)
    winners = sum(1 for score in scores if score == best_score)
    return 1.0 / max(winners, 1) if scores[export_position] == best_score else 0.0


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(record["normalized_margin"]) for record in records]
    return {
        "games": len(records),
        "win_rate": fmean(_record_win_rate(record) for record in records) if records else 0.0,
        "mean_score_margin": fmean(margins) if margins else 0.0,
        "worst_decile_score_margin": _worst_decile(margins),
    }


def _per_opponent_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["mode"] != "2p":
            continue
        opponent_id = str(record["opponents"][0])
        grouped.setdefault(opponent_id, []).append(record)
    return {
        opponent_id: _summarize_records(opponent_records)
        for opponent_id, opponent_records in sorted(grouped.items())
    }


def _mode_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["mode"]), []).append(record)
    return {mode: _summarize_records(mode_records) for mode, mode_records in sorted(grouped.items())}


def _two_player_position_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record["mode"] != "2p":
            continue
        export_position = int(record.get("export_position", 0))
        grouped.setdefault(export_position, []).append(record)
    positions = {
        f"player_{position}": _summarize_records(position_records)
        for position, position_records in sorted(grouped.items())
    }
    gaps: dict[str, float] = {}
    if 0 in grouped and 1 in grouped:
        player_0 = positions["player_0"]
        player_1 = positions["player_1"]
        gaps = {
            "win_rate_gap": abs(float(player_0["win_rate"]) - float(player_1["win_rate"])),
            "mean_score_margin_gap": abs(
                float(player_0["mean_score_margin"]) - float(player_1["mean_score_margin"])
            ),
            "worst_decile_score_margin_gap": abs(
                float(player_0["worst_decile_score_margin"]) - float(player_1["worst_decile_score_margin"])
            ),
        }
    return {
        "positions": positions,
        "gaps": gaps,
    }


def _holdout_hall_of_fame_validation(
    export_path: Path,
    *,
    candidate_spec: AgentSpec,
    hall_of_fame: list[AgentSpec],
    holdout_seeds: list[int],
    cfg: FinalSelectionConfig,
) -> dict[str, Any]:
    opponents = [spec for spec in hall_of_fame if not _same_agent_identity(spec, candidate_spec)]
    if not holdout_seeds:
        return {"enabled": False, "reason": "holdout_seeds_empty"}
    if not opponents:
        return {
            "enabled": False,
            "reason": "hall_of_fame_empty",
            "holdout_seeds": list(holdout_seeds),
            "hall_of_fame_opponents": [],
            "games": 0,
        }

    records: list[dict[str, Any]] = []

    for seed in holdout_seeds:
        for opponent in opponents:
            for export_position in (0, 1):
                scores = _run_export_match(
                    export_path,
                    [opponent],
                    seed=seed,
                    episode_steps=cfg.episode_steps,
                    enable_comets=cfg.enable_comets,
                    export_position=export_position,
                )
                records.append(
                    {
                        "mode": "2p",
                        "seed": seed,
                        "export_position": export_position,
                        "opponents": [opponent.id],
                        "scores": scores,
                        "normalized_margin": _normalized_margin(scores, export_position),
                    }
                )

        if cfg.include_4p and len(opponents) >= 3:
            for lobby in combinations(opponents, 3):
                scores = _run_export_match(
                    export_path,
                    list(lobby),
                    seed=seed,
                    episode_steps=cfg.episode_steps,
                    enable_comets=cfg.enable_comets,
                    export_position=0,
                )
                records.append(
                    {
                        "mode": "4p",
                        "seed": seed,
                        "export_position": 0,
                        "opponents": [spec.id for spec in lobby],
                        "scores": scores,
                        "normalized_margin": _normalized_margin(scores, 0),
                    }
                )

    overall_summary = _summarize_records(records)
    per_opponent_2p = _per_opponent_summary(records)
    two_player_position_summary = _two_player_position_summary(records)
    return {
        "enabled": True,
        "holdout_seeds": list(holdout_seeds),
        "hall_of_fame_opponents": [spec.id for spec in opponents],
        **overall_summary,
        "mode_summary": _mode_summary(records),
        "per_opponent_2p": per_opponent_2p,
        "worst_matchup_2p": {
            "min_win_rate": min((summary["win_rate"] for summary in per_opponent_2p.values()), default=0.0),
            "min_worst_decile_score_margin": min(
                (summary["worst_decile_score_margin"] for summary in per_opponent_2p.values()),
                default=0.0,
            ),
        },
        "two_player_position_summary": two_player_position_summary,
        "records": records,
    }


def run_objective_validation(
    manifest: dict[str, list[AgentSpec]],
    selection_cfg: FinalSelectionConfig,
    validation_cfg: ObjectiveValidationConfig,
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    holdout_seeds, holdout_seed_source = _resolve_holdout_seeds(selection_cfg, validation_cfg)

    selection_report = run_final_selection(manifest, selection_cfg, out_dir=out_path)
    export_ids = [item["candidate_id"] for item in selection_report["exports"]]
    exported_rankings = {item["candidate_id"]: item for item in selection_report["ranking"] if item["candidate_id"] in export_ids}

    candidate_checks = []
    for rank, export in enumerate(selection_report["exports"], start=1):
        candidate_id = export["candidate_id"]
        ranking = exported_rankings[candidate_id]
        candidate_spec = next(spec for group in manifest.values() for spec in group if spec.id == candidate_id)
        export_path = out_path / f"candidate_{rank}_submission.py"
        runtime_validations = []
        if validation_cfg.require_2p_self_play:
            runtime_validations.append(
                _export_runtime_validation(
                    export_path,
                    num_players=2,
                    seeds=holdout_seeds,
                    episode_steps=selection_cfg.episode_steps,
                    enable_comets=selection_cfg.enable_comets,
                )
            )
        if validation_cfg.require_4p_self_play:
            runtime_validations.append(
                _export_runtime_validation(
                    export_path,
                    num_players=4,
                    seeds=holdout_seeds,
                    episode_steps=selection_cfg.episode_steps,
                    enable_comets=selection_cfg.enable_comets,
                )
            )
        checks = {
            "candidate_id": candidate_id,
            "rank": rank,
            "selection_win_rate_ok": float(ranking["win_rate"]) >= validation_cfg.min_selection_win_rate,
            "mean_score_margin_ok": float(ranking["mean_score_margin"]) >= validation_cfg.min_mean_score_margin,
            "worst_decile_ok": float(ranking["worst_decile_score_margin"]) >= validation_cfg.min_worst_decile_score_margin,
            "ranking": ranking,
            "runtime_validations": runtime_validations,
        }
        checks["crash_rate_ok"] = all(
            validation["crash_rate"] <= validation_cfg.max_crash_rate for validation in runtime_validations
        )
        checks["timeout_rate_ok"] = all(
            validation["timeout_rate"] <= validation_cfg.max_timeout_rate for validation in runtime_validations
        )
        checks["invalid_action_rate_ok"] = all(
            validation["invalid_action_rate"] <= validation_cfg.max_invalid_action_rate for validation in runtime_validations
        )
        checks["self_play"] = []
        if validation_cfg.require_2p_self_play:
            checks["self_play"].append(_self_play_validation(export_path, 2))
        if validation_cfg.require_4p_self_play:
            checks["self_play"].append(_self_play_validation(export_path, 4))
        checks["self_play_ok"] = all(result["all_done"] for result in checks["self_play"])
        checks["holdout_validation"] = _holdout_hall_of_fame_validation(
            export_path,
            candidate_spec=candidate_spec,
            hall_of_fame=manifest["hall_of_fame"],
            holdout_seeds=holdout_seeds,
            cfg=selection_cfg,
        )
        checks["holdout_vs_hall_of_fame"] = checks["holdout_validation"]
        holdout = checks["holdout_validation"]
        if validation_cfg.require_holdout_hall_of_fame:
            checks["holdout_hall_of_fame_ok"] = bool(holdout.get("enabled")) and (
                float(holdout.get("win_rate", 0.0)) >= validation_cfg.min_holdout_win_rate
                and float(holdout.get("mean_score_margin", -1.0)) >= validation_cfg.min_holdout_mean_score_margin
                and float(holdout.get("worst_decile_score_margin", -1.0))
                >= validation_cfg.min_holdout_worst_decile_score_margin
            )
        else:
            checks["holdout_hall_of_fame_ok"] = True
        checks["holdout_vs_hall_of_fame_ok"] = checks["holdout_hall_of_fame_ok"]
        holdout_enabled = bool(holdout.get("enabled"))
        checks["selection_vs_holdout_gap"] = {
            "win_rate_gap": float(ranking["win_rate"]) - float(holdout.get("win_rate", 0.0)) if holdout_enabled else None,
            "mean_score_margin_gap": float(ranking["mean_score_margin"]) - float(holdout.get("mean_score_margin", 0.0))
            if holdout_enabled
            else None,
            "worst_decile_score_margin_gap": float(ranking["worst_decile_score_margin"])
            - float(holdout.get("worst_decile_score_margin", 0.0))
            if holdout_enabled
            else None,
        }
        checks["generalization_gap_ok"] = holdout_enabled and (
            float(checks["selection_vs_holdout_gap"]["win_rate_gap"]) <= validation_cfg.max_selection_holdout_win_rate_gap
            and float(checks["selection_vs_holdout_gap"]["mean_score_margin_gap"])
            <= validation_cfg.max_selection_holdout_mean_score_margin_gap
            and float(checks["selection_vs_holdout_gap"]["worst_decile_score_margin_gap"])
            <= validation_cfg.max_selection_holdout_worst_decile_score_margin_gap
        )
        per_opponent_2p = holdout.get("per_opponent_2p", {})
        checks["per_opponent_holdout_ok"] = holdout_enabled and bool(per_opponent_2p) and all(
            float(summary["win_rate"]) >= validation_cfg.min_holdout_per_opponent_win_rate
            and float(summary["worst_decile_score_margin"])
            >= validation_cfg.min_holdout_per_opponent_worst_decile_score_margin
            for summary in per_opponent_2p.values()
        )
        position_gaps = holdout.get("two_player_position_summary", {}).get("gaps", {})
        checks["position_balance_ok"] = holdout_enabled and bool(position_gaps) and (
            float(position_gaps["win_rate_gap"]) <= validation_cfg.max_holdout_position_win_rate_gap
            and float(position_gaps["mean_score_margin_gap"]) <= validation_cfg.max_holdout_position_mean_score_margin_gap
            and float(position_gaps["worst_decile_score_margin_gap"])
            <= validation_cfg.max_holdout_position_worst_decile_score_margin_gap
        )
        checks["passed"] = all(
            checks[key]
            for key in (
                "crash_rate_ok",
                "timeout_rate_ok",
                "invalid_action_rate_ok",
                "selection_win_rate_ok",
                "mean_score_margin_ok",
                "worst_decile_ok",
                "self_play_ok",
                "holdout_hall_of_fame_ok",
                "generalization_gap_ok",
                "per_opponent_holdout_ok",
                "position_balance_ok",
            )
        )
        candidate_checks.append(checks)

    report = {
        "objective_ready": len(export_ids) == 2 and all(item["passed"] for item in candidate_checks),
        "selection_summary": selection_report["summary"],
        "holdout_summary": {
            "seeds": holdout_seeds,
            "seed_source": holdout_seed_source,
            "hall_of_fame_size": len(manifest["hall_of_fame"]),
        },
        "criteria": asdict(validation_cfg),
        "exports": selection_report["exports"],
        "candidate_checks": candidate_checks,
    }
    report_path = out_path / "objective_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-config", default="configs/final_selection.yaml")
    parser.add_argument("--validation-config", default="configs/objective_validation.yaml")
    parser.add_argument("--out-dir", default="artifacts/final_candidates")
    args = parser.parse_args()

    report = run_objective_validation(
        load_population_manifest(args.manifest),
        load_final_selection_config(args.selection_config),
        load_objective_validation_config(args.validation_config),
        out_dir=args.out_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
