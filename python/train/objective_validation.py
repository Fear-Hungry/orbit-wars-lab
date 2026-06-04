from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from kaggle_environments import make
from python.league.evaluation import AgentSpec, load_population_manifest
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.observation import to_official_observation
from python.train.evaluate_population import _moves_are_legal
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
    require_2p_self_play: bool = True
    require_4p_self_play: bool = True


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
                    if elapsed > act_timeout:
                        timeouts += 1
                        moves = []
                    if not isinstance(moves, list) or not _moves_are_legal(state, player, moves):
                        invalid_actions += 1
                        moves = []
                except Exception:
                    crashes += 1
                    moves = []
                actions[player] = moves

            outcomes, states = backend.step_with_states([actions])
            outcome = outcomes[0]
            state = states[0]
            if outcome["done"]:
                break
        all_done = all_done and bool(outcome["done"])

    total_turns = max(decision_turns, 1)
    return {
        "players": num_players,
        "games": games,
        "decision_turns": decision_turns,
        "crash_rate": crashes / total_turns,
        "timeout_rate": timeouts / total_turns,
        "invalid_action_rate": invalid_actions / total_turns,
        "all_done": all_done,
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

    selection_report = run_final_selection(manifest, selection_cfg, out_dir=out_path)
    export_ids = [item["candidate_id"] for item in selection_report["exports"]]
    exported_rankings = {item["candidate_id"]: item for item in selection_report["ranking"] if item["candidate_id"] in export_ids}

    candidate_checks = []
    for rank, export in enumerate(selection_report["exports"], start=1):
        candidate_id = export["candidate_id"]
        ranking = exported_rankings[candidate_id]
        export_path = out_path / f"candidate_{rank}_submission.py"
        runtime_validations = []
        if validation_cfg.require_2p_self_play:
            runtime_validations.append(
                _export_runtime_validation(
                    export_path,
                    num_players=2,
                    seeds=selection_cfg.retained_seeds,
                    episode_steps=selection_cfg.episode_steps,
                    enable_comets=selection_cfg.enable_comets,
                )
            )
        if validation_cfg.require_4p_self_play:
            runtime_validations.append(
                _export_runtime_validation(
                    export_path,
                    num_players=4,
                    seeds=selection_cfg.retained_seeds,
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
            )
        )
        candidate_checks.append(checks)

    report = {
        "objective_ready": len(export_ids) == 2 and all(item["passed"] for item in candidate_checks),
        "selection_summary": selection_report["summary"],
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
