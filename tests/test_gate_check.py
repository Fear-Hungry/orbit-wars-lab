from pathlib import Path

from scripts.gate_check import (
    DEFAULT_CONFIG,
    GateConfig,
    _final_seed_list,
    _gate_critical_matchups,
    _gate_floors,
    _gate_holdout,
    _gate_regression,
    _load_config,
)


def _cfg() -> GateConfig:
    return GateConfig(
        episode_steps=500,
        enable_comets=True,
        act_timeout=1.0,
        benchmark_seeds=8,
        technical_seeds=[0, 1, 2, 3],
        holdout_seeds=[17, 53],
        final_seed_start=100,
        final_seeds=20,
        opponents=["greedy", "weak_random"],
        floors={"greedy": 0.90, "weak_random": 0.85, "four_player": 0.50},
        regression={
            "max_2p_win_rate_drop": 0.05,
            "max_mean_score_margin_drop": 0.10,
            "max_worst_decile_score_margin_drop": 0.10,
        },
        min_holdout_worst_decile=-0.30,
        critical_matchups=[],
    )


def _report(*, greedy: float, weak_random: float, four_player: float, margins: list[float]) -> dict:
    two_player_records = [{"normalized_margin": margin} for margin in margins]
    return {
        "formats": [
            {
                "format": "2p",
                "opponents": [
                    {"opponent": "greedy", "summary": {"win_rate": greedy}, "records": two_player_records[:1]},
                    {
                        "opponent": "weak_random",
                        "summary": {"win_rate": weak_random},
                        "records": two_player_records[1:],
                    },
                ],
            },
            {
                "format": "4p",
                "summary": {"win_rate": four_player},
                "records": [{"normalized_margin": four_player}],
            },
        ]
    }


def test_gate_floors_requires_each_opponent_to_pass():
    gate = _gate_floors(_report(greedy=0.95, weak_random=0.80, four_player=0.75, margins=[1.0, 0.0]), _cfg())

    assert not gate["passed"]
    assert [check for check in gate["checks"] if not check["passed"]][0]["opponent"] == "weak_random"


def test_gate_regression_compares_candidate_against_baseline():
    baseline = _report(greedy=0.90, weak_random=0.90, four_player=0.75, margins=[0.5, 0.5, 0.5, 0.5])
    candidate = _report(greedy=0.90, weak_random=0.70, four_player=0.75, margins=[0.5, 0.5, -0.5, -0.5])

    gate = _gate_regression(candidate, baseline, _cfg())

    assert not gate["passed"]
    assert {check["metric"] for check in gate["checks"] if not check["passed"]} == {
        "win_rate_2p_mean",
        "mean_score_margin",
        "worst_decile_score_margin",
    }


def test_gate_holdout_enforces_worst_decile_floor():
    report = _report(greedy=1.0, weak_random=1.0, four_player=1.0, margins=[1.0, -1.0, 1.0, 1.0])

    gate = _gate_holdout(report, _cfg())

    assert not gate["passed"]
    assert gate["metrics"]["worst_decile_score_margin"] == -1.0


def test_final_seed_list_uses_holdout_range_offset():
    assert _final_seed_list(_cfg()) == list(range(100, 120))


def test_gate_critical_matchups_passes_when_no_cases_are_configured():
    gate = _gate_critical_matchups(Path("unused.py"), _cfg())

    assert gate == {"name": "gate_2b_critical_matchups", "passed": True, "checks": []}


def test_default_gate_tracks_known_weak_random_seed_two_failure():
    cfg = _load_config(DEFAULT_CONFIG)

    assert {
        "opponent": "weak_random",
        "seed": 2,
        "submission_player": 1,
        "min_win_points": 1.0,
        "min_normalized_margin": 0.0,
    } in cfg.critical_matchups
