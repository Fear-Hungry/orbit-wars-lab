from scripts.gate_check import (
    DEFAULT_CONFIG,
    GateConfig,
    _final_seed_list,
    _gate_floors,
    _gate_holdout,
    _gate_no_silent_fallbacks,
    _gate_regression,
    _load_config,
)


def _cfg() -> GateConfig:
    return GateConfig(
        episode_steps=500,
        enable_comets=True,
        act_timeout=1.0,
        benchmark_seeds=8,
        jobs=1,
        technical_seeds=[0, 1, 2, 3],
        holdout_seeds=[17, 53],
        final_seed_start=100,
        final_seeds=20,
        opponents=["greedy", "weak_random"],
        hall_of_fame_opponents=[],
        floors={"greedy": 0.95, "weak_random": 0.95, "hall_of_fame": 0.55, "four_player": 0.70},
        regression={
            "max_2p_win_rate_drop": 0.05,
            "max_mean_score_margin_drop": 0.10,
            "max_worst_decile_score_margin_drop": 0.10,
        },
        min_holdout_worst_decile=0.10,
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


def test_gate_floors_checks_hall_of_fame_group(tmp_path):
    hof_path = tmp_path / "submission_prev.py"
    hof_path.write_text("def agent(obs):\n    return []\n", encoding="utf-8")
    cfg = GateConfig(
        episode_steps=500,
        enable_comets=True,
        act_timeout=1.0,
        benchmark_seeds=8,
        jobs=1,
        technical_seeds=[0, 1, 2, 3],
        holdout_seeds=[17, 53],
        final_seed_start=100,
        final_seeds=20,
        opponents=["greedy"],
        hall_of_fame_opponents=[str(hof_path)],
        floors={"greedy": 0.95, "hall_of_fame": 0.55},
        regression={
            "max_2p_win_rate_drop": 0.05,
            "max_mean_score_margin_drop": 0.10,
            "max_worst_decile_score_margin_drop": 0.10,
        },
        min_holdout_worst_decile=0.10,
    )
    report = {
        "formats": [
            {
                "format": "2p",
                "opponents": [
                    {"opponent": "greedy", "summary": {"win_rate": 1.0}, "records": []},
                    {"opponent": "submission_prev", "summary": {"win_rate": 0.50}, "records": []},
                ],
            }
        ]
    }

    gate = _gate_floors(report, cfg)

    assert not gate["passed"]
    failed = [check for check in gate["checks"] if not check["passed"]]
    assert failed[0]["opponent"] == "hall_of_fame"


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


def test_gate_no_silent_fallbacks_rejects_nonzero_fallback_rate():
    report = {
        "formats": [
            {
                "format": "2p",
                "opponents": [
                    {
                        "opponent": "producer",
                        "summary": {
                            "fallback_rate": 0.25,
                            "policy_illegal_move_rate": 0.0,
                            "fallback_error_rate": 0.0,
                        },
                    }
                ],
            }
        ]
    }

    gate = _gate_no_silent_fallbacks(report)

    assert not gate["passed"]
    failed = [check for check in gate["checks"] if not check["passed"]]
    assert failed == [
        {
            "opponent": "producer",
            "metric": "fallback_rate",
            "value": 0.25,
            "maximum": 0.0,
            "passed": False,
        }
    ]


def test_gate_no_silent_fallbacks_rejects_report_without_checked_formats():
    gate = _gate_no_silent_fallbacks({"formats": []})

    assert not gate["passed"]
    assert gate["checks"] == []


def test_gate_no_silent_fallbacks_rejects_missing_instrumentation_rate():
    report = {
        "formats": [
            {
                "format": "4p",
                "summary": {
                    "fallback_rate": 0.0,
                    "policy_illegal_move_rate": 0.0,
                    "fallback_error_rate": 0.0,
                    "instrumentation_missing_rate": 1.0,
                },
            }
        ]
    }

    gate = _gate_no_silent_fallbacks(report)

    assert not gate["passed"]
    failed = [check for check in gate["checks"] if not check["passed"]]
    assert failed == [
        {
            "opponent": "four_player",
            "metric": "instrumentation_missing_rate",
            "value": 1.0,
            "maximum": 0.0,
            "passed": False,
        }
    ]


def test_final_seed_list_uses_holdout_range_offset():
    assert _final_seed_list(_cfg()) == list(range(100, 120))


def test_default_gate_uses_tight_general_floors_without_seed_specific_matchups():
    cfg = _load_config(DEFAULT_CONFIG)

    assert cfg.floors == {
        "greedy": 0.95,
        "defensive": 0.85,
        "rush": 0.85,
        "anti_meta": 0.95,
        "weak_random": 0.95,
        "hall_of_fame": 0.55,
        "four_player": 0.70,
    }
    assert cfg.hall_of_fame_opponents == ["artifacts/hof/submission_v_old.py"]
    assert cfg.jobs == 8
    assert cfg.min_holdout_worst_decile == 0.10
