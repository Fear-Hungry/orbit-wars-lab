"""Decision engine: short-circuits, bands, gates, schema (synthetic)."""
from __future__ import annotations

import scripts.league_submission_selector as sel
from scripts.league_agents import LB_ANCHORS


def _pair(wins, losses, *, ties=0):
    decisive = wins + losses
    appearances = decisive + ties
    return {
        "appearances": appearances, "audited": appearances, "wins": wins,
        "losses": losses, "ties": ties, "decisive": decisive,
        "win_rate": wins / appearances if appearances else 0.0,
        "decisive_win_rate": wins / decisive if decisive else None,
        "nonloss_rate": (wins + ties) / appearances if appearances else 0.0,
        "annihilation_rate": 0.0, "bad_status": 0,
        "faults": {"crashes": 0, "timeouts": 0, "invalid_moves": 0},
    }


def _summary(name, *, inc_wins=80, inc_losses=20, latency=200.0, verdict="PASS_LOCAL",
             win_4p=0.35, ann_4p=0.10, templates=(0.35, 0.30), faults=0):
    pairwise = {
        "pgs_holdwave": _pair(inc_wins, inc_losses),
        "producer": _pair(70, 30),
        "oep": _pair(65, 35),
        "pgs_allscripts": _pair(90, 10),
        "pgs_bigwave": _pair(60, 40),
    }
    score_2p = sum(p["win_rate"] for p in pairwise.values()) / len(pairwise)
    return {
        "candidate": name,
        "verdict": verdict,
        "score_2p_fixed": score_2p,
        "score_2p_peer": None,
        "score_4p_fixed": win_4p,
        "adv_2p_fixed": 2 * score_2p - 1,
        "adv_4p_fixed": (win_4p - 0.25) / 0.75,
        "field_advantage": 0.0,
        "overall": {
            "faults": {"crashes": faults, "timeouts": 0, "invalid_moves": 0},
            "bad_status": 0, "audited": 1, "appearances": 1,
        },
        "four_player": {"win_rate": win_4p, "annihilation_rate": ann_4p,
                        "appearances": 64},
        "four_player_templates": {
            f"line{i}": {"win_rate": t, "appearances": 16} for i, t in enumerate(templates)
        },
        "pairwise_fixed": pairwise,
        "pairwise_peer": {},
        "buckets": {},
        "worst_bucket_score": 0.55,
        "latency_p95_max": latency,
        "latency_audited": True,
        "checks": [
            {"name": "no_faults", "passed": faults == 0, "severity": "fail", "details": {}},
            {"name": "all_status_done", "passed": True, "severity": "fail", "details": {}},
        ],
        "risk_penalty": 0.0,
        "risk_components": {},
    }


def _report(candidates, *, seed_split="selector", profile="selector"):
    return {
        "incumbent": "pgs_holdwave",
        "references": ["producer", "oep", "pgs_allscripts", "pgs_bigwave"],
        "candidates": candidates,
        "settings": {"seed_split": seed_split, "profile": profile},
        "local_veto_passes": [n for n, s in candidates.items()
                              if s["verdict"] == "PASS_LOCAL"],
        "selection_status": "VETO_ONLY",
        "promotion_order_valid": False,
    }


def _calibration(valid=True, code_hash="abc"):
    return {
        "calibration_version": "cal_test",
        "calibration_valid": valid,
        "code_hash": code_hash,
        "seed_split": "validation",
        "spearman_selector_lb": 0.7,
        "weights": None,
        "lb_anchors": dict(LB_ANCHORS),
    }


_OK_PREFLIGHT = {"valid": True, "problems": {}, "smoke": False}


def test_invalid_reference_pool_short_circuits():
    decision = sel.decide(
        _report({"cand": _summary("cand")}), _calibration(),
        preflight={"valid": False, "problems": {"ext_lb1050": "missing"}, "smoke": False},
    )
    assert decision["decision"] == "INVALID_REFERENCE_POOL"
    assert decision["chosen_candidate"] is None


def test_calibration_failed_when_invalid_or_hash_mismatch():
    report = _report({"cand": _summary("cand")})
    invalid = sel.decide(report, _calibration(valid=False), preflight=_OK_PREFLIGHT)
    assert invalid["decision"] == "CALIBRATION_FAILED"

    mismatch = sel.decide(report, _calibration(code_hash="old"),
                          preflight=_OK_PREFLIGHT, current_code_hash="new")
    assert mismatch["decision"] == "CALIBRATION_FAILED"


def test_dev_split_report_is_rejected():
    report = _report({"cand": _summary("cand")}, seed_split="dev")
    decision = sel.decide(report, _calibration(), preflight=_OK_PREFLIGHT)
    assert decision["decision"] == "NO_TECHNICALLY_VALID_CANDIDATE"
    rule = next(r for r in decision["rules"] if r["rule"] == "report_is_selector_holdout")
    assert not rule["passed"]


def test_submit_when_confident_and_separated():
    decision = sel.decide(
        _report({"cand": _summary("cand", inc_wins=80, inc_losses=20)}),
        _calibration(), preflight=_OK_PREFLIGHT,
    )
    assert decision["decision"] == "SUBMIT_CANDIDATE"
    assert decision["chosen_candidate"] == "cand"
    assert decision["confidence"]["p_beats_incumbent"] > 0.99
    # etapa 18 schema
    for key in ("selector_version", "selector_valid", "calibration_valid",
                "fallback_choice", "reason", "eligible_candidates",
                "rejected_candidates", "score_components", "risk", "calibration",
                "rules", "seed_split"):
        assert key in decision


def test_run_more_games_band_on_thin_undecided_h2h():
    # 52-48 over 100 decisive games: covers the minimum but does not separate
    # (p ~0.65 < 0.80) and evidence is still thin -> more games, not a guess
    decision = sel.decide(
        _report({"cand": _summary("cand", inc_wins=52, inc_losses=48)}),
        _calibration(), preflight=_OK_PREFLIGHT,
    )
    assert decision["decision"] == "RUN_MORE_GAMES"


def test_keep_incumbent_when_tied_with_plenty_of_evidence():
    # 102-98 over 200 decisive games: undecided with ample evidence -> incumbent
    decision = sel.decide(
        _report({"cand": _summary("cand", inc_wins=102, inc_losses=98)}),
        _calibration(), preflight=_OK_PREFLIGHT,
    )
    assert decision["decision"] == "KEEP_INCUMBENT"
    assert decision["chosen_candidate"] is None
    assert decision["fallback_choice"] == "pgs_holdwave"


def test_choice_level_4p_gates_stricter_than_pass():
    # passes the ruler (verdict PASS_LOCAL) but 4p win rate below the CHOICE bar
    weak_4p = _summary("cand", win_4p=0.26, templates=(0.26, 0.26))
    decision = sel.decide(_report({"cand": weak_4p}), _calibration(),
                          preflight=_OK_PREFLIGHT)
    assert decision["decision"] == "NO_TECHNICALLY_VALID_CANDIDATE"
    assert "choice_4p_winrate" in decision["rejected_candidates"]["cand"]


def test_latency_above_500_blocks_choice():
    slow = _summary("cand", latency=650.0)
    decision = sel.decide(_report({"cand": slow}), _calibration(), preflight=_OK_PREFLIGHT)
    assert decision["decision"] == "NO_TECHNICALLY_VALID_CANDIDATE"
    assert "latency_p95_le_500" in decision["rejected_candidates"]["cand"]


def test_never_chooses_rejected_floor_fixture():
    # backtest item (a): a REJECT_LOCAL candidate can never be chosen
    floor = _summary("pgs_allscripts_v2", verdict="REJECT_LOCAL", faults=1)
    strong = _summary("cand", inc_wins=80, inc_losses=20)
    decision = sel.decide(_report({"pgs_allscripts_v2": floor, "cand": strong}),
                          _calibration(), preflight=_OK_PREFLIGHT)
    assert decision["chosen_candidate"] == "cand"
    assert "pgs_allscripts_v2" in decision["rejected_candidates"]


def test_tiebreak_required_then_resolves_or_keeps_incumbent():
    a = _summary("cand_a", inc_wins=80, inc_losses=20)
    b = _summary("cand_b", inc_wins=79, inc_losses=21)
    report = _report({"cand_a": a, "cand_b": b})

    undecided = sel.decide(report, _calibration(), preflight=_OK_PREFLIGHT)
    assert undecided["decision"] == "RUN_MORE_GAMES"
    assert "tiebreak" in undecided["reason"]

    resolved = sel.decide(report, _calibration(), preflight=_OK_PREFLIGHT,
                          tiebreak_p_a_over_b=0.85)
    assert resolved["decision"] == "SUBMIT_CANDIDATE"

    still_tied = sel.decide(report, _calibration(), preflight=_OK_PREFLIGHT,
                            tiebreak_p_a_over_b=0.55)
    assert still_tied["decision"] == "KEEP_INCUMBENT"


def test_preflight_flags_unknown_and_missing(tmp_path, monkeypatch):
    result = sel.preflight_references(["producer", "definitely_not_a_bot"])
    assert not result["valid"]
    assert "definitely_not_a_bot" in result["problems"]

    ok = sel.preflight_references(["producer", "pgs_holdwave"])
    assert ok["valid"]
