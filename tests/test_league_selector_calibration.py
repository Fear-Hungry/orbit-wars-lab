"""Calibration logic (synthetic — no games run)."""
from __future__ import annotations

import json

from scripts.league_selector_calibration import (
    calibration_checks,
    check_scoring_isolation,
    match_reusable,
)
from scripts.league_submit_ruler import MatchTask


def _row(name, lb, score, *, gates=True, wins=50, losses=50):
    return {
        "name": name,
        "lb": lb,
        "features": {},
        "selector_score": score,
        "verdict": "PASS_LOCAL" if gates else "REJECT_LOCAL",
        "hard_gates_passed": gates,
        "fixed_record": {"wins": wins, "losses": losses},
    }


def _healthy_rows():
    # selector_score ordered like the LB; allscripts rejected; holdwave on top
    return [
        _row("pgs_holdwave", 1228.8, 0.60, wins=70, losses=30),
        _row("oep", 1182.7, 0.45),
        _row("producer", 1173.1, 0.40),
        _row("pgs_wave_s100", 1146.1, 0.35),
        _row("pgs_hold", 1057.6, 0.20, wins=40, losses=60),
        _row("pgs_allscripts", 1021.5, -0.30, gates=False, wins=10, losses=90),
    ]


def test_calibration_valid_requires_all_checks():
    checks, rho, inversions = calibration_checks(_healthy_rows())
    assert rho >= 0.6
    assert not inversions
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_spearman_and_grave_inversion_fail_loud():
    rows = _healthy_rows()
    # invert the two extremes: holdwave (LB 1228.8) scored at the floor and
    # pgs_hold (LB 1057.6) near the top — a >75-LB-gap inversion
    rows[0]["selector_score"] = -0.5
    rows[4]["selector_score"] = 0.55
    checks, rho, inversions = calibration_checks(rows)
    by_name = {c["name"]: c for c in checks}
    assert not by_name["spearman_ge_min"]["passed"] or not by_name["no_grave_inversion"]["passed"]
    assert inversions, "a 171-point LB gap ordered backwards must be flagged"


def test_allscripts_passing_gates_invalidates_calibration():
    rows = _healthy_rows()
    rows[-1]["hard_gates_passed"] = True
    rows[-1]["verdict"] = "PASS_LOCAL"
    checks, _, _ = calibration_checks(rows)
    by_name = {c["name"]: c for c in checks}
    assert not by_name["allscripts_rejected"]["passed"]


def test_holdwave_tied_with_top_is_acceptable():
    rows = _healthy_rows()
    # oep edges holdwave on score, but the records are statistically tied
    rows[1]["selector_score"] = 0.61
    rows[1]["fixed_record"] = {"wins": 68, "losses": 32}
    checks, _, _ = calibration_checks(rows)
    by_name = {c["name"]: c for c in checks}
    assert by_name["holdwave_top_or_tied"]["passed"]
    assert by_name["holdwave_top_or_tied"]["details"]["p_holdwave_above_top"] >= 0.30


def test_pgs_hold_above_producer_is_the_historical_false_positive():
    rows = _healthy_rows()
    rows[4]["selector_score"] = 0.50  # pgs_hold above producer's 0.40
    checks, _, _ = calibration_checks(rows)
    by_name = {c["name"]: c for c in checks}
    assert not by_name["pgs_hold_not_above_producer"]["passed"]


def _payload(names, seed_base, seeds):
    games = []
    for seed in range(seed_base, seed_base + seeds):
        for order in (list(names), list(reversed(names))):
            games.append({
                "seed": seed, "seats": order, "final_ships": [10.0, 5.0],
                "winner": order[0], "winner_seat": 0, "tie": False,
                "died_at": [None, None], "agent_status": ["DONE", "DONE"],
                "faults": {},
            })
    return {"mode": "2p", "agents": list(names), "seed_base": seed_base,
            "seed_count": seeds, "steps": 500, "games": games}


def test_reuse_matches_only_identical_task_signatures(tmp_path):
    names = ("pgs_hold", "producer")
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_payload(names, 100, 2)))
    prior = [
        {"mode": "2p", "names": list(names), "seeds": 2, "seed_base": 100,
         "steps": 500, "out": str(good), "returncode": 0, "candidate": "pgs_hold",
         "label": "x"},
        # same matchup, DIFFERENT seed base -> must not be adopted
        {"mode": "2p", "names": list(names), "seeds": 2, "seed_base": 999,
         "steps": 500, "out": str(good), "returncode": 0, "candidate": "pgs_hold",
         "label": "y"},
    ]
    task = MatchTask(label="pgs_hold__2p__producer", mode="2p", candidate="pgs_hold",
                     names=names, seeds=2, seed_base=100, steps=500,
                     out=tmp_path / "new.json")
    other = MatchTask(label="pgs_hold__2p__oep", mode="2p", candidate="pgs_hold",
                      names=("pgs_hold", "oep"), seeds=2, seed_base=100, steps=500,
                      out=tmp_path / "other.json")

    reused, remaining = match_reusable([task, other], prior)
    assert [r["label"] for r in reused] == ["pgs_hold__2p__producer"]
    assert reused[0]["reused"] is True
    assert [t.label for t in remaining] == ["pgs_hold__2p__oep"]


def test_reuse_rejects_corrupt_or_partial_payloads(tmp_path):
    names = ("pgs_hold", "producer")
    partial = tmp_path / "partial.json"
    payload = _payload(names, 100, 2)
    payload["games"] = payload["games"][:1]  # stale/incomplete
    partial.write_text(json.dumps(payload))
    prior = [{"mode": "2p", "names": list(names), "seeds": 2, "seed_base": 100,
              "steps": 500, "out": str(partial), "returncode": 0,
              "candidate": "pgs_hold", "label": "x"}]
    task = MatchTask(label="t", mode="2p", candidate="pgs_hold", names=names,
                     seeds=2, seed_base=100, steps=500, out=tmp_path / "new.json")
    reused, remaining = match_reusable([task], prior)
    assert not reused and remaining == [task]


def test_panel_invariant_scoring_check(tmp_path):
    names = ("pgs_hold", "producer")
    out = tmp_path / "g.json"
    out.write_text(json.dumps(_payload(names, 100, 2)))
    task_results = [
        {"mode": "2p", "names": list(names), "seeds": 2, "seed_base": 100,
         "steps": 500, "out": str(out), "returncode": 0, "candidate": "pgs_hold",
         "label": "a", "role": "fixed_2p"},
        # another candidate's rows must not leak into pgs_hold's score
        {"mode": "2p", "names": ["pgs_wave_s100", "producer"], "seeds": 2,
         "seed_base": 100, "steps": 500, "out": str(out), "returncode": 0,
         "candidate": "pgs_wave_s100", "label": "b", "role": "fixed_2p"},
    ]
    kwargs = dict(incumbent="pgs_holdwave", min_decisive_2p=1,
                  min_producer_winrate=0.5, min_incumbent_winrate=0.5,
                  min_floor_winrate=0.6, max_annihilation_rate_4p=0.3, weight_2p=0.46)
    check = check_scoring_isolation("pgs_hold", task_results, kwargs)
    assert check["passed"], check["details"]
