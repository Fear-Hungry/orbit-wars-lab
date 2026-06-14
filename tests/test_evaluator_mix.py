"""Field-mix weighting of the research-loop evaluator + the loop fitness scalar.

The "mix" gate (0.54·4p + 0.46·2p) is the only instrument that can match a
mixed-regime leaderboard — pure-4p collapses the hold-family to a tied floor and
pure-2p inverts vs the LB (docstrings in evaluator.py / eval2p.py). The weighting
arithmetic is the whole point, so we test it directly with the simulator stubbed
out (deterministic, no Rust, no 500-step runs).
"""
from __future__ import annotations

import scripts.research_loop.evaluator as ev
from scripts.research_loop.evaluator import FIELD_2P, FIELD_4P, evaluate
from scripts.research_loop.genome import fitness


def _stub(death, margin, planets=2.0):
    def _run(opp, pgs_config, seeds, steps, enable_comets, opponent="producer"):
        return {"name": opp, "death_rate": death, "mean_margin": margin,
                "mean_final_planets": planets, "timeouts": 0, "elapsed_s": 0.0}
    return _run


def test_field_weights_sum_to_one():
    assert abs(FIELD_4P + FIELD_2P - 1.0) < 1e-9


def test_mix_is_field_weighted_blend(monkeypatch):
    # 4p reports death=0.6, 2p reports death=0.2 -> mix = .54*.6 + .46*.2
    monkeypatch.setattr(ev, "run_config", _stub(0.6, -0.1))
    monkeypatch.setattr(ev, "run_config_2p", _stub(0.2, 0.5))
    r = evaluate({"scripts": "hold"}, seeds=2, steps=10,
                 pool=("producer", "rush"), seats="mix", verbose=False)
    assert abs(r["death_rate"] - (FIELD_4P * 0.6 + FIELD_2P * 0.2)) < 1e-9
    assert abs(r["mean_margin"] - (FIELD_4P * -0.1 + FIELD_2P * 0.5)) < 1e-9
    assert r["seats"] == "mix"
    # components preserved so a caller can see WHICH regime drove the score.
    assert set(r["components"]) == {"4p", "2p"}
    assert r["components"]["4p"]["death_rate"] == 0.6
    assert r["components"]["2p"]["death_rate"] == 0.2


def test_mix_per_opponent_also_blended(monkeypatch):
    monkeypatch.setattr(ev, "run_config", _stub(0.6, -0.1))
    monkeypatch.setattr(ev, "run_config_2p", _stub(0.2, 0.5))
    r = evaluate({"scripts": "hold"}, seeds=2, steps=10,
                 pool=("producer",), seats="mix", verbose=False)
    blended = r["per_opponent"]["producer"]["mean_margin"]
    assert abs(blended - (FIELD_4P * -0.1 + FIELD_2P * 0.5)) < 1e-9


def test_pure_4p_uses_only_run_config(monkeypatch):
    monkeypatch.setattr(ev, "run_config", _stub(0.6, -0.1))
    # run_config_2p would raise if called -> proves seats=4 never touches 2p.
    monkeypatch.setattr(ev, "run_config_2p",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("2p called")))
    r = evaluate({"scripts": "hold"}, seeds=2, steps=10,
                 pool=("producer",), seats=4, verbose=False)
    assert r["death_rate"] == 0.6


def test_fitness_weights_death_double_and_penalizes_timeouts():
    # margin - 2*death - timeout_pen
    assert abs(fitness({"mean_margin": 0.5, "death_rate": 0.1}) - (0.5 - 0.2)) < 1e-9
    assert abs(fitness({"mean_margin": 0.5, "death_rate": 0.1, "timeouts": 4})
               - (0.5 - 0.2 - 1.0)) < 1e-9


def test_fitness_prefers_survival_over_raw_margin():
    safe = fitness({"mean_margin": 0.0, "death_rate": 0.1})
    risky = fitness({"mean_margin": 0.3, "death_rate": 0.5})
    assert safe > risky
