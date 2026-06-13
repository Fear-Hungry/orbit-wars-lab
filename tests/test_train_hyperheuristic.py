from __future__ import annotations

from scripts.train_hyperheuristic import (
    PRIMITIVES,
    evaluate,
    holdout_verdict,
    learn_per_bucket,
    regret_table,
    selector_pick,
    split_seeds,
)

P, T, H, R = PRIMITIVES


def _rec(seed, bucket, **m):
    margins = {f: 0.0 for f in PRIMITIVES}
    margins["oep"] = 0.0
    margins.update(m)
    return {"seed": seed, "bucket": bucket, "margins": margins}


def test_split_seeds_disjoint_halves():
    assert split_seeds([0, 1, 2, 3]) == ([0, 1], [2, 3])
    assert split_seeds([0, 1, 2, 3, 4]) == ([0, 1], [2, 3, 4])


def test_learn_per_bucket_picks_best_mean():
    train = [
        _rec(0, "expand", **{P: 0.2, T: -0.1, H: -0.3, R: 0.0}),
        _rec(0, "ahead", **{P: -0.2, T: 0.0, H: 0.3, R: -0.1}),
    ]
    learned = learn_per_bucket(train, PRIMITIVES)
    assert learned["per_bucket"]["expand"] == P
    assert learned["per_bucket"]["ahead"] == H


def test_selector_pick_uses_fallback_for_unknown_bucket():
    learned = {"per_bucket": {"expand": P}, "fallback": R}
    assert selector_pick("expand", learned) == P
    assert selector_pick("never_seen", learned) == R


def test_evaluate_oracle_ge_selected_and_regret_nonnegative():
    learned = {"per_bucket": {"expand": P}, "fallback": P}
    rec = _rec(2, "expand", **{P: 0.1, T: 0.4, H: -0.2, R: -0.2})
    rec["margins"]["oep"] = 0.05
    holdout = [rec]
    m = evaluate(holdout, learned, PRIMITIVES)
    # selector picks P (0.1); oracle picks T (0.4).
    assert abs(m["selector_choice_predicted"] - 0.1) < 1e-9
    assert abs(m["oracle_choice_known_outcome"] - 0.4) < 1e-9
    assert m["regret"] >= 0.0
    assert abs(m["selected_vs_oep_delta"] - (0.1 - 0.05)) < 1e-9


def test_regret_table_pick_rates():
    learned = {"per_bucket": {"expand": P}, "fallback": P}
    holdout = [
        _rec(2, "expand", **{P: 0.3, T: 0.1, H: -0.1, R: -0.1}),  # oracle=P, sel=P
        _rec(3, "expand", **{P: -0.1, T: 0.2, H: -0.1, R: -0.1}),  # oracle=T, sel=P
    ]
    table = regret_table(holdout, learned, PRIMITIVES)
    assert table[P]["selected_pick_rate"] == 1.0  # selector always picks P here
    assert table[P]["oracle_pick_rate"] == 0.5  # P is oracle pick in 1/2
    assert table[T]["oracle_pick_rate"] == 0.5


def test_holdout_verdict_bands():
    assert holdout_verdict(-0.10) == "SELECTOR_DOES_NOT_CAPTURE"
    assert holdout_verdict(-0.045) == "SELECTOR_DOES_NOT_CAPTURE"
    assert holdout_verdict(-0.02) == "SELECTOR_HAS_SIGNAL"
    assert holdout_verdict(0.03) == "SELECTOR_CROSSES_FLOOR"
