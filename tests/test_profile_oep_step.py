from __future__ import annotations

from scripts.profile_oep_step import _merge_selection, _percentile, _summarize_selection


def test_percentile_interpolates_sorted_values() -> None:
    assert _percentile([10.0, 20.0, 30.0], 0.0) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 1.0) == 30.0
    assert _percentile([10.0, 20.0], 0.25) == 12.5


def test_percentile_handles_empty_and_singleton_values() -> None:
    assert _percentile([], 0.95) == 0.0
    assert _percentile([42.0], 0.95) == 42.0


def test_selection_summary_aggregates_weighted_deltas() -> None:
    target: dict[str, float] = {}

    _merge_selection(
        target,
        {
            "decisions": 2.0,
            "oep_choices": 1.0,
            "producer_choices": 1.0,
            "oep_nonempty_rate": 1.0,
            "producer_nonempty_rate": 0.5,
            "mean_fitness_delta_oep_minus_producer": 0.25,
            "min_fitness_delta_oep_minus_producer": -1.0,
            "max_fitness_delta_oep_minus_producer": 1.5,
            "mean_ordinal_win_rate_oep_vs_producer": 0.8,
            "min_ordinal_win_rate_oep_vs_producer": 0.6,
            "max_ordinal_win_rate_oep_vs_producer": 1.0,
            "ordinal_wins_oep_vs_producer": 8.0,
            "ordinal_variants_oep_vs_producer": 10.0,
            "plan_memory_variant_calls": 4.0,
            "plan_memory_variant_choices": 1.0,
            "plan_memory_variant_candidates": 8.0,
            "beam_first_calls": 2.0,
            "beam_first_choices": 1.0,
            "beam_first_candidates": 6.0,
            "beam_pair_calls": 2.0,
            "beam_pair_choices": 1.0,
            "beam_pair_candidates": 10.0,
            "early_decisions": 2.0,
            "early_oep_choice_rate": 0.5,
            "early_producer_choice_rate": 0.5,
            "early_mean_fitness_delta_oep_minus_producer": 0.25,
        },
    )
    _merge_selection(
        target,
        {
            "decisions": 1.0,
            "oep_choices": 0.0,
            "producer_choices": 1.0,
            "oep_nonempty_rate": 0.0,
            "producer_nonempty_rate": 1.0,
            "mean_fitness_delta_oep_minus_producer": -1.0,
            "min_fitness_delta_oep_minus_producer": -1.0,
            "max_fitness_delta_oep_minus_producer": -1.0,
            "mean_ordinal_win_rate_oep_vs_producer": 0.2,
            "min_ordinal_win_rate_oep_vs_producer": 0.2,
            "max_ordinal_win_rate_oep_vs_producer": 0.2,
            "ordinal_wins_oep_vs_producer": 1.0,
            "ordinal_variants_oep_vs_producer": 5.0,
            "plan_memory_variant_calls": 2.0,
            "plan_memory_variant_choices": 1.0,
            "plan_memory_variant_candidates": 4.0,
            "beam_first_calls": 1.0,
            "beam_first_choices": 0.0,
            "beam_first_candidates": 3.0,
            "beam_pair_calls": 1.0,
            "beam_pair_choices": 1.0,
            "beam_pair_candidates": 5.0,
            "early_decisions": 1.0,
            "early_oep_choice_rate": 0.0,
            "early_producer_choice_rate": 1.0,
            "early_mean_fitness_delta_oep_minus_producer": -1.0,
        },
    )

    summary = _summarize_selection(target)

    assert summary["decisions"] == 3.0
    assert summary["oep_choice_rate"] == 1.0 / 3.0
    assert summary["producer_choice_rate"] == 2.0 / 3.0
    assert summary["oep_nonempty_rate"] == 2.0 / 3.0
    assert summary["producer_nonempty_rate"] == 2.0 / 3.0
    assert summary["mean_fitness_delta_oep_minus_producer"] == -0.5 / 3.0
    assert summary["min_fitness_delta_oep_minus_producer"] == -1.0
    assert summary["max_fitness_delta_oep_minus_producer"] == 1.5
    assert summary["mean_ordinal_win_rate_oep_vs_producer"] == 1.8 / 3.0
    assert summary["min_ordinal_win_rate_oep_vs_producer"] == 0.2
    assert summary["max_ordinal_win_rate_oep_vs_producer"] == 1.0
    assert summary["ordinal_wins_oep_vs_producer"] == 9.0
    assert summary["ordinal_variants_oep_vs_producer"] == 15.0
    assert summary["plan_memory_variant_calls"] == 6.0
    assert summary["plan_memory_variant_choices"] == 2.0
    assert summary["plan_memory_variant_candidates"] == 12.0
    assert summary["plan_memory_variant_choice_rate"] == 2.0 / 6.0
    assert summary["beam_first_calls"] == 3.0
    assert summary["beam_first_choices"] == 1.0
    assert summary["beam_first_candidates"] == 9.0
    assert summary["beam_first_choice_rate"] == 1.0 / 3.0
    assert summary["beam_pair_calls"] == 3.0
    assert summary["beam_pair_choices"] == 2.0
    assert summary["beam_pair_candidates"] == 15.0
    assert summary["beam_pair_choice_rate"] == 2.0 / 3.0
    assert summary["early_decisions"] == 3.0
    assert summary["early_oep_choice_rate"] == 1.0 / 3.0
    assert summary["early_producer_choice_rate"] == 2.0 / 3.0
    assert summary["early_mean_fitness_delta_oep_minus_producer"] == -0.5 / 3.0
