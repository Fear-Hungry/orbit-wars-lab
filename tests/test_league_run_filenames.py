"""Regression: per-match output filenames must be unique per (round, names).

The old scheme truncated agent names to 6 chars, so distinct seat compositions
collided (pgs_hold/pgs_holdwave -> "pgs_ho"; pgs_wave_s100/pgs_wave_s50 ->
"pgs_wa") and the second write silently overwrote the first (round 92 lost a
match this way). match_filename now appends a hash of the FULL names tuple.
"""
from __future__ import annotations

from scripts import league_run

match_filename = league_run.match_filename


def test_truncation_collision_resolved() -> None:
    # confirmed real-world collision: both truncate to r0001_2p_pgs_ho_pgs_wa
    a = match_filename(1, ["pgs_hold", "pgs_wave_s100"])
    b = match_filename(1, ["pgs_holdwave", "pgs_wave_s50"])
    assert a != b


def test_pool_pairs_never_collide() -> None:
    # every distinct composition (incl. seat order and the 3-way pgs_wa group)
    # must map to a distinct filename within the same round
    names = ["producer", "pgs_hold", "pgs_holdwave", "pgs_wave_s100",
             "pgs_wave_s50", "pgs_wave_4pfloor", "brep", "brep_league3"]
    pairs = [[x, y] for x in names for y in names if x != y]
    files = [match_filename(7, p) for p in pairs]
    assert len(files) == len(set(files))


def test_deterministic_and_readable() -> None:
    names = ["pgs_holdwave", "brep", "ext_lb1050", "producer"]
    a = match_filename(92, names)
    assert a == match_filename(92, list(names))  # rerun/resume: same name
    assert a.startswith("r0092_4p_pgs_ho_brep_ext_lb_produc_")  # prefix kept
    assert a.endswith(".json")
    assert a != match_filename(93, names)  # round still disambiguates


def test_fresh_league_state_starts_from_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(league_run, "STATE", tmp_path / "state.json")
    assert league_run.load_state() == {"next_seed": 0, "round": 0}
