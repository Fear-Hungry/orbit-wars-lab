"""Regression: league_match must MEASURE per-game faults and never fake a winner.

Locked-down bugs (2026-06-10):
- BUG A: scripts/league_match.py counted crashes only in an aggregate dict —
  the per-game JSON had no fault fields, so a probe agent that always crashed
  produced a "valid-looking" game; invalid move entries (len<3 / non-finite)
  were dropped uncounted; decisions slower than Kaggle's actTimeout=1s left no
  trace. Now each game dict carries "faults": {name: {crashes, timeouts,
  invalid_moves}}.
- BUG C (2026-06-11): the key used to be OMITTED when clean, which made a
  clean post-fix game indistinguishable from a PRE-instrumentation game (all
  5k old games read as "clean" in the report). Now the key is ALWAYS written
  ({} when clean); a missing key means UNAUDITED.
- BUG D (2026-06-11): faults were MEASURED but Kaggle semantics were not
  APPLIED — a crashing/over-budget agent kept acting and could WIN locally
  while on Kaggle it would be dead (status ERROR/TIMEOUT, reward None).
  Now: exception => ERROR (dead rest of game); act overrun beyond the
  remainingOverageTime bank (12s, kaggle schemas.json) => DeadlineExceeded =>
  TIMEOUT (dead, that turn's moves lost); errored seats are excluded from the
  winner argmax; per-seat "agent_status" is recorded. Invalid move entries
  stay penalty-free (official process_moves skips them) but match the exact
  len == 3 rule.
- BUG B: ``winner = argmax(tot)`` turned a tied final total into a fake seat-0
  win (26 inconsistencies in old artifacts). Now a shared maximum is recorded
  as winner_seat=-1, winner=None, tie=true.
"""
from __future__ import annotations

import time


def _fake_make(name):
    if name == "crasher":
        def crasher(obs):
            raise RuntimeError("boom")
        return crasher
    if name == "badmove":
        def badmove(obs):
            # too-short + non-finite + too-long (official drops len != 3
            # exactly) = 3 invalid per step
            return [[1.0], [float("nan"), 0.0, 1.0], [1.0, 0.0, 1.0, 9.9]]
        return badmove
    if name == "ghostmover":
        def ghostmover(obs):
            # well-formed entry, but planet 999999 is never ours: semantically
            # illegal per rules.moves_are_legal — engine ignores it silently
            return [[999999.0, 0.0, 10.0]]
        return ghostmover
    if name == "overbudget":
        def overbudget(obs):
            own = [p for p in obs.get("planets", []) if int(p[1]) == int(obs.get("player", 0))]
            if not own:
                return []
            src = own[0]
            ships = int(src[5])
            if ships < 2:
                return []
            # Each entry is individually legal, but the turn spends too much
            # from the same source. This used to pass because league_match
            # called moves_are_legal one move at a time.
            return [[int(src[0]), 0.0, ships], [int(src[0]), 0.1, 1]]
        return overbudget
    if name == "sleeper":
        def sleeper(obs):
            time.sleep(1.05)  # just over Kaggle actTimeout=1s
            return []
        return sleeper

    def passer(obs):
        return []
    return passer


def test_crashing_agent_is_dead_and_cannot_win(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["crasher", "passer"], [123], 5, {}, {})
    assert len(games) == 1
    g = games[0]
    # Kaggle: first raise => status ERROR, agent never acts again
    assert g["faults"]["crasher"]["crashes"] == 1
    assert "passer" not in g["faults"], "clean agents must not appear in faults"
    assert g["agent_status"] == ["ERROR", "DONE"]
    # errored seat is excluded from the winner argmax (reward=None on Kaggle)
    assert g["winner"] == "passer" and g["winner_seat"] == 1
    assert g["tie"] is False


def test_invalid_moves_are_counted_not_silently_dropped(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["badmove", "passer"], [123], 3, {}, {})
    assert games[0]["faults"]["badmove"]["invalid_moves"] == 9  # 3 bad entries x 3 steps
    # invalid entries carry NO Kaggle penalty (official process_moves skips
    # them): the agent stays alive and eligible
    assert games[0]["agent_status"] == ["DONE", "DONE"]


def test_semantically_illegal_move_counted_as_invalid(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["ghostmover", "passer"], [123], 3, {}, {})
    assert games[0]["faults"]["ghostmover"]["invalid_moves"] == 3  # 1 illegal entry x 3 steps


def test_aggregate_overbudget_move_counted_as_invalid(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["overbudget", "passer"], [123], 3, {}, {})
    assert games[0]["faults"]["overbudget"]["invalid_moves"] >= 1


def test_slow_decision_counted_as_timeout_but_bank_absorbs(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["sleeper", "passer"], [123], 1, {}, {})
    assert games[0]["faults"]["sleeper"]["timeouts"] == 1
    # 0.05s overrun is drawn from the 12s overage bank — agent stays alive
    # (Kaggle only kills when the overrun exceeds remainingOverageTime)
    assert games[0]["agent_status"] == ["DONE", "DONE"]


def test_overage_exhaustion_kills_with_timeout_status(monkeypatch):
    import scripts.league_match as lm

    calls = []

    def fake_make(name):
        if name == "sleeper":
            def sleeper(obs):
                calls.append(1)
                time.sleep(1.05)
                return []
            return sleeper
        return lambda obs: []

    monkeypatch.setattr(lm, "make", fake_make)
    # bank smaller than the 0.05s overrun: the FIRST slow act already exceeds
    # it -> DeadlineExceeded -> status TIMEOUT, dead for the rest of the game
    monkeypatch.setattr(lm, "OVERAGE_BANK_S", 0.01)
    games = lm.play_batch(["sleeper", "passer"], [123], 3, {}, {})
    g = games[0]
    assert len(calls) == 1, "TIMEOUT agent must not act again"
    assert g["faults"]["sleeper"]["timeouts"] == 1
    assert g["agent_status"] == ["TIMEOUT", "DONE"]
    assert g["winner"] == "passer", "errored seat cannot win"


def test_tied_totals_recorded_as_tie_not_seat0_win(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    # force a tie regardless of map: both seats end with the same total
    monkeypatch.setattr(lm, "_totals", lambda state, num_players: [10.0] * num_players)
    games = lm.play_batch(["passer", "passer"], [123], 2, {}, {})
    g = games[0]
    assert g["tie"] is True
    assert g["winner_seat"] == -1
    assert g["winner"] is None


def test_normal_game_carries_empty_faults_key_and_tie_flag(monkeypatch):
    import scripts.league_match as lm

    monkeypatch.setattr(lm, "make", _fake_make)
    games = lm.play_batch(["passer", "passer"], [123], 3, {}, {})
    g = games[0]
    assert g["faults"] == {}, (
        "clean game must carry an EMPTY faults key — a missing key means "
        "pre-instrumentation/unaudited, not clean")
    assert g["tie"] in (True, False)
    if not g["tie"]:
        assert g["winner_seat"] >= 0 or max(g["final_ships"]) <= 0
