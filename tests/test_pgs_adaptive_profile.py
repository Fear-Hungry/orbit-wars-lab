"""Unit coverage for the PGS v3 online opponent profiler."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class _FakeMovement:
    tracked_fleet_ids: torch.Tensor
    owner: torch.Tensor
    target_slot: torch.Tensor
    ships: torch.Tensor
    eta: torch.Tensor


def _movement(
    *,
    fleet_ids: list[int],
    owners: list[int],
    targets: list[int],
    ships: list[float],
    etas: list[float] | None = None,
) -> _FakeMovement:
    if etas is None:
        etas = [3.0] * len(fleet_ids)
    return _FakeMovement(
        tracked_fleet_ids=torch.tensor(fleet_ids, dtype=torch.long),
        owner=torch.tensor(owners, dtype=torch.long),
        target_slot=torch.tensor(targets, dtype=torch.long),
        ships=torch.tensor(ships, dtype=torch.float32),
        eta=torch.tensor(etas, dtype=torch.float32),
    )


def _stat_float(stat: Any, *names: str) -> float:
    for name in names:
        if hasattr(stat, name):
            return float(getattr(stat, name))
    raise AssertionError(f"{type(stat).__name__} must expose one of: {', '.join(names)}")


def _profile_total(stat: Any) -> float:
    for name in ("total", "total_ships", "profile_total", "observed_ships", "ships_seen"):
        if hasattr(stat, name):
            return float(getattr(stat, name))
    return (
        _stat_float(stat, "to_neutral", "to_neutral_ships", "neutral_ships", "ships_to_neutral")
        + _stat_float(stat, "to_me", "to_me_ships", "pressure_ships", "ships_to_me")
        + _stat_float(stat, "to_other", "to_other_ships", "other_ships", "ships_to_enemy")
    )


def _label_text(label: Any) -> str:
    if isinstance(label, tuple) and label:
        return _label_text(label[0])
    if isinstance(label, str):
        return label.lower()
    if hasattr(label, "name"):
        return str(label.name).lower()
    return str(label).lower()


def test_pgs_config_adaptive_mode_initializes_online_profile_state() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    cfg = PGSConfig(adaptive_mode=True)
    assert cfg.adaptive_mode is True
    assert PGSConfig().adaptive_mode is False

    runtime = PGSRuntime(cfg)

    assert runtime._seen_fleet_ids == set()
    assert runtime._opp_profiles == {}


def test_update_opponent_profiles_accumulates_by_target_owner_and_dedupes() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(adaptive_mode=True))
    owner0 = torch.tensor([0, 1, -1, 2], dtype=torch.long)

    first = _movement(
        fleet_ids=[100, 101, 102, 103],
        owners=[1, 1, 1, 0],
        targets=[2, 0, 3, 2],
        ships=[12.0, 8.0, 7.0, 99.0],
        etas=[3.0, 4.0, 5.0, 2.0],
    )
    runtime._update_opponent_profiles(movement=first, owner0=owner0, me=0, step_now=10)

    stat = runtime._opp_profiles[1]
    assert is_dataclass(stat)
    assert _stat_float(
        stat, "to_neutral", "to_neutral_ships", "neutral_ships", "ships_to_neutral"
    ) == pytest.approx(12.0)
    assert _stat_float(stat, "to_me", "to_me_ships", "pressure_ships", "ships_to_me") == (
        pytest.approx(8.0)
    )
    assert _stat_float(stat, "to_other", "to_other_ships", "other_ships", "ships_to_enemy") == (
        pytest.approx(7.0)
    )
    assert _profile_total(stat) == pytest.approx(27.0)

    runtime._update_opponent_profiles(movement=first, owner0=owner0, me=0, step_now=11)
    assert _profile_total(runtime._opp_profiles[1]) == pytest.approx(27.0)

    second = _movement(
        fleet_ids=[100, 101, 102, 104],
        owners=[1, 1, 1, 1],
        targets=[2, 0, 3, 0],
        ships=[12.0, 8.0, 7.0, 5.0],
        etas=[2.0, 3.0, 4.0, 5.0],
    )
    runtime._update_opponent_profiles(movement=second, owner0=owner0, me=0, step_now=12)

    stat = runtime._opp_profiles[1]
    assert _stat_float(stat, "to_me", "to_me_ships", "pressure_ships", "ships_to_me") == (
        pytest.approx(13.0)
    )
    assert _profile_total(stat) == pytest.approx(32.0)

    seen_ids = {
        int(item[-1] if isinstance(item, tuple) else item) for item in runtime._seen_fleet_ids
    }
    assert {100, 101, 102, 104}.issubset(seen_ids)


def test_classify_opponent_distinguishes_expanders_from_pressure_players() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    owner0 = torch.tensor([0, -1, 1], dtype=torch.long)

    expander = PGSRuntime(PGSConfig(adaptive_mode=True))
    expander._update_opponent_profiles(
        movement=_movement(
            fleet_ids=[200, 201, 202, 203, 204, 205, 206],
            owners=[1, 1, 1, 1, 1, 1, 1],
            targets=[1, 1, 1, 1, 1, 1, 1],
            ships=[22.0, 24.0, 20.0, 26.0, 28.0, 21.0, 25.0],
        ),
        owner0=owner0,
        me=0,
        step_now=18,
    )
    expander_label = _label_text(
        expander._classify_opponent(expander._opp_profiles[1], step_now=18)
    )
    assert any(term in expander_label for term in ("expand", "neutral", "econ"))

    pressure = PGSRuntime(PGSConfig(adaptive_mode=True))
    pressure._update_opponent_profiles(
        movement=_movement(
            fleet_ids=[300, 301, 302, 303, 304, 305, 306],
            owners=[1, 1, 1, 1, 1, 1, 1],
            targets=[0, 0, 0, 0, 0, 0, 0],
            ships=[9.0, 11.0, 13.0, 10.0, 12.0, 14.0, 15.0],
        ),
        owner0=owner0,
        me=0,
        step_now=18,
    )
    pressure_label = _label_text(
        pressure._classify_opponent(pressure._opp_profiles[1], step_now=18)
    )
    assert any(term in pressure_label for term in ("press", "aggr", "rush", "attack"))
