"""Unit coverage for adaptive mission assembly primitives."""

from __future__ import annotations

import torch


def _entries(source: list[int], target: list[int], ships: list[float]):
    from orbit_lite.movement_step import LaunchEntries

    return LaunchEntries(
        source_slots=torch.tensor(source, dtype=torch.long),
        target_slots=torch.tensor(target, dtype=torch.long),
        ships=torch.tensor(ships, dtype=torch.float32),
        angle=torch.zeros(len(source), dtype=torch.float32),
        eta=torch.ones(len(source), dtype=torch.float32),
        valid=torch.ones(len(source), dtype=torch.bool),
    )


def _mission(
    name: str,
    *,
    source: int,
    target: int,
    ships: float,
    priority: float,
    kind: str,
):
    from bots.pgs.planner import MissionCandidate

    return MissionCandidate(
        name=name,
        entries=_entries([source], [target], [ships]),
        replace_sources=frozenset({source}),
        exclusive_targets=frozenset({target}),
        priority=priority,
        kind=kind,
    )


def test_effective_policy_exposes_evac_and_recapture_flags() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(
        PGSConfig(
            mission_mode=True,
            enabled_missions=" rescue,EVAC,recapture ",
        )
    )

    policy = runtime._effective_policy(me=0, opp_ids=[1], player_count=2, step_now=20)

    assert policy.rescue_enabled is True
    assert policy.evac_enabled is True
    assert policy.recapture_enabled is True


def test_assemble_missions_replaces_conflicting_sources_and_targets() -> None:
    from bots.pgs.planner import MissionCandidate, PGSRuntime

    runtime = PGSRuntime()
    base = _entries([0, 1, 2], [5, 6, 7], [10.0, 11.0, 12.0])
    mission = MissionCandidate(
        name="rescue:6",
        entries=_entries([1], [6], [4.0]),
        replace_sources=frozenset({1}),
        exclusive_targets=frozenset({6}),
        priority=1000.0,
        kind="rescue",
    )

    assembled = runtime._assemble_missions(base, [mission])

    assert assembled.source_slots.tolist() == [0, 2, 1]
    assert assembled.target_slots.tolist() == [5, 7, 6]
    assert assembled.ships.tolist() == [10.0, 12.0, 4.0]


def test_select_missions_greedy_skips_replace_source_conflicts() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(max_selected_missions=3))
    base = _entries([0, 1, 2], [5, 6, 7], [10.0, 10.0, 10.0])
    evac = _mission(
        "evac:1",
        source=1,
        target=20,
        ships=25.0,
        priority=100.0,
        kind="evac",
    )
    recapture_conflict = _mission(
        "recapture:1",
        source=1,
        target=21,
        ships=200.0,
        priority=90.0,
        kind="recapture",
    )
    recapture_clear = _mission(
        "recapture:2",
        source=2,
        target=22,
        ships=25.0,
        priority=80.0,
        kind="recapture",
    )

    selected = runtime._select_missions_greedy(
        base=base,
        candidates=[evac, recapture_conflict, recapture_clear],
        value_fn=lambda entries: float(entries.ships.sum().item()),
        min_advantage=0.1,
    )
    assembled = runtime._assemble_missions(base, selected)

    assert [mission.name for mission in selected] == ["evac:1", "recapture:2"]
    assert assembled.source_slots.tolist().count(1) == 1
    assert assembled.source_slots.tolist().count(2) == 1
    assert assembled.target_slots.tolist() == [5, 20, 22]
