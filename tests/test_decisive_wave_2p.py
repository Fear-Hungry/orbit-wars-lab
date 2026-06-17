"""G3.2 Phase 2 — decisive-wave concentration for even_attrition_2p.

Pure-tensor unit tests of PGSRuntime._decisive_wave_filter (the focus-fire core
selector). No engine: we hand-build owner/ships/prod + a LaunchEntries table and
assert the filter (a) withholds NON-core enemy attacks, (b) holds the core wave
until it is decisive (or ages out), (c) never touches defence/expansion, and
(d) is inert outside the even-share regime. The off-by-default invariant (the
live pgs_holdwave is unchanged) is asserted on the config.
"""
from __future__ import annotations

import torch

from bots.pgs.planner import PGSConfig, PGSRuntime
from orbit_lite.movement_step import LaunchEntries

ME = 0


def _entries(rows):
    """rows: list of (source, target, ships, valid)."""
    src = torch.tensor([r[0] for r in rows], dtype=torch.long)
    tgt = torch.tensor([r[1] for r in rows], dtype=torch.long)
    shp = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    val = torch.tensor([r[3] for r in rows], dtype=torch.bool)
    z = torch.zeros(len(rows), dtype=torch.float32)
    return LaunchEntries(source_slots=src, target_slots=tgt, ships=shp,
                         angle=z.clone(), eta=z.clone() + 1.0, valid=val)


def _runtime(**cfg):
    base = dict(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                decisive_wave_2p=True, decisive_wave_start_step=200,
                decisive_wave_even_band=0.30, decisive_wave_min=100.0,
                decisive_wave_max_delay=20)
    base.update(cfg)
    r = PGSRuntime(PGSConfig(**base))
    r._decisive_pending = {}
    return r


# owner0: planet0=me, planet1=enemy CORE (prod 5), planet2=enemy weak (prod 2),
# planet3=neutral. Even ship share (me 50 vs enemy 50).
OWNER0 = torch.tensor([ME, 1, 1, -1], dtype=torch.long)
SHIPS = torch.tensor([50.0, 30.0, 20.0, 0.0])
PROD = torch.tensor([4.0, 5.0, 2.0, 3.0])


def _kept_targets(out: LaunchEntries) -> set[int]:
    return {int(t) for t, v in zip(out.target_slots, out.valid) if bool(v)}


def test_withholds_noncore_and_holds_subdecisive_core():
    r = _runtime()
    base = _entries([
        (0, 1, 30.0, True),   # core attack, sub-decisive (<100) -> HOLD
        (0, 2, 20.0, True),   # weak-enemy attack -> WITHHELD (concentrate)
        (0, 3, 10.0, True),   # neutral expansion -> KEPT
    ])
    out = r._decisive_wave_filter(base, OWNER0, SHIPS, PROD, ME, step_now=200)
    kept = _kept_targets(out)
    assert kept == {3}, kept                      # only expansion survives
    assert r._decisive_pending == {1: 200}        # core recorded as held


def test_releases_decisive_core_wave():
    r = _runtime()
    base = _entries([
        (0, 1, 120.0, True),  # core attack, decisive (>=100) -> FIRE
        (0, 2, 20.0, True),   # weak-enemy attack -> WITHHELD
        (0, 3, 10.0, True),   # expansion -> KEPT
    ])
    out = r._decisive_wave_filter(base, OWNER0, SHIPS, PROD, ME, step_now=200)
    assert _kept_targets(out) == {1, 3}
    assert r._decisive_pending == {}              # fired, nothing pending


def test_age_out_forces_release():
    r = _runtime()
    r._decisive_pending = {1: 175}                # held since step 175
    base = _entries([(0, 1, 30.0, True), (0, 2, 20.0, True)])
    out = r._decisive_wave_filter(base, OWNER0, SHIPS, PROD, ME, step_now=200)  # age 25 >= 20
    assert 1 in _kept_targets(out)                # core released despite small


def test_defence_never_withheld():
    r = _runtime()
    # add an attack-on-own-planet recapture? defence = target owned by me.
    base = _entries([
        (1, 0, 40.0, True),   # defence/reinforce of my planet 0 -> KEPT
        (0, 2, 20.0, True),   # weak-enemy attack -> WITHHELD
        (0, 1, 30.0, True),   # core attack sub-decisive -> HOLD
    ])
    out = r._decisive_wave_filter(base, OWNER0, SHIPS, PROD, ME, step_now=200)
    assert 0 in _kept_targets(out)                # defence survives concentration


def test_inert_when_position_lopsided():
    r = _runtime()
    lop = torch.tensor([90.0, 5.0, 5.0, 0.0])     # share 0.9 -> outside even band
    base = _entries([(0, 1, 30.0, True), (0, 2, 20.0, True)])
    out = r._decisive_wave_filter(base, OWNER0, lop, PROD, ME, step_now=200)
    assert _kept_targets(out) == {1, 2}           # untouched
    assert r._decisive_pending == {}


def test_off_by_default_invariant():
    # The live submission config must not enable the decisive wave.
    assert PGSConfig().decisive_wave_2p is False
    assert PGSConfig(scripts="hold", wave_min_ships=60.0,
                     wave_start_step=150).decisive_wave_2p is False
