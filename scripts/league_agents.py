"""Agent pool for the local league (H-P4 field ruler).

Every entry is a FACTORY returning a fresh callable(official_obs_dict) -> moves,
one instance per game (cross-contamination gotcha: never share runtimes).

LB_ANCHORS holds the real leaderboard score of OUR submissions of the same
config — the league is only trusted if its ranking reproduces these (Spearman +
the hard requirement that pgs_allscripts lands clearly below producer/oep/brep).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# LEAGUE = VETO ONLY (rule since the 2026-06-10 falsification). The league
# discards obvious floors (allscripts-class) but must NOT promote between close
# configs: pgs_hold and pgs_wave_s100 both passed P(>=producer)=1.00 and landed
# 115-135 LB points BELOW producer (1057.6 / ~1036-1109). BT rank is relative
# to THIS pool (8/10 producer-lineage; ext bots too weak to discriminate the
# top — Balduzzi 2018 arXiv:1806.02643). Promotion between close configs needs
# an LB probe or a pool with style exploiters (rusher / pgs_bigwave below).
# GATE_REFERENCE is the veto floor; INCUMBENT is the live LB champion the
# report prints head-to-head against (promotion sanity, never sufficient).
GATE_REFERENCE = "producer"
INCUMBENT = "pgs_holdwave"  # LB record 1228.8 (ref=53537753)

# real Kaggle ratings of our submitted configs (refreshed 2026-06-12 via
# Kaggle CLI). NEVER trust these without a refresh: s100 moved 1036->1109->1146
# ->1136 across reads.
LB_ANCHORS = {
    "producer": 1173.1,       # ref=53366194
    "oep": 1182.7,            # ref=53433131 (resubmit 53582886 w/ new wrapper: 1161.5)
    "brep": 1156.1,           # ref=53513962
    "pgs_allscripts": 1021.5, # ref=53519882 (accidental all-scripts default)
    "pgs_holdwave": 1228.8,   # ref=53537753 (T+7h, ESTABILIZADO — nível do recorde)
    "pgs_hold": 1057.6,       # ref=53541125 (stable across refreshes T+2h..T+5h)
    "pgs_wave_s100": 1136.3,  # ref=53542864 (CLI 2026-06-12; 1036->1109->1138->1146->1136)
    # TENSION (2026-06-12): TWO holdwave-config resubmits now read ~1145
    # (53542884: 1147.5; 53582859 new wrapper: 1144.0) vs the 1228.8 original —
    # spread ~85 > the ±60 noise budget. Either the original was a lucky high
    # or the field moved. Calibration checks that assume "holdwave is top"
    # inherit this uncertainty; surface it, don't silently re-anchor.
}

# Style buckets for the submission selector. A candidate's mean score hides
# style-specific collapses (a bot that beats our lineage but folds to rush
# pressure is not submittable); the selector aggregates per bucket and treats a
# total failure in any CRITICAL bucket as disqualifying, whatever the mean says.
# four_player_survivor / counterpunch are reserved: no pool member fits yet.
AGENT_BUCKETS = {
    "producer": "own_lineage",
    "oep": "own_lineage",
    "brep": "own_lineage",
    "pgs_hold": "own_lineage",
    "pgs_holdwave": "own_lineage",
    "pgs_holdwave_half2p": "own_lineage",
    "pgs_wave_s100": "own_lineage",
    "pgs_wave_s50": "own_lineage",
    "pgs_wave_4pfloor": "own_lineage",
    "pgs_valuenet": "own_lineage",
    "pgs_allscripts": "rejected_floor",
    "rusher": "rush_pressure",
    "rush": "rush_pressure",
    "pgs_bigwave": "bigwave_hoard",
    "ext_lb1050": "external_lb_proxy",
    "ext_hellburner": "external_lb_proxy",
}
CRITICAL_BUCKETS = {"rejected_floor", "rush_pressure", "bigwave_hoard", "external_lb_proxy"}


def bucket_of(name: str) -> str | None:
    return AGENT_BUCKETS.get(name)

_EXT_N = [0]

_FORBIDDEN_SUBMISSION_STATS = {
    "fallbacks",
    "timeouts",
    "timeout_thread_blocks",
    "fallback_errors",
    "illegal_moves",
    "policy_illegal_moves",
    "invalid_actions",
}


def _submission_stats_snapshot(mod) -> dict[str, int]:
    stats = getattr(mod, "SUBMISSION_STATS", None)
    if not isinstance(stats, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in stats.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _forbidden_submission_stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    bad: dict[str, int] = {}
    for key, value in after.items():
        if key in _FORBIDDEN_SUBMISSION_STATS or key.endswith("_fallbacks"):
            delta = int(value) - int(before.get(key, 0))
            if delta > 0:
                bad[key] = delta
    return bad


def _fresh_module(path: Path, tag: str):
    _EXT_N[0] += 1
    spec = importlib.util.spec_from_file_location(f"_league_{tag}_{_EXT_N[0]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _external(path: str):
    p = ROOT / path
    return lambda: _fresh_module(p, p.parent.name.replace("-", "_")).agent


def _producer():
    from bots.producer.agent import make_agent

    return make_agent()


def _oep():
    from bots.oep.agent import make_agent

    return make_agent()


def _pgs(**cfg):
    from bots.pgs.planner import PGSConfig, PGSRuntime

    return PGSRuntime(PGSConfig(**cfg)).act


def _rusher(**kw):
    from bots.exploiters.rusher import make_agent

    return make_agent(**kw)


def _heuristic(name: str):
    from python.agents.registry import get_heuristic_policies

    policy = get_heuristic_policies()[name]

    def act(obs):
        player = int(obs.get("player", 0))
        moves = policy(obs, player)
        return list(moves) if isinstance(moves, list) else []

    return act


class _TarballIsolation:
    """Per-tarball import context. Tarballs bundle bare-named modules
    (_brep_weights, _producer_agent, _upstream, orbit_lite, ...); with the
    cache dir permanently on sys.path those names hit the GLOBAL sys.modules
    cache, so two tarballs share whichever copy imported first — including
    LAZY imports during act() (the real main.py imports _brep_weights on first
    use). Each tarball therefore keeps its bundled modules in a private
    overlay, swapped into sys.modules only while its code runs. Safe because
    league_match calls agents sequentially in one thread."""

    def __init__(self, cache: Path):
        self._cache = str(cache)
        self._owned = set()
        for entry in cache.iterdir():
            if entry.name == "main.py":
                continue
            if entry.suffix == ".py":
                self._owned.add(entry.stem)
            elif (entry / "__init__.py").exists():
                self._owned.add(entry.name)
        self._overlay: dict[str, object] = {}

    def _owns(self, key: str) -> bool:
        return key.split(".", 1)[0] in self._owned

    @contextlib.contextmanager
    def active(self):
        saved = {}
        for key in [k for k in sys.modules if self._owns(k)]:
            saved[key] = sys.modules.pop(key)
        sys.modules.update(self._overlay)
        inserted = self._cache not in sys.path
        if inserted:
            sys.path.insert(0, self._cache)
        try:
            yield
        finally:
            if inserted:
                sys.path.remove(self._cache)
            for key, mod in list(sys.modules.items()):
                mod_file = getattr(mod, "__file__", None) or ""
                if self._owns(key) or mod_file.startswith(self._cache + "/"):
                    self._overlay[key] = sys.modules.pop(key)
            sys.modules.update(saved)


def _ensure_tarball_cache(tar_path: Path, cache_name: str) -> Path:
    digest = hashlib.sha1(tar_path.read_bytes()).hexdigest()[:12]
    key = f"{cache_name}-{digest}"
    cache = ROOT / "artifacts" / "league" / "cache" / key
    complete = cache / ".complete"
    if complete.exists() and (cache / "main.py").exists():
        return cache

    cache_parent = cache.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    locks = cache_parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"{key}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if complete.exists() and (cache / "main.py").exists():
                return cache
            if cache.exists():
                shutil.rmtree(cache)
            tmp = cache_parent / f".{key}.{os.getpid()}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tar_path) as tf:
                    # filter="data" blocks path traversal / symlink escapes from
                    # a hostile tarball (and is the post-3.14 mandatory default
                    # anyway).
                    tf.extractall(tmp, filter="data")
                main_py = tmp / "main.py"
                if not main_py.exists():
                    raise ValueError(f"tarball missing main.py: {tar_path}")
                py_compile.compile(str(main_py), doraise=True)
                (tmp / ".complete").write_text("ok\n", encoding="utf-8")
                os.replace(tmp, cache)
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            return cache
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _tarball_agent(tar_path: Path, cache_name: str):
    """Load a self-contained submission tarball as a league agent (fresh main
    module and fresh bundled-module overlay per instance; bare-named deps stay
    private — see _TarballIsolation — so multiple tarballs never share
    weights/models/runtime state through sys.modules or sys.path ordering).

    The cache dir is keyed by the tarball's CONTENT hash: re-exporting a
    tarball under the same name gets a fresh extraction AND a fresh import
    overlay. The overlay is intentionally PER INSTANCE, not per tarball hash:
    PGS/OEP submissions have module-global runtimes inside bundled packages, so
    two games using the same tarball still need isolated package state. Old hash
    dirs become orphans — never reused, safe to leave. A missing tarball fails
    LOUD even when a stale cache exists (no-silent-fallback rule)."""
    cache = _ensure_tarball_cache(tar_path, cache_name)
    iso = _TarballIsolation(cache)
    with iso.active():
        mod = _fresh_module(cache / "main.py", cache_name)
    inner = mod.agent

    def act(obs):
        before = _submission_stats_snapshot(mod)
        with iso.active():
            result = inner(obs)
        bad = _forbidden_submission_stats_delta(before, _submission_stats_snapshot(mod))
        if bad:
            raise RuntimeError(f"{cache_name} submission degradation counters changed: {bad}")
        return result

    return act


_BREP_TAR = Path.home() / "projects/Kaggle/orbit-wars-lab-B/artifacts/submission_brep.tar.gz"

# Single source of truth for file-backed references; the selector preflight
# (league_submission_selector.py) checks these BEFORE any multi-hour run so a
# missing external dies at decision time, not 6 hours into a schedule.
REQUIRED_EXTERNAL_PATHS = {
    "ext_lb1050": "artifacts/opponents/top5_proxy/lb-1050-heuristic-simulation-agent-test-3/agent.py",
    "ext_hellburner": "artifacts/opponents/top5_proxy/hellburner/agent.py",
    "brep": _BREP_TAR,
}


def external_path(name: str) -> Path | None:
    raw = REQUIRED_EXTERNAL_PATHS.get(name)
    if raw is None:
        return None
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _brep():
    return _tarball_agent(_BREP_TAR, "brep")


_PGS_V3_BASE = {
    "scripts": "hold",
    "wave_min_ships": 60.0,
    "wave_start_step": 150,
    "floor_in_4p": True,
}


def register_submission_file(name: str, path: str | Path) -> None:
    """Register a Kaggle-format ``agent(obs)`` file as a league bot."""

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    FACTORIES[str(name)] = (lambda f=p, n=str(name): _fresh_module(f, n).agent)


def register_submission_tarball(name: str, path: str | Path) -> None:
    """Register a Kaggle-format tarball as a league bot."""

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    FACTORIES[str(name)] = (lambda f=p, n=str(name): _tarball_agent(f, n))


FACTORIES = {
    "producer": lambda: _producer(),
    "oep": lambda: _oep(),
    "brep": lambda: _brep(),
    "greedy": lambda: _heuristic("greedy"),
    "rush": lambda: _heuristic("rush"),
    "pgs_hold": lambda: _pgs(scripts="hold"),
    "pgs_holdwave": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150),
    # Drenagem-dupla (commit 8459f7b) seat-rotated promotion gate (scripts/
    # gate_drain_h2h.py). pgs_hold_fix == the live pgs_holdwave submission (fix
    # ON); pgs_hold_prefix == byte-identical config but with the fix DISABLED
    # (source_spend_budget=None == pre-fix behaviour). disable_drain_fix is a
    # per-instance PGSConfig knob, so the two play head-to-head in one process.
    "pgs_hold_fix": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150),
    "pgs_hold_prefix": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                    disable_drain_fix=True),
    # G3.2 Phase 2: pgs_holdwave + decisive-wave concentration for even_attrition_2p
    # (focus-fire enemy core in even 2p late-game). Candidate for the seat-rotated
    # 2p ruler vs incumbent pgs_holdwave. Off-by-default knob; submission unchanged.
    "pgs_decisive2p": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                   decisive_wave_2p=True),
    # Passo 5 tuning (even_attrition_2p): up to 3 sequential A/B cycles vs
    # pgs_holdwave, NOT a grid. Each variant only changes ONE decisive_wave knob
    # off the baseline (start_step=200, min=80, even_band=0.30, max_delay=20).
    # Cycle 1 — wave start_step: earlier (more late-game to act) vs later.
    "pgs_dw_s150": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                decisive_wave_2p=True, decisive_wave_start_step=150),
    "pgs_dw_s250": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                decisive_wave_2p=True, decisive_wave_start_step=250),
    # Item 5 (kingmaker/overextension 4p survival): pgs_holdwave + H9 threat-value
    # 4p portfolio (auto-adds `reinforce` in 4p so the forward per-enemy threat
    # value can SELECT survival plans; 2p frozen = scripts="hold"). Candidate for
    # the 4p-HEAVY seat-rotated ruler vs incumbent pgs_holdwave — re-validating H9
    # with the corrected death+margin verdict (the old h9_4p_gate was death-only).
    "pgs_h9threat": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                 threat_value_4p=True),
    "pgs_holdwave_half2p": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                        half_in_2p=True),
    "pgs_allscripts": lambda: _pgs(),
    "ext_lb1050": _external(str(REQUIRED_EXTERNAL_PATHS["ext_lb1050"])),
    "ext_hellburner": _external(str(REQUIRED_EXTERNAL_PATHS["ext_hellburner"])),
    # H-P5 league-guided wave round (one round, pre-registered; /goal 2026-06-10)
    "pgs_wave_s100": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=100),
    "pgs_wave_s50": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=50),
    # Rank-19 threat-aware targeting (general + ρ(eta) ramp) — margin sweep for the
    # daily LB submission choice, ranked by the +0.80 diverse-pool survival gate.
    "pgs_reactive_m4": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                    reactive_reinforce_margin=0.4),
    "pgs_reactive_m6": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                    reactive_reinforce_margin=0.6),
    "pgs_reactive_m8": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                    reactive_reinforce_margin=0.8),
    # Beta sweep at the rank-19-proven eta (3/12, 1-indexed). reinforce_size_beta=2.2
    # is The Producer V2's published value; we validate around it on the +0.80 gate.
    "pgs_reactive_b15": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                     reactive_reinforce_margin=1.5),
    "pgs_reactive_b22": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                     reactive_reinforce_margin=2.2),
    "pgs_reactive_b30": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                     reactive_reinforce_margin=3.0),
    # No-wave threat-aware (closer to bare Producer V2/1443, which has no wave). Test
    # whether our wave helps or is redundant with the threat-aware discipline.
    "pgs_nowave_b15": lambda: _pgs(scripts="hold", reactive_reinforce_margin=1.5),
    "pgs_nowave_b22": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2),
    # Weakest-enemy 4p targeting (kvatsa5) stacked on threat-aware b15. Sweep the mult.
    "pgs_we13": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                             reactive_reinforce_margin=1.5, weakest_enemy_4p_mult=1.3),
    "pgs_we15": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                             reactive_reinforce_margin=1.5, weakest_enemy_4p_mult=1.5),
    "pgs_we20": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                             reactive_reinforce_margin=1.5, weakest_enemy_4p_mult=2.0),
    # Exposed-target (kvatsa5 snipe) + full stack on the LB-PROVEN threat-aware b22.
    "pgs_exp": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                            reactive_reinforce_margin=2.2, exposed_target_mult=2.0),
    "pgs_b22_we": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                               reactive_reinforce_margin=2.2, weakest_enemy_4p_mult=1.5),
    "pgs_fullstack": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                  reactive_reinforce_margin=2.2, weakest_enemy_4p_mult=1.5,
                                  exposed_target_mult=2.0),
    # Threat-aware DEFENSE on the nowave threat-aware base (top-5 lever): margin sweep.
    "pgs_nowave_ta": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2),
    "pgs_def03": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2, reactive_defense_margin=0.3),
    "pgs_def05": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2, reactive_defense_margin=0.5),
    "pgs_def08": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2, reactive_defense_margin=0.8),
    "pgs_def12": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2, reactive_defense_margin=1.2),
    "pgs_def16": lambda: _pgs(scripts="hold", reactive_reinforce_margin=2.2, reactive_defense_margin=1.6),
    # Wave AMPLIFICATION toward the elite "few BIG waves + hoard 2-5x" style (the PROVEN
    # lever +178). Bigger wave_min_ships = bigger waves / more hoard. holdwave base = w60.
    "pgs_w80":  lambda: _pgs(scripts="hold", wave_min_ships=80.0,  wave_start_step=150),
    "pgs_w100": lambda: _pgs(scripts="hold", wave_min_ships=100.0, wave_start_step=150, reactive_reinforce_margin=2.2),
    "pgs_w120": lambda: _pgs(scripts="hold", wave_min_ships=120.0, wave_start_step=150, reactive_reinforce_margin=2.2),
    # Wave-timing search (goal/seat-rotated branch 2026-06-19). The ONLY lever with
    # a real LB signal is wave timing (monotone on LB: nowave 1057 < s100 1146 <
    # s150 1228). These probe LATER starts / BIGGER waves around the holdwave
    # optimum (s150,min60). Off-by-default factories; submission unchanged. Promote
    # only if a variant DOMINATES holdwave vs the external field proxies.
    "pgs_wave_s120": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=120),
    "pgs_wave_s175": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=175),
    "pgs_wave_s200": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=200),
    "pgs_wave_min45_s150": lambda: _pgs(scripts="hold", wave_min_ships=45.0, wave_start_step=150),
    "pgs_wave_min80_s150": lambda: _pgs(scripts="hold", wave_min_ships=80.0, wave_start_step=150, reactive_reinforce_margin=2.2),
    "pgs_wave_min80_s175": lambda: _pgs(scripts="hold", wave_min_ships=80.0, wave_start_step=175),
    "pgs_wave_4pfloor": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                     floor_in_4p=True),
    # PGS v3 pre-registered league variants. They keep the operational
    # hold+wave profile, force the Producer floor in 4p, and opt into the new
    # planner hooks incrementally for 2p evaluation.
    "pgs_v3_profile_only": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=False,
        mission_mode=False,
        enabled_missions="",
        value_mode="scalar",
    ),
    "pgs_v3_adaptive_arbiter": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=False,
        enabled_missions="",
        value_mode="scalar",
    ),
    "pgs_v3_adaptive_defense": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=True,
        enabled_missions="rescue",
        max_mission_candidates=10,
        max_selected_missions=1,
        value_mode="scalar",
    ),
    "pgs_v3_adaptive_full2p": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=True,
        enabled_missions="rescue,punish,hammer",
        max_mission_candidates=8,
        max_selected_missions=1,
        hammer_top_targets=3,
        hammer_top_sources=4,
        deadline_ms=450.0,
        deadline_guard_ms=100.0,
        value_mode="scalar",
    ),
    # Etapas E/G do plano v2 (2026-06-12): cada variante isola UM delta sobre
    # o full2p para a ordem de avaliação atribuir ganho/perda ao mecanismo certo.
    "pgs_v3_timeline2p": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=True,
        enabled_missions="rescue,punish,hammer",
        max_mission_candidates=8,
        max_selected_missions=1,
        hammer_top_targets=3,
        hammer_top_sources=4,
        deadline_ms=450.0,
        deadline_guard_ms=100.0,
        value_mode="timeline",
    ),
    "pgs_v3_hoard2p": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=True,
        enabled_missions="rescue,punish,hammer,hold_source",
        max_mission_candidates=8,
        max_selected_missions=2,
        hammer_top_targets=3,
        hammer_top_sources=4,
        deadline_ms=450.0,
        deadline_guard_ms=100.0,
        value_mode="scalar",
    ),
    "pgs_v3_waveactive2p": lambda: _pgs(
        **_PGS_V3_BASE,
        adaptive_mode=True,
        adaptive_reply_models=True,
        mission_mode=True,
        enabled_missions="rescue,punish,hammer",
        max_mission_candidates=8,
        max_selected_missions=1,
        hammer_top_targets=3,
        hammer_top_sources=4,
        deadline_ms=450.0,
        deadline_guard_ms=100.0,
        value_mode="scalar",
        wave_release_on_age=False,
    ),
    # H7 E4: holdwave base + learned value net plugged into the search (scores the
    # post-launch board instead of margin-at-H). Tests if the learned value unblocks
    # 4p deviation + improves survival. defend_in_4p on so reinforce/evac are candidates.
    "pgs_valuenet": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                 defend_in_4p=True,
                                 value_net_path=str(ROOT / "artifacts/h7/value_net.pt")),
    # H11 (2026-06-17): same as pgs_valuenet but the value net is the ATTENTION
    # arch (AttnValueNet) — sees pairwise threat (which fleet threatens which
    # planet), the structure mean-pool was blind to (E5 4p-death failure DB 166).
    # load_value_net auto-builds the attn arch from the checkpoint's "arch" tag.
    # CANONICAL path = artifacts/h7/value_net_attn.pt (goal.md "single path" rule);
    # retrained fresh 2026-06-18 (train_value_net --arch attn --epochs 30, E3 PASS
    # Spearman +0.653 > baseline +0.637). nash_gate._candidate_checkpoint hashes
    # this same path so the gate report's hash matches the net that actually plays.
    "pgs_valuenet_attn": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                      defend_in_4p=True,
                                      value_net_path=str(ROOT / "artifacts/h7/value_net_attn.pt")),
    # style exploiters (league-only, NEVER submit): cover the field's loss axes
    # absent from the producer-lineage pool (2026-06-10 falsification).
    "rusher": lambda: _rusher(attack_from=50, cadence=4),       # early all-in (annihilates hold-family regime)
    "pgs_bigwave": lambda: _pgs(scripts="hold", wave_min_ships=100.0, wave_start_step=50,
                                wave_max_delay=25),             # elite proxy: hoard + few BIG waves (lb taxonomy)
}

# Any tarball dropped in artifacts/league/tarballs/<name>.tar.gz auto-registers
# as a league bot "<name>" — the cross-worktree contract (worktree B exports its
# champions here; self-contained, no shared code needed).
def _register_league_artifacts() -> None:
    for tar in sorted((ROOT / "artifacts" / "league" / "tarballs").glob("*.tar.gz")):
        name = tar.stem.replace(".tar", "")
        FACTORIES[name] = (lambda t=tar, n=name: _tarball_agent(t, n))

    for py in sorted((ROOT / "artifacts" / "league" / "submissions").glob("*.py")):
        if py.stem not in FACTORIES:
            register_submission_file(py.stem, py)

    # ARL auto-research survivors: PGS-config genomes dropped as JSON by the
    # Auto-Research Loop handoff (scripts/research_loop/arl.py --research). Each
    # registers as a `_pgs(**genome)` factory so the seat-rotated ruler can run
    # it by name. Per-file guard: a malformed/non-dict genome is skipped, never
    # breaks this import (which would break the whole eval stack).
    for cfg in sorted((ROOT / "artifacts" / "research_loop" / "candidates").glob("*.json")):
        name = cfg.stem
        if name in FACTORIES:
            continue
        try:
            genome = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(genome, dict):
            FACTORIES[name] = (lambda g=dict(genome): _pgs(**g))


_register_league_artifacts()

# Exported .py submissions (e.g. DRL promotion-gate candidates) auto-register too.
for _py in sorted((ROOT / "artifacts" / "league" / "submissions").glob("*.py")):
    register_submission_file(_py.stem, _py)


def make(name: str):
    return FACTORIES[name]()
