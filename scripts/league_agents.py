"""Agent pool for the local league (H-P4 field ruler).

Every entry is a FACTORY returning a fresh callable(official_obs_dict) -> moves,
one instance per game (cross-contamination gotcha: never share runtimes).

LB_ANCHORS holds the real leaderboard score of OUR submissions of the same
config — the league is only trusted if its ranking reproduces these (Spearman +
the hard requirement that pgs_allscripts lands clearly below producer/oep/brep).
"""
from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Submission-gate reference: candidates must reach P(cand >= GATE_REFERENCE).
# Flip to the new champion only AFTER its LB score stabilizes (project rule);
# producer stays an LB_ANCHOR regardless (longest stable rating we own).
GATE_REFERENCE = "producer"  # TODO flip to pgs_holdwave/pgs_hold when LB stabilizes

# real Kaggle ratings of our submitted configs (refreshed 2026-06-10 ~17:20)
LB_ANCHORS = {
    "producer": 1173.1,       # ref=53366194
    "oep": 1182.7,            # ref=53433131
    "brep": 1156.1,           # ref=53513962
    "pgs_allscripts": 1021.5, # ref=53519882 (accidental all-scripts default)
    "pgs_holdwave": 1243.8,   # ref=53537753 (T+3.3h, ~30 eps — semi-stable, refresh later)
    # pgs_hold (ref=53541125): 951 @ T+1h15 — still in placement, NOT anchored yet
}

_EXT_N = [0]


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


def _tarball_agent(tar_path: Path, cache_name: str):
    """Load a self-contained submission tarball as a league agent (fresh module
    per instance; the extracted dir goes on sys.path so its bundled packages
    shadow nothing — each tarball gets its own cache dir)."""
    cache = ROOT / "artifacts" / "league" / "cache" / cache_name
    if not (cache / "main.py").exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(cache)
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    mod = _fresh_module(cache / "main.py", cache_name)
    return mod.agent


_BREP_TAR = Path.home() / "projects/Kaggle/orbit-wars-lab-B/artifacts/submission_brep.tar.gz"


def _brep():
    return _tarball_agent(_BREP_TAR, "brep")


FACTORIES = {
    "producer": lambda: _producer(),
    "oep": lambda: _oep(),
    "brep": lambda: _brep(),
    "pgs_hold": lambda: _pgs(scripts="hold"),
    "pgs_holdwave": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150),
    "pgs_allscripts": lambda: _pgs(),
    "ext_lb1050": _external("artifacts/opponents/top5_proxy/lb-1050-heuristic-simulation-agent-test-3/agent.py"),
    "ext_hellburner": _external("artifacts/opponents/top5_proxy/hellburner/agent.py"),
    # H-P5 league-guided wave round (one round, pre-registered; /goal 2026-06-10)
    "pgs_wave_s100": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=100),
    "pgs_wave_s50": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=50),
    "pgs_wave_4pfloor": lambda: _pgs(scripts="hold", wave_min_ships=60.0, wave_start_step=150,
                                     floor_in_4p=True),
}

# Any tarball dropped in artifacts/league/tarballs/<name>.tar.gz auto-registers
# as a league bot "<name>" — the cross-worktree contract (worktree B exports its
# champions here; self-contained, no shared code needed).
for _tar in sorted((ROOT / "artifacts" / "league" / "tarballs").glob("*.tar.gz")):
    _name = _tar.stem.replace(".tar", "")
    FACTORIES[_name] = (lambda t=_tar, n=_name: _tarball_agent(t, n))


def make(name: str):
    return FACTORIES[name]()
