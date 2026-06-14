"""League tarball isolation: multiple submission tarballs bundle the same
bare-named modules (_brep_weights, _producer_agent, orbit_lite, ...). They must
NOT share those modules through the global sys.modules cache — including LAZY
imports that only run during act() (the real brep main.py imports
_brep_weights on first use), otherwise which weights a bot uses depends on
make() order and the local measurement is invalid.
"""
from __future__ import annotations

import hashlib
import sys
import tarfile

import pytest
from scripts import league_agents


def _make_tarball(tmp_path, name, sentinel):
    src = tmp_path / f"src_{name}"
    src.mkdir()
    (src / "_brep_weights.py").write_text(f"WEIGHTS_B64 = {sentinel!r}\n")
    (src / "main.py").write_text(
        "def agent(obs):\n"
        "    import _brep_weights  # lazy, like the real submission\n"
        "    return _brep_weights.WEIGHTS_B64\n"
    )
    tar = tmp_path / f"{name}.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in sorted(src.iterdir()):
            tf.add(f, arcname=f.name)
    return tar


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(league_agents, "ROOT", tmp_path)
    return tmp_path


def test_interleaved_tarballs_keep_their_own_weights(fake_root, tmp_path):
    tar_a = _make_tarball(tmp_path, "iso_a", "WEIGHTS_A")
    tar_b = _make_tarball(tmp_path, "iso_b", "WEIGHTS_B")
    agent_a = league_agents._tarball_agent(tar_a, "iso_a")
    agent_b = league_agents._tarball_agent(tar_b, "iso_b")
    # interleaved acts: the lazy import inside act() must resolve per tarball,
    # in both orders, and stay stable after the other tarball has run
    assert agent_a({}) == "WEIGHTS_A"
    assert agent_b({}) == "WEIGHTS_B"
    assert agent_a({}) == "WEIGHTS_A"
    assert agent_b({}) == "WEIGHTS_B"


def test_tarball_modules_do_not_leak_or_shadow_globals(fake_root, tmp_path):
    sentinel = object()  # stands in for a repo-side module with a colliding name
    sys.modules["_brep_weights"] = sentinel
    try:
        tar = _make_tarball(tmp_path, "iso_c", "WEIGHTS_C")
        agent = league_agents._tarball_agent(tar, "iso_c")
        # the tarball must see its OWN bundled module, not the global one
        assert agent({}) == "WEIGHTS_C"
        # ... and the global namespace must come back untouched
        assert sys.modules["_brep_weights"] is sentinel
        digest = hashlib.sha1(tar.read_bytes()).hexdigest()[:12]
        cache = fake_root / "artifacts" / "league" / "cache" / f"iso_c-{digest}"
        assert cache.exists() and str(cache) not in sys.path
    finally:
        del sys.modules["_brep_weights"]


def test_same_tarball_instances_do_not_share_module_runtime(fake_root, tmp_path):
    src = tmp_path / "src_stateful"
    src.mkdir()
    (src / "_state.py").write_text("COUNTER = 0\n")
    (src / "main.py").write_text(
        "def agent(obs):\n"
        "    import _state\n"
        "    _state.COUNTER += 1\n"
        "    return _state.COUNTER\n"
    )
    tar = tmp_path / "stateful.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in sorted(src.iterdir()):
            tf.add(f, arcname=f.name)

    agent_a = league_agents._tarball_agent(tar, "stateful")
    agent_b = league_agents._tarball_agent(tar, "stateful")

    assert agent_a({}) == 1
    assert agent_b({}) == 1
    assert agent_a({}) == 2
    assert agent_b({}) == 2


def test_reexported_tarball_invalidates_cache(fake_root, tmp_path):
    """Regression 2026-06-11: cache keyed by NAME skipped extraction whenever
    cache/main.py existed — re-exporting a tarball under the same path kept
    running the OLD extracted code ("testei o bot novo, mas era cache velho").
    The cache is now keyed by the tarball's content hash."""
    tar = _make_tarball(tmp_path, "champ", "V1")
    agent_v1 = league_agents._tarball_agent(tar, "champ")
    assert agent_v1({}) == "V1"
    # re-export: same tarball path/name, new content
    tar.write_bytes(_make_tarball(tmp_path, "champ_v2", "V2").read_bytes())
    agent_v2 = league_agents._tarball_agent(tar, "champ")
    assert agent_v2({}) == "V2", "re-exported tarball must invalidate the cache"
    # instances already constructed keep their own version (no retroactive swap)
    assert agent_v1({}) == "V1"


def test_partial_tarball_cache_without_complete_marker_is_rebuilt(fake_root, tmp_path):
    tar = _make_tarball(tmp_path, "race", "FRESH")
    digest = hashlib.sha1(tar.read_bytes()).hexdigest()[:12]
    cache = fake_root / "artifacts" / "league" / "cache" / f"race-{digest}"
    cache.mkdir(parents=True)
    (cache / "main.py").write_text('def agent(obs):\n    return "PARTIAL\n')

    agent = league_agents._tarball_agent(tar, "race")

    assert agent({}) == "FRESH"
    assert (cache / ".complete").exists()


def test_missing_tarball_fails_loud_even_with_warm_cache(fake_root, tmp_path):
    """No-silent-fallback: without the source tarball there is no way to know
    WHICH cached version would run — fail loud instead of guessing."""
    import pytest

    tar = _make_tarball(tmp_path, "champ3", "V1")
    league_agents._tarball_agent(tar, "champ3")  # warms the cache
    tar.unlink()
    with pytest.raises(FileNotFoundError):
        league_agents._tarball_agent(tar, "champ3")


def test_tarball_submission_stats_fallbacks_fail_loud(fake_root, tmp_path):
    src = tmp_path / "src_fallback"
    src.mkdir()
    (src / "main.py").write_text(
        "SUBMISSION_STATS = {'calls': 0, 'fallbacks': 0}\n"
        "def agent(obs):\n"
        "    SUBMISSION_STATS['calls'] += 1\n"
        "    SUBMISSION_STATS['fallbacks'] += 1\n"
        "    return []\n"
    )
    tar = tmp_path / "fallback.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in sorted(src.iterdir()):
            tf.add(f, arcname=f.name)

    agent = league_agents._tarball_agent(tar, "fallback")

    with pytest.raises(RuntimeError, match="degradation counters"):
        agent({})


def test_autoload_prefers_tarball_over_same_named_submission_py(fake_root, tmp_path, monkeypatch):
    tarballs = fake_root / "artifacts" / "league" / "tarballs"
    submissions = fake_root / "artifacts" / "league" / "submissions"
    tarballs.mkdir(parents=True)
    submissions.mkdir(parents=True)
    tar = _make_tarball(tmp_path, "champ", "TARBALL")
    (tarballs / "champ.tar.gz").write_bytes(tar.read_bytes())
    (submissions / "champ.py").write_text("def agent(obs):\n    return 'PY'\n")
    monkeypatch.setattr(league_agents, "FACTORIES", {})

    league_agents._register_league_artifacts()

    assert league_agents.make("champ")({}) == "TARBALL"
