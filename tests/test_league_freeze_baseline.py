"""Baseline freeze (selector etapa 1): immutable, self-contained, hash-bound."""
from __future__ import annotations

import json

import pytest
from scripts.league_freeze_baseline import build_manifest, freeze


def _fixture_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    games = {"games": [], "mode": "2p"}
    (run / "a__2p__producer.json").write_text(json.dumps(games))
    task_results = [{
        "candidate": "a", "label": "a__2p__producer", "mode": "2p",
        "names": ["a", "producer"], "returncode": 0,
        "out": "artifacts/league/submit_ruler/x/a__2p__producer.json",
        "seed_base": 5433500, "seeds": 48, "steps": 500, "chunk_size": 8,
    }]
    tr = run / "task_results.json"
    tr.write_text(json.dumps(task_results))
    return tr


def test_manifest_captures_settings_hashes_and_anchors(tmp_path):
    tr = _fixture_run(tmp_path)
    manifest = build_manifest(tr, label="b1")
    assert manifest["label"] == "b1"
    assert manifest["task_count"] == 1
    assert manifest["candidates"] == ["a"]
    assert manifest["settings"]["seed_base"] == [5433500]
    assert manifest["settings"]["seeds"] == [48]
    assert len(manifest["task_results_sha1"]) == 40
    assert manifest["git_head"]
    # LB anchors of the day are part of the frozen context
    assert "pgs_holdwave" in manifest["lb_anchors"]
    assert manifest["incumbent"]


def test_freeze_copies_games_and_refuses_overwrite(tmp_path):
    tr = _fixture_run(tmp_path)
    out_dir = tmp_path / "baselines"
    dest = freeze(tr, label="b1", out_dir=out_dir)
    assert (dest / "manifest.json").exists()
    assert (dest / "task_results.json").exists()
    assert (dest / "games" / "a__2p__producer.json").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert list(manifest["game_files"]) == ["a__2p__producer.json"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        freeze(tr, label="b1", out_dir=out_dir)
    freeze(tr, label="b1", out_dir=out_dir, force=True)  # explicit opt-in works


def test_freeze_fails_loud_on_missing_game_files(tmp_path):
    tr = _fixture_run(tmp_path)
    (tr.parent / "a__2p__producer.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        freeze(tr, label="b1", out_dir=tmp_path / "baselines")
