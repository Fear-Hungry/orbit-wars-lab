from pathlib import Path

import pytest
from scripts.select_ppo_checkpoint import _expand_checkpoints, checkpoint_id, score_metrics


def test_checkpoint_id_is_stable_for_paths():
    assert checkpoint_id(Path("artifacts/ppo/phase0-seed1.65536.pt")) == "phase0_seed1_65536"


def test_score_metrics_rewards_win_rate_and_penalizes_runtime_failures():
    clean = {
        "win_rate_vs_league": 0.5,
        "normalized_score_margin": 0.1,
        "robustness_across_seeds": 0.8,
        "crash_rate": 0.0,
        "timeout_rate": 0.0,
        "invalid_action_rate": 0.0,
    }
    broken = {**clean, "crash_rate": 0.5}

    assert score_metrics(clean) > score_metrics(broken)


def test_expand_checkpoints_requires_match(tmp_path: Path):
    checkpoint = tmp_path / "agent.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    assert _expand_checkpoints([str(tmp_path / "*.pt")]) == [checkpoint.resolve()]
    with pytest.raises(ValueError, match="no checkpoint files"):
        _expand_checkpoints([str(tmp_path / "*.missing")])
