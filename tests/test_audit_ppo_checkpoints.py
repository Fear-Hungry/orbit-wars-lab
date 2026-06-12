from __future__ import annotations

from pathlib import Path

import torch
from python.agents.policy import FlatActorCritic
from python.orbit_wars_gym.encoding import observation_dim
from scripts import audit_ppo_checkpoints as audit


def _checkpoint(path: Path) -> None:
    torch.save(
        {
            "model_state_dict": FlatActorCritic(observation_dim()).state_dict(),
            "summary": {"arch": "flat"},
            "config": {
                "policy_track": "phase0_2p",
                "num_players": 2,
                "total_timesteps": 1024,
                "opponents": ["producer"],
            },
        },
        path,
    )


def test_checkpoint_metadata_reads_arch_and_training_context(tmp_path: Path):
    checkpoint = tmp_path / "agent.pt"
    _checkpoint(checkpoint)

    meta = audit._checkpoint_metadata(checkpoint)

    assert meta["valid_checkpoint"]
    assert meta["arch"] == "flat"
    assert meta["policy_track"] == "phase0_2p"
    assert meta["num_players"] == 2
    assert meta["total_timesteps"] == 1024
    assert meta["opponents"] == ["producer"]


def test_audit_ppo_checkpoints_records_parity_and_sorts(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "nested" / "agent.pt"
    checkpoint.parent.mkdir()
    _checkpoint(checkpoint)

    monkeypatch.setattr(
        audit,
        "check_checkpoint_export_parity",
        lambda checkpoint_path, **kwargs: {
            "checkpoint": str(checkpoint_path),
            "submission": str(tmp_path / "audit" / "submissions" / "agent.py"),
            "tarball": str(tmp_path / "audit" / "submissions" / "agent.tar.gz"),
            "passed": True,
            "checked_observations": 12,
            "mismatches": [],
        },
    )

    report = audit.audit_ppo_checkpoints(
        [str(tmp_path / "**" / "*.pt")],
        out_dir=tmp_path / "audit",
        max_checkpoints=None,
        parity_seeds=1,
        parity_steps=1,
        run_gate=False,
        gate_profile="quick",
        gate_seeds=None,
        gate_steps=None,
        gate_jobs=1,
    )

    assert report["checkpoint_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["checkpoint"] == str(checkpoint.resolve())
    assert candidate["export_parity"]["passed"] is True
    assert candidate["export_parity"]["checked_observations"] == 12
    assert candidate["export_parity"]["tarball"].endswith("agent.tar.gz")
    assert candidate["gate_row"]["verdict"] == "NOT_RUN"
