from __future__ import annotations

import tarfile
from pathlib import Path

import torch
from python.agents.policy import EntityActorCritic, FlatActorCritic
from python.orbit_wars_gym.encoding import observation_dim
from scripts.ppo_export_parity import check_checkpoint_export_parity


def _write_checkpoint(path: Path, model, *, arch: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "summary": {
                "arch": arch,
                "decoder": {
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
                    "max_moves_per_turn": 4,
                    "min_ships_to_launch": 2,
                    "reserve_home_ships": 8,
                },
            },
        },
        path,
    )


def test_flat_checkpoint_export_matches_local_policy(tmp_path: Path):
    torch.manual_seed(7)
    checkpoint = tmp_path / "flat.pt"
    _write_checkpoint(checkpoint, FlatActorCritic(observation_dim()), arch="flat")

    report = check_checkpoint_export_parity(
        checkpoint,
        submission_path=tmp_path / "flat_submission.py",
        seeds=[0],
        steps=1,
        player_counts=(2,),
    )

    assert report["passed"]
    assert report["checked_observations"] > 0
    assert report["mismatches"] == []
    assert Path(report["tarball"]).exists()
    with tarfile.open(report["tarball"]) as tar:
        assert "main.py" in {member.name for member in tar.getmembers()}


def test_entity_checkpoint_export_matches_local_policy(tmp_path: Path):
    torch.manual_seed(11)
    checkpoint = tmp_path / "entity.pt"
    _write_checkpoint(checkpoint, EntityActorCritic(observation_dim()), arch="entity")

    report = check_checkpoint_export_parity(
        checkpoint,
        submission_path=tmp_path / "entity_submission.py",
        seeds=[0],
        steps=1,
        player_counts=(2,),
    )

    assert report["passed"]
    assert report["checked_observations"] > 0
    assert Path(report["tarball"]).exists()


def test_hybrid_template_4p_export_validates_without_ppo_4p_parity(tmp_path: Path):
    torch.manual_seed(13)
    checkpoint = tmp_path / "flat_2p.pt"
    _write_checkpoint(checkpoint, FlatActorCritic(observation_dim()), arch="flat")

    report = check_checkpoint_export_parity(
        checkpoint,
        submission_path=tmp_path / "hybrid_template_submission.py",
        seeds=[0],
        steps=1,
        player_counts=(2, 4),
        four_player_policy="template",
    )

    assert report["passed"]
    assert report["four_player_policy"] == "template"
    assert report["checkpoint_4p"] is None
    assert report["mismatches"] == []


def test_dual_checkpoint_export_matches_2p_and_4p_policies(tmp_path: Path):
    torch.manual_seed(17)
    checkpoint_2p = tmp_path / "flat_2p.pt"
    _write_checkpoint(checkpoint_2p, FlatActorCritic(observation_dim()), arch="flat")
    torch.manual_seed(19)
    checkpoint_4p = tmp_path / "flat_4p.pt"
    _write_checkpoint(checkpoint_4p, FlatActorCritic(observation_dim()), arch="flat")

    report = check_checkpoint_export_parity(
        checkpoint_2p,
        submission_path=tmp_path / "dual_submission.py",
        checkpoint_4p_path=checkpoint_4p,
        seeds=[0],
        steps=1,
        player_counts=(2, 4),
        four_player_policy="neural",
    )

    assert report["passed"]
    assert report["checkpoint_4p"] == str(checkpoint_4p)
    assert report["four_player_policy"] == "neural"
