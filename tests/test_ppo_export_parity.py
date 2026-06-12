from __future__ import annotations

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
