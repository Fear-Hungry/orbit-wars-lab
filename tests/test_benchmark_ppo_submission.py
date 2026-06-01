from __future__ import annotations

from pathlib import Path

import torch
from python.agents.policy import FlatActorCritic
from python.orbit_wars_gym.encoding import observation_dim
from scripts.benchmark_ppo_submission import benchmark_exported_checkpoint


def test_benchmark_exported_checkpoint_runs_submission_runtime(tmp_path: Path):
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": FlatActorCritic(observation_dim()).state_dict(),
            "summary": {
                "decoder": {
                    "fractions": [0.10, 0.25, 0.50, 0.75],
                    "angle_offsets": [-0.261799, -0.130899, 0.0, 0.130899, 0.261799],
                    "max_moves_per_turn": 2,
                    "min_ships_to_launch": 2,
                    "reserve_home_ships": 8,
                }
            },
        },
        checkpoint,
    )

    report = benchmark_exported_checkpoint(
        checkpoint,
        submission_out=tmp_path / "submission.py",
        opponents=["weak_random"],
        seeds=[0],
        episode_steps=32,
        enable_comets=True,
        act_timeout=1.0,
        include_4p=False,
    )

    assert report["checkpoint"] == str(checkpoint)
    assert Path(report["submission"]).exists()
    assert report["submission_size_bytes"] > 0
    assert report["summary"]["games"] == 2.0
    assert report["summary"]["crash_rate"] == 0.0
    assert report["summary"]["timeout_rate"] == 0.0
    assert report["summary"]["invalid_action_rate"] == 0.0
    assert report["formats"][0]["opponents"][0]["opponent"] == "weak_random"
