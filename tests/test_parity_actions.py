"""Step-by-step parity under real actions: Rust simulator vs official Kaggle env.

The passive parity tests only advance with empty actions, so they never exercise
fleet launch, swept fleet/planet collision, combat, capture, reinforcement or
fleet-vs-comet interaction. This drives identical pseudo-random legal launches
into both the official ``kaggle_environments`` interpreter and the Rust backend
(loaded from the official initial state) and asserts the full state matches every
step, within windows that do not cross hidden-seed comet spawns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")

from scripts.parity_probe_actions import run  # noqa: E402


@pytest.mark.parametrize("num_players", [2, 4])
def test_action_parity_pre_comet_window(num_players: int) -> None:
    # Window 0..49 has no comets: isolates launch, swept collision, combat,
    # capture and reinforcement against the official interpreter.
    report = run(
        num_players=num_players,
        episodes=6,
        start_step=0,
        steps=49,
        enable_comets=False,
        launch_prob=0.6,
        atol=1e-6,
    )
    assert report["passed"], report["failures"]
    assert report["checked_steps"] > 0


@pytest.mark.parametrize("num_players", [2, 4])
def test_action_parity_comet_window(num_players: int) -> None:
    # Window 51..149 carries live comets (spawned at step 50): exercises
    # fleet-vs-comet swept collision and comet capture/expiry timing.
    report = run(
        num_players=num_players,
        episodes=6,
        start_step=51,
        steps=98,
        enable_comets=True,
        launch_prob=0.6,
        atol=1e-6,
    )
    assert report["passed"], report["failures"]
    assert report["checked_steps"] > 0
