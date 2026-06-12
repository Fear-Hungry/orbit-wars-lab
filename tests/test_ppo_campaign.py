from __future__ import annotations

from pathlib import Path

import pytest
from scripts import ppo_campaign


def test_pfsp_reweights_frontier_opponents_and_composite_lineups():
    weighted = ppo_campaign._pfsp_reweighted_opponents(
        (
            "producer+oep+pgs_holdwave",
            "oep+rush",
            "brep",
        ),
        {
            "producer": {"decisive_win_rate": 0.5},
            "oep": {"decisive_win_rate": 0.0},
            "pgs_holdwave": {"decisive_win_rate": 0.25},
            "rush": {"decisive_win_rate": 0.95},
            "brep": {"decisive_win_rate": 0.4},
        },
        max_repeats=4,
    )

    assert weighted.count("producer+oep+pgs_holdwave") == 4
    assert weighted.count("brep") == 4
    assert weighted.count("oep+rush") == 1


def test_ppo_campaign_phase5_uses_four_player_training(monkeypatch, tmp_path: Path):
    calls = []
    margin_include_4p = []
    margin_jobs = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "episodes_observed": 4.0,
            "completed_episodes": 2.0,
            "mean_return": 0.25,
            "mean_early_survival_rate": 0.75,
            "last_bc_anchor_loss": 1.25,
            "last_bc_anchor_examples": 8.0,
            "last_bc_anchor_launch_rate": 0.5,
            "last_bc_anchor_mean_quant_error": 0.2,
            "learner_seat_rotation": cfg.learner_seat_rotation,
        }

    monkeypatch.setattr(ppo_campaign, "train_phase5_4p", fake_train)

    def fake_margin(_checkpoint, **kwargs):
        margin_include_4p.append(kwargs["include_4p"])
        margin_jobs.append(kwargs["jobs"])
        return {
            "games": 5.0,
            "mean_score_margin": 0.1,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        }

    monkeypatch.setattr(ppo_campaign, "_margin", fake_margin)

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "pgs_holdwave"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        train_episode_steps=40,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        eval_jobs=1,
        training_track="phase5_4p",
    )

    assert calls[0].policy_track == "phase5_4p"
    assert calls[0].num_players == 4
    assert calls[0].learner_seat_rotation is True
    assert calls[0].normalized_margin_scale_start > calls[0].normalized_margin_scale_end > 0.0
    assert margin_include_4p == [True]
    assert report["history"][0]["eval_include_4p"] is True
    assert margin_jobs == [1]
    assert report["eval_jobs"] == 1
    assert report["history"][0]["learner_seat_rotation"] is True
    assert report["learner_seat_rotation"] is True
    assert report["history"][0]["eval_games"] == 5.0
    assert report["best_checkpoint"] == str(tmp_path / "campaign" / "best.pt")


def test_ppo_campaign_mixed_track_runs_2p_then_4p_substages(monkeypatch, tmp_path: Path):
    calls = []
    margin_include_4p = []

    def fake_train_2p(cfg):
        calls.append(("2p", cfg))
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.25,
            "last_entropy": 2.0,
            "episodes_observed": 2.0,
            "completed_episodes": 1.0,
            "mean_return": 0.1,
            "mean_early_survival_rate": 1.0,
            "decoder": {"max_moves_per_turn": cfg.decoder_max_moves_per_turn},
        }

    def fake_train_4p(cfg):
        calls.append(("4p", cfg))
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.5,
            "episodes_observed": 3.0,
            "completed_episodes": 2.0,
            "mean_return": 0.2,
            "mean_early_survival_rate": 0.75,
            "learner_seat_rotation": cfg.learner_seat_rotation,
            "decoder": {"max_moves_per_turn": cfg.decoder_max_moves_per_turn},
        }

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train_2p)
    monkeypatch.setattr(ppo_campaign, "train_phase5_4p", fake_train_4p)

    def fake_margin(_checkpoint, **kwargs):
        margin_include_4p.append(kwargs["include_4p"])
        return {
            "games": 2.0,
            "mean_score_margin": 0.3,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        }

    monkeypatch.setattr(ppo_campaign, "_margin", fake_margin)

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=10,
        mixed_2p_fraction=0.4,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        training_track="mixed_2p4p",
    )

    assert [label for label, _cfg in calls] == ["2p", "4p"]
    assert calls[0][1].policy_track == "phase0_2p"
    assert calls[0][1].num_players == 2
    assert calls[0][1].total_timesteps == 4
    assert calls[0][1].checkpoint_in == str(tmp_path / "init.pt")
    assert calls[0][1].checkpoint_out == str(tmp_path / "campaign" / "chunk00_2p.pt")
    assert calls[1][1].policy_track == "phase5_4p"
    assert calls[1][1].num_players == 4
    assert calls[1][1].total_timesteps == 6
    assert calls[1][1].checkpoint_in == str(tmp_path / "campaign" / "chunk00_2p.pt")
    assert calls[1][1].checkpoint_out == str(tmp_path / "campaign" / "chunk00.pt")
    assert calls[1][1].learner_seat_rotation is True
    assert margin_include_4p == [True]
    row = report["history"][0]
    assert row["episodes_observed"] == 5.0
    assert row["completed_episodes"] == 3.0
    assert row["eval_include_4p"] is True
    assert row["learner_seat_rotation"] is True
    assert [stage["track"] for stage in row["training_stages"]] == ["phase0_2p", "phase5_4p"]
    assert [stage["timesteps"] for stage in row["training_stages"]] == [4, 6]
    assert report["training_track"] == "mixed_2p4p"
    assert report["mixed_2p_fraction"] == 0.4
    assert report["best_checkpoint"] == str(tmp_path / "campaign" / "best.pt")


def test_ppo_campaign_passes_conservative_ppo_hparams(monkeypatch, tmp_path: Path):
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "episodes_observed": 4.0,
            "completed_episodes": 2.0,
            "mean_return": 0.25,
            "mean_early_survival_rate": 0.75,
            "last_bc_anchor_loss": 1.25,
            "last_bc_anchor_examples": 8.0,
            "last_bc_anchor_launch_rate": 0.5,
            "last_bc_anchor_mean_quant_error": 0.2,
            "last_bc_anchor_effective_coef": cfg.bc_anchor_coef,
        }

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "games": 2.0,
            "mean_score_margin": 0.2,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.004,
        train_episode_steps=40,
        learning_rate=1e-4,
        clip_coef=0.1,
        update_epochs=1,
        minibatch_size=64,
        bc_anchor_coef=0.07,
        bc_anchor_coef_end=0.02,
        bc_anchor_teacher="brep",
        bc_anchor_max_quant_error=2.5,
        patience=1,
        seed=0,
    )

    assert calls[0].learning_rate == 1e-4
    assert calls[0].clip_coef == 0.1
    assert calls[0].update_epochs == 1
    assert calls[0].minibatch_size == 64
    assert calls[0].episode_steps == 40
    assert calls[0].bc_anchor_coef == 0.07
    assert calls[0].bc_anchor_coef_end == pytest.approx(0.02)
    assert calls[0].bc_anchor_teacher == "brep"
    assert calls[0].bc_anchor_max_quant_error == 2.5
    assert report["learning_rate"] == 1e-4
    assert report["clip_coef"] == 0.1
    assert report["update_epochs"] == 1
    assert report["minibatch_size"] == 64
    assert report["train_episode_steps"] == 40
    assert report["history"][0]["train_episode_steps"] == 40
    assert report["history"][0]["episodes_observed"] == 4.0
    assert report["history"][0]["completed_episodes"] == 2.0
    assert report["history"][0]["terminal_reward_observed"] is True
    assert report["history"][0]["mean_train_return"] == 0.25
    assert report["history"][0]["mean_early_survival_rate"] == 0.75
    assert report["bc_anchor_coef"] == 0.07
    assert report["bc_anchor_coef_end"] == pytest.approx(0.02)
    assert report["bc_anchor_teacher"] == "brep"
    assert report["bc_anchor_max_quant_error"] == 2.5
    assert report["history"][0]["bc_anchor_loss"] == 1.25
    assert report["history"][0]["bc_anchor_examples"] == 8.0
    assert report["history"][0]["bc_anchor_launch_rate"] == 0.5
    assert report["history"][0]["bc_anchor_mean_quant_error"] == 0.2
    assert report["history"][0]["bc_anchor_effective_coef"] == 0.07
    assert report["history"][0]["learning_rate"] == 1e-4


def test_ppo_campaign_can_override_decoder_and_disable_checkpoint_inheritance(
    monkeypatch, tmp_path: Path
):
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "episodes_observed": 2.0,
            "completed_episodes": 1.0,
            "decoder": {"max_moves_per_turn": cfg.decoder_max_moves_per_turn},
        }

    monkeypatch.setattr(ppo_campaign, "train_phase5_4p", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "games": 2.0,
            "mean_score_margin": 0.2,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "single_action_init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        training_track="phase5_4p",
        decoder_max_moves_per_turn=8,
        inherit_checkpoint_decoder=False,
        min_decoder_max_moves_per_turn=2,
    )

    assert calls[0].decoder_max_moves_per_turn == 8
    assert calls[0].inherit_checkpoint_decoder is False
    row = report["history"][0]
    assert row["decoder_max_moves_per_turn"] == 8
    assert row["min_decoder_max_moves_per_turn"] == 2
    assert row["decoder_capacity_ok"] is True
    assert row["inherit_checkpoint_decoder"] is False
    assert report["decoder_max_moves_per_turn"] == 8
    assert report["inherit_checkpoint_decoder"] is False


def test_ppo_campaign_decoder_capacity_guard_blocks_selection(monkeypatch, tmp_path: Path):
    def fake_train(cfg):
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "episodes_observed": 2.0,
            "completed_episodes": 1.0,
            "decoder": {"max_moves_per_turn": 1},
        }

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "games": 2.0,
            "mean_score_margin": 0.8,
            "win_rate": 1.0,
            "invalid_action_rate": 0.0,
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "single_action_init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        min_decoder_max_moves_per_turn=2,
    )

    row = report["history"][0]
    assert row["decoder_max_moves_per_turn"] == 1
    assert row["min_decoder_max_moves_per_turn"] == 2
    assert row["decoder_capacity_ok"] is False
    assert row["train_signal_verdict"] == "REJECT_DECODER_CAPACITY"
    assert report["best_checkpoint"] is None
    assert report["best_margin"] is None
    assert report["best_observed_score"] is None


def test_ppo_campaign_strict_gate_without_pass_has_no_best_margin(monkeypatch, tmp_path: Path):
    def fake_train(cfg):
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {"last_explained_variance": 0.5, "last_entropy": 1.0}

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "mean_score_margin": -0.4,
            "win_rate": 0.0,
            "invalid_action_rate": 0.0,
        },
    )
    monkeypatch.setattr(
        ppo_campaign,
        "run_drl_promotion_gate",
        lambda **_kwargs: {
            "ranking": [
                {
                    "candidate": "ppo_chunk",
                    "verdict": "REJECT_LOCAL",
                    "overall_score": 0.1,
                    "score_2p": 0.2,
                    "score_4p": 0.05,
                }
            ],
            "prepared_candidates": {"ppo_chunk": {}},
            "candidates": {
                "ppo_chunk": {
                    "pairwise": {
                        "producer": {
                            "win_rate": 0.5,
                            "decisive_win_rate": 0.75,
                            "nonloss_rate": 0.625,
                            "faults": {"crashes": 0, "invalid_moves": 0, "timeouts": 0},
                        }
                    }
                }
            },
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        strict_drl_gate=True,
    )

    assert report["best_checkpoint"] is None
    assert report["best_margin"] is None
    assert report["best_observed_score"] == 0.1
    assert report["history"][0]["gate_score_2p"] == 0.2
    assert report["history"][0]["gate_score_4p"] == 0.05
    assert report["history"][0]["gate_pairwise"]["producer"]["decisive_win_rate"] == 0.75
    assert report["history"][0]["gate_pairwise"]["producer"]["faults"]["crashes"] == 0
    assert "Infinity" not in (tmp_path / "campaign" / "campaign_report.json").read_text()


def test_ppo_campaign_terminal_reward_guard_skips_gate_and_selection(monkeypatch, tmp_path: Path):
    def fake_train(cfg):
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "episodes_observed": 3.0,
            "completed_episodes": 0.0,
            "mean_return": -0.1,
            "mean_early_survival_rate": 1.0,
        }

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "mean_score_margin": 0.4,
            "win_rate": 1.0,
            "invalid_action_rate": 0.0,
        },
    )

    def fail_gate(**_kwargs):
        raise AssertionError("terminal-reward guard should skip the expensive gate")

    monkeypatch.setattr(ppo_campaign, "run_drl_promotion_gate", fail_gate)

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=1,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        strict_drl_gate=True,
        require_terminal_reward=True,
    )

    row = report["history"][0]
    assert row["terminal_reward_observed"] is False
    assert row["require_terminal_reward"] is True
    assert row["train_signal_verdict"] == "REJECT_NO_TERMINAL_REWARD"
    assert row["gate_verdict"] == "SKIP_NO_TERMINAL_REWARD"
    assert row["gate_report"] is None
    assert report["best_checkpoint"] is None
    assert report["best_margin"] is None
    assert report["best_observed_score"] is None


def test_ppo_campaign_strict_gate_continues_while_rejected_score_improves(
    monkeypatch, tmp_path: Path
):
    trained = []

    def fake_train(cfg):
        trained.append(cfg.checkpoint_out)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {"last_explained_variance": 0.5, "last_entropy": 1.0}

    gate_scores = iter([0.1, 0.2])

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "mean_score_margin": -0.4,
            "win_rate": 0.0,
            "invalid_action_rate": 0.0,
        },
    )

    def fake_gate(**_kwargs):
        score = next(gate_scores)
        return {
            "ranking": [
                {
                    "candidate": "ppo_chunk",
                    "verdict": "REJECT_LOCAL",
                    "overall_score": score,
                }
            ],
            "prepared_candidates": {"ppo_chunk": {}},
        }

    monkeypatch.setattr(ppo_campaign, "run_drl_promotion_gate", fake_gate)

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=2,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        strict_drl_gate=True,
    )

    assert len(trained) == 2
    assert len(report["history"]) == 2
    assert report["best_checkpoint"] is None
    assert report["best_margin"] is None
    assert report["best_observed_score"] == 0.2
    assert report["best_observed_checkpoint"] == str(tmp_path / "campaign" / "chunk01.pt")


def test_ppo_campaign_kl_anchor_defaults_to_initial_checkpoint(monkeypatch, tmp_path: Path):
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {"last_explained_variance": 0.5, "last_entropy": 1.0, "last_kl_to_ref": 0.25}

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "mean_score_margin": 0.1,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        },
    )
    init = tmp_path / "init.pt"

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(init),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=2,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        kl_to_ref_coef=0.03,
        patience=2,
        seed=0,
    )

    assert [call.kl_to_ref_coef for call in calls] == [0.03, 0.03]
    assert [call.ref_checkpoint for call in calls] == [str(init), str(init)]
    assert calls[0].checkpoint_in == str(init)
    assert calls[1].checkpoint_in == str(tmp_path / "campaign" / "chunk00.pt")
    assert report["kl_to_ref_coef"] == 0.03
    assert report["ref_checkpoint"] == str(init)
    assert report["history"][0]["kl_to_ref"] == 0.25


def test_ppo_campaign_bc_anchor_decay_spans_chunks(monkeypatch, tmp_path: Path):
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {
            "last_explained_variance": 0.5,
            "last_entropy": 1.0,
            "last_bc_anchor_effective_coef": cfg.bc_anchor_coef_end,
        }

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "mean_score_margin": 0.1,
            "win_rate": 0.5,
            "invalid_action_rate": 0.0,
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=2,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        bc_anchor_coef=0.08,
        bc_anchor_coef_end=0.02,
        bc_anchor_teacher="brep",
        patience=2,
        seed=0,
    )

    assert [call.bc_anchor_coef for call in calls] == [0.08, 0.05]
    assert [call.bc_anchor_coef_end for call in calls] == pytest.approx([0.05, 0.02])
    assert report["history"][0]["bc_anchor_chunk_start_coef"] == 0.08
    assert report["history"][0]["bc_anchor_chunk_end_coef"] == 0.05
    assert report["history"][1]["bc_anchor_chunk_start_coef"] == 0.05
    assert report["history"][1]["bc_anchor_chunk_end_coef"] == pytest.approx(0.02)


def test_ppo_campaign_pfsp_uses_gate_pairwise_for_next_chunk(monkeypatch, tmp_path: Path):
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {"last_explained_variance": 0.5, "last_entropy": 1.0}

    monkeypatch.setattr(ppo_campaign, "train_phase0", fake_train)
    monkeypatch.setattr(
        ppo_campaign,
        "_margin",
        lambda *_args, **_kwargs: {
            "games": 2.0,
            "mean_score_margin": 0.1,
            "win_rate": 0.0,
            "invalid_action_rate": 0.0,
        },
    )
    monkeypatch.setattr(
        ppo_campaign,
        "run_drl_promotion_gate",
        lambda **_kwargs: {
            "ranking": [
                {
                    "candidate": "ppo_chunk",
                    "verdict": "REJECT_LOCAL",
                    "overall_score": 0.1,
                }
            ],
            "prepared_candidates": {"ppo_chunk": {}},
            "candidates": {
                "ppo_chunk": {
                    "pairwise": {
                        "producer": {"decisive_win_rate": 0.5},
                        "oep": {"decisive_win_rate": 0.0},
                    }
                }
            },
        },
    )

    report = ppo_campaign.run_campaign(
        init_checkpoint=str(tmp_path / "init.pt"),
        out_dir=tmp_path / "campaign",
        opponents=("producer", "oep"),
        eval_opponents=["producer"],
        chunks=2,
        chunk_timesteps=8,
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=2,
        seed=0,
        strict_drl_gate=True,
        pfsp=True,
        pfsp_max_repeats=3,
    )

    assert calls[0].opponents == ("producer", "oep")
    assert calls[1].opponents == ("producer", "producer", "producer", "oep")
    assert report["history"][0]["next_training_opponents"] == [
        "producer",
        "producer",
        "producer",
        "oep",
    ]
    assert report["history"][1]["training_opponents"] == ["producer", "producer", "producer", "oep"]
