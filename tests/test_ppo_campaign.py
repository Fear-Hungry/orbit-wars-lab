from __future__ import annotations

from pathlib import Path

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

    def fake_train(cfg):
        calls.append(cfg)
        Path(cfg.checkpoint_out).write_bytes(b"checkpoint")
        return {"last_explained_variance": 0.5, "last_entropy": 1.0}

    monkeypatch.setattr(ppo_campaign, "train_phase5_4p", fake_train)
    def fake_margin(_checkpoint, **kwargs):
        margin_include_4p.append(kwargs["include_4p"])
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
        rollout_steps=4,
        eval_seeds=[0],
        eval_episode_steps=8,
        ent_coef=0.01,
        patience=1,
        seed=0,
        training_track="phase5_4p",
    )

    assert calls[0].policy_track == "phase5_4p"
    assert calls[0].num_players == 4
    assert calls[0].normalized_margin_scale_start > calls[0].normalized_margin_scale_end > 0.0
    assert margin_include_4p == [True]
    assert report["history"][0]["eval_include_4p"] is True
    assert report["history"][0]["eval_games"] == 5.0
    assert report["best_checkpoint"] == str(tmp_path / "campaign" / "best.pt")


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
                }
            ],
            "prepared_candidates": {"ppo_chunk": {}},
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
    assert "Infinity" not in (tmp_path / "campaign" / "campaign_report.json").read_text()


def test_ppo_campaign_strict_gate_continues_while_rejected_score_improves(monkeypatch, tmp_path: Path):
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
    assert report["history"][0]["next_training_opponents"] == ["producer", "producer", "producer", "oep"]
    assert report["history"][1]["training_opponents"] == ["producer", "producer", "producer", "oep"]
