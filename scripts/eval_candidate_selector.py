"""In-process eval of a trained candidate_selector policy (Frente B). EVAL ONLY.

The candidate_selector is not exportable yet (a submission would bundle the factory
+ all experts), so this plays the policy greedily (argmax candidate) inside the
candidate-mode gym env vs a chosen opponent and reports the normalized score margin.
Evaluates the bar (producer) AND held-out opponents the policy never trained against
(defensive/rush) — the goal's anti-overfitting check.
"""
from __future__ import annotations

import argparse
import json
from statistics import fmean

import torch

from python.agents.policy import CandidateSelectorActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustConfig
from python.orbit_wars_gym.encoding import observation_dim
from python.orbit_wars_gym.gym_env import OrbitWarsGymEnv
from python.orbit_wars_gym.rules import normalized_margin


def _load_policy(path: str) -> CandidateSelectorActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CandidateSelectorActorCritic(observation_dim())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _eval_vs(model, opponent: str, seeds: int, episode_steps: int) -> dict:
    margins, wins = [], []
    invalid = 0.0
    for seed in range(max(1, seeds)):
        env = OrbitWarsGymEnv(
            action_mode="candidate",
            opponent_policy=make_isolated_opponent(opponent),
            rust_cfg=RustConfig(episode_steps=episode_steps, enable_comets=True),
        )
        obs, _ = env.reset(seed=seed)
        scores = [0.0, 0.0]
        for _ in range(episode_steps + 1):
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                idx = int(model.forward(ot)["candidate"].argmax(-1).item())  # greedy
            obs, _, terminated, truncated, info = env.step(idx)
            if info.get("scores"):
                scores = [float(x) for x in info["scores"]]
            if terminated or truncated:
                break
        margins.append(normalized_margin(scores, 0))
        wins.append(1.0 if scores[0] > max(scores[1:]) else 0.0)
    return {
        "opponent": opponent,
        "mean_score_margin": fmean(margins),
        "win_rate": fmean(wins),
        "seeds": len(margins),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponents", nargs="+", default=["producer", "defensive", "rush"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episode-steps", type=int, default=256)
    args = ap.parse_args()

    model = _load_policy(args.checkpoint)
    results = [_eval_vs(model, opp, args.seeds, args.episode_steps) for opp in args.opponents]
    report = {"checkpoint": args.checkpoint, "per_opponent": results}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
