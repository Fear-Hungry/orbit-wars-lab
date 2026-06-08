"""Diagnose the -0.25 Producer plateau (Frente B): candidate ceiling vs suboptimal policy.

Plays a FIXED candidate index (no learned policy) inside the candidate-mode env vs the
Producer opponent and reports the normalized margin per fixed choice:
  always-producer (idx 1) ~ 0     => candidate-producer mirrors the Producer opponent =>
                                      the LEARNED policy (-0.25) is SUBOPTIMAL (local optimum) -> B4/more directed training.
  always-producer ~ -0.25         => candidate-producer != Producer-opponent (extraction/2nd-mover
                                      asymmetry) => CANDIDATE CEILING -> need a better candidate, not more PPO.
"""
from statistics import fmean
from python.agents.candidate_factory import CANDIDATE_NAMES
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustConfig
from python.orbit_wars_gym.gym_env import OrbitWarsGymEnv
from python.orbit_wars_gym.rules import normalized_margin


def eval_fixed(idx, seeds=8, steps=256):
    margins, wins = [], []
    for seed in range(seeds):
        env = OrbitWarsGymEnv(action_mode="candidate", opponent_policy=make_isolated_opponent("producer"),
                              rust_cfg=RustConfig(episode_steps=steps, enable_comets=True))
        obs, _ = env.reset(seed=seed)
        scores = [0.0, 0.0]
        for _ in range(steps + 1):
            obs, _, term, trunc, info = env.step(idx)
            if info.get("scores"):
                scores = [float(x) for x in info["scores"]]
            if term or trunc:
                break
        margins.append(normalized_margin(scores, 0))
        wins.append(1.0 if scores[0] > scores[1] else 0.0)
    return fmean(margins), fmean(wins)


if __name__ == "__main__":
    for idx in (1, 2, 0):  # producer, oep, no_op
        m, w = eval_fixed(idx)
        print(f"always-{CANDIDATE_NAMES[idx]:9s} (idx {idx}) vs Producer: margin={m:+.4f} win={w:.3f}")
