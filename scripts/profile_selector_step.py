"""[DEBUG-perf] Profile where a candidate-selector single-env step spends time.
Hypothesis: the Python candidate factory (runs ~6 experts incl Producer/OEP per
step) dominates, NOT the Rust sim or the net — so GPU would not accelerate it.
Throwaway diagnostic; delete after."""
from __future__ import annotations
import cProfile
import pstats
import time
import io

import torch

from python.train.train_ppo import build_phase0_env, _build_policy, PHASE0_OPPONENTS
from python.orbit_wars_gym.encoding import observation_dim, EncoderConfig
from python.agents.candidate_factory import CandidateFactory

N = 300
env = build_phase0_env(seed=0, num_players=2, opponent_name="producer", action_mode="candidate")
obs, _ = env.reset(seed=0)
obs_dim = observation_dim(EncoderConfig())
model = _build_policy("candidate_selector", obs_dim, default_candidate=1)
model.eval()

# Direct component timing -----------------------------------------------------
# env.step() ALSO runs the opponent (producer) internally; the selector's own
# candidate generation is the separate cf.candidates() call. Time both.
cf = CandidateFactory()
t_cf = t_step = 0.0
for i in range(N):
    t1 = time.perf_counter()
    cands = cf.candidates(env.state, 0)
    t2 = time.perf_counter(); t_cf += t2 - t1
    obs, r, term, trunc, info = env.step(1)  # pick candidate idx 1 (producer)
    t3 = time.perf_counter(); t_step += t3 - t2
    if term or trunc:
        obs, _ = env.reset(seed=i + 1)

total = t_cf + t_step
print(f"[DEBUG-perf] over {N} steps:")
print(f"[DEBUG-perf]   candidate_factory  : {t_cf:7.3f}s  ({100*t_cf/total:5.1f}%)  {N/t_cf:7.1f}/s")
print(f"[DEBUG-perf]   env.step(Rust+opp+rwd): {t_step:7.3f}s  ({100*t_step/total:5.1f}%)  {N/t_step:7.1f}/s")
print(f"[DEBUG-perf]   TOTAL steps/s      : {N/total:7.1f}/s")

# cProfile top hotspots -------------------------------------------------------
def run():
    o = env.reset(seed=0)[0]
    for i in range(N):
        cf.candidates(env.state, 0)
        env.step(1)
        if env.state is None:
            env.reset(seed=i)

pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
print("[DEBUG-perf] top cumulative:")
print("\n".join(s.getvalue().splitlines()[:28]))
