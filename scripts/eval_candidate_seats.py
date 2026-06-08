"""Seat-ALTERNATED eval of a candidate_selector policy (honest margin, no player-0 bias).

eval_candidate_selector fixes the policy as player 0, which carries a large player-0
seat advantage (always-producer scored +0.38 there despite being a pure Producer mirror
= 0 seat-neutral). This runs each game in BOTH seatings (policy as p0 AND p1) and averages,
cancelling the seat bias — the real margin vs the opponent.
"""
import argparse, json
from statistics import fmean
import torch
from python.agents.candidate_factory import CandidateFactory
from python.agents.policy import CandidateSelectorActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import EncoderConfig, encode_state, observation_dim
from python.orbit_wars_gym.rules import moves_are_legal, normalized_margin


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CandidateSelectorActorCritic(observation_dim()); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def _game(model, opponent, seed, policy_seat, steps, enc):
    factory = CandidateFactory()
    opp = make_isolated_opponent(opponent)
    be = RustBatchBackend(num_envs=1, num_players=2, seed=seed,
                          config=RustConfig(episode_steps=steps, enable_comets=True))
    state = be.reset(seed)[0]; scores = [0.0, 0.0]
    for _ in range(steps + 1):
        acts = [[], []]
        for seat in range(2):
            if seat == policy_seat:
                cands = factory.candidates(state, seat)
                obs = torch.as_tensor(encode_state(state, player=seat, cfg=enc), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    idx = int(model.forward(obs)["candidate"].argmax(-1).item())
                mv = cands[idx]["moves"]
            else:
                mv = opp(state, seat)
                if not isinstance(mv, list) or not moves_are_legal(state, seat, mv):
                    mv = []
            acts[seat] = mv
        outs, states = be.step_with_states([acts]); state = states[0]
        if outs[0].get("scores"):
            scores = [float(x) for x in outs[0]["scores"]]
        if outs[0]["done"]:
            break
    return normalized_margin(scores, policy_seat)


def eval_vs(model, opponent, seeds, steps):
    enc = EncoderConfig(); margins = []
    for seed in range(seeds):
        for seat in (0, 1):  # both seatings -> cancels seat bias
            margins.append(_game(model, opponent, seed, seat, steps, enc))
    return fmean(margins)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponents", nargs="+", default=["producer", "oep"])
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--episode-steps", type=int, default=256)
    a = ap.parse_args(); model = _load(a.checkpoint)
    for opp in a.opponents:
        print(f"{opp:9s}: seat-neutral margin = {eval_vs(model, opp, a.seeds, a.episode_steps):+.4f}")
    # sanity: always-producer mirror should be ~0 seat-neutral
