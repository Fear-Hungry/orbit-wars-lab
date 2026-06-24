"""MCTS (PUCT / AlphaZero-style) over the GridNet policy + value net.

The shallow lookahead (fixed depth, k candidates, pick-best-by-value) did not cross
the reactive→planner cliff. A real MCTS builds a SELECTIVE search tree: PUCT
balances the policy prior, the value estimate, and exploration, expanding promising
lines much deeper than fixed-depth lookahead. Each node caches its engine state
(reset_from_states); expansion samples k actions from the policy as priors; leaves
are evaluated by the value net; values back up the path. If a heavy MCTS beats the
producer where the bare policy and shallow lookahead are annihilated, deep search —
the AlphaZero ingredient — is the missing piece (then it'd be worth the training
loop). If it stalls too, the combinatorial branching + reactive prior is the wall.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from python.agents.policy import GridNetActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.action_decoder import decode_gridnet_action, gridnet_planet_mask
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import DEFAULT_ENCODER_CONFIG, encode_state, observation_dim


class _Node:
    __slots__ = ("state", "actions", "priors", "N", "W", "children", "value")

    def __init__(self, state):
        self.state = state
        self.actions: list = []      # k sampled per-planet actions
        self.priors: list = []       # policy prior per action
        self.N: list = []            # visit count per action
        self.W: list = []            # value sum per action
        self.children: dict = {}     # action idx -> _Node
        self.value = 0.0


def _value_of(model, states, player, device):
    obs = torch.as_tensor(
        np.stack([encode_state(s, player, DEFAULT_ENCODER_CONFIG) for s in states]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        return model.forward(obs)["value"].cpu().numpy()


def _expand(node, model, player, k, device):
    obs = torch.as_tensor(encode_state(node.state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32, device=device).unsqueeze(0).repeat(k, 1)
    mask = torch.as_tensor(gridnet_planet_mask(node.state, player), dtype=torch.bool, device=device).unsqueeze(0).repeat(k, 1)
    with torch.no_grad():
        a, lp, _, v = model.get_action_and_value(obs, masks={"planet": mask})
    node.actions = list(a.cpu().numpy())
    node.priors = list(np.exp(lp.cpu().numpy()))  # ~relative prior
    tot = sum(node.priors) or 1.0
    node.priors = [p / tot for p in node.priors]
    node.N = [0] * k
    node.W = [0.0] * k
    node.value = float(v[0].item())


def _puct(node, c):
    total = sum(node.N) + 1
    best, best_score = 0, -1e9
    for i in range(len(node.actions)):
        q = node.W[i] / node.N[i] if node.N[i] > 0 else 0.0
        u = c * node.priors[i] * math.sqrt(total) / (1 + node.N[i])
        if q + u > best_score:
            best_score, best = q + u, i
    return best


def _simulate_step(sim, state, action, player, opp_model, opp):
    sim.reset_from_states([state])
    rows = [[0.0, float(player), float(m[0]), float(m[1]), float(m[2])] for m in decode_gridnet_action(state, player, action)]
    for m in opp_model(state, opp):
        rows.append([0.0, float(opp), float(m[0]), float(m[1]), float(m[2])])
    arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
    _, st = sim.step_flat_with_states(arr)
    return st[0]


def mcts_action(model, root_state, player, *, opp_model, n_sim, k, c_puct, max_depth, sim, device):
    root = _Node(root_state)
    _expand(root, model, player, k, device)
    opp = 1 - player
    for _ in range(n_sim):
        node, path, depth = root, [], 0
        while True:
            ai = _puct(node, c_puct)
            path.append((node, ai))
            if ai not in node.children:
                child_state = _simulate_step(sim, node.state, node.actions[ai], player, opp_model, opp)
                child = _Node(child_state)
                node.children[ai] = child
                _expand(child, model, player, k, device)
                value = child.value
                break
            node = node.children[ai]
            depth += 1
            if depth >= max_depth:
                value = node.value
                break
        for n, ai in path:
            n.N[ai] += 1
            n.W[ai] += value
    return root.actions[int(np.argmax(root.N))]


def play(model, opponent_name, seat, seed, *, n_sim, k, c_puct, max_depth, steps, device, model_opp=True):
    b = RustBatchBackend(num_envs=1, num_players=2, seed=seed, config=RustConfig(episode_steps=steps, enable_comets=True))
    s = b.reset(seed)[0]
    real_opp = make_isolated_opponent(opponent_name)
    sim_opp = make_isolated_opponent(opponent_name) if model_opp else (lambda st, p: decode_gridnet_action(st, p, _greedy(model, st, p, device)))
    sim = RustBatchBackend(num_envs=1, num_players=2, seed=0, config=RustConfig(enable_comets=True))
    last = [0.0, 0.0]
    op = 1 - seat
    for _ in range(steps):
        a = mcts_action(model, s, seat, opp_model=sim_opp, n_sim=n_sim, k=k, c_puct=c_puct, max_depth=max_depth, sim=sim, device=device)
        rows = [[0.0, float(seat), float(m[0]), float(m[1]), float(m[2])] for m in decode_gridnet_action(s, seat, a)]
        rows += [[0.0, float(op), float(m[0]), float(m[1]), float(m[2])] for m in real_opp(s, op)]
        arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        out, st = b.step_flat_with_states(arr)
        s = st[0]
        sc = out[0].get("scores") or out[0].get("rewards")
        if sc:
            last = list(sc)
        if out[0].get("done"):
            break
    s0, s1 = float(last[0]), float(last[1])
    d = max(abs(s0) + abs(s1), 1.0)
    return (s0 - s1) / d if seat == 0 else (s1 - s0) / d


def _greedy(model, state, player, device):
    mask = torch.as_tensor(gridnet_planet_mask(state, player), dtype=torch.bool, device=device).unsqueeze(0)
    obs = torch.as_tensor(encode_state(state, player, DEFAULT_ENCODER_CONFIG), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.forward(obs)
        launch = torch.where(mask, out["launch"].argmax(-1), torch.zeros_like(out["launch"].argmax(-1)))
        return torch.stack([launch, out["target"].argmax(-1), out["frac"].argmax(-1), out["offset"].argmax(-1)], dim=-1)[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="artifacts/bc/gridnet_bc_big512.pt")
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--n-sim", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    sm = ck["summary"]
    model = GridNetActorCritic(observation_dim(), entity_hidden=sm["entity_hidden"], hidden=sm["hidden"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    ms = []
    for sd in range(args.seeds):
        ms.append(0.5 * (play(model, args.opponent, 0, sd, n_sim=args.n_sim, k=args.k, c_puct=args.c_puct, max_depth=args.max_depth, steps=args.steps, device=dev)
                         + play(model, args.opponent, 1, sd, n_sim=args.n_sim, k=args.k, c_puct=args.c_puct, max_depth=args.max_depth, steps=args.steps, device=dev)))
    print(f"GridNet+MCTS(n_sim={args.n_sim},k={args.k},d={args.max_depth}) vs {args.opponent}: margem={np.mean(ms):+.3f}")


if __name__ == "__main__":
    main()
