from __future__ import annotations

import argparse
import time

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.backend import RustBatchBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-players", type=int, default=2)
    args = parser.parse_args()

    sim = RustBatchBackend(num_envs=args.num_envs, num_players=args.num_players, seed=0)
    states = sim.reset(42)
    t0 = time.perf_counter()
    turns = 0
    for _ in range(args.steps):
        actions = []
        for state in states:
            actions.append([greedy_moves(state, pid) for pid in range(args.num_players)])
        _, states = sim.step_with_states(actions)
        turns += args.num_envs
    dt = time.perf_counter() - t0
    print({"envs": args.num_envs, "steps": args.steps, "turns": turns, "seconds": dt, "turns_per_second": turns / max(dt, 1e-9)})


if __name__ == "__main__":
    main()
