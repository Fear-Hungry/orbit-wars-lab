from __future__ import annotations

from orbit_wars_gym.action_decoder import greedy_moves
from orbit_wars_gym.backend import RustBatchBackend


def main():
    sim = RustBatchBackend(num_envs=2, num_players=2, seed=0)
    states = sim.reset(123)
    print("initial_states", len(states), "planets", len(states[0]["planets"]))
    for _ in range(5):
        actions = []
        for state in states:
            actions.append([greedy_moves(state, 0), greedy_moves(state, 1)])
        outcomes = sim.step(actions)
        states = sim.states()
    print("last_outcomes", outcomes)


if __name__ == "__main__":
    main()
