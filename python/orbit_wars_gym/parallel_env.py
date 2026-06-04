from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .action_decoder import decode_discrete_action
from .backend import RustBatchBackend, RustConfig
from .encoding import EncoderConfig, encode_state, observation_dim


class OrbitWarsParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv for simultaneous multiagent self-play."""

    metadata = {"name": "orbit_wars_parallel_v0"}

    def __init__(self, num_players: int = 2, seed: int = 0, encoder_cfg: EncoderConfig | None = None, rust_cfg: RustConfig | None = None):
        self.num_players = num_players
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]
        self.encoder_cfg = encoder_cfg or EncoderConfig()
        self.backend = RustBatchBackend(num_envs=1, num_players=num_players, seed=seed, config=rust_cfg)
        self.state: dict[str, Any] | None = None

    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(observation_dim(self.encoder_cfg),), dtype=np.float32)

    def action_space(self, agent):
        return spaces.MultiDiscrete([16, 32, 4, 5])

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.agents = self.possible_agents[:]
        self.state = self.backend.reset(0 if seed is None else seed)[0]
        observations = {agent: encode_state(self.state, i, self.encoder_cfg) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        assert self.state is not None
        packed = [[[] for _ in range(self.num_players)]]
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                packed[0][i] = decode_discrete_action(self.state, i, actions[agent])
        outcomes, states = self.backend.step_with_states(packed)
        self.state = states[0]
        done = bool(outcomes[0]["done"])
        rewards = {agent: float(outcomes[0]["rewards"][i]) if done else 0.0 for i, agent in enumerate(self.possible_agents)}
        terminations = {agent: done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        observations = {agent: encode_state(self.state, i, self.encoder_cfg) for i, agent in enumerate(self.possible_agents)}
        infos = {agent: {"scores": outcomes[0].get("scores", [])} for agent in self.possible_agents}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos
