from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FlatActorCritic(nn.Module):
    """First baseline policy.

    This is intentionally small. Replace with entity transformer after the flat
    pipeline proves that the simulator, reward and decoder are sane.
    """

    def __init__(self, obs_dim: int, source_n: int = 16, target_n: int = 32, frac_n: int = 4, offset_n: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.source = nn.Linear(256, source_n)
        self.target = nn.Linear(256, target_n)
        self.frac = nn.Linear(256, frac_n)
        self.offset = nn.Linear(256, offset_n)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        return {
            "source": self.source(h),
            "target": self.target(h),
            "frac": self.frac(h),
            "offset": self.offset(h),
            "value": self.value(h).squeeze(-1),
        }

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        out = self.forward(obs)
        dists = [Categorical(logits=out[k]) for k in ("source", "target", "frac", "offset")]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=-1)
        logprob = torch.stack([d.log_prob(action[:, i]) for i, d in enumerate(dists)], dim=-1).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return action, logprob, entropy, out["value"]
