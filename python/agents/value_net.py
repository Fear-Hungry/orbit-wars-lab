"""H7 — learned state VALUE net (E1). Replaces the hand-coded ``_plan_value``
inside the PGS search; defense/4p-survival emerge from valuing positions, not
margin-at-H (diagnosis DB 118/166).

Reuses the proven permutation-invariant ENTITY encoder of
:class:`python.agents.policy.EntityActorCritic` (planet/fleet per-entity MLP +
masked-mean pool + trunk), but keeps ONLY a value head. Consumes the same flat
observation as the policy (``orbit_wars_gym.encoding.encode_state`` → dim
``GLOBAL_F + PLANET_N*PLANET_F + FLEET_N*FLEET_F``), which is perspective-aware
(self/enemy/neutral/OTHER), so it works for 2p AND 4p out of the box.

Output is ``tanh``-bounded to [-1, 1] — the win/lose/tie target from the acting
player's perspective. CPU-only inference (submission invariant D10/D11).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from python.agents.policy import FLEET_F, FLEET_N, GLOBAL_F, PLANET_F, PLANET_N

OBS_DIM = GLOBAL_F + PLANET_N * PLANET_F + FLEET_N * FLEET_F  # 3912


class EntityValueNet(nn.Module):
    """State -> scalar value in [-1, 1] from the acting player's perspective."""

    def __init__(self, entity_hidden: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.planet_mlp = nn.Sequential(
            nn.Linear(PLANET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        self.fleet_mlp = nn.Sequential(
            nn.Linear(FLEET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(GLOBAL_F + 2 * entity_hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.value = nn.Linear(hidden, 1)

    @staticmethod
    def _masked_mean(emb: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        weight = present.unsqueeze(-1)
        total = (emb * weight).sum(dim=1)
        count = weight.sum(dim=1).clamp_min(1.0)
        return total / count

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, OBS_DIM) flat. Returns (B,) value in [-1, 1]."""
        b = obs.shape[0]
        glob = obs[:, :GLOBAL_F]
        p_end = GLOBAL_F + PLANET_N * PLANET_F
        planets = obs[:, GLOBAL_F:p_end].reshape(b, PLANET_N, PLANET_F)
        fleets = obs[:, p_end:].reshape(b, FLEET_N, FLEET_F)
        planet_pool = self._masked_mean(self.planet_mlp(planets), planets[:, :, 0])
        fleet_pool = self._masked_mean(self.fleet_mlp(fleets), fleets[:, :, 0])
        h = self.trunk(torch.cat([glob, planet_pool, fleet_pool], dim=-1))
        return torch.tanh(self.value(h).squeeze(-1))


def load_value_net(path: str, device: str = "cpu") -> EntityValueNet:
    net = EntityValueNet()
    sd = torch.load(path, map_location=device)
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.to(device).eval()
    return net


@torch.no_grad()
def value_of_states(net: EntityValueNet, obs_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Convenience: (N, OBS_DIM) numpy -> (N,) values."""
    t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    return net(t).cpu().numpy()
