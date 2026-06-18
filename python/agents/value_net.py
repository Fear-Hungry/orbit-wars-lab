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


class AttnValueNet(nn.Module):
    """H11 — value net that ATTENDS over the combined {planets ∪ fleets} entity
    set, so it can see PAIRWISE threat (which fleet threatens which planet) — the
    structure masked-mean pooling is blind to (H7/E5 failure DB 166: the mean-pool
    net could not differentiate hold/launch/defend in 4p and bled deaths).

    One multi-head self-attention layer over all entities (presence-masked, with a
    learned planet/fleet type embedding), residual+norm, then masked-mean pool each
    type into the same trunk as :class:`EntityValueNet`. Drop-in: same flat
    ``OBS_DIM`` input, scalar ``tanh`` output, CPU-only inference. One layer keeps
    the per-call forward cheap enough for the in-search PGS budget (actTimeout 1s)."""

    def __init__(self, entity_hidden: int = 64, hidden: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.planet_mlp = nn.Sequential(
            nn.Linear(PLANET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        self.fleet_mlp = nn.Sequential(
            nn.Linear(FLEET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        self.type_emb = nn.Embedding(2, entity_hidden)  # 0=planet, 1=fleet
        self.attn = nn.MultiheadAttention(entity_hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(entity_hidden)
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
        b = obs.shape[0]
        glob = obs[:, :GLOBAL_F]
        p_end = GLOBAL_F + PLANET_N * PLANET_F
        planets = obs[:, GLOBAL_F:p_end].reshape(b, PLANET_N, PLANET_F)
        fleets = obs[:, p_end:].reshape(b, FLEET_N, FLEET_F)
        p_present = planets[:, :, 0]
        f_present = fleets[:, :, 0]
        p_emb = self.planet_mlp(planets) + self.type_emb.weight[0]
        f_emb = self.fleet_mlp(fleets) + self.type_emb.weight[1]
        ent = torch.cat([p_emb, f_emb], dim=1)               # (B, P+F, H)
        present = torch.cat([p_present, f_present], dim=1)    # (B, P+F)
        key_pad = present < 0.5                               # True = ignore (absent)
        attended, _ = self.attn(ent, ent, ent, key_padding_mask=key_pad, need_weights=False)
        ent = self.norm(ent + attended)
        planet_pool = self._masked_mean(ent[:, :PLANET_N, :], p_present)
        fleet_pool = self._masked_mean(ent[:, PLANET_N:, :], f_present)
        h = self.trunk(torch.cat([glob, planet_pool, fleet_pool], dim=-1))
        return torch.tanh(self.value(h).squeeze(-1))


def build_value_net(arch: str = "mean", **kw) -> nn.Module:
    """Construct a value net by architecture tag: 'mean' (EntityValueNet, the H7
    masked-mean baseline) or 'attn' (AttnValueNet, the H11 pairwise-threat fix)."""
    if arch == "attn":
        return AttnValueNet(**kw)
    if arch == "mean":
        return EntityValueNet(**kw)
    raise ValueError(f"unknown value-net arch {arch!r}")


def load_value_net(path: str, device: str = "cpu") -> nn.Module:
    sd = torch.load(path, map_location=device)
    # backward-compatible: old checkpoints are bare state_dicts / {"model": ...}
    # of the mean-pool EntityValueNet (no "arch" key).
    arch = sd.get("arch", "mean") if isinstance(sd, dict) else "mean"
    net = build_value_net(arch)
    net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    net.to(device).eval()
    return net


@torch.no_grad()
def value_of_states(net: EntityValueNet, obs_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Convenience: (N, OBS_DIM) numpy -> (N,) values."""
    t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    return net(t).cpu().numpy()
