from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Action layout: [launch, source, target, frac, offset]. The leading binary
# ``launch`` head gates whether a turn launches at all; experts pass ~60-81% of
# turns (see todo P1.5), which the move-only space could not represent. The move
# heads only contribute to log-prob / entropy when launch == 1, so
# ``P(pass) = P(launch=0)`` is independent of source/target/frac/offset.
MOVE_KEYS = ("source", "target", "frac", "offset")

# Flat-observation layout (see orbit_wars_gym.encoding): 8 global features, then
# ``PLANET_N`` planet rows of ``PLANET_F`` and ``FLEET_N`` fleet rows of
# ``FLEET_F``. The first element of each entity row is a 1.0/0.0 presence flag.
GLOBAL_F = 8
PLANET_N, PLANET_F = 96, 14
FLEET_N, FLEET_F = 256, 10


def _masked_categoricals(
    out: dict[str, torch.Tensor], masks: dict[str, torch.Tensor] | None
) -> tuple[Categorical, list[Categorical]]:
    """Build the launch + move Categoricals with the action mask applied.

    The same mask must be supplied at sampling and update time so the PPO
    importance ratio stays correct (see action_masks).
    """

    def _logits(key: str) -> torch.Tensor:
        logits = out[key]
        if masks is not None and key in masks:
            logits = logits.masked_fill(~masks[key].bool(), float("-inf"))
        return logits

    launch_dist = Categorical(logits=_logits("launch"))
    move_dists = [Categorical(logits=_logits(k)) for k in MOVE_KEYS]
    return launch_dist, move_dists


def _safe_categorical_kl(cur: Categorical, ref: Categorical) -> torch.Tensor:
    """KL(cur || ref) per row, robust to masked positions.

    cur/ref .logits are log-softmax with ``-inf`` at masked positions, so the raw
    ``logp - logq`` is ``(-inf) - (-inf) = nan`` there. We zero the diff with
    ``where`` BEFORE multiplying by ``p`` — zeroing only the final term still leaks
    a ``0 * nan = nan`` gradient into ``p`` (and thus the logits). Cleaning ``diff``
    first keeps both the value and the gradient finite at masked positions.
    """
    diff = cur.logits - ref.logits
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    return (cur.probs * diff).sum(-1)


def launch_gated_kl(
    cur_out: dict[str, torch.Tensor],
    ref_out: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Per-sample KL(cur || ref) for the launch-gated multi-discrete policy.

    Mirrors the entropy gating (line below): ``KL(launch) + P_cur(launch=1) *
    Σ KL(move)`` — the move heads only matter when the turn actually launches.
    Both policies must share the same ``masks``. Used as an anti-drift anchor to
    a reference (e.g. BC-init) policy during PPO fine-tuning.
    """
    cur_launch, cur_moves = _masked_categoricals(cur_out, masks)
    ref_launch, ref_moves = _masked_categoricals(ref_out, masks)
    kl = _safe_categorical_kl(cur_launch, ref_launch)
    p_launch = cur_launch.probs[:, 1]
    move_kl = torch.stack(
        [_safe_categorical_kl(c, r) for c, r in zip(cur_moves, ref_moves, strict=True)], dim=-1
    ).sum(-1)
    return kl + p_launch * move_kl


def _action_value_from_heads(
    out: dict[str, torch.Tensor],
    action: torch.Tensor | None,
    masks: dict[str, torch.Tensor] | None,
):
    """Shared launch-gated sampling/log-prob/entropy used by every policy head set."""
    launch_dist, move_dists = _masked_categoricals(out, masks)
    if action is None:
        launch_a = launch_dist.sample()
        move_a = torch.stack([d.sample() for d in move_dists], dim=-1)
        action = torch.cat([launch_a.unsqueeze(-1), move_a], dim=-1)

    launch_a = action[:, 0]
    is_launch = launch_a.to(out["value"].dtype)
    launch_logprob = launch_dist.log_prob(launch_a)
    move_logprob = torch.stack(
        [d.log_prob(action[:, i + 1]) for i, d in enumerate(move_dists)], dim=-1
    ).sum(-1)
    logprob = launch_logprob + is_launch * move_logprob

    p_launch = launch_dist.probs[:, 1]
    move_entropy = torch.stack([d.entropy() for d in move_dists], dim=-1).sum(-1)
    entropy = launch_dist.entropy() + p_launch * move_entropy
    return action, logprob, entropy, out["value"]


class _Heads(nn.Module):
    """Launch + move + value heads on top of a fixed-width trunk feature."""

    def __init__(self, hidden: int, source_n: int, target_n: int, frac_n: int, offset_n: int):
        super().__init__()
        self.launch = nn.Linear(hidden, 2)
        self.source = nn.Linear(hidden, source_n)
        self.target = nn.Linear(hidden, target_n)
        self.frac = nn.Linear(hidden, frac_n)
        self.offset = nn.Linear(hidden, offset_n)
        self.value = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "launch": self.launch(h),
            "source": self.source(h),
            "target": self.target(h),
            "frac": self.frac(h),
            "offset": self.offset(h),
            "value": self.value(h).squeeze(-1),
        }


class FlatActorCritic(nn.Module):
    """First baseline policy.

    This is intentionally small. Replace with entity transformer after the flat
    pipeline proves that the simulator, reward and decoder are sane.
    """

    # State-dict keys (net.0.*, launch.*, source.*, ...) are load-bearing: the
    # submission exporter reads them by name, so they must not be renamed.
    MOVE_KEYS = MOVE_KEYS

    def __init__(self, obs_dim: int, source_n: int = 16, target_n: int = 32, frac_n: int = 4, offset_n: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.launch = nn.Linear(256, 2)
        self.source = nn.Linear(256, source_n)
        self.target = nn.Linear(256, target_n)
        self.frac = nn.Linear(256, frac_n)
        self.offset = nn.Linear(256, offset_n)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        return {
            "launch": self.launch(h),
            "source": self.source(h),
            "target": self.target(h),
            "frac": self.frac(h),
            "offset": self.offset(h),
            "value": self.value(h).squeeze(-1),
        }

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ):
        return _action_value_from_heads(self.forward(obs), action, masks)


class EntityActorCritic(nn.Module):
    """Permutation-invariant entity encoder (todo T4).

    Consumes the SAME flat observation as :class:`FlatActorCritic` but reshapes it
    back into per-planet / per-fleet rows, applies a shared per-entity MLP, and
    masked-mean-pools each entity type (using the row's presence flag) before the
    trunk + heads. Pooling makes the encoder invariant to entity order/slot index
    — one of the asymmetries flagged in P5 — without changing the dataset, the
    decoder, or the action space, so it can be compared head-to-head with the flat
    baseline on the same behavioral-cloning data.
    """

    MOVE_KEYS = MOVE_KEYS

    def __init__(
        self,
        obs_dim: int,
        source_n: int = 16,
        target_n: int = 32,
        frac_n: int = 4,
        offset_n: int = 5,
        entity_hidden: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        expected = GLOBAL_F + PLANET_N * PLANET_F + FLEET_N * FLEET_F
        if obs_dim != expected:
            raise ValueError(f"EntityActorCritic expects flat obs_dim {expected}, got {obs_dim}")
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
        self.heads = _Heads(hidden, source_n, target_n, frac_n, offset_n)

    @staticmethod
    def _masked_mean(emb: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # emb: (B, N, H); present: (B, N) in {0,1}. Mean over present entities.
        weight = present.unsqueeze(-1)
        total = (emb * weight).sum(dim=1)
        count = weight.sum(dim=1).clamp_min(1.0)
        return total / count

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        b = obs.shape[0]
        glob = obs[:, :GLOBAL_F]
        p_end = GLOBAL_F + PLANET_N * PLANET_F
        planets = obs[:, GLOBAL_F:p_end].reshape(b, PLANET_N, PLANET_F)
        fleets = obs[:, p_end:].reshape(b, FLEET_N, FLEET_F)
        planet_pool = self._masked_mean(self.planet_mlp(planets), planets[:, :, 0])
        fleet_pool = self._masked_mean(self.fleet_mlp(fleets), fleets[:, :, 0])
        h = self.trunk(torch.cat([glob, planet_pool, fleet_pool], dim=-1))
        return self.heads(h)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ):
        return _action_value_from_heads(self.forward(obs), action, masks)


class ProducerResidualBranchActorCritic(nn.Module):
    """BReP — batchable per-slot branching edits over a BASE plan (Tavakoli 2017).

    Ported from commit 19f0cf0 (the original BReP over Producer). Reuses the entity
    encoder but replaces the move heads with ``k_max`` INDEPENDENT ``Discrete(n_edit)``
    branches, one per BASE-move slot. Edit 0 = KEEP reproduces the base move EXACTLY,
    so a KEEP-initialised policy == base agent == guaranteed parity FLOOR; PPO can
    only learn BENEFICIAL deviations (cancel/scale). The base plan is computed once
    per env in the batched rollout; the net runs ZERO experts.

    This worktree generalises the BASE from Producer to ANY ``(state, player) -> moves``
    agent (e.g. the holdwave PGS) via ``Phase0TrainingConfig.base_agent`` — the arch
    here is base-agnostic (it only sizes the per-slot branches), the base choice lives
    in the rollout decoder ``_apply_residual_edits``.

    State-dict keys (planet_mlp/fleet_mlp/trunk/edit/value) are its OWN — a distinct
    arch, so it does not collide with EntityActorCritic's exporter keys.
    """

    # Per-slot edit codes over the base move (v2): 0=KEEP, 1=CANCEL, ship-SCALE
    # {2:x0.25, 3:x0.5, 4:x1.5, 5:x2.0}. The decoder (_apply_residual_edits) owns it.
    N_EDIT = 6

    def __init__(
        self,
        obs_dim: int,
        k_max: int = 16,
        n_edit: int = N_EDIT,
        entity_hidden: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        expected = GLOBAL_F + PLANET_N * PLANET_F + FLEET_N * FLEET_F
        if obs_dim != expected:
            raise ValueError(
                f"ProducerResidualBranchActorCritic expects flat obs_dim {expected}, got {obs_dim}"
            )
        self.k_max = int(k_max)
        self.n_edit = int(n_edit)
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
        self.edit = nn.Linear(hidden, self.k_max * self.n_edit)
        self.value = nn.Linear(hidden, 1)
        # KEEP-init: zero the edit head so the untrained output is dominated by a
        # high bias on KEEP (index 0) → every slot KEEPs → exact base = parity floor.
        with torch.no_grad():
            self.edit.weight.zero_()
            bias = self.edit.bias.view(self.k_max, self.n_edit)
            bias.zero_()
            bias[:, 0] = 5.0

    @staticmethod
    def _masked_mean(emb: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        weight = present.unsqueeze(-1)
        total = (emb * weight).sum(dim=1)
        count = weight.sum(dim=1).clamp_min(1.0)
        return total / count

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        b = obs.shape[0]
        glob = obs[:, :GLOBAL_F]
        p_end = GLOBAL_F + PLANET_N * PLANET_F
        planets = obs[:, GLOBAL_F:p_end].reshape(b, PLANET_N, PLANET_F)
        fleets = obs[:, p_end:].reshape(b, FLEET_N, FLEET_F)
        planet_pool = self._masked_mean(self.planet_mlp(planets), planets[:, :, 0])
        fleet_pool = self._masked_mean(self.fleet_mlp(fleets), fleets[:, :, 0])
        h = self.trunk(torch.cat([glob, planet_pool, fleet_pool], dim=-1))
        edit_logits = self.edit(h).reshape(b, self.k_max, self.n_edit)
        return {"edit": edit_logits, "value": self.value(h).squeeze(-1)}

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ):
        """``k_max`` independent Discrete(n_edit) branches. ``masks['edit']`` is
        ``(B, k_max)`` boolean marking ACTIVE slots (= real base moves); inactive
        slots are forced to KEEP(0) and contribute 0 to log-prob/entropy. The SAME
        mask must be passed at sampling and update so the PPO ratio stays correct.
        Returns ``(action[B,k_max], logprob[B], entropy[B], value[B])``."""
        out = self.forward(obs)
        logits = out["edit"]
        b, k, _ = logits.shape
        if masks is not None and "edit" in masks:
            active = masks["edit"].bool()
        else:
            active = torch.ones(b, k, dtype=torch.bool, device=logits.device)
        dist = Categorical(logits=logits)  # batch_shape (B, k_max)
        if action is None:
            sampled = dist.sample()
            action = torch.where(active, sampled, torch.zeros_like(sampled))
        a = action.long()
        active_f = active.to(out["value"].dtype)
        logprob = (dist.log_prob(a) * active_f).sum(-1)
        entropy = (dist.entropy() * active_f).sum(-1)
        return a, logprob, entropy, out["value"]
