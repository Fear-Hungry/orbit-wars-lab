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
        keep_init_bias: float = 5.0,
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
        # The argmax is KEEP for ANY keep_init_bias > 0 (weights zeroed), so the
        # parity floor holds regardless of magnitude; a SMALLER bias only softens the
        # sampling temperature so PPO can explore (and escape to) edits faster — a
        # bias of 5.0 makes KEEP ~97% of samples, so edits never get enough signal.
        with torch.no_grad():
            self.edit.weight.zero_()
            bias = self.edit.bias.view(self.k_max, self.n_edit)
            bias.zero_()
            bias[:, 0] = float(keep_init_bias)

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


# ===================== GridNet per-planet policy (GN1) =====================
# GridNet (Huang & Ontañón 2021, arXiv:2105.13807): predict an action for EVERY
# planet slot in one forward, then a per-planet mask keeps only the player's
# launchable planets active. This replaces the single global (source,target,frac,
# offset) tuple — whose shared ranks homogenised every source (the target head
# collapsed to 0.04-0.06 acc, killing the direct decoder) — with INDEPENDENT
# per-planet decisions, so multi-source plans (hammer) are representable. Encoder-
# decoder form (the ablation's GridNet-favourable arch): per-planet embedding +
# pooled global context -> per-planet heads.
GN_TARGET_N = 32
GN_FRAC_N = 4
GN_OFFSET_N = 5
# per-planet logits: launch(2) + target + frac + offset
GN_PER_PLANET_LOGITS = 2 + GN_TARGET_N + GN_FRAC_N + GN_OFFSET_N


class GridNetActorCritic(nn.Module):
    """Per-planet launch policy with full per-planet masking.

    Action is ``(B, PLANET_N, 4)``: each planet slot emits
    ``[launch, target_rank, frac_idx, offset_idx]``. ``masks['planet']`` is
    ``(B, PLANET_N)`` marking the player's launchable planets; inactive slots are
    forced to launch=0 (no-op) and contribute 0 to log-prob/entropy. The move
    sub-heads (target/frac/offset) of a slot count only when that slot launches —
    a per-planet version of the launch gating. The SAME mask must be passed at
    sampling and update for the PPO ratio to stay correct.
    """

    def __init__(
        self,
        obs_dim: int,
        target_n: int = GN_TARGET_N,
        frac_n: int = GN_FRAC_N,
        offset_n: int = GN_OFFSET_N,
        entity_hidden: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        expected = GLOBAL_F + PLANET_N * PLANET_F + FLEET_N * FLEET_F
        if obs_dim != expected:
            raise ValueError(f"GridNetActorCritic expects flat obs_dim {expected}, got {obs_dim}")
        self.target_n = int(target_n)
        self.frac_n = int(frac_n)
        self.offset_n = int(offset_n)
        self.per_planet_logits = 2 + self.target_n + self.frac_n + self.offset_n
        self.planet_mlp = nn.Sequential(
            nn.Linear(PLANET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        self.fleet_mlp = nn.Sequential(
            nn.Linear(FLEET_F, entity_hidden), nn.Tanh(),
            nn.Linear(entity_hidden, entity_hidden), nn.Tanh(),
        )
        # global context (encoder): pooled planets+fleets+globals
        self.trunk = nn.Sequential(
            nn.Linear(GLOBAL_F + 2 * entity_hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # per-planet decoder: planet embedding + broadcast global context -> heads
        self.planet_decoder = nn.Sequential(
            nn.Linear(entity_hidden + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.per_planet_logits),
        )
        self.value = nn.Linear(hidden, 1)

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
        planet_emb = self.planet_mlp(planets)  # (B, PLANET_N, H)
        planet_pool = self._masked_mean(planet_emb, planets[:, :, 0])
        fleet_pool = self._masked_mean(self.fleet_mlp(fleets), fleets[:, :, 0])
        ctx = self.trunk(torch.cat([glob, planet_pool, fleet_pool], dim=-1))  # (B, hidden)
        ctx_b = ctx.unsqueeze(1).expand(-1, PLANET_N, -1)  # (B, PLANET_N, hidden)
        per_planet = self.planet_decoder(torch.cat([planet_emb, ctx_b], dim=-1))  # (B, PLANET_N, L)
        t, f, o = self.target_n, self.frac_n, self.offset_n
        return {
            "launch": per_planet[:, :, 0:2],
            "target": per_planet[:, :, 2:2 + t],
            "frac": per_planet[:, :, 2 + t:2 + t + f],
            "offset": per_planet[:, :, 2 + t + f:2 + t + f + o],
            "value": self.value(ctx).squeeze(-1),
        }

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ):
        """Per-planet multi-discrete. ``masks['planet']`` (B, PLANET_N) marks active
        planets. Returns ``(action[B,PLANET_N,4], logprob[B], entropy[B], value[B])``."""
        out = self.forward(obs)
        b = out["launch"].shape[0]
        if masks is not None and "planet" in masks:
            active = masks["planet"].bool()
        else:
            active = torch.ones(b, PLANET_N, dtype=torch.bool, device=out["launch"].device)
        active_f = active.to(out["value"].dtype)

        launch_d = Categorical(logits=out["launch"])
        target_d = Categorical(logits=out["target"])
        frac_d = Categorical(logits=out["frac"])
        offset_d = Categorical(logits=out["offset"])
        if action is None:
            launch_a = torch.where(active, launch_d.sample(), torch.zeros(b, PLANET_N, dtype=torch.long, device=active.device))
            target_a = target_d.sample()
            frac_a = frac_d.sample()
            offset_a = offset_d.sample()
            action = torch.stack([launch_a, target_a, frac_a, offset_a], dim=-1)
        a = action.long()
        launch_a, target_a, frac_a, offset_a = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        # launched gate per planet: move sub-heads count only where launch==1 AND active
        launched = (launch_a == 1).to(out["value"].dtype) * active_f
        lp = launch_d.log_prob(launch_a) * active_f
        lp = lp + (target_d.log_prob(target_a) + frac_d.log_prob(frac_a) + offset_d.log_prob(offset_a)) * launched
        ent = launch_d.entropy() * active_f
        ent = ent + (target_d.entropy() + frac_d.entropy() + offset_d.entropy()) * launched
        return a, lp.sum(-1), ent.sum(-1), out["value"]


def gridnet_gated_kl(
    cur_out: dict[str, torch.Tensor],
    ref_out: dict[str, torch.Tensor],
    planet_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-planet KL(cur||ref) for the GridNet policy, gated like the rollout/loss:
    KL(launch) + P_cur(launch=1)*Σ KL(move heads), summed over active planets.
    Anchors PPO to a reference (BC) so RL can't degrade the good BC minimum
    (AlphaStar/RLHF device). ``planet_mask`` is (B, PLANET_N) bool."""
    active = planet_mask.to(cur_out["value"].dtype)

    def _kl(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        clp = torch.log_softmax(c, dim=-1)
        rlp = torch.log_softmax(r, dim=-1)
        return (clp.exp() * (clp - rlp)).sum(-1)

    launch_kl = _kl(cur_out["launch"], ref_out["launch"])  # (B, N)
    p_launch = torch.softmax(cur_out["launch"], dim=-1)[..., 1]  # (B, N)
    move_kl = (
        _kl(cur_out["target"], ref_out["target"])
        + _kl(cur_out["frac"], ref_out["frac"])
        + _kl(cur_out["offset"], ref_out["offset"])
    )
    per_planet = launch_kl + p_launch * move_kl
    return (per_planet * active).sum(-1)  # (B,)
