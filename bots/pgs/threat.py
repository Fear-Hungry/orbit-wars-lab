"""H9 — per-enemy forward threat readout for 4p (max-n / non-paranoid).

Root cause of the 4p floor (DB id=118, PGS_DBG4P probe): the endpoint value
``margin(ships+territory) at horizon H`` is INSENSITIVE to launch/hold/defend in
4p, because ships are conserved and launches do not RESOLVE (capture/die) by H.
The H7 value-net (DB 166/172) inherited the same blindness via mean-pool over a
single ``enemy`` bucket: it cannot tell which of the three opponents finishes us.

This module does NOT add a learned net. It reads the FULL projected trajectory
(``PlanetGarrisonStatus`` over ticks 0..H) and the engine's per-owner arrival
ledger (``arrivals_by_owner`` [P, H, A], populated when ``track_fleets=True``),
split PER ENEMY. The decisive aggregations are max/min over enemies and over
ticks — never a mean — so the worst single opponent and the worst single tick are
visible instead of averaged away.

Falsified by ``scripts/h9_threat_probe.py``: if these features do NOT separate
launch/hold/defend where margin@H is flat, H9 is dead (escalate to H11 attention,
DB 170) — exactly as H7 was killed at E5 rather than carried forward on faith.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from orbit_lite.movement import PlanetGarrisonStatus


@dataclass(frozen=True)
class ThreatFeatures:
    """Forward, per-enemy 4p threat. All scalars; ready to feed a veto/arbiter.

    ``*_incoming`` are total enemy ships arriving onto planets we currently hold,
    over [0, H). ``min_my_*`` are floors over the trajectory (survival), not the
    endpoint. ``planets_lost_to_one`` / ``time_to_first_loss`` capture *which*
    enemy finishes us and *when* — the signal the endpoint margin throws away.
    """

    n_enemies: float
    total_enemy_incoming: float        # sum over enemies (the OLD sum-pool view; kept for contrast only)
    max_enemy_incoming: float          # worst enemy's incoming on us  <-- decisive (max, not mean)
    second_worst_incoming: float       # two-enemy focus risk
    max_enemy_peak_tick: float         # worst single-tick incoming from the worst enemy
    max_planets_lost_to_one: float     # most of our planets captured by a SINGLE enemy by H
    time_to_first_loss: float          # earliest tick we lose a held planet (H+1 == never)
    min_my_planet_count: float         # min over ticks of planets we own (0 => annihilation path)
    annihilated: float                 # 1.0 if min_my_planet_count hits 0
    min_my_ships: float                # min over ticks of our total ships (survival floor)

    def threat_value(
        self,
        prod_weight: float = 1.0,
        death_penalty: float = 1e4,
        *,
        planet_weight: float = 100.0,
        first_loss_weight: float = 2.0,
        incoming_weight: float = 0.5,
        ships_weight: float = 0.01,
    ) -> float:
        """Trajectory-sensitive SURVIVAL scalar (contrast against margin@H).

        v1 FAILED the 500-step gate (DB 234/235): a large ``min_my_ships`` term made
        the search prefer EVAC/attack (empty the planet to "preserve ships in transit")
        -> lost every planet -> 100% annihilation. SURVIVAL in this game = keeping
        PLANETS alive to the end, not hoarding ships. So territory dominates: holding
        planets through the worst tick and delaying first loss; ships are a tiny
        tie-breaker. Scaled so saving one planet (+100) clears the arbiter margin (~25).

        The four term weights are exposed as keyword args (defaults reproduce the
        shipped v2 scalar exactly) so the research loop can mutate the threat-value
        trade-off without touching the search; see ``scripts/research_loop/``.
        """
        return (
            planet_weight * self.min_my_planet_count
            + first_loss_weight * self.time_to_first_loss
            - incoming_weight * self.max_enemy_incoming
            + ships_weight * self.min_my_ships
            - death_penalty * self.annihilated
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_threat_features(status: PlanetGarrisonStatus, me: int, *, horizon: int | None = None) -> ThreatFeatures:
    """Read per-enemy forward threat from a projected garrison status.

    ``status.owner``/``status.ships`` are ``[P, H+1]``; ``status.arrivals_by_owner``
    is ``[P, H, A]`` (None when fleet tracking is off — then incoming is reported
    as 0 and only ownership-trajectory features are populated).
    """
    owner = status.owner            # [P, H+1] long
    ships = status.ships.float()    # [P, H+1]
    P, Hp1 = owner.shape
    H = (Hp1 - 1) if horizon is None else int(horizon)
    H = max(1, min(H, Hp1 - 1))

    mine_t = owner == int(me)               # [P, H+1] bool
    mine0 = mine_t[:, 0]                     # planets we hold at t=0
    my_count_t = mine_t.float().sum(dim=0)   # [H+1]
    my_ships_t = (ships * mine_t.float()).sum(dim=0)  # [H+1]

    min_my_planet_count = float(my_count_t[: H + 1].min().item())
    min_my_ships = float(my_ships_t[: H + 1].min().item())
    annihilated = 1.0 if min_my_planet_count <= 0.0 else 0.0

    # time-to-first-loss: earliest tick a t0-held planet stops being ours.
    if bool(mine0.any()):
        held = mine_t[mine0]                 # [K, H+1]
        lost = ~held                         # [K, H+1]
        # first tick lost per planet (H+1 if never), then min across planets.
        tick_idx = torch.arange(Hp1, device=owner.device).expand_as(lost)
        big = torch.full_like(tick_idx, Hp1)
        first_lost = torch.where(lost, tick_idx, big).min(dim=1).values  # [K]
        time_to_first_loss = float(first_lost.min().item())
    else:
        time_to_first_loss = float(Hp1)

    # per-enemy aggregations (max over enemies, never mean).
    enemy_ids = sorted({int(o) for o in owner[:, 0].unique().tolist() if int(o) >= 0 and int(o) != int(me)})
    # include enemies that only appear later in the trajectory (e.g. capture our planet)
    for o in owner[:, : H + 1].unique().tolist():
        oi = int(o)
        if oi >= 0 and oi != int(me) and oi not in enemy_ids:
            enemy_ids.append(oi)

    arr = status.arrivals_by_owner  # [P, H, A] or None
    incoming_per_enemy: list[float] = []
    peak_per_enemy: list[float] = []
    lost_per_enemy: list[float] = []
    mine0_f = mine0.float().unsqueeze(1)  # [P, 1]
    owner_H = owner[:, H]
    for e in enemy_ids:
        if arr is not None and e < arr.shape[-1]:
            inc = arr[..., e][:, :H]              # [P, H] arrivals from e
            on_mine = inc * mine0_f               # only onto planets we hold now
            incoming_per_enemy.append(float(on_mine.sum().item()))
            peak_per_enemy.append(float(on_mine.sum(dim=0).max().item()) if H > 0 else 0.0)
        else:
            incoming_per_enemy.append(0.0)
            peak_per_enemy.append(0.0)
        lost_per_enemy.append(float(((owner_H == e) & mine0).float().sum().item()))

    ordered = sorted(incoming_per_enemy, reverse=True)
    max_enemy_incoming = ordered[0] if ordered else 0.0
    second_worst_incoming = ordered[1] if len(ordered) > 1 else 0.0
    if incoming_per_enemy:
        worst_idx = max(range(len(incoming_per_enemy)), key=lambda i: incoming_per_enemy[i])
        max_enemy_peak_tick = peak_per_enemy[worst_idx]
    else:
        max_enemy_peak_tick = 0.0

    return ThreatFeatures(
        n_enemies=float(len(enemy_ids)),
        total_enemy_incoming=float(sum(incoming_per_enemy)),
        max_enemy_incoming=max_enemy_incoming,
        second_worst_incoming=second_worst_incoming,
        max_enemy_peak_tick=max_enemy_peak_tick,
        max_planets_lost_to_one=max(lost_per_enemy) if lost_per_enemy else 0.0,
        time_to_first_loss=time_to_first_loss,
        min_my_planet_count=min_my_planet_count,
        annihilated=annihilated,
        min_my_ships=min_my_ships,
    )
