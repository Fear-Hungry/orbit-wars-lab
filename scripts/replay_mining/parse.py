"""Parse an Orbit Wars replay into a flat feature record and classify losses.

Replay schema (from the episode JSON `specification.observation`):
  planets : [id, owner, x, y, radius, ships, production]   owner -1 == neutral
  fleets  : [id, owner, x, y, angle, from_planet_id, ships]
  action  : [[from_planet_id, direction_angle, num_ships], ...]
  comet_planet_ids : planet ids that are (temporary) comets
  remainingOverageTime : banked time; agent gets TIMEOUT when it drops < 0

Score proxy = total ships a player controls (planet garrisons + in-flight
fleets). Production and planets-owned are tracked alongside. Per step we read
the fullest observation available (it is shared/full-state), so an eliminated
agent's frozen view does not corrupt the trajectory.

The loss classifier is rule-based (evidence strength: PARTIAL — thresholds are
hand-tuned, not learned). Each loss gets one PRIMARY class by priority plus the
full set of triggered flags and the metrics behind them, so a human can audit
every call.
"""
from __future__ import annotations

from typing import Any

NEUTRAL = -1
OPENING_STEP = 60          # "early" cutoff
# Thresholds are expressed RELATIVE to fair share (1/n_players), so 2p and 4p
# are comparable: rel = share * n_players, where 1.0 == fair, >1 == ahead.
BEHIND_REL = 0.80          # below fair share == losing the exchange
DOMINATE_REL = 1.50        # well above fair share == was clearly winning
COLLAPSE_REL = 0.55        # ended well below fair share == lost decisively
HOARD_FRAC = 0.45          # one planet holding >45% of our ships == hoarding

# Primary-class priority (first match wins).
PRIORITY = [
    "timeout_crash",
    "comet",
    "overextension",
    "kingmaker_4p",
    "bad_opening",
    "even_attrition_collapse",
    "no_defense_recapture",
    "late_redistribution",
    "unclassified",
]

CLASS_LABELS = {
    "timeout_crash": "Timeout / crash (DQ)",
    "comet": "Comet collision/loss",
    "overextension": "Overextension (peaked then collapsed)",
    "kingmaker_4p": "4p kingmaker / ganged",
    "bad_opening": "Bad opening (behind early, never recovered)",
    "even_attrition_collapse": "Even game lost in attrition endgame",
    "no_defense_recapture": "No defense / never recaptured",
    "late_redistribution": "Late redistribution (idle reserve hoarded)",
    "unclassified": "Unclassified loss",
}


def _full_obs(step: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the fullest observation in a step (shared full-state)."""
    best: dict[str, Any] = {}
    for agent in step:
        obs = agent.get("observation") or {}
        if len(obs.get("planets") or []) >= len(best.get("planets") or []):
            if obs.get("planets") is not None:
                best = obs
    return best


def _player_state(obs: dict[str, Any], n_players: int) -> dict[str, Any]:
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    pl_ships = [0] * n_players
    fl_ships = [0] * n_players
    owned = [0] * n_players
    prod = [0] * n_players
    max_stack = [0] * n_players
    for p in planets:
        owner = p[1]
        if 0 <= owner < n_players:
            pl_ships[owner] += p[5]
            owned[owner] += 1
            prod[owner] += p[6]
            if p[5] > max_stack[owner]:
                max_stack[owner] = p[5]
    for f in fleets:
        owner = f[1]
        if 0 <= owner < n_players:
            fl_ships[owner] += f[6]
    total = [pl_ships[i] + fl_ships[i] for i in range(n_players)]
    return {
        "total": total,
        "planet_ships": pl_ships,
        "fleet_ships": fl_ships,
        "owned": owned,
        "prod": prod,
        "max_stack": max_stack,
    }


def parse_replay(replay: dict[str, Any], our_team_name: str = "Marcus Vinicius",
                 our_index_hint: int | None = None) -> dict[str, Any]:
    info = replay.get("info") or {}
    team_names = info.get("TeamNames") or []
    rewards = replay.get("rewards") or []
    n_players = len(rewards) or len(replay["steps"][0])

    # Our player index: prefer team-name match, fall back to the hint.
    our_idx = our_index_hint if our_index_hint is not None else 0
    for i, name in enumerate(team_names):
        if name == our_team_name:
            our_idx = i
            break

    steps = replay.get("steps") or []
    n_steps = len(steps)
    cfg = replay.get("configuration") or {}

    # Per-step aggregates.
    our_share: list[float] = []
    our_total: list[int] = []
    our_owned: list[int] = []
    our_maxstack: list[int] = []
    rank_at: list[int] = []           # our rank by ships (1 == best)
    ships_sent: list[int] = []
    comet_active: list[bool] = []
    comet_on_ours: list[bool] = []
    prev_owner: dict[int, int] = {}
    planets_lost = 0
    recaptures = 0
    elim_step: int | None = None

    for t, step in enumerate(steps):
        obs = _full_obs(step)
        st = _player_state(obs, n_players)
        tot = st["total"]
        ssum = sum(tot) or 1
        our_share.append(tot[our_idx] / ssum)
        our_total.append(tot[our_idx])
        our_owned.append(st["owned"][our_idx])
        our_maxstack.append(st["max_stack"][our_idx])
        rank_at.append(1 + sum(1 for v in tot if v > tot[our_idx]))
        if st["owned"][our_idx] == 0 and elim_step is None and t > 5:
            elim_step = t

        # Our action this step.
        act = []
        if our_idx < len(step):
            act = step[our_idx].get("action") or []
        sent = 0
        for mv in act:
            try:
                sent += int(mv[2])
            except (IndexError, TypeError, ValueError):
                pass
        ships_sent.append(sent)

        # Comets.
        comet_ids = set(obs.get("comet_planet_ids") or [])
        comet_active.append(bool(comet_ids))
        ours_comet = any(
            p[0] in comet_ids and p[1] == our_idx for p in (obs.get("planets") or [])
        )
        comet_on_ours.append(ours_comet)

        # Ownership transitions for our planets.
        for p in obs.get("planets") or []:
            pid, owner = p[0], p[1]
            was = prev_owner.get(pid)
            if was is not None:
                if was == our_idx and owner != our_idx:
                    planets_lost += 1
                elif was != our_idx and owner == our_idx:
                    recaptures += 1
            prev_owner[pid] = owner

    # Derived metrics.
    opening_idx = min(OPENING_STEP, n_steps - 1)
    opening_share = our_share[opening_idx] if our_share else 0.0
    peak_share = max(our_share) if our_share else 0.0
    peak_step = our_share.index(peak_share) if our_share else 0
    final_share = our_share[-1] if our_share else 0.0
    final_reward = rewards[our_idx] if our_idx < len(rewards) else None
    final_rank = rank_at[-1] if rank_at else None
    mid_rank = rank_at[opening_idx] if rank_at else None
    best_rank = min(rank_at) if rank_at else None
    total_sent = sum(ships_sent)

    # Biggest single-step ship drop (collapse moment).
    max_drop = 0
    drop_step = 0
    for t in range(1, len(our_total)):
        d = our_total[t - 1] - our_total[t]
        if d > max_drop:
            max_drop = d
            drop_step = t
    comet_near_drop = any(
        comet_on_ours[t] for t in range(max(0, drop_step - 4), min(len(comet_on_ours), drop_step + 2))
    )

    # Our status (timeout/crash).
    statuses = replay.get("statuses") or []
    our_status = statuses[our_idx] if our_idx < len(statuses) else None
    min_overage = None
    last_active = -1
    for t, step in enumerate(steps):
        if our_idx < len(step):
            ag = step[our_idx]
            ov = (ag.get("observation") or {}).get("remainingOverageTime")
            if ov is not None:
                min_overage = ov if min_overage is None else min(min_overage, ov)
            if ag.get("status") == "ACTIVE":
                last_active = t

    # Late-game send activity while behind (idle-reserve detection).
    late = range(max(0, n_steps - 80), n_steps)
    behind_late = [t for t in late if our_share[t] < 0.5 and our_total[t] > 0]
    if behind_late:
        send_activity = sum(ships_sent[t] for t in behind_late) / max(
            1, sum(our_total[t] for t in behind_late)
        )
        hoard_late = max(
            (our_maxstack[t] / our_total[t]) for t in behind_late if our_total[t] > 0
        )
    else:
        send_activity = 0.0
        hoard_late = 0.0

    return {
        "episode_id": info.get("EpisodeId"),
        "seed": info.get("seed", cfg.get("seed")),
        "n_players": n_players,
        "format": f"{n_players}p",
        "our_index": our_idx,
        "our_team": our_team_name,
        "opponents": [n for i, n in enumerate(team_names) if i != our_idx],
        "n_steps": n_steps,
        "final_reward": final_reward,
        "final_rank": final_rank,
        "our_status": our_status,
        "min_overage": min_overage,
        "last_active_step": last_active,
        "opening_share": round(opening_share, 4),
        "peak_share": round(peak_share, 4),
        "peak_step": peak_step,
        "final_share": round(final_share, 4),
        "mid_rank": mid_rank,
        "best_rank": best_rank,
        "planets_lost": planets_lost,
        "recaptures": recaptures,
        "elim_step": elim_step,
        "max_ship_drop": max_drop,
        "drop_step": drop_step,
        "comet_steps": sum(comet_active),
        "comet_on_ours_steps": sum(comet_on_ours),
        "comet_near_collapse": comet_near_drop,
        "total_ships_sent": total_sent,
        "send_activity_late": round(send_activity, 4),
        "hoard_frac_late": round(hoard_late, 4),
    }


def classify_loss(f: dict[str, Any]) -> dict[str, Any]:
    """Map a parsed feature record to a primary loss class + all triggered flags.

    Shares are normalised to fair share (rel = share * n_players) so the same
    thresholds apply to 2p and 4p.
    """
    flags: list[str] = []
    n = f["n_players"] or 2
    n_steps = f["n_steps"] or 1
    opening_rel = f["opening_share"] * n
    peak_rel = f["peak_share"] * n
    final_rel = f["final_share"] * n

    # timeout / crash: DQ status, time bank exhausted, or stopped acting while
    # still alive (not a normal in-game elimination).
    timeout = (
        (f.get("our_status") not in (None, "DONE"))
        or (f.get("min_overage") is not None and f["min_overage"] <= 0.0)
        or (f.get("last_active_step", -1) >= 0 and f["last_active_step"] < n_steps - 2
            and f.get("elim_step") is None)
    )
    if timeout:
        flags.append("timeout_crash")

    # comet: a comet sat on one of our planets right around the decisive collapse
    if f.get("comet_near_collapse") and f.get("max_ship_drop", 0) > 30:
        flags.append("comet")

    # overextension: dominated at the peak (rel >= 1.5) mid-game, then collapsed
    lost_after_peak = f["peak_step"] < n_steps - 10 and peak_rel >= DOMINATE_REL
    if (
        lost_after_peak
        and final_rel < COLLAPSE_REL
        and f["total_ships_sent"] > 0
        and f["recaptures"] <= f["planets_lost"]
    ):
        flags.append("overextension")

    # 4p kingmaker / ganged: still in the top half at the opening cutoff (mid_rank
    # <= 2 — genuinely contending, not a step-0 tie) then dropped to last /
    # eliminated. The headline 4p failure: overtaken/ganged after contending.
    if f["n_players"] == 4 and peak_rel < DOMINATE_REL:
        mid_rank = f.get("mid_rank")
        contended = mid_rank is not None and mid_rank <= 2
        ended_last = f["final_rank"] == 4 or f.get("elim_step") is not None
        if contended and ended_last:
            flags.append("kingmaker_4p")

    # bad opening: behind early and never led. 2p -> below fair share at the
    # opening cutoff; 4p -> already in the bottom half (mid_rank >= 3), i.e. lost
    # the expansion race.
    behind_early = opening_rel < BEHIND_REL or (
        f["n_players"] == 4 and f.get("mid_rank") is not None and f["mid_rank"] >= 3
    )
    if behind_early and peak_rel < DOMINATE_REL:
        flags.append("bad_opening")

    # even attrition collapse: even start, never dominated, lost decisively by the
    # end (typically a long planet-trading grind we lose in the endgame).
    if (
        BEHIND_REL <= opening_rel
        and peak_rel < DOMINATE_REL
        and final_rel < COLLAPSE_REL
    ):
        flags.append("even_attrition_collapse")

    # no defense / recapture: bled planets, essentially never recaptured
    if f["planets_lost"] >= 6 and f["recaptures"] <= max(1, f["planets_lost"] // 5):
        flags.append("no_defense_recapture")

    # late redistribution: still had material, hoarded an idle reserve, barely sent
    if (
        final_rel >= 0.40
        and f["send_activity_late"] < 0.05
        and f["hoard_frac_late"] >= HOARD_FRAC
        and f.get("elim_step") is None
    ):
        flags.append("late_redistribution")

    if not flags:
        flags.append("unclassified")

    primary = next(c for c in PRIORITY if c in flags)
    return {"primary_class": primary, "flags": flags}
