"""Direct eval for a *v1* BReP checkpoint (n_edit=4) — no submission export.

The live code is BReP v2 (N_EDIT=6, scale table in train_ppo._apply_residual_edits),
which can no longer load or decode v1 checkpoints (brep_gpu/*, brep_seat/*). This
shim rebuilds the v1 setup exactly: model constructed with n_edit=4 and the v1 edit
table recovered byte-for-byte from the packaged c05 submission (_brep_apply in
artifacts/submission_brep.tar.gz!main.py):

    0 = KEEP, 1 = CANCEL, 2 = REDUCE (ships // 2), 3 = BOOST (min(avail-1, 2*ships))

Note REDUCE is integer floor-div, NOT v2's round(ships*0.5) — they differ on odd
ship counts, so reusing the v2 decoder with n_edit=4 would mis-measure the policy.

Same protocol as eval_brep_direct: greedy edits, BOTH seats, 500 steps, batched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from python.train.train_ppo import _moves_to_flat_rows
from python.agents.policy import ProducerResidualBranchActorCritic
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.encoding import observation_dim, EncoderConfig
from python.orbit_wars_gym.entities import (
    fleet_owner, fleet_ships, planet_id, planet_owner, planet_ships,
)

N_EDIT_V1 = 4


def _apply_residual_edits_v1(
    state: dict[str, Any], base_moves: Sequence[Sequence[float]], edits: Sequence[int], k_max: int
) -> list[list[float]]:
    """v1 decode table, recovered from the c05 submission's _brep_apply."""
    ships_by_id = {planet_id(p): planet_ships(p) for p in state.get("planets", [])}
    out: list[list[float]] = []
    for i, mv in enumerate(base_moves):
        ships = int(mv[2])
        if i >= k_max:
            out.append([mv[0], mv[1], float(ships)])
            continue
        e = int(edits[i])
        if e == 1:  # CANCEL
            continue
        if e == 2:  # REDUCE
            out.append([mv[0], mv[1], float(max(1, ships // 2))])
        elif e == 3:  # BOOST
            avail = int(ships_by_id.get(int(mv[0]), ships))
            boosted = min(max(1, avail - 1), ships * 2)
            out.append([mv[0], mv[1], float(boosted if boosted > 0 else ships)])
        else:  # KEEP (0) or unknown
            out.append([mv[0], mv[1], float(ships)])
    return out


def _ships(state, player):
    own = enemy = 0.0
    for p in state.get("planets", []):
        o = planet_owner(p)
        if o == player:
            own += planet_ships(p)
        elif o >= 0:
            enemy += planet_ships(p)
    for f in state.get("fleets", []):
        o = fleet_owner(f)
        if o == player:
            own += fleet_ships(f)
        elif o >= 0:
            enemy += fleet_ships(f)
    return own, enemy


def _play_seat(model, k_max, agent_seat, seeds, episode_steps, enable_comets, device, opponent="producer"):
    opp_seat = 1 - agent_seat
    n = len(seeds)
    backend = RustBatchBackend(num_envs=n, num_players=2, seed=int(seeds[0]),
                               config=RustConfig(enable_comets=enable_comets))
    states = backend.reset(int(seeds[0]))
    death_steps: list[int | None] = [None] * n
    agent_base = [make_isolated_opponent("producer") for _ in range(n)]
    opp_policy = [make_isolated_opponent(opponent) for _ in range(n)]
    for step in range(episode_steps):
        obs = np.asarray(backend.encoded_states(agent_seat))
        base_moves = [[list(m) for m in agent_base[i](states[i], agent_seat)] for i in range(n)]
        mask = np.zeros((n, k_max), dtype=bool)
        for i in range(n):
            mask[i, : min(len(base_moves[i]), k_max)] = True
        with torch.no_grad():
            logits = model.forward(torch.as_tensor(obs, dtype=torch.float32, device=device))["edit"]
            greedy = logits.argmax(-1)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            greedy = torch.where(mask_t, greedy, torch.zeros_like(greedy))
        edits = greedy.cpu().numpy()
        rows = []
        for i in range(n):
            agent_moves = _apply_residual_edits_v1(states[i], base_moves[i], edits[i], k_max)
            opp_moves = opp_policy[i](states[i], opp_seat)
            rows.extend(_moves_to_flat_rows(i, agent_seat, agent_moves))
            rows.extend(_moves_to_flat_rows(i, opp_seat, opp_moves))
        flat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
        backend.step_flat_with_encoded_states(flat, agent_seat)
        states = backend.states()
        for i in range(n):
            if death_steps[i] is None:
                own, _ = _ships(states[i], agent_seat)
                if own <= 0:
                    death_steps[i] = step + 1
    margins = []
    for i in range(n):
        own, enemy = _ships(states[i], agent_seat)
        margins.append((own - enemy) / (own + enemy) if (own + enemy) > 0 else 0.0)
    return margins, death_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=96)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opponent", default="producer")
    ap.add_argument("--no-comets", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch = ckpt["config"]["policy_arch"]
    if arch != "producer_residual":
        raise SystemExit(f"checkpoint arch is {arch!r}, expected producer_residual")
    head_out = ckpt["model_state_dict"]["edit.weight"].shape[0]
    model = ProducerResidualBranchActorCritic(observation_dim(EncoderConfig()), n_edit=N_EDIT_V1)
    if head_out != model.k_max * model.n_edit:
        raise SystemExit(f"edit head is {head_out}, expected {model.k_max * model.n_edit} (v1 n_edit=4)")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    k_max = model.k_max
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    enable_comets = not args.no_comets

    m0, d0 = _play_seat(model, k_max, 0, seeds, args.episode_steps, enable_comets, args.device, args.opponent)
    m1, d1 = _play_seat(model, k_max, 1, seeds, args.episode_steps, enable_comets, args.device, args.opponent)
    seat0, seat1 = float(np.mean(m0)), float(np.mean(m1))
    overall = float(np.mean(m0 + m1))
    deaths = [d for d in d0 + d1 if d is not None]
    result = {
        "checkpoint": args.checkpoint,
        "decoder": "v1 (KEEP/CANCEL/REDUCE//2/BOOSTx2)",
        "mean_score_margin": overall,
        "seat0_margin": seat0,
        "seat1_margin": seat1,
        "annihilated": len(deaths),
        "survival_rate": 1.0 - len(deaths) / max(len(d0) + len(d1), 1),
        "first_death_step": min(deaths) if deaths else None,
        "mean_death_step": float(np.mean(deaths)) if deaths else None,
        "seat0_annihilated": sum(1 for d in d0 if d is not None),
        "seat1_annihilated": sum(1 for d in d1 if d is not None),
        "seeds": args.seeds,
        "episode_steps": args.episode_steps,
        "opponent": args.opponent,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
