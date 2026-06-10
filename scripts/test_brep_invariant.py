"""Regression test for BReP correctness (diagnose Phase 5).
Critical invariant: KEEP-all edits applied to the Producer base plan == the EXACT
Producer plan (the parity floor). Also checks edit semantics + net shapes/init."""
from __future__ import annotations
import numpy as np
import torch

from python.train.train_ppo import _apply_residual_edits, build_phase0_env, _build_policy
from python.agents.registry import get_isolated_opponents
from python.orbit_wars_gym.encoding import observation_dim, EncoderConfig

K_MAX = 16
ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond


# --- get a real state + Producer base plan -----------------------------------
env = build_phase0_env(seed=3, num_players=2, opponent_name="producer")
env.reset(seed=3)
prod = get_isolated_opponents("producer", 1)[0]
# advance ~30 turns (planets accumulate ships) so Producer actually emits moves
for _ in range(30):
    env.step(np.array([0, 0, 0, 0, 0]))  # player-0 pass; opponent (producer) plays internally
    if env.state is None:
        env.reset(seed=3)
state = env.state
base = [list(m) for m in prod(state, 0)]
print(f"Producer base plan: {len(base)} moves; sample: {base[:2]}")
check("producer emits >=1 base move", len(base) >= 1)

# --- INVARIANT: KEEP-all == exact base ---------------------------------------
keep = _apply_residual_edits(state, base, [0] * len(base), K_MAX)
check("KEEP-all reproduces base length", len(keep) == len(base))
exact = all(
    int(a[0]) == int(b[0]) and abs(float(a[1]) - float(b[1])) < 1e-9 and int(a[2]) == int(b[2])
    for a, b in zip(keep, base)
)
check("KEEP-all reproduces base EXACTLY (parity floor)", exact)

# --- edit semantics -----------------------------------------------------------
cancel = _apply_residual_edits(state, base, [1] * len(base), K_MAX)
check("CANCEL-all drops every move", len(cancel) == 0)
half = _apply_residual_edits(state, base, [3] * len(base), K_MAX)  # code 3 = x0.5
check("code 3 (x0.5) ~halves ships (>=1)", all(int(r[2]) == max(1, round(int(b[2]) * 0.5)) for r, b in zip(half, base)))
for code in (2, 3, 4, 5):  # all scale codes must yield legal >=1-ship moves
    sc = _apply_residual_edits(state, base, [code] * len(base), K_MAX)
    check(f"scale code {code} yields >=1 ships, legal length", all(int(x[2]) >= 1 for x in sc) and len(sc) == len(base))

# --- net: shapes + KEEP-init argmax ------------------------------------------
obs_dim = observation_dim(EncoderConfig())
model = _build_policy("producer_residual", obs_dim)
model.eval()
obs = torch.tensor(np.asarray(env.backend.encoded_states(0)), dtype=torch.float32)
out = model.forward(obs)
check("edit logits shape (B,K_MAX,N_EDIT)", tuple(out["edit"].shape) == (obs.shape[0], K_MAX, model.n_edit))
check("KEEP-init: argmax == KEEP(0) for all slots", bool((out["edit"].argmax(-1) == 0).all()))

# --- get_action_and_value with mask: KEEP-init samples KEEP on active slots ---
mask = torch.zeros(obs.shape[0], K_MAX, dtype=torch.bool)
mask[:, : len(base)] = True
torch.manual_seed(0)
a, logp, ent, val = model.get_action_and_value(obs, masks={"edit": mask})
check("action shape (B,K_MAX)", tuple(a.shape) == (obs.shape[0], K_MAX))
check("inactive slots forced KEEP(0)", bool((a[:, len(base):] == 0).all()))
check("logprob & entropy are (B,)", logp.shape == (obs.shape[0],) and ent.shape == (obs.shape[0],))
# update-time consistency: same mask+action -> same logprob
_, logp2, _, _ = model.get_action_and_value(obs, action=a, masks={"edit": mask})
check("logprob reproducible at update time", torch.allclose(logp, logp2, atol=1e-6))

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
raise SystemExit(0 if ok else 1)
