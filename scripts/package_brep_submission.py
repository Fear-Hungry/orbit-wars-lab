"""Package a BReP (producer_residual) checkpoint as a Kaggle tar.gz submission.

BReP edits the Producer base plan at inference, so the submission bundles the REAL
Producer (torch + orbit_lite, exactly as the training base plan) plus a pure-Python
BReP net (weights embedded, no torch needed for the tiny edit head). Structure:
  main.py            = submission_template + BReP runtime (this script renders it)
  _producer_agent.py = bots/producer/agent.py   (base-plan provider)
  _upstream.py       = bots/producer/_upstream.py
  orbit_lite/        = Producer deps
The allowed deps are torch + orbit_lite (D11 only forbids orbit_wars_core/_py).
"""
from __future__ import annotations

import argparse
import base64
import json
import tarfile
import zlib
from pathlib import Path
from typing import Any

from scripts.export_submission import validate_submission_template


def _residual_payload(checkpoint_path: str) -> dict[str, Any]:
    import torch

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    arch = ck.get("summary", {}).get("arch") or ck.get("config", {}).get("policy_arch")
    if arch != "producer_residual":
        raise ValueError(f"checkpoint arch is {arch!r}, expected producer_residual")
    keys = [
        f"{p}.{i}.{w}"
        for p in ("planet_mlp", "fleet_mlp", "trunk")
        for i in (0, 2)
        for w in ("weight", "bias")
    ] + ["edit.weight", "edit.bias"]
    missing = [k for k in keys if k not in state]
    if missing:
        raise ValueError(f"checkpoint missing BReP tensors: {missing}")
    # Build the model and load to read k_max/n_edit authoritatively (and validate the
    # checkpoint matches the current arch — a shape mismatch raises here, loudly).
    from python.agents.policy import ProducerResidualBranchActorCritic
    from python.orbit_wars_gym.encoding import EncoderConfig, observation_dim

    model = ProducerResidualBranchActorCritic(observation_dim(EncoderConfig()))
    model.load_state_dict(state)
    return {
        "k_max": int(model.k_max),
        "n_edit": int(model.n_edit),
        "weights": {k: state[k].detach().cpu().tolist() for k in keys},
    }


def _encoded_weights(payload: dict[str, Any]) -> str:
    return base64.b64encode(
        zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=9)
    ).decode("ascii")


def _runtime_src() -> str:
    # NOTE: nothing heavy runs at module load — Kaggle's agentTimeout=2s covers the
    # cold import (torch + orbit_lite + Producer). The 1.5MB weights blob lives in a
    # SEPARATE module (_brep_weights) and is decoded LAZILY on the first agent call
    # (inside actTimeout=1s), keeping init lean. Producer setup is GUARDED so a load
    # failure degrades to a valid agent instead of an unrecoverable submission ERROR.
    return '''

# ===================== BReP runtime (producer_residual) =====================
import base64 as _b64
import json as _json
import zlib as _zlib
from math import tanh as _tanh, log as _log, cos as _cos, sin as _sin

_BREP_MAX_PLANETS = 96
_BREP_MAX_FLEETS = 256
_BREP_CENTER = 50.0
_BREP_POLICY = None  # lazily populated from _brep_weights on first use

try:
    import _producer_agent  # bundled Producer (torch + orbit_lite) — the base plan
    _BREP_PRODUCER = _producer_agent.make_agent()  # isolated per-game runtime, like training
except Exception:
    _BREP_PRODUCER = None  # guarded: agent() falls back to fallback_greedy if Producer dies at load


def _brep_load_policy():
    global _BREP_POLICY
    if _BREP_POLICY is None:
        import _brep_weights  # separate module so the 1.5MB blob is off the init path
        _BREP_POLICY = _json.loads(_zlib.decompress(_b64.b64decode(_brep_weights.WEIGHTS_B64)).decode("utf-8"))
    return _BREP_POLICY


def _brep_owner_rel(owner, player):
    if owner == -1:
        return (0.0, 0.0, 1.0, 0.0)
    if owner == player:
        return (1.0, 0.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0, 0.0)


def _brep_encode(obs, player):
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    own_ships = sum(_planet_ships(p) for p in planets if _planet_owner(p) == player)
    enemy_ships = sum(_planet_ships(p) for p in planets if _planet_owner(p) not in (-1, player))
    own_prod = sum(_planet_production(p) for p in planets if _planet_owner(p) == player)
    enemy_prod = sum(_planet_production(p) for p in planets if _planet_owner(p) not in (-1, player))
    out = [
        float(obs.get("step", obs.get("turn", 0))) / 500.0,
        float(obs.get("angular_velocity", 0.0)),
        len(planets) / max(_BREP_MAX_PLANETS, 1),
        len(fleets) / max(_BREP_MAX_FLEETS, 1),
        _log(1.0 + max(float(own_ships), 0.0)) / 8.0,
        _log(1.0 + max(float(enemy_ships), 0.0)) / 8.0,
        float(own_prod) / 64.0,
        float(enemy_prod) / 64.0,
    ]
    for idx in range(_BREP_MAX_PLANETS):
        if idx >= len(planets):
            out.extend([0.0] * 14)
            continue
        p = planets[idx]
        x = _planet_x(p); y = _planet_y(p)
        dx = (x - _BREP_CENTER) / _BREP_CENTER; dy = (y - _BREP_CENTER) / _BREP_CENTER
        osf, oen, one, oot = _brep_owner_rel(_planet_owner(p), player)
        out.extend([1.0, osf, oen, one, oot, x / 100.0, y / 100.0, dx, dy,
                    (dx * dx + dy * dy) ** 0.5, _planet_radius(p) / 10.0,
                    _log(1.0 + max(float(_planet_ships(p)), 0.0)) / 8.0,
                    float(_planet_production(p)) / 5.0, float(_planet_id(p)) / 512.0])
    for idx in range(_BREP_MAX_FLEETS):
        if idx >= len(fleets):
            out.extend([0.0] * 10)
            continue
        f = fleets[idx]
        ang = _fleet_angle(f)
        osf, oen, one, _o = _brep_owner_rel(_fleet_owner(f), player)
        out.extend([1.0, osf, oen, one, _fleet_x(f) / 100.0, _fleet_y(f) / 100.0,
                    _cos(ang), _sin(ang), _log(1.0 + max(float(_fleet_ships(f)), 0.0)) / 8.0,
                    float(_fleet_from_planet_id(f)) / 512.0])
    return out


def _brep_linear(vec, weight, bias):
    out = []
    for row, b in zip(weight, bias):
        t = float(b)
        for l, r in zip(vec, row):
            t += l * r
        out.append(t)
    return out


def _brep_tanh(vec):
    return [_tanh(v) for v in vec]


def _brep_argmax(vals):
    bi = 0; bv = vals[0]
    for i, v in enumerate(vals[1:], start=1):
        if v > bv:
            bi = i; bv = v
    return bi


def _brep_pool(rows, prefix):
    w = _brep_load_policy()["weights"]
    width = len(w[prefix + ".2.bias"])
    acc = [0.0] * width; cnt = 0.0
    for row in rows:
        if row[0] <= 0.0:
            continue
        h = _brep_tanh(_brep_linear(row, w[prefix + ".0.weight"], w[prefix + ".0.bias"]))
        h = _brep_tanh(_brep_linear(h, w[prefix + ".2.weight"], w[prefix + ".2.bias"]))
        for j in range(width):
            acc[j] += h[j]
        cnt += 1.0
    if cnt > 0.0:
        acc = [a / cnt for a in acc]
    return acc


def _brep_edits(obs, player, n_base):
    pol = _brep_load_policy()
    w = pol["weights"]
    kmax = pol["k_max"]; nedit = pol["n_edit"]
    flat = _brep_encode(obs, player)
    glob = flat[:8]
    prows = [flat[8 + i * 14:8 + (i + 1) * 14] for i in range(_BREP_MAX_PLANETS)]
    foff = 8 + _BREP_MAX_PLANETS * 14
    frows = [flat[foff + i * 10:foff + (i + 1) * 10] for i in range(_BREP_MAX_FLEETS)]
    trunk_in = list(glob) + _brep_pool(prows, "planet_mlp") + _brep_pool(frows, "fleet_mlp")
    h = _brep_tanh(_brep_linear(trunk_in, w["trunk.0.weight"], w["trunk.0.bias"]))
    h = _brep_tanh(_brep_linear(h, w["trunk.2.weight"], w["trunk.2.bias"]))
    logits = _brep_linear(h, w["edit.weight"], w["edit.bias"])
    edits = []
    for s in range(kmax):
        if s < n_base:
            edits.append(_brep_argmax(logits[s * nedit:(s + 1) * nedit]))
        else:
            edits.append(0)  # inactive slot -> KEEP
    return edits


_BREP_SCALES = {2: 0.25, 3: 0.5, 4: 1.5, 5: 2.0}  # mirrors train_ppo._EDIT_SCALES


def _brep_apply(obs, base, edits, kmax):
    ships_by_id = {_planet_id(p): _planet_ships(p) for p in obs.get("planets", [])}
    out = []
    for i, mv in enumerate(base):
        ships = int(mv[2])
        if i >= kmax:
            out.append([mv[0], mv[1], float(ships)]); continue
        e = int(edits[i])
        if e == 1:  # CANCEL
            continue
        scale = _BREP_SCALES.get(e)
        if scale is None:  # KEEP (0) or unknown -> exact Producer move
            out.append([mv[0], mv[1], float(ships)]); continue
        scaled = int(round(ships * scale))
        if scale > 1.0:
            scaled = min(scaled, max(1, int(ships_by_id.get(int(mv[0]), ships)) - 1))
        out.append([mv[0], mv[1], float(max(1, scaled))])
    return out


# kaggle_environments.get_last_callable picks the LAST callable in the module
# namespace ([-1] of env.values()). The template already defines `agent`, and
# reassigning that existing key keeps its EARLY insertion position — so without
# this `del`, the last callable would be a _brep_* helper and Kaggle would call
# the WRONG function (the cause of the validation ERROR). del + redefine lands
# `agent` last. `*_` tolerates Kaggle passing (observation, configuration).
try:
    del agent
except NameError:
    pass


def agent(obs, *_):
    _submission_stats_increment("calls")
    player = int(obs.get("player", 0))
    # Producer base plan is the FLOOR (== training base, ~baseline strength). If the
    # bundled Producer failed to load, degrade to the template greedy (never ERROR).
    if _BREP_PRODUCER is None:
        _submission_stats_increment("fallbacks")
        return fallback_greedy(obs)
    try:
        base = [list(m) for m in _BREP_PRODUCER(obs)]
    except Exception:
        _submission_stats_increment("fallbacks")
        return fallback_greedy(obs)
    # BReP edits are BEST-EFFORT on top of the base: any net failure (lazy weight
    # load, etc.) falls back to the pure Producer plan, which is still valid+strong.
    try:
        pol = _brep_load_policy()
        edits = _brep_edits(obs, player, len(base))
        moves = _brep_apply(obs, base, edits, pol["k_max"])
        if not _moves_are_legal(obs, player, moves):
            _submission_stats_increment("illegal_moves")
            moves = base
    except Exception:
        _submission_stats_increment("net_fallbacks")
        moves = base
    if not _moves_are_legal(obs, player, moves):
        _submission_stats_increment("fallbacks")
        return fallback_greedy(obs)
    return list(moves)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--template", default="python/submission/submission_template.py")
    ap.add_argument("--out", default="artifacts/submission_brep.tar.gz")
    ap.add_argument("--main-out", default="artifacts/brep_main.py", help="also write the rendered main.py for inspection/testing")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    template = (root / args.template).read_text(encoding="utf-8")
    validate_submission_template(template)
    payload = _residual_payload(str(root / args.checkpoint) if not Path(args.checkpoint).is_absolute() else args.checkpoint)
    main_py = template + _runtime_src()

    main_path = root / args.main_out
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_text(main_py, encoding="utf-8")
    # weights as a SEPARATE module (off the init path; lazily imported on first call)
    weights_path = main_path.with_name("brep_weights_module.py")
    weights_path.write_text('WEIGHTS_B64 = "' + _encoded_weights(payload) + '"\n', encoding="utf-8")

    out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(main_path, arcname="main.py", recursive=False)
        tar.add(weights_path, arcname="_brep_weights.py", recursive=False)
        tar.add(root / "bots/producer/agent.py", arcname="_producer_agent.py", recursive=False)
        tar.add(root / "bots/producer/_upstream.py", arcname="_upstream.py", recursive=False)
        for p in sorted((root / "orbit_lite").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                tar.add(p, arcname=str(Path("orbit_lite") / p.relative_to(root / "orbit_lite")), recursive=False)
    print(json.dumps({"wrote": str(out), "main_py": str(main_path),
                      "main_kb": round(main_path.stat().st_size / 1024, 1),
                      "weights_kb": round(weights_path.stat().st_size / 1024, 1),
                      "k_max": payload["k_max"], "n_edit": payload["n_edit"],
                      "size_kb": round(out.stat().st_size / 1024, 1)}, indent=2))


if __name__ == "__main__":
    main()
