from __future__ import annotations

import argparse
import ast
import base64
import json
import zlib
from pathlib import Path
from typing import Any


def validate_submission_template(template: str) -> None:
    tree = ast.parse(template)
    fallback_defined = False
    agent_has_fallback = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fallback_greedy":
            fallback_defined = True
        if not isinstance(node, ast.FunctionDef) or node.name != "agent":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Try):
                continue
            for handler in stmt.handlers:
                for fallback_stmt in handler.body:
                    if not isinstance(fallback_stmt, ast.Return):
                        continue
                    call = fallback_stmt.value
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    if isinstance(func, ast.Name) and func.id == "fallback_greedy":
                        agent_has_fallback = True
                        break

    if not fallback_defined or not agent_has_fallback:
        raise ValueError("submission template must define fallback_greedy and return it from agent() exception handling")


def _tensor_payload(value: Any) -> list[Any]:
    return value.detach().cpu().tolist()


def _decoder_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
    if isinstance(summary.get("decoder"), dict):
        return dict(summary["decoder"])
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    mapping = {
        "decoder_fractions": "fractions",
        "decoder_angle_offsets": "angle_offsets",
        "decoder_max_moves_per_turn": "max_moves_per_turn",
        "decoder_min_ships_to_launch": "min_ships_to_launch",
        "decoder_reserve_home_ships": "reserve_home_ships",
    }
    payload: dict[str, Any] = {}
    for source, target in mapping.items():
        if source in config:
            payload[target] = config[source]
    return {
        "fractions": payload.get("fractions", [0.10, 0.25, 0.50, 0.75]),
        "angle_offsets": payload.get("angle_offsets", [-0.261799, -0.130899, 0.0, 0.130899, 0.261799]),
        "max_moves_per_turn": int(payload.get("max_moves_per_turn", 8)),
        "min_ships_to_launch": int(payload.get("min_ships_to_launch", 2)),
        "reserve_home_ships": int(payload.get("reserve_home_ships", 8)),
    }


def _load_checkpoint_payload(path: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only outside dev env
        raise RuntimeError("exporting PPO checkpoints requires torch in the local export environment") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid PPO checkpoint: {path}")
    state = checkpoint["model_state_dict"]
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}
    arch = str(summary.get("arch", "flat"))

    _HEAD = ("launch", "source", "target", "frac", "offset")
    if arch == "entity":
        required = tuple(
            f"{p}.{i}.{w}"
            for p in ("planet_mlp", "fleet_mlp", "trunk")
            for i in (0, 2)
            for w in ("weight", "bias")
        ) + tuple(f"heads.{h}.{w}" for h in _HEAD for w in ("weight", "bias"))
    else:
        required = (
            "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
        ) + tuple(f"{h}.{w}" for h in _HEAD for w in ("weight", "bias"))

    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing {arch} policy tensors: {missing}")
    return {
        "arch": arch,
        "decoder": _decoder_payload(checkpoint),
        "weights": {key: _tensor_payload(state[key]) for key in required},
    }


_FLAT_ACTION_SRC = '''def _neural_action(obs, player):
    weights = _NEURAL_POLICY["weights"]
    hidden = _tanh_vec(_linear(_encode_state_flat(obs, player), weights["net.0.weight"], weights["net.0.bias"]))
    hidden = _tanh_vec(_linear(hidden, weights["net.2.weight"], weights["net.2.bias"]))
    planets = obs.get("planets", [])
    min_ships = int(_NEURAL_POLICY["decoder"].get("min_ships_to_launch", 2))
    n_launchable = sum(1 for planet in planets if _planet_owner(planet) == player and _planet_ships(planet) >= min_ships)
    planet_count = len(planets)
    launch = _argmax(_linear(hidden, weights["launch.weight"], weights["launch.bias"])) if n_launchable > 0 else 0
    _ftr = _NEURAL_POLICY["decoder"].get("force_target_rank")
    target_rank = int(_ftr) if _ftr is not None else _masked_argmax(_linear(hidden, weights["target.weight"], weights["target.bias"]), max(planet_count - 1, 0))
    return [
        launch,
        _masked_argmax(_linear(hidden, weights["source.weight"], weights["source.bias"]), n_launchable),
        target_rank,
        _argmax(_linear(hidden, weights["frac.weight"], weights["frac.bias"])),
        _argmax(_linear(hidden, weights["offset.weight"], weights["offset.bias"])),
    ]'''


_ENTITY_ACTION_SRC = '''def _entity_pool(rows, prefix):
    weights = _NEURAL_POLICY["weights"]
    width = len(weights[prefix + ".2.bias"])
    acc = [0.0] * width
    count = 0.0
    for row in rows:
        if row[0] <= 0.0:
            continue
        h = _tanh_vec(_linear(row, weights[prefix + ".0.weight"], weights[prefix + ".0.bias"]))
        h = _tanh_vec(_linear(h, weights[prefix + ".2.weight"], weights[prefix + ".2.bias"]))
        for j in range(width):
            acc[j] += h[j]
        count += 1.0
    if count > 0.0:
        acc = [value / count for value in acc]
    return acc


def _neural_action(obs, player):
    weights = _NEURAL_POLICY["weights"]
    flat = _encode_state_flat(obs, player)
    glob = flat[:8]
    planet_rows = [flat[8 + i * 14:8 + (i + 1) * 14] for i in range(_NEURAL_MAX_PLANETS)]
    foff = 8 + _NEURAL_MAX_PLANETS * 14
    fleet_rows = [flat[foff + i * 10:foff + (i + 1) * 10] for i in range(_NEURAL_MAX_FLEETS)]
    trunk_in = list(glob) + _entity_pool(planet_rows, "planet_mlp") + _entity_pool(fleet_rows, "fleet_mlp")
    hidden = _tanh_vec(_linear(trunk_in, weights["trunk.0.weight"], weights["trunk.0.bias"]))
    hidden = _tanh_vec(_linear(hidden, weights["trunk.2.weight"], weights["trunk.2.bias"]))
    planets = obs.get("planets", [])
    min_ships = int(_NEURAL_POLICY["decoder"].get("min_ships_to_launch", 2))
    n_launchable = sum(1 for planet in planets if _planet_owner(planet) == player and _planet_ships(planet) >= min_ships)
    planet_count = len(planets)
    launch = _argmax(_linear(hidden, weights["heads.launch.weight"], weights["heads.launch.bias"])) if n_launchable > 0 else 0
    _ftr = _NEURAL_POLICY["decoder"].get("force_target_rank")
    target_rank = int(_ftr) if _ftr is not None else _masked_argmax(_linear(hidden, weights["heads.target.weight"], weights["heads.target.bias"]), max(planet_count - 1, 0))
    return [
        launch,
        _masked_argmax(_linear(hidden, weights["heads.source.weight"], weights["heads.source.bias"]), n_launchable),
        target_rank,
        _argmax(_linear(hidden, weights["heads.frac.weight"], weights["heads.frac.bias"])),
        _argmax(_linear(hidden, weights["heads.offset.weight"], weights["heads.offset.bias"])),
    ]'''


def _policy_action_source(payload: dict[str, Any], *, function_name: str, policy_name: str) -> str:
    action_src = _ENTITY_ACTION_SRC if payload.get("arch") == "entity" else _FLAT_ACTION_SRC
    return action_src.replace("def _neural_action(obs, player):", f"def {function_name}(obs, player):").replace(
        "_NEURAL_POLICY", policy_name
    )


def _neural_runtime_source(
    payload: dict[str, Any],
    *,
    four_player_payload: dict[str, Any] | None = None,
    four_player_policy: str = "neural",
) -> str:
    if four_player_policy not in {"neural", "template"}:
        raise ValueError("four_player_policy must be 'neural' or 'template'")
    encoded = base64.b64encode(zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=9)).decode("ascii")
    action_src = _policy_action_source(payload, function_name="_neural_action", policy_name="_NEURAL_POLICY")
    encoded_4p = (
        base64.b64encode(
            zlib.compress(json.dumps(four_player_payload, separators=(",", ":")).encode("utf-8"), level=9)
        ).decode("ascii")
        if four_player_payload is not None
        else None
    )
    policy_4p_src = (
        f'_NEURAL_POLICY_4P = json.loads(zlib.decompress(base64.b64decode("{encoded_4p}")).decode("utf-8"))\n'
        + _policy_action_source(four_player_payload, function_name="_neural_action_4p", policy_name="_NEURAL_POLICY_4P")
        if four_player_payload is not None
        else "_NEURAL_POLICY_4P = None\n"
    )
    return f'''

import base64
import json
import zlib
from math import floor, tanh

_NEURAL_POLICY = json.loads(zlib.decompress(base64.b64decode("{encoded}")).decode("utf-8"))
{policy_4p_src}
_FOUR_PLAYER_POLICY = "{four_player_policy}"
_BASE_TEMPLATE_AGENT = agent
_NEURAL_MAX_PLANETS = 96
_NEURAL_MAX_FLEETS = 256


def _owner_rel(owner, player):
    if owner == -1:
        return (0.0, 0.0, 1.0, 0.0)
    if owner == player:
        return (1.0, 0.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0, 0.0)


def _encode_state_flat(obs, player):
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    own_ships = sum(_planet_ships(planet) for planet in planets if _planet_owner(planet) == player)
    enemy_ships = sum(_planet_ships(planet) for planet in planets if _planet_owner(planet) not in (-1, player))
    own_prod = sum(_planet_production(planet) for planet in planets if _planet_owner(planet) == player)
    enemy_prod = sum(_planet_production(planet) for planet in planets if _planet_owner(planet) not in (-1, player))
    out = [
        float(obs.get("step", obs.get("turn", 0))) / 500.0,
        float(obs.get("angular_velocity", 0.0)),
        len(planets) / max(_NEURAL_MAX_PLANETS, 1),
        len(fleets) / max(_NEURAL_MAX_FLEETS, 1),
        log(1.0 + max(float(own_ships), 0.0)) / 8.0,
        log(1.0 + max(float(enemy_ships), 0.0)) / 8.0,
        float(own_prod) / 64.0,
        float(enemy_prod) / 64.0,
    ]
    for idx in range(_NEURAL_MAX_PLANETS):
        if idx >= len(planets):
            out.extend([0.0] * 14)
            continue
        planet = planets[idx]
        owner = _planet_owner(planet)
        x = _planet_x(planet)
        y = _planet_y(planet)
        dx = (x - CENTER) / CENTER
        dy = (y - CENTER) / CENTER
        owner_self, owner_enemy, owner_neutral, owner_other = _owner_rel(owner, player)
        out.extend([
            1.0,
            owner_self,
            owner_enemy,
            owner_neutral,
            owner_other,
            x / 100.0,
            y / 100.0,
            dx,
            dy,
            (dx * dx + dy * dy) ** 0.5,
            _planet_radius(planet) / 10.0,
            log(1.0 + max(float(_planet_ships(planet)), 0.0)) / 8.0,
            float(_planet_production(planet)) / 5.0,
            float(_planet_id(planet)) / 512.0,
        ])
    for idx in range(_NEURAL_MAX_FLEETS):
        if idx >= len(fleets):
            out.extend([0.0] * 10)
            continue
        fleet = fleets[idx]
        owner = _fleet_owner(fleet)
        angle = _fleet_angle(fleet)
        owner_self, owner_enemy, owner_neutral, _owner_other = _owner_rel(owner, player)
        out.extend([
            1.0,
            owner_self,
            owner_enemy,
            owner_neutral,
            _fleet_x(fleet) / 100.0,
            _fleet_y(fleet) / 100.0,
            cos(angle),
            sin(angle),
            log(1.0 + max(float(_fleet_ships(fleet)), 0.0)) / 8.0,
            float(_fleet_from_planet_id(fleet)) / 512.0,
        ])
    return out


def _linear(vec, weight, bias):
    out = []
    for row, item_bias in zip(weight, bias):
        total = float(item_bias)
        for left, right in zip(vec, row):
            total += left * right
        out.append(total)
    return out


def _tanh_vec(vec):
    return [tanh(value) for value in vec]


def _argmax(values):
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def _masked_argmax(values, valid_count):
    # Train/inference parity with action_masks: argmax only over valid ranks.
    if valid_count <= 0 or valid_count >= len(values):
        return _argmax(values)
    return _argmax(values[:valid_count])


{action_src}


def _observed_player_count(obs):
    raw = obs.get("num_players", None)
    try:
        value = int(raw)
        if value in (2, 4):
            return value
    except Exception:
        pass
    max_owner = int(obs.get("player", 0))
    for row in obs.get("planets", []):
        try:
            owner = int(row[1] if not isinstance(row, dict) else row.get("owner", -1))
        except Exception:
            continue
        if owner >= 0:
            max_owner = max(max_owner, owner)
    for row in obs.get("fleets", []):
        try:
            owner = int(row[1] if not isinstance(row, dict) else row.get("owner", -1))
        except Exception:
            continue
        if owner >= 0:
            max_owner = max(max_owner, owner)
    return 4 if max_owner >= 3 else 2


def _neural_decode_with_policy(policy, obs, player, action):
    decoder = policy["decoder"]
    fractions = decoder.get("fractions", [0.10, 0.25, 0.50, 0.75])
    angle_offsets = decoder.get("angle_offsets", [-0.261799, -0.130899, 0.0, 0.130899, 0.261799])
    max_moves = int(decoder.get("max_moves_per_turn", 8))
    min_ships = int(decoder.get("min_ships_to_launch", 2))
    reserve = int(decoder.get("reserve_home_ships", 8))
    planets = obs.get("planets", [])
    own = [planet for planet in planets if _planet_owner(planet) == player and _planet_ships(planet) >= min_ships]
    if not own:
        return []
    # Action layout: [launch, source_rank, target_rank, fraction_idx, offset_idx].
    # launch == 0 is an explicit pass (see todo P1.5).
    if int(action[0]) == 0:
        return []
    source_rank, target_rank, fraction_idx, offset_idx = [int(value) for value in action[1:5]]
    own.sort(key=lambda planet: (_planet_ships(planet), _planet_production(planet)), reverse=True)
    offset = source_rank % len(own)
    ranked_sources = own[offset:] + own[:offset]
    moves = []
    used_targets = set()
    for source in ranked_sources:
        if len(moves) >= max_moves:
            break
        candidates = [planet for planet in planets if _planet_id(planet) != _planet_id(source)]
        if not candidates:
            continue
        source_xy = (_planet_x(source), _planet_y(source))
        max_launch = max(1, int(_planet_ships(source) * float(fractions[-1])))

        def target_score(planet):
            target_xy = _predict_target_xy(obs, source_xy, planet, max_launch)
            dist = _distance(source_xy, target_xy)
            owner = _planet_owner(planet)
            enemy_bonus = 8.0 if owner not in (-1, player) else 0.0
            neutral_bonus = 4.0 if owner == -1 else 0.0
            repeat_penalty = 3.0 if _planet_id(planet) in used_targets else 0.0
            return (
                float(_planet_production(planet)) * 10.0
                + enemy_bonus
                + neutral_bonus
                - repeat_penalty
                - 0.15 * dist
                - 0.12 * float(_planet_ships(planet))
            )

        candidates.sort(key=target_score, reverse=True)
        target = candidates[target_rank % len(candidates)]
        fraction = float(fractions[fraction_idx % len(fractions)])
        ships = int(max(0, floor(float(_planet_ships(source)) * fraction)))
        if _planet_ships(source) - ships < reserve and len(own) <= 2:
            ships = max(0, _planet_ships(source) - reserve)
        if ships <= 0:
            continue
        target_xy = _predict_target_xy(obs, source_xy, target, ships)
        base = _sun_safe_angle(source_xy, target_xy, _angle(source_xy, target_xy))
        angle = base + float(angle_offsets[offset_idx % len(angle_offsets)])
        moves.append([_planet_id(source), float(angle), int(ships)])
        used_targets.add(_planet_id(target))
    return moves


def _neural_decode(obs, player, action):
    return _neural_decode_with_policy(_NEURAL_POLICY, obs, player, action)


def _neural_decode_4p(obs, player, action):
    policy = _NEURAL_POLICY_4P if _NEURAL_POLICY_4P is not None else _NEURAL_POLICY
    return _neural_decode_with_policy(policy, obs, player, action)


def agent(obs):
    if _observed_player_count(obs) >= 4 and _FOUR_PLAYER_POLICY == "template":
        return list(_BASE_TEMPLATE_AGENT(obs))
    _submission_stats_increment("calls")
    try:
        player = int(obs.get("player", 0))
        if _observed_player_count(obs) >= 4 and _NEURAL_POLICY_4P is not None:
            moves = _neural_decode_4p(obs, player, _neural_action_4p(obs, player))
        else:
            moves = _neural_decode(obs, player, _neural_action(obs, player))
        if not _moves_are_legal(obs, player, moves):
            _submission_stats_increment("illegal_moves")
            raise ValueError("neural policy produced illegal moves")
        return list(moves)
    except Exception:
        _submission_stats_increment("fallbacks")
        return fallback_greedy(obs)
'''


def render_submission(
    template: str,
    checkpoint: str | None = None,
    *,
    checkpoint_4p: str | None = None,
    four_player_policy: str = "neural",
) -> str:
    validate_submission_template(template)
    if checkpoint is None:
        if checkpoint_4p is not None:
            raise ValueError("checkpoint_4p requires checkpoint")
        return template
    four_player_payload = _load_checkpoint_payload(checkpoint_4p) if checkpoint_4p is not None else None
    return template + _neural_runtime_source(
        _load_checkpoint_payload(checkpoint),
        four_player_payload=four_player_payload,
        four_player_policy=four_player_policy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-4p", default=None)
    parser.add_argument("--four-player-policy", choices=("neural", "template"), default="neural")
    parser.add_argument("--out", default="submission.py")
    args = parser.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.checkpoint_4p is not None and not Path(args.checkpoint_4p).exists():
        raise SystemExit(f"4p checkpoint not found: {args.checkpoint_4p}")

    template = Path("python/submission/submission_template.py").read_text(encoding="utf-8")
    rendered = render_submission(
        template,
        args.checkpoint,
        checkpoint_4p=args.checkpoint_4p,
        four_player_policy=args.four_player_policy,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print({"wrote": args.out, "checkpoint": args.checkpoint})


if __name__ == "__main__":
    main()
