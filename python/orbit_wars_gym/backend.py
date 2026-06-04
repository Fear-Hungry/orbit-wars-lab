from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

_U32 = struct.Struct("<I")
_MOVE = struct.Struct("<idi")


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class RustConfig:
    episode_steps: int = 500
    act_timeout: float = 1.0
    ship_speed: float = 6.0
    comet_speed: float = 4.0
    enable_comets: bool = True
    max_planets: int = 96
    max_fleets: int = 4096


class RustBatchBackend:
    """Thin debug wrapper around the PyO3 module.

    The public wrapper prefers binary actions plus MessagePack payloads and
    keeps JSON fallbacks for older local extensions.
    """

    def __init__(self, num_envs: int, num_players: int = 2, seed: int = 0, config: RustConfig | None = None):
        if num_players not in (2, 4):
            raise ValueError("Orbit Wars supports only 2 or 4 players")
        try:
            import orbit_wars_rs  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(
                "Rust backend not available. Run: maturin develop --release -m crates/orbit_wars_py/Cargo.toml"
            ) from exc

        self._rs = orbit_wars_rs
        cfg = config or RustConfig()
        py_cfg = orbit_wars_rs.PyConfig(
            cfg.episode_steps,
            cfg.act_timeout,
            cfg.ship_speed,
            cfg.comet_speed,
            cfg.enable_comets,
            cfg.max_planets,
            cfg.max_fleets,
        )
        self.sim = orbit_wars_rs.PyBatchSimulator(num_envs, num_players, seed, py_cfg)
        self.num_envs = num_envs
        self.num_players = num_players

    def reset(self, seed: int) -> list[dict[str, Any]]:
        if hasattr(self.sim, "reset_msgpack"):
            return msgpack.unpackb(self.sim.reset_msgpack(seed), raw=False, strict_map_key=False)
        return json.loads(self.sim.reset_json(seed))

    def states(self) -> list[dict[str, Any]]:
        if hasattr(self.sim, "states_msgpack"):
            return msgpack.unpackb(self.sim.states_msgpack(), raw=False, strict_map_key=False)
        return json.loads(self.sim.states_json())

    def step(self, actions: list[list[list[list[float]]]]) -> list[dict[str, Any]]:
        if hasattr(self.sim, "step_msgpack"):
            return msgpack.unpackb(self.sim.step_msgpack(_pack_actions_binary(actions)), raw=False, strict_map_key=False)
        return json.loads(self.sim.step_json(json.dumps(actions)))

    def step_with_states(self, actions: list[list[list[list[float]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if hasattr(self.sim, "step_with_states_msgpack"):
            payload = self.sim.step_with_states_msgpack(_pack_actions_binary(actions))
            outcomes, states = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            return outcomes, states
        if not hasattr(self.sim, "step_with_states_json"):
            return self.step(actions), self.states()
        payload = json.loads(self.sim.step_with_states_json(json.dumps(actions)))
        return payload[0], payload[1]

    def encoded_states(
        self,
        player: int,
        *,
        max_planets: int = 96,
        max_fleets: int = 256,
        include_fleets: bool = True,
    ) -> np.ndarray:
        """Return PPO observations directly from Rust when available.

        The fast path avoids `GameState -> MessagePack -> dict -> encode_state`
        for loops that only need the flat observation tensor. Older extensions
        fall back to the Python encoder with the same public shape.
        """

        dim = 8 + max_planets * 14 + max_fleets * 10
        if hasattr(self.sim, "encoded_states"):
            flat = np.asarray(
                self.sim.encoded_states(int(player), int(max_planets), int(max_fleets), bool(include_fleets)),
                dtype=np.float32,
            )
            return flat.reshape(self.num_envs, dim)

        from .encoding import EncoderConfig, encode_state

        cfg = EncoderConfig(max_planets=max_planets, max_fleets=max_fleets, include_fleets=include_fleets)
        return np.stack([encode_state(state, player, cfg) for state in self.states()]).astype(np.float32, copy=False)


def _pack_actions_binary(actions: list[list[list[list[float]]]]) -> bytes:
    out = bytearray()
    out.extend(_U32.pack(len(actions)))
    for env_actions in actions:
        out.extend(_U32.pack(len(env_actions)))
        for player_actions in env_actions:
            out.extend(_U32.pack(len(player_actions)))
            for move in player_actions:
                out.extend(_MOVE.pack(int(move[0]), float(move[1]), int(move[2])))
    return bytes(out)
