from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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

    The current binding uses JSON for simplicity. This is acceptable for smoke
    tests and parity probes. For serious training, replace step_json/states_json
    with ndarray zero-copy buffers.
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
        return json.loads(self.sim.reset_json(seed))

    def states(self) -> list[dict[str, Any]]:
        return json.loads(self.sim.states_json())

    def step(self, actions: list[list[list[list[float]]]]) -> list[dict[str, Any]]:
        return json.loads(self.sim.step_json(json.dumps(actions)))

    def step_with_states(self, actions: list[list[list[list[float]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not hasattr(self.sim, "step_with_states_json"):
            return self.step(actions), self.states()
        payload = json.loads(self.sim.step_with_states_json(json.dumps(actions)))
        return payload[0], payload[1]
