from __future__ import annotations

from statistics import mean

from python.orbit_wars_gym.snapshots import official_seeded_initial_snapshot

from orbit_wars_gym.backend import RustBatchBackend


def _field(entity, index: int, key: str):
    if isinstance(entity, dict):
        return entity[key]
    return entity[index]


def _is_rotating(planet) -> bool:
    x = float(_field(planet, 2, "x"))
    y = float(_field(planet, 3, "y"))
    radius = float(_field(planet, 4, "radius"))
    orbital = ((x - 50.0) ** 2 + (y - 50.0) ** 2) ** 0.5
    return orbital + radius < 50.0


def _summary(state: dict) -> dict[str, float]:
    planets = state["planets"]
    groups = [planets[idx : idx + 4] for idx in range(0, len(planets), 4)]
    return {
        "planets": float(len(planets)),
        "groups": float(len(groups)),
        "static_groups": float(sum(1 for group in groups if not _is_rotating(group[0]))),
        "orbiting_groups": float(sum(1 for group in groups if _is_rotating(group[0]))),
        "mean_production": mean(float(_field(planet, 6, "production")) for planet in planets),
        "mean_ships": mean(float(_field(planet, 5, "ships")) for planet in planets),
        "angular_velocity": float(state["angular_velocity"]),
    }


def _aggregate(samples: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(sample[key] for sample in samples) for key in samples[0]}


def test_training_generator_distribution_tracks_official_openings():
    # Distributional check, not per-seed parity: the Rust generator (ChaCha8) and the
    # official generator (Python Mersenne Twister) draw different maps for the same
    # integer seed by design (see docs/PARITY.md — do NOT reproduce the Python RNG in
    # Rust). Tolerances below compare aggregate distributions, so the sample must be
    # large enough that cross-RNG variance averages out; 16 seeds was under-powered.
    seeds = list(range(256))
    for players in (2, 4):
        backend = RustBatchBackend(num_envs=1, num_players=players, seed=0)
        local_samples = [_summary(backend.reset(seed)[0]) for seed in seeds]
        official_samples = [_summary(official_seeded_initial_snapshot(players, seed)) for seed in seeds]

        local = _aggregate(local_samples)
        official = _aggregate(official_samples)

        assert abs(local["groups"] - official["groups"]) <= 1.0
        assert abs(local["static_groups"] - official["static_groups"]) <= 0.75
        assert abs(local["orbiting_groups"] - official["orbiting_groups"]) <= 0.75
        assert abs(local["mean_production"] - official["mean_production"]) <= 0.2
        assert abs(local["mean_ships"] - official["mean_ships"]) <= 4.0
        assert abs(local["angular_velocity"] - official["angular_velocity"]) <= 0.01
