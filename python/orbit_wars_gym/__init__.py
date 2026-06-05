"""Orbit Wars gym — Rust-backed simulator for TRAINING and evaluation only.

This is the fast path: it drives the Rust core (``orbit_wars_core`` via PyO3) to
run thousands of games per generation. It is NOT submitted to Kaggle — the
submission cannot depend on Rust (see DECISIONS.md D10/D11). For the pure-Python
engine that ships in the submission, see the separate ``orbit_lite`` package.

Reached from the repo root as ``orbit_wars_gym`` via the import shim at the repo
root. Heavy optional deps (gymnasium, pettingzoo) are imported lazily.
"""

__all__ = ["OrbitWarsGymEnv", "OrbitWarsParallelEnv"]


def __getattr__(name):
    if name == "OrbitWarsGymEnv":
        from .gym_env import OrbitWarsGymEnv
        return OrbitWarsGymEnv
    if name == "OrbitWarsParallelEnv":
        from .parallel_env import OrbitWarsParallelEnv
        return OrbitWarsParallelEnv
    raise AttributeError(name)
