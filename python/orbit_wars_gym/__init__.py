"""Orbit Wars training package.

Heavy optional dependencies such as gymnasium and pettingzoo are imported lazily.
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
