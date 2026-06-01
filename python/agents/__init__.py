__all__ = [
    "HEURISTIC_NAMES",
    "get_heuristic_policies",
    "greedy_agent",
    "defensive_agent",
    "rush_agent",
    "anti_meta_agent",
    "weak_random_agent",
]


def __getattr__(name):
    if name in {"HEURISTIC_NAMES", "get_heuristic_policies"}:
        from . import registry

        return getattr(registry, name)
    if name in {
        "greedy_agent",
        "defensive_agent",
        "rush_agent",
        "anti_meta_agent",
        "weak_random_agent",
    }:
        from . import heuristics

        return getattr(heuristics, name)
    raise AttributeError(name)
