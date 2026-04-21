from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


def make_round_robin(ids: Sequence[str]) -> list[tuple[str, str]]:
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


def make_elo_nearby_pairs(ids: Sequence[str], ratings: dict[str, float], pairs_per_agent: int = 4, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    result: set[tuple[str, str]] = set()
    sorted_ids = sorted(ids, key=lambda x: ratings.get(x, 1000.0))
    for idx, aid in enumerate(sorted_ids):
        lo = max(0, idx - 8)
        hi = min(len(sorted_ids), idx + 9)
        candidates = [x for x in sorted_ids[lo:hi] if x != aid]
        rng.shuffle(candidates)
        for bid in candidates[:pairs_per_agent]:
            result.add(tuple(sorted((aid, bid))))
    return sorted(result)


def make_elo_diverse_pairs(
    ids: Sequence[str],
    ratings: dict[str, float],
    behaviors: dict[str, Any],
    pairs_per_agent: int = 4,
    seed: int = 0,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    result: set[tuple[str, str]] = set()
    sorted_ids = sorted(ids, key=lambda x: ratings.get(x, 1000.0))

    def diversity_score(left: str, right: str) -> int:
        left_behavior = behaviors.get(left)
        right_behavior = behaviors.get(right)
        if left_behavior is None or right_behavior is None:
            return 0
        return (
            abs(int(left_behavior.expansion_bin) - int(right_behavior.expansion_bin))
            + abs(int(left_behavior.aggression_bin) - int(right_behavior.aggression_bin))
            + abs(int(left_behavior.defense_bin) - int(right_behavior.defense_bin))
            + abs(int(left_behavior.fleet_size_bin) - int(right_behavior.fleet_size_bin))
        )

    for idx, aid in enumerate(sorted_ids):
        lo = max(0, idx - 8)
        hi = min(len(sorted_ids), idx + 9)
        candidates = [candidate for candidate in sorted_ids[lo:hi] if candidate != aid]
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda bid: (
                diversity_score(aid, bid),
                -abs(ratings.get(aid, 1000.0) - ratings.get(bid, 1000.0)),
            ),
            reverse=True,
        )
        for bid in candidates[:pairs_per_agent]:
            result.add(tuple(sorted((aid, bid))))
    return sorted(result)
