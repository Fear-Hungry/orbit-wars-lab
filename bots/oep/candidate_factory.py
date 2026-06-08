"""Family H — CandidateFactory.

The narrow line "Producer/OEP + selection/rollout over the same kind of plan"
saturated (E1/E2/C1/E3 all stalled or regressed vs the -0.045 floor). The
measured bottleneck is a *poor candidate generator*, not the selector: E3 added
diverse candidates + rollout value and regressed to -0.201; the shallow beam /
plan-memory variants were chosen 1.3%/2.3% of the time — minimal perturbation
of the greedy plan, not real exploration.

This module reframes the agent as a **factory of diverse candidates**. Each
candidate *family* is a function ``obs -> moves`` with the exact contract of the
Kaggle ``agent(obs)`` entrypoint (``[] == deliberately launch nothing``), so:

* Producer and OEP plug in unchanged as two candidates among many — not the
  centre of the universe;
* new Family-H generators (production-projected attack, timeline-risk, hammer /
  multiprong, regroup / dominance, RHEA macro search, ...) are just new
  functions of the same shape, with no coupling to the planner's internal
  tensor state (which keeps the simulator-parity invariant intact).

The oracle (H1, ``scripts/oracle_candidates.py``) ranks candidates by their true
downstream margin to decide whether the *generator* or the *selector* is the
bottleneck. The runtime hyper-heuristic (H7) decides which family earns compute
in a given state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Obs = dict[str, Any]
Move = list[float]
Moves = list[Move]

#: A candidate family: given a player's observation, return that player's moves
#: for this step. ``[]`` means "launch nothing this turn" (a real decision, not
#: a failure). Same shape as ``bots/*/agent.py:agent(obs)``.
CandidateGenerator = Callable[[Obs], Moves]


@dataclass
class PlanCandidate:
    """One candidate plan for the current decision."""

    family: str
    moves: Moves = field(default_factory=list)


def _producer_generator() -> CandidateGenerator:
    # Fresh, isolated Producer runtime (own memory) — never shared across
    # families or envs. Reuses the vendored Producer, not a reimplementation.
    from bots.producer.agent import make_agent

    return make_agent()


def _oep_generator() -> CandidateGenerator:
    from bots.oep.agent import make_agent

    return make_agent()


#: Registry of available families. Each entry builds a *fresh, isolated*
#: generator so candidates never share runtime/memory. H2-H7 register here as
#: they land (see ``register_family``).
_FAMILY_BUILDERS: dict[str, Callable[[], CandidateGenerator]] = {
    "producer": _producer_generator,
    "oep": _oep_generator,
}

#: Families enumerated by default. The two known archetypes (Producer = floor,
#: OEP = search) plus the Family-H generators (H2-H5). The oracle measures
#: whether the H families lift ``oracle_best_margin`` above the Producer mirror.
DEFAULT_FAMILIES: tuple[str, ...] = (
    "producer",
    "oep",
    "production_projected_attack",
    "timeline_risk",
    "hammer_multiprong",
    "regroup_dominance",
    "rhea_macro",
    "hyperheuristic",
)


def _register_builtin_families() -> None:
    # Importing these modules registers H2-H7 into ``_FAMILY_BUILDERS``. Done
    # lazily at the bottom of the module so the registry functions exist first
    # (avoids the import cycle family_h/rhea/hyperheuristic -> candidate_factory).
    from bots.oep import family_h as _family_h  # noqa: F401
    from bots.oep import hyperheuristic as _hyperheuristic  # noqa: F401
    from bots.oep import rhea as _rhea  # noqa: F401


def available_families() -> tuple[str, ...]:
    """Names of every registered candidate family, in registration order."""

    return tuple(_FAMILY_BUILDERS)


def register_family(name: str, builder: Callable[[], CandidateGenerator]) -> None:
    """Register a new candidate family (H2-H7 call this at import time).

    ``builder`` must return a *fresh* ``obs -> moves`` generator each call so the
    factory can hand out isolated, side-effect-free instances.
    """

    if name in _FAMILY_BUILDERS:
        raise ValueError(f"candidate family already registered: {name!r}")
    _FAMILY_BUILDERS[name] = builder


class CandidateFactory:
    """Produces multiple :class:`PlanCandidate` per state from chosen families.

    Generators are **stateful** (Producer/OEP accumulate per-step memory). Two
    usage modes:

    * **Per-step play** (runtime / oracle rollouts): build one factory and call
      :meth:`candidates` once per consecutive game step — memory tracks the game
      correctly.
    * **Isolated-state sampling**: pass ``fresh=True`` to rebuild generators on
      every call so an arbitrary, non-consecutive state is judged without stale
      memory leaking in.
    """

    def __init__(self, families: tuple[str, ...] = DEFAULT_FAMILIES) -> None:
        unknown = [name for name in families if name not in _FAMILY_BUILDERS]
        if unknown:
            raise ValueError(f"unknown candidate families: {unknown}")
        self._families: tuple[str, ...] = tuple(families)
        self._generators: dict[str, CandidateGenerator] = {}

    @property
    def families(self) -> tuple[str, ...]:
        return self._families

    def _generator(self, family: str, *, fresh: bool) -> CandidateGenerator:
        if fresh:
            return _FAMILY_BUILDERS[family]()
        gen = self._generators.get(family)
        if gen is None:
            gen = _FAMILY_BUILDERS[family]()
            self._generators[family] = gen
        return gen

    def candidates(self, obs: Obs, *, fresh: bool = False) -> list[PlanCandidate]:
        """Return one :class:`PlanCandidate` per configured family for ``obs``."""

        out: list[PlanCandidate] = []
        for family in self._families:
            moves = self._generator(family, fresh=fresh)(obs)
            out.append(PlanCandidate(family=family, moves=list(moves) if moves else []))
        return out


# Register H2-H5 at import so DEFAULT_FAMILIES and the oracle see the full set.
_register_builtin_families()
