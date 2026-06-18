"""Pure metrics parser + keep/discard policy for the Auto-Research Loop (ARL).

This module is the AUDITABLE core of the loop and is deliberately **pure** (no
torch, no Rust ``.so``, no DuckDB, no evaluation) so it can be unit-tested in
milliseconds and so the keep/discard logic can be read in one place.

Two responsibilities, both demanded by ``goal.md``:

1. ``parse_metrics`` — normalise whatever the evaluator (or a saved report)
   produced into a single ``ParsedMetrics`` shape, classify *faults*
   (timeout/invalid/bad-status/fallback/exception/p95-over-budget) and decide
   whether the sample is even valid.
2. ``keep_or_discard`` — the promotion policy. It returns exactly one of the
   five contract decisions::

       promoted | rejected | inconclusive | needs_more_seeds | technical_fail

   **Invariant (goal.md):** ``technical_fail`` is checked FIRST and NEVER becomes
   a competitive ``rejected`` — a broken run is a harness problem, not evidence
   that the candidate is weak. Only ``promoted``/``rejected`` are *competitive*
   verdicts; the other three are non-committal.

The policy encodes the project's hard-won lessons as guardrails:
- promotion requires enough seeds (memory: "never decide by 12-16 seeds" — single
  pass fitness inflates ~3-4x), else ``needs_more_seeds``;
- a flat top (delta within a noise band) is ``inconclusive``, never a promotion
  (memory: local_league_is_submission_gate — the LB top is flat and noisy);
- any fault zeroes the verdict to ``technical_fail``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# The five decisions of the iteration contract (goal.md §"Arquitetura alvo").
DECISIONS = ("promoted", "rejected", "inconclusive", "needs_more_seeds", "technical_fail")
# Only these two are competitive verdicts about candidate strength.
COMPETITIVE_DECISIONS = ("promoted", "rejected")

# Canonical fault counters parse_metrics looks for. Any > 0 (or p95_over_budget)
# forces technical_fail. p95_ms / act_timeout_ms are context, not counters.
_FAULT_COUNTERS = ("timeouts", "invalid_moves", "bad_status", "fallbacks", "exceptions")


def status_for(decision: str) -> str:
    """Map a contract decision to the shared ``experiments`` status column.

    ``promoted`` -> ``applied``; ``rejected`` -> ``rejected``; everything else
    (including ``technical_fail``) -> ``logged``. Crucially ``technical_fail``
    must NOT land in ``rejected`` — see module docstring.
    """
    if decision == "promoted":
        return "applied"
    if decision == "rejected":
        return "rejected"
    return "logged"


def _finite(x) -> float | None:
    """Return ``float(x)`` if it is a finite number, else ``None``."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int0(x) -> int:
    """Best-effort non-negative int (None/garbage -> 0)."""
    try:
        return max(0, int(x))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class ParsedMetrics:
    """Normalised view of one evaluation. ``valid`` gates competitive verdicts."""

    valid: bool
    death_rate: float | None
    mean_margin: float | None
    mean_final_planets: float | None
    n_seeds: int
    steps: int | None
    seats: object
    pool: tuple
    per_opponent: dict
    faults: dict
    note: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    def fault_summary(self) -> str:
        hits = [f"{k}={self.faults.get(k, 0)}" for k in _FAULT_COUNTERS if _int0(self.faults.get(k)) > 0]
        if self.faults.get("p95_over_budget"):
            hits.append(f"p95={self.faults.get('p95_ms')}ms>budget={self.faults.get('act_timeout_ms')}ms")
        return ", ".join(hits)


def has_faults(faults: dict) -> bool:
    """True if any canonical fault counter is positive or p95 exceeds budget."""
    if any(_int0(faults.get(k)) > 0 for k in _FAULT_COUNTERS):
        return True
    return bool(faults.get("p95_over_budget"))


def parse_metrics(raw, *, act_timeout_ms: float = 1000.0, n_seeds: int | None = None) -> ParsedMetrics:
    """Normalise an evaluator/report payload into ``ParsedMetrics``.

    ``raw`` may be a dict (the evaluator's return) or a JSON string (a saved
    report). Missing/degenerate ``death_rate``/``mean_margin`` -> ``valid=False``
    (no valid sample). Fault counters default to 0 when absent so a clean run is
    fault-free, but any present-and-positive counter is surfaced verbatim.

    ``n_seeds`` overrides the seed count parsed from the payload (the runner knows
    the budget it asked for; the payload echoes it back but we trust the caller).
    """
    note = ""
    if raw is None:
        # "No eval ran" (dry-run / no metrics) — a missing sample, NOT a fault.
        # Routing this through technical_fail would mislabel every dry-run.
        return ParsedMetrics(
            valid=False, death_rate=None, mean_margin=None, mean_final_planets=None,
            n_seeds=_int0(n_seeds), steps=None, seats=None, pool=(), per_opponent={},
            faults={"timeouts": 0, "invalid_moves": 0, "bad_status": 0, "fallbacks": 0,
                    "exceptions": 0, "p95_ms": None, "act_timeout_ms": float(act_timeout_ms),
                    "p95_over_budget": 0},
            note="no eval ran (dry-run / no metrics)", raw={},
        )
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return ParsedMetrics(
                valid=False, death_rate=None, mean_margin=None, mean_final_planets=None,
                n_seeds=_int0(n_seeds), steps=None, seats=None, pool=(), per_opponent={},
                faults={"exceptions": 1, "p95_ms": None, "act_timeout_ms": float(act_timeout_ms),
                        "p95_over_budget": 0},
                note="unparseable metrics payload (not JSON)", raw={},
            )
    if not isinstance(raw, dict):
        return ParsedMetrics(
            valid=False, death_rate=None, mean_margin=None, mean_final_planets=None,
            n_seeds=_int0(n_seeds), steps=None, seats=None, pool=(), per_opponent={},
            faults={"exceptions": 1, "p95_ms": None, "act_timeout_ms": float(act_timeout_ms),
                    "p95_over_budget": 0},
            note=f"metrics payload is {type(raw).__name__}, expected dict", raw={},
        )

    death = _finite(raw.get("death_rate"))
    margin = _finite(raw.get("mean_margin"))
    planets = _finite(raw.get("mean_final_planets"))
    seeds = _int0(n_seeds) if n_seeds is not None else _int0(raw.get("seeds"))
    steps = raw.get("steps")
    steps = int(steps) if isinstance(steps, (int, float)) else None
    pool = tuple(raw.get("pool", ())) if isinstance(raw.get("pool"), (list, tuple)) else ()
    per_opp = raw.get("per_opponent") if isinstance(raw.get("per_opponent"), dict) else {}

    p95 = raw.get("p95_ms", raw.get("p95"))
    p95_ms = _finite(p95)
    faults = {k: _int0(raw.get(k, raw.get(_ALIASES.get(k, k), 0))) for k in _FAULT_COUNTERS}
    if raw.get("error"):
        faults["exceptions"] = max(faults["exceptions"], 1)
        note = (note + " | " if note else "") + f"error: {str(raw.get('error'))[:120]}"
    faults["p95_ms"] = p95_ms
    faults["act_timeout_ms"] = float(act_timeout_ms)
    faults["p95_over_budget"] = int(p95_ms is not None and p95_ms > act_timeout_ms)

    valid = death is not None and margin is not None
    if not valid and not note:
        note = "no valid sample (death_rate/mean_margin missing or non-finite)"

    return ParsedMetrics(
        valid=valid, death_rate=death, mean_margin=margin, mean_final_planets=planets,
        n_seeds=seeds, steps=steps, seats=raw.get("seats"), pool=pool, per_opponent=per_opp,
        faults=faults, note=note, raw=raw,
    )


# Tolerated alternate spellings for fault counters seen in different report shapes.
_ALIASES = {
    "bad_status": "bad_statuses",
    "fallbacks": "fallback_errors",
}


@dataclass(frozen=True)
class Decision:
    """Outcome of the keep/discard policy for one candidate."""

    decision: str
    reason: str
    fitness: float | None = None
    parent_fitness: float | None = None
    delta: float | None = None

    @property
    def competitive(self) -> bool:
        return self.decision in COMPETITIVE_DECISIONS

    @property
    def status(self) -> str:
        return status_for(self.decision)


def keep_or_discard(
    metrics: ParsedMetrics,
    *,
    fitness: float | None,
    parent_fitness: float | None,
    min_promotion_seeds: int,
    noise_band: float,
) -> Decision:
    """Pure promotion policy → exactly one of ``DECISIONS``.

    Order of checks (the order IS the guarantee):

    1. **Faults dominate** → ``technical_fail`` (never ``rejected``).
    2. No valid sample → ``needs_more_seeds`` if under the seed floor, else
       ``inconclusive``.
    3. No parent bar to compare against → ``inconclusive`` (non-competitive).
    4. Under the seed floor (even with a valid sample) → ``needs_more_seeds``.
       This is what keeps a *smoke* run from ever promoting competitively.
    5. ``delta = fitness - parent_fitness``:
       ``> +noise_band`` → ``promoted``; ``< -noise_band`` → ``rejected``;
       otherwise (flat top) → ``inconclusive``.
    """
    f = metrics.faults or {}
    if has_faults(f):
        return Decision("technical_fail", f"faults present ({metrics.fault_summary()}); "
                        "harness/runtime problem, not competitive evidence")

    if not metrics.valid or fitness is None:
        if metrics.n_seeds < min_promotion_seeds:
            return Decision("needs_more_seeds",
                            f"no valid sample and only {metrics.n_seeds} seeds "
                            f"(< {min_promotion_seeds}); {metrics.note}".strip())
        return Decision("inconclusive", f"no valid sample at {metrics.n_seeds} seeds; {metrics.note}".strip())

    if parent_fitness is None:
        return Decision("inconclusive", "no parent fitness to compare against (baseline measurement)",
                        fitness=fitness)

    if metrics.n_seeds < min_promotion_seeds:
        return Decision("needs_more_seeds",
                        f"{metrics.n_seeds} seeds < promotion floor {min_promotion_seeds}; "
                        "sample too small to promote (triage-inflation guard)",
                        fitness=fitness, parent_fitness=parent_fitness,
                        delta=fitness - parent_fitness)

    delta = fitness - parent_fitness
    if delta > noise_band:
        return Decision("promoted", f"fitness {fitness:+.4f} beats parent {parent_fitness:+.4f} "
                        f"by {delta:+.4f} (> noise band {noise_band:.4f})",
                        fitness=fitness, parent_fitness=parent_fitness, delta=delta)
    if delta < -noise_band:
        return Decision("rejected", f"fitness {fitness:+.4f} below parent {parent_fitness:+.4f} "
                        f"by {delta:+.4f} (< -noise band {noise_band:.4f})",
                        fitness=fitness, parent_fitness=parent_fitness, delta=delta)
    return Decision("inconclusive", f"fitness {fitness:+.4f} within noise band "
                    f"({noise_band:.4f}) of parent {parent_fitness:+.4f} (delta {delta:+.4f}); "
                    "cannot discriminate",
                    fitness=fitness, parent_fitness=parent_fitness, delta=delta)


# --------------------------------------------------------------------------- #
# Promotion handoff (Mode 3) — pure selection + command building.
#
# The ARL's local verdict is NOT a competitive promotion (the ruler is a FALSE
# PASS — see docs/auto_research_pipeline.md §10). So the handoff only PRE-FILTERS
# which candidates are worth the expensive *seat-rotated* ruler: those that
# survived the veto AND beat the parent on the (unverified) local fitness. The
# real verdict is league_submit_ruler.py, run by a human.
# --------------------------------------------------------------------------- #
def select_survivors(iterations, *, noise_band: float) -> list:
    """Iterations worth handing to the seat-rotated ruler.

    Keep a candidate iff it was not vetoed (decision ``promoted``/``inconclusive``
    — note ``promoted`` is downgraded to ``inconclusive`` when the ruler is
    untrusted, so we filter on the raw ``delta``, not the label) AND it beat the
    parent by more than the noise band. Excludes ``rejected`` (delta < -band),
    ``technical_fail`` (delta None) and ``needs_more_seeds`` (sample too small).
    """
    out = []
    for it in iterations:
        if it.get("decision") not in ("promoted", "inconclusive"):
            continue
        d = it.get("delta")
        if isinstance(d, (int, float)) and d > noise_band:
            out.append(it)
    return out


def candidate_name(run_id: str) -> str:
    """Filesystem/FACTORIES-safe candidate name for a survivor (``arl_...``)."""
    safe = "".join(c if c.isalnum() else "_" for c in str(run_id))
    return ("arl_" + safe).strip("_")


def build_promotion_command(names, *, profile: str = "strong") -> str:
    """The seat-rotated ruler command for the survivors (empty string if none)."""
    names = [n for n in names if n]
    if not names:
        return ""
    return (".venv/bin/python scripts/league_submit_ruler.py "
            f"--candidates {' '.join(names)} --profile {profile}")
