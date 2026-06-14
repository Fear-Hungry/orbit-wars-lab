"""Route ladder for the autopilot — the map the six-hats verdict steers along.

Each route is one hypothesis about how to give the fitness gate discriminating
power (it lost it: the 4p gate tied the whole hold-family — FALSE PASS 2026-06-13).
The autopilot runs a route, builds a CHECKPOINT from the measured artifact, and a
six-hats judgment decides the transition: succeed → on_success, exhaust → on_exhaust,
or continue (retry, escalating seeds). Cross-regime was ARCHIVED by the six-hats
analysis (anchors are same-regime → poor calibration coverage).

A route is a plain dict (this module is imported, not serialized). ``success_when``
is a predicate on the parsed artifact. Sentinels STOP_WIN / STOP_EXHAUSTED end the run.
``blocked_reason`` marks a route whose action isn't wired yet: the autopilot emits a
"blocked" checkpoint instead of launching, so the six-hats can decide to wire it (the
/loop re-invokes the model, which implements it) or pivot.
"""
from __future__ import annotations

ENTRY = "calibrate_2p"


def _calibrated(a: dict) -> bool:
    """A gate is calibrated only if it SEPARATES the competitive anchors (not tied),
    orders them WITH the LB (rho >= 0.3), AND has no inversions beyond the LB noise
    band. "Separated" alone is a FALSE PASS — the 2p gate proved you can break the tie
    and still rank backwards (six-hats validation analysis, 2026-06-14)."""
    return ((not a.get("competitive_tied", True))
            and (a.get("rho") if a.get("rho") is not None else -1) >= 0.3
            and not a.get("inversions"))


ROUTES: dict[str, dict] = {
    # ── Route 1: the cheap, falsifiable test the six-hats picked first ──────────
    "calibrate_2p": {
        "kind": "calibrate",
        "desc": "Calibrate the 4 LB anchors in 2p (1v1). Tests the unverified belief "
                "that PGS SEPARATES in 2p where it collapses in 4p.",
        "cmd": ".venv/bin/python -m scripts.research_loop.calibrate --seats 2 "
               "--seeds {seeds} --steps 500 --pool producer,oep,rush,greedy",
        "artifact": "artifacts/research_loop/calibration_2p.json",
        "seeds_schedule": [6, 12],          # attempt 1 → 6 seeds; retry → 12 (anti-noise)
        "max_attempts": 2,
        "success_when": _calibrated,
        "on_success": "search_2p",
        # six-hats (2026-06-14): pure-regime can't match a mixed-regime LB → pivot to
        # the field-mix gate, NOT to bigwave (which the memory says orders wrong).
        "on_exhaust": "calibrate_mix",
    },
    # ── Route 2 (six-hats-derived): the field-MIX gate — the principled fix ─────
    "calibrate_mix": {
        "kind": "calibrate",
        "desc": "Calibrate with the field-weighted gate (0.54·4p + 0.46·2p). The LB is a "
                "regime MIXTURE; pure-4p collapses and pure-2p inverts — only the mix can "
                "match it. Doubles compute (runs both seat counts).",
        "cmd": ".venv/bin/python -m scripts.research_loop.calibrate --seats mix "
               "--seeds {seeds} --steps 500 --pool producer,oep,rush,greedy",
        "artifact": "artifacts/research_loop/calibration_mix.json",
        "seeds_schedule": [6, 10],
        "max_attempts": 2,
        "success_when": _calibrated,
        "on_success": "search_mix",
        "on_exhaust": "calibrate_field_pool",
    },
    # ── Route 2: add the one exploiter that (allegedly) splits the hold family ──
    "calibrate_2p_bigwave": {
        "kind": "calibrate",
        "desc": "2p calibration with pgs_bigwave added to the pool (the only exploiter "
                "the memory says discriminates the hold family — but may order it WRONG).",
        "cmd": ".venv/bin/python -m scripts.research_loop.calibrate --seats 2 "
               "--seeds {seeds} --steps 500 --pool producer,oep,rush,greedy,pgs_bigwave",
        "artifact": "artifacts/research_loop/calibration_2p.json",
        "seeds_schedule": [6, 12],
        "max_attempts": 2,
        "success_when": _calibrated,
        "on_success": "search_2p",
        "on_exhaust": "calibrate_field_pool",
        # ORPHANED from the main path by the six-hats analysis (memory: bigwave orders
        # the hold-family WRONG). Kept for reference / manual rewire only.
        "blocked_reason": "pgs_bigwave is league-only; needs wiring into "
                          "get_isolated_opponents / the eval pool before this route runs.",
    },
    # ── Route 3: align the pool with the real field (attacks population bias) ───
    "calibrate_field_pool": {
        "kind": "calibrate",
        "desc": "Calibrate against a field-representative opponent pool derived from real "
                "episodes (EpisodeService) — attacks the population bias that falsified the league.",
        "cmd": None,
        "artifact": "artifacts/research_loop/calibration_field.json",
        "seeds_schedule": [8],
        "max_attempts": 1,
        "success_when": _calibrated,
        "on_success": "search_mix",
        "on_exhaust": "STOP_EXHAUSTED",
        "blocked_reason": "needs a field-derived opponent pool built from EpisodeService "
                          "episodes; not wired. Last rung — if this fails the ladder is spent.",
    },
    # ── Exploit a calibrated signal: autonomous search on a TRUSTED gate ────────
    "search_2p": {
        "kind": "search",
        "desc": "Run self_research in 2p on the now-calibrated gate (the only mode where "
                "the fitness was shown to order the LB). NEVER submits.",
        "cmd": ".venv/bin/python -m scripts.research_loop.self_research --seats 2 "
               "--max-hours {hours} --seeds 6 --steps 500 --candidates-per-batch 4 "
               "--confirm-seeds 12 --pool producer,oep,rush,greedy",
        "artifact": "artifacts/research_loop/self_research_state.json",
        "hours_schedule": [3, 3],
        "max_attempts": 2,
        # success = champion beats the holdwave LB-champion on the gate AND held confirmation
        "success_when": lambda a: bool(a.get("beats_holdwave_anchor"))
                                  and bool((a.get("confirm") or {}).get("held_vs_start")),
        "on_success": "STOP_WIN",
        "on_exhaust": "STOP_PLATEAU",
    },
    "search_mix": {
        "kind": "search",
        "desc": "Run self_research on the calibrated field-MIX gate (the strongest "
                "validated signal). NEVER submits.",
        "cmd": ".venv/bin/python -m scripts.research_loop.self_research --seats mix "
               "--max-hours {hours} --seeds 6 --steps 500 --candidates-per-batch 4 "
               "--confirm-seeds 10 --pool producer,oep,rush,greedy",
        "artifact": "artifacts/research_loop/self_research_state.json",
        "hours_schedule": [4, 4],
        "max_attempts": 2,
        "success_when": lambda a: bool(a.get("beats_holdwave_anchor"))
                                  and bool((a.get("confirm") or {}).get("held_vs_start")),
        "on_success": "STOP_WIN",
        "on_exhaust": "STOP_PLATEAU",
    },
}

TERMINALS = {
    "STOP_WIN": "A candidate beat the holdwave LB-champion on a CALIBRATED gate and held "
                "confirmation. Strongest result yet — surface for human review + 1/day submit.",
    "STOP_EXHAUSTED": "Every calibration route failed to give the gate discriminating power. "
                      "The robustness-gate family is the wrong instrument; rethink fitness.",
    "STOP_PLATEAU": "The gate is calibrated but the autonomous search plateaued without "
                    "beating the champion. Surface the best candidate; widen search space next.",
}


def is_terminal(route_name: str) -> bool:
    return route_name in TERMINALS
