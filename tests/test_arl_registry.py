"""Pin registry.select_parents' frontier hygiene (scripts/research_loop/registry.py).

A smoke/dry-run row is a WIRING check at a tiny budget; its fitness is not
comparable to a full research eval, so it must never become a parent (promotion
bar). Otherwise a 24-seed/500-step candidate is judged apples-to-oranges against
a 2-seed smoke parent. This test locks that exclusion in.
"""
from __future__ import annotations

from python.lab.experiments import add_experiment
from scripts.research_loop import registry


def _add(db, idea, fitness, status):
    add_experiment(db_path=db, date="2026-06-18", idea=idea,
                   result=f"fitness={fitness:.4f} | decision=x", status=status, tags="ARL")


def test_select_parents_excludes_smoke_even_with_higher_fitness(tmp_path):
    db = tmp_path / "e.duckdb"
    # research candidate (a valid parent) ...
    _add(db, 'ARL[research] perturb x | genome={"scripts":"hold","wave_min_ships":60.0}', -0.5000, "rejected")
    # ... and a smoke row with HIGHER fitness that must still be excluded.
    _add(db, 'ARL[smoke] perturb x | genome={"scripts":"hold","wave_min_ships":99.0}', -0.4000, "logged")
    _add(db, 'ARL[dry-run] perturb x | genome={"scripts":"hold","wave_min_ships":42.0}', -0.1000, "logged")

    parents = registry.select_parents(5, db_path=db)
    assert len(parents) == 1, "only the research row is a valid parent"
    genome, fit = parents[0]
    assert fit == -0.5000 and genome["wave_min_ships"] == 60.0


def test_select_parents_keeps_legacy_research_rows(tmp_path):
    db = tmp_path / "e.duckdb"
    # legacy runner rows (no mode tag) stay eligible.
    _add(db, 'ARL g0c0 genome={"scripts":"hold","wave_min_ships":60.0}', -0.5000, "applied")
    parents = registry.select_parents(5, db_path=db)
    assert len(parents) == 1 and parents[0][1] == -0.5000
