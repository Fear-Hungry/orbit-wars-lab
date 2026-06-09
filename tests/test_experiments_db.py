"""Regression tests for the DuckDB-backed experiment store (python/lab/experiments.py).

experiments.duckdb is the canonical (git-tracked) store; experiments are added
via add(). parse_md/import_md bulk-load from a markdown dump and are tested here
against a fixture, NOT the live store. Tasks come from todo.md (git-tracked).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

import python.lab.experiments as ex  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TODO = REPO / "todo.md"
_DATE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*\|")
_TOP_CB_RE = re.compile(r"^- \[[ x~]\]")

SAMPLE_MD = """# Experimentos

## Resultados recentes

```text
2026-06-01 | A ideia aplicada | rtk cmd a | antes 0.1 | depois 0.3 | resultado A | aplicar como default
2026-06-02 | G1 ideia rejeitada | rtk git archive ref | tar -x foo | b2 | a2 | piorou | rejeitar, regrediu
```

## Próximas hipóteses

```text
2026-06-03 | C ideia futura | rtk cmd c | - | - | - | testar depois
```
"""


def _sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "EXPERIMENTS.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    return p


def _count_date_rows(text: str) -> int:
    return sum(1 for line in text.splitlines() if _DATE_RE.match(line))


def _count_toplevel_tasks(text: str) -> int:
    n, in_manual = 0, False
    for line in text.splitlines():
        if line.startswith("# "):
            in_manual = "MANUAL" in line
        if not in_manual and _TOP_CB_RE.match(line):
            n += 1
    return n


# --- markdown parser (against a controlled fixture) ---


def test_parse_md_fields_and_status(tmp_path):
    rows = ex.parse_md(_sample_md(tmp_path))
    assert len(rows) == 3
    by_idea = {r["idea"]: r for r in rows}
    assert by_idea["A ideia aplicada"]["status"] == "applied"
    assert by_idea["G1 ideia rejeitada"]["status"] == "rejected"
    assert by_idea["C ideia futura"]["status"] == "todo"  # under Próximas hipóteses
    assert by_idea["G1 ideia rejeitada"]["tags"] == "G1"


def test_parse_md_folds_command_pipe(tmp_path):
    rows = ex.parse_md(_sample_md(tmp_path))
    rej = next(r for r in rows if r["idea"] == "G1 ideia rejeitada")
    # the command held an internal ' | ' (git archive ... | tar -x ...): must stay in command,
    # not leak into the structured before/after/result/decision fields.
    assert "tar -x foo" in rej["command"]
    assert rej["metric_before"] == "b2"
    assert rej["decision"].startswith("rejeitar")


def test_import_md_roundtrips(tmp_path):
    db = tmp_path / "experiments.duckdb"
    n = ex.import_md(_sample_md(tmp_path), db)
    assert n == 3
    con = ex.connect(db)
    try:
        (count,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
    finally:
        con.close()
    assert count == 3


def test_export_is_standalone_and_preserves_rows(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(_sample_md(tmp_path), db)
    out = tmp_path / "dump.md"
    ex.export_md(db, out)
    assert _count_date_rows(out.read_text(encoding="utf-8")) == 3


# --- DB-native add flow (the canonical way now) ---


def test_add_experiment_is_tracked(tmp_path):
    db = tmp_path / "experiments.duckdb"
    new_id = ex.add_experiment(db, date="2026-06-09", idea="PZ native add", status="todo")
    con = ex.connect(db)
    try:
        (total,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        (status,) = con.execute("SELECT status FROM experiments WHERE id = ?", [new_id]).fetchone()
    finally:
        con.close()
    assert total == 1
    assert status == "todo"


def test_add_requires_date_and_idea(tmp_path):
    db = tmp_path / "experiments.duckdb"
    with pytest.raises(ValueError):
        ex.add_experiment(db, idea="missing date")


def test_report_has_all_sections(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(_sample_md(tmp_path), db)
    report = ex.report_md(db)
    for needle in ("# Relatório de Experimentos", "Aplicados", "Rejeitados", "Pendentes"):
        assert needle in report


# --- todo.md task tracking (separate `tasks` table; must not touch experiments) ---


def test_parse_todo_matches_toplevel_checkboxes():
    text = TODO.read_text(encoding="utf-8")
    tasks = ex.parse_todo(TODO)
    assert len(tasks) == _count_toplevel_tasks(text)
    assert len(tasks) > 0
    assert all(t["status"] in {"todo", "done", "wip"} for t in tasks)


def test_parse_todo_skips_manual_reference_section():
    assert not any("MANUAL" in (t["section"] or "") for t in ex.parse_todo(TODO))


def test_parse_todo_folds_subitems_into_notes():
    assert any(t["notes"] for t in ex.parse_todo(TODO))


def test_import_tasks_is_isolated_from_experiments(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.add_experiment(db, date="2026-06-09", idea="native exp", status="logged")
    n_tasks = ex.import_tasks(TODO, db)
    con = ex.connect(db)
    try:
        (exp_count,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        (task_count,) = con.execute("SELECT COUNT(*) FROM tasks").fetchone()
    finally:
        con.close()
    assert task_count == n_tasks > 0
    assert exp_count == 1  # importing tasks must NOT touch experiments


def test_report_includes_tasks_section(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.add_experiment(db, date="2026-06-09", idea="native exp", status="applied")
    ex.import_tasks(TODO, db)
    report = ex.report_md(db)
    assert "# Tarefas (todo.md)" in report
    assert "A fazer" in report and "Feitas" in report
