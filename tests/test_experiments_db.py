"""Regression tests for the DuckDB-backed experiment store (python/lab/experiments.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

import python.lab.experiments as ex  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MD = REPO / "EXPERIMENTS.md"
TODO = REPO / "todo.md"

_DATE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*\|")
_TOP_CB_RE = re.compile(r"^- \[[ x~]\]")


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


def test_parse_count_matches_markdown():
    rows = ex.parse_md(MD)
    assert len(rows) == _count_date_rows(MD.read_text(encoding="utf-8"))
    assert len(rows) > 0


def test_every_row_has_valid_status():
    rows = ex.parse_md(MD)
    valid = {"todo", "applied", "rejected", "logged"}
    assert all(r["status"] in valid for r in rows)
    # the markdown has both done (Resultados recentes) and pending (Próximas hipóteses)
    statuses = {r["status"] for r in rows}
    assert "todo" in statuses


def test_command_with_internal_pipe_is_folded_not_split():
    # the "git archive <ref> | tar -x" row must keep its pipe inside the command,
    # not leak into the structured before/after/result/decision fields.
    rows = ex.parse_md(MD)
    piped = [r for r in rows if "git archive" in r["command"] and "tar" in r["command"]]
    assert piped, "expected a row whose command contains a folded ' | '"
    assert "|" in piped[0]["command"]


def test_import_roundtrips_into_duckdb(tmp_path):
    db = tmp_path / "experiments.duckdb"
    n = ex.import_md(MD, db)
    assert n == _count_date_rows(MD.read_text(encoding="utf-8"))
    con = ex.connect(db)
    try:
        (count,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        distinct_status = {s for (s,) in con.execute("SELECT DISTINCT status FROM experiments").fetchall()}
    finally:
        con.close()
    assert count == n
    assert distinct_status <= {"todo", "applied", "rejected", "logged"}


def test_export_preserves_row_count(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(MD, db)
    out = tmp_path / "regenerated.md"
    ex.export_md(db, MD, out)
    assert _count_date_rows(out.read_text(encoding="utf-8")) == _count_date_rows(MD.read_text(encoding="utf-8"))


def test_report_has_all_sections(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(MD, db)
    report = ex.report_md(db)
    for needle in ("# Relatório de Experimentos", "Aplicados", "Rejeitados", "Pendentes", "Por família"):
        assert needle in report


def test_add_experiment_is_tracked(tmp_path):
    db = tmp_path / "experiments.duckdb"
    base = ex.import_md(MD, db)
    new_id = ex.add_experiment(db, date="2026-06-09", idea="PZ test hypothesis", status="todo")
    con = ex.connect(db)
    try:
        (total,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        (status,) = con.execute("SELECT status FROM experiments WHERE id = ?", [new_id]).fetchone()
    finally:
        con.close()
    assert total == base + 1
    assert status == "todo"


def test_add_requires_date_and_idea(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(MD, db)
    with pytest.raises(ValueError):
        ex.add_experiment(db, idea="missing date")


# --- todo.md task tracking (separate `tasks` table; must not touch experiments) ---


def test_parse_todo_matches_toplevel_checkboxes():
    text = TODO.read_text(encoding="utf-8")
    tasks = ex.parse_todo(TODO)
    assert len(tasks) == _count_toplevel_tasks(text)
    assert len(tasks) > 0
    assert all(t["status"] in {"todo", "done", "wip"} for t in tasks)


def test_parse_todo_skips_manual_reference_section():
    # the "# MANUAL" section is reference prose, never tracked as tasks
    assert not any("MANUAL" in (t["section"] or "") for t in ex.parse_todo(TODO))


def test_parse_todo_folds_subitems_into_notes():
    assert any(t["notes"] for t in ex.parse_todo(TODO))


def test_import_tasks_is_isolated_from_experiments(tmp_path):
    db = tmp_path / "experiments.duckdb"
    n_exp = ex.import_md(MD, db)
    n_tasks = ex.import_tasks(TODO, db)
    con = ex.connect(db)
    try:
        (exp_count,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        (task_count,) = con.execute("SELECT COUNT(*) FROM tasks").fetchone()
    finally:
        con.close()
    assert task_count == n_tasks > 0
    # importing tasks must NOT change the experiment rows (no conflict / no pollution)
    assert exp_count == n_exp == _count_date_rows(MD.read_text(encoding="utf-8"))


def test_report_includes_tasks_section(tmp_path):
    db = tmp_path / "experiments.duckdb"
    ex.import_md(MD, db)
    ex.import_tasks(TODO, db)
    report = ex.report_md(db)
    assert "# Tarefas (todo.md)" in report
    assert "A fazer" in report and "Feitas" in report
