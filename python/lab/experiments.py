"""Experiment tracking store backed by DuckDB.

``experiments.duckdb`` is the **canonical, git-tracked store** of tested
hypotheses (it replaced the old hand-written 120 KB ``EXPERIMENTS.md``). Add
experiments with ``add`` (then commit the ``.duckdb``); read with
``list``/``query``/``stats``/``report``; dump a markdown view with ``export``.
``parse_md``/``import_md`` remain available to bulk-load from a markdown dump.

Experiment fields (the ``add`` columns; commands may contain their own ``|``,
which ``parse_md`` folds back into the command field)::

    date | idea | command | before | after | result | decision

Status is derived so the log is trackable:
- rows under ``## Próximas hipóteses`` -> ``todo`` (not done yet);
- rows under ``## Resultados recentes`` -> ``applied`` / ``rejected`` / ``logged``
  inferred from the decision text.

Work tasks from ``todo.md`` are tracked too, in a SEPARATE ``tasks`` table (same
DB file) so they never pollute the experiment metrics. ``import`` rebuilds both;
``report`` renders experiments and tasks in distinct sections.

CLI::

    python -m python.lab.experiments add --date 2026-06-09 --idea "..." [--command ...] [--decision ...] [--status ...]
    python -m python.lab.experiments import [--todo todo.md] [--md dump.md] [--db experiments.duckdb]
    python -m python.lab.experiments list   [--status todo] [--since 2026-06-01] [--limit N]
    python -m python.lab.experiments tasks  [--status todo|wip|done]
    python -m python.lab.experiments query  "SELECT ... FROM experiments ..."
    python -m python.lab.experiments stats
    python -m python.lab.experiments export [--out EXPERIMENTS.export.md]
    python -m python.lab.experiments report [--out docs/EXPERIMENTS_REPORT.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import duckdb

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MD = REPO / "EXPERIMENTS.md"
DEFAULT_TODO = REPO / "todo.md"
DEFAULT_DB = REPO / "experiments.duckdb"

_DATE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*\|")
_TAG_RE = re.compile(r"^([A-Z][A-Za-z]*\d*[A-Za-z]?\d*|P\d|T\d|G\d|E\d|C\d)\b")
# todo.md checklist line: indent, [ ]/[x]/[~], text. Sub-items (indented) are
# folded into the parent task's notes; the MANUAL/reference section has none.
_CHECKBOX_RE = re.compile(r"^(\s*)- \[([ x~])\]\s+(.*)")
_BOX_STATUS = {"x": "done", " ": "todo", "~": "wip"}

TODO_SECTION_KEY = "hipótese"  # substring of "## Próximas hipóteses"

COLUMNS = (
    "id",
    "date",
    "idea",
    "command",
    "metric_before",
    "metric_after",
    "result",
    "decision",
    "section",
    "status",
    "tags",
    "source_line",
    "raw",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY,
    date          DATE,
    idea          TEXT,
    command       TEXT,
    metric_before TEXT,
    metric_after  TEXT,
    result        TEXT,
    decision      TEXT,
    section       TEXT,
    status        TEXT,
    tags          TEXT,
    source_line   INTEGER,
    raw           TEXT
);
"""

# Work tasks from todo.md live in a SEPARATE table so they never pollute the
# experiment metrics (applied/rejected/...). Same DB file, unified by `report`.
TASK_COLUMNS = ("id", "date", "task", "status", "section", "notes", "tags", "source_line", "raw")

_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY,
    date        DATE,
    task        TEXT,
    status      TEXT,
    section     TEXT,
    notes       TEXT,
    tags        TEXT,
    source_line INTEGER,
    raw         TEXT
);
"""


def _derive_status(section: str, decision: str) -> str:
    if TODO_SECTION_KEY in (section or "").lower():
        return "todo"
    d = (decision or "").lower()
    if any(k in d for k in ("rejeit", "descart", "abandon", "regred", "não promov", "nao promov", "não adot", "nao adot")):
        return "rejected"
    if any(k in d for k in ("aplic", "promov", "manter", "commit", "adot", "merge", "congel", "default")):
        return "applied"
    return "logged"


def _derive_tags(idea: str) -> str:
    m = _TAG_RE.match((idea or "").strip())
    return m.group(1) if m else ""


def parse_md(md_path: Path) -> list[dict[str, Any]]:
    """Parse EXPERIMENTS.md into a list of experiment row dicts (lossless via ``raw``)."""
    rows: list[dict[str, Any]] = []
    section = ""
    next_id = 1
    for lineno, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = _DATE_RE.match(line)
        if not m:
            continue
        stripped = line.rstrip()
        parts = stripped.split(" | ")
        if len(parts) >= 7:
            date = parts[0].strip()
            idea = parts[1].strip()
            decision = parts[-1].strip()
            result = parts[-2].strip()
            metric_after = parts[-3].strip()
            metric_before = parts[-4].strip()
            command = " | ".join(parts[2:-4]).strip()  # folds back pipes inside commands
        else:  # malformed line: keep raw, best-effort scalar fields
            date = m.group(1)
            idea = stripped.split("|", 2)[1].strip() if "|" in stripped else ""
            command = metric_before = metric_after = result = decision = ""
        status = _derive_status(section, decision)
        rows.append(
            {
                "id": next_id,
                "date": date,
                "idea": idea,
                "command": command,
                "metric_before": metric_before,
                "metric_after": metric_after,
                "result": result,
                "decision": decision,
                "section": section,
                "status": status,
                "tags": _derive_tags(idea),
                "source_line": lineno,
                "raw": stripped,
            }
        )
        next_id += 1
    return rows


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute(_SCHEMA)
    con.execute(_TASKS_SCHEMA)
    return con


def import_md(md_path: Path = DEFAULT_MD, db_path: Path = DEFAULT_DB, *, replace: bool = True) -> int:
    """Parse the markdown log and (re)load it into DuckDB. Returns row count."""
    rows = parse_md(md_path)
    con = connect(db_path)
    try:
        if replace:
            con.execute("DELETE FROM experiments")
        con.executemany(
            f"INSERT INTO experiments ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})",
            [[r[c] for c in COLUMNS] for r in rows],
        )
        (count,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
    finally:
        con.close()
    return int(count)


def parse_todo(todo_path: Path) -> list[dict[str, Any]]:
    """Parse todo.md top-level checklist items into task rows.

    Sub-items (indented checkboxes, e.g. ``verificar:`` criteria) are folded into
    the parent task's ``notes``. The ``# MANUAL`` reference section is skipped.
    Status: ``[x]`` -> done, ``[ ]`` -> todo, ``[~]`` -> wip.
    """
    rows: list[dict[str, Any]] = []
    section = ""
    section_date: str | None = None
    in_manual = False
    current: dict[str, Any] | None = None
    subs: list[str] = []

    def _flush() -> None:
        if current is not None:
            current["notes"] = " ; ".join(subs)
            rows.append(current)

    for lineno, line in enumerate(todo_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("# "):  # level-1 header
            in_manual = "MANUAL" in line
            section = line[2:].strip()
            d = re.search(r"\d{4}-\d{2}-\d{2}", line)
            section_date = d.group(0) if d else None
            continue
        if line.startswith("## "):  # level-2 header
            section = line[3:].strip()
            d = re.search(r"\d{4}-\d{2}-\d{2}", line)
            if d:
                section_date = d.group(0)
            continue
        if in_manual:
            continue
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        indent, box, text = m.groups()
        text = text.replace("**", "").strip()
        if len(indent) == 0:  # new top-level task
            _flush()
            subs = []
            current = {
                "date": section_date,
                "task": text,
                "status": _BOX_STATUS[box],
                "section": section,
                "notes": "",
                "tags": _derive_tags(text),
                "source_line": lineno,
                "raw": line.rstrip(),
            }
        elif current is not None:  # sub-item -> fold into the current task
            subs.append(f"[{box if box.strip() else ' '}] {text}")
    _flush()
    return rows


def import_tasks(todo_path: Path = DEFAULT_TODO, db_path: Path = DEFAULT_DB) -> int:
    """(Re)load todo.md tasks into the ``tasks`` table. Returns row count."""
    rows = parse_todo(todo_path)
    for i, r in enumerate(rows, 1):
        r["id"] = i
    con = connect(db_path)
    try:
        con.execute("DELETE FROM tasks")
        con.executemany(
            f"INSERT INTO tasks ({', '.join(TASK_COLUMNS)}) VALUES ({', '.join('?' for _ in TASK_COLUMNS)})",
            [[r[c] for c in TASK_COLUMNS] for r in rows],
        )
        (count,) = con.execute("SELECT COUNT(*) FROM tasks").fetchone()
    finally:
        con.close()
    return int(count)


def _next_id(con: duckdb.DuckDBPyConnection) -> int:
    (mx,) = con.execute("SELECT COALESCE(MAX(id), 0) FROM experiments").fetchone()
    return int(mx) + 1


def add_experiment(db_path: Path = DEFAULT_DB, **fields: Any) -> int:
    """Insert one experiment. Required: date, idea. Returns the new id."""
    if not fields.get("date") or not fields.get("idea"):
        raise ValueError("add_experiment requires at least --date and --idea")
    con = connect(db_path)
    try:
        new_id = _next_id(con)
        section = fields.get("section") or ("Próximas hipóteses" if fields.get("status") == "todo" else "Resultados recentes")
        row = {
            "id": new_id,
            "date": fields["date"],
            "idea": fields["idea"],
            "command": fields.get("command", ""),
            "metric_before": fields.get("metric_before", ""),
            "metric_after": fields.get("metric_after", ""),
            "result": fields.get("result", ""),
            "decision": fields.get("decision", ""),
            "section": section,
            "status": fields.get("status") or _derive_status(section, fields.get("decision", "")),
            "tags": fields.get("tags") or _derive_tags(fields["idea"]),
            "source_line": None,
            "raw": "",
        }
        con.execute(
            f"INSERT INTO experiments ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})",
            [row[c] for c in COLUMNS],
        )
    finally:
        con.close()
    return new_id


def export_md(db_path: Path = DEFAULT_DB, out_path: Path | None = None) -> Path:
    """Dump the experiments table to a standalone markdown view (grouped by
    section), for reading/recovery. Generated read-only — the DB is canonical.
    """
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT section, raw, date, idea, command, metric_before, metric_after, result, decision "
            "FROM experiments ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    by_section: dict[str, list[str]] = {}
    order: list[str] = []
    for (section, raw, date, idea, command, b, a, result, decision) in rows:
        line = raw or f"{date} | {idea} | {command} | {b} | {a} | {result} | {decision}"
        if section not in by_section:
            by_section[section] = []
            order.append(section)
        by_section[section].append(line)

    out_lines = ["# Experimentos (gerado de experiments.duckdb — não editar à mão)", ""]
    for section in order:
        out_lines += [f"## {section or '(sem seção)'}", "", "```text", *by_section[section], "```", ""]
    target = out_path or (REPO / "EXPERIMENTS.export.md")
    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return target


def _md_cell(value: Any, width: int = 80) -> str:
    """Sanitize a value for a markdown table cell (no newlines / unescaped pipes)."""
    s = " ".join(str(value or "").split())
    s = s.replace("|", "\\|")
    return (s[: width - 1] + "…") if len(s) > width else s


def report_md(db_path: Path = DEFAULT_DB, out_path: Path | None = None) -> str:
    """Build a human-readable results report (markdown) from the DB.

    Sections: executive summary, applied wins (with before→after), rejected dead
    ends (so they are not retried), pending, and a per-family breakdown.
    """
    con = connect(db_path)
    try:
        (total,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
        dmin, dmax = con.execute("SELECT MIN(date), MAX(date) FROM experiments").fetchone()
        status_counts = dict(con.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall())
        applied = con.execute(
            "SELECT date, tags, idea, metric_before, metric_after, decision FROM experiments "
            "WHERE status = 'applied' ORDER BY date DESC, id DESC"
        ).fetchall()
        rejected = con.execute(
            "SELECT date, tags, idea, result, decision FROM experiments "
            "WHERE status = 'rejected' ORDER BY date DESC, id DESC"
        ).fetchall()
        todo = con.execute(
            "SELECT date, tags, idea FROM experiments WHERE status = 'todo' ORDER BY date DESC, id DESC"
        ).fetchall()
        by_tag = con.execute(
            "SELECT tags, COUNT(*) AS total, "
            "SUM(CASE WHEN status='applied' THEN 1 ELSE 0 END) AS applied, "
            "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected, "
            "SUM(CASE WHEN status='todo' THEN 1 ELSE 0 END) AS todo "
            "FROM experiments WHERE tags <> '' GROUP BY tags ORDER BY total DESC, tags"
        ).fetchall()
        task_counts = dict(con.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall())
        tasks_todo = con.execute(
            "SELECT section, task, notes FROM tasks WHERE status IN ('todo', 'wip') ORDER BY id"
        ).fetchall()
        tasks_done = con.execute("SELECT section, task FROM tasks WHERE status = 'done' ORDER BY id").fetchall()
    finally:
        con.close()

    sc = status_counts
    out: list[str] = []
    out.append("# Relatório de Experimentos")
    out.append("")
    out.append(
        f"> Gerado de `experiments.duckdb` (store git-tracked) · **{total}** experimentos · "
        f"período **{dmin}** → **{dmax}**. Adicione com `python -m python.lab.experiments add`."
    )
    out.append("")
    out.append("## Resumo")
    out.append("")
    out.append("| status | n | % |")
    out.append("|---|---:|---:|")
    for st in ("applied", "rejected", "logged", "todo"):
        n = int(sc.get(st, 0))
        pct = (100.0 * n / total) if total else 0.0
        label = {"applied": "✅ aplicados", "rejected": "❌ rejeitados", "logged": "📋 logados", "todo": "⏳ pendentes"}[st]
        out.append(f"| {label} | {n} | {pct:.0f}% |")
    out.append("")

    out.append(f"## ✅ Aplicados ({int(sc.get('applied', 0))}) — o que entrou")
    out.append("")
    out.append("| data | tag | ideia | antes → depois | decisão |")
    out.append("|---|---|---|---|---|")
    for date, tags, idea, before, after, decision in applied:
        ba = f"{_md_cell(before, 24)} → {_md_cell(after, 24)}" if (before or after) else "—"
        out.append(f"| {date} | {_md_cell(tags, 10)} | {_md_cell(idea, 70)} | {ba} | {_md_cell(decision, 40)} |")
    out.append("")

    out.append(f"## ❌ Rejeitados ({int(sc.get('rejected', 0))}) — becos sem saída (não repetir)")
    out.append("")
    out.append("| data | tag | ideia | resultado | decisão |")
    out.append("|---|---|---|---|---|")
    for date, tags, idea, result, decision in rejected:
        out.append(
            f"| {date} | {_md_cell(tags, 10)} | {_md_cell(idea, 60)} | {_md_cell(result, 40)} | {_md_cell(decision, 36)} |"
        )
    out.append("")

    out.append(f"## ⏳ Pendentes ({int(sc.get('todo', 0))}) — ainda não feito")
    out.append("")
    if todo:
        out.append("| data | tag | ideia |")
        out.append("|---|---|---|")
        for date, tags, idea in todo:
            out.append(f"| {date} | {_md_cell(tags, 10)} | {_md_cell(idea, 90)} |")
    else:
        out.append("_(nenhum)_")
    out.append("")

    out.append("## Por família (tag)")
    out.append("")
    out.append("| tag | total | ✅ | ❌ | ⏳ |")
    out.append("|---|---:|---:|---:|---:|")
    for tags, t, ap, rj, td in by_tag:
        out.append(f"| {_md_cell(tags, 14)} | {int(t)} | {int(ap or 0)} | {int(rj or 0)} | {int(td or 0)} |")
    out.append("")

    # ---- Work tasks (todo.md) — separate table, kept apart from experiment metrics ----
    tc = task_counts
    n_todo = int(tc.get("todo", 0)) + int(tc.get("wip", 0))
    out.append("---")
    out.append("")
    out.append(f"# Tarefas (todo.md) — {int(tc.get('done', 0))} feitas · {n_todo} a fazer")
    out.append("")
    out.append(f"## ⏳ A fazer ({n_todo})")
    out.append("")
    if tasks_todo:
        out.append("| seção | tarefa | critério / notas |")
        out.append("|---|---|---|")
        for section, task, notes in tasks_todo:
            out.append(f"| {_md_cell(section, 28)} | {_md_cell(task, 70)} | {_md_cell(notes, 60)} |")
    else:
        out.append("_(nenhuma)_")
    out.append("")
    out.append(f"## ✅ Feitas ({int(tc.get('done', 0))})")
    out.append("")
    if tasks_done:
        out.append("| seção | tarefa |")
        out.append("|---|---|")
        for section, task in tasks_done:
            out.append(f"| {_md_cell(section, 28)} | {_md_cell(task, 90)} |")
    else:
        out.append("_(nenhuma)_")
    out.append("")

    report = "\n".join(out) + "\n"
    if out_path is not None:
        out_path.write_text(report, encoding="utf-8")
    return report


def _print_rows(rows: list[tuple], headers: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*(str(c) for c in r)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments", description="DuckDB-backed experiment tracker")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import", help="refresh the tasks table from todo.md (experiments are native to the committed DB)")
    p_imp.add_argument("--todo", type=Path, default=DEFAULT_TODO)
    p_imp.add_argument("--md", type=Path, default=None, help="optional: also (re)load experiments from a markdown dump")

    p_list = sub.add_parser("list", help="list experiments (compact)")
    p_list.add_argument("--status", choices=("todo", "applied", "rejected", "logged"))
    p_list.add_argument("--tag")
    p_list.add_argument("--since")
    p_list.add_argument("--limit", type=int, default=30)

    p_q = sub.add_parser("query", help="run an arbitrary read-only SQL query")
    p_q.add_argument("sql")

    p_tasks = sub.add_parser("tasks", help="list work tasks from todo.md")
    p_tasks.add_argument("--status", choices=("todo", "wip", "done"))
    p_tasks.add_argument("--limit", type=int, default=40)

    sub.add_parser("stats", help="counts by status / section / tag (experiments + tasks)")

    p_add = sub.add_parser("add", help="add one experiment")
    for f in ("date", "idea", "command", "metric-before", "metric-after", "result", "decision", "status", "tags", "section"):
        p_add.add_argument(f"--{f}", default=None)

    p_exp = sub.add_parser("export", help="dump experiments to a standalone markdown view (read-only)")
    p_exp.add_argument("--out", type=Path, default=None, help="output path (default: EXPERIMENTS.export.md)")

    p_rep = sub.add_parser("report", help="results report (applied/rejected/pending + per-family) as markdown")
    p_rep.add_argument("--out", type=Path, default=None, help="write to this path (default: print to stdout)")

    args = parser.parse_args(argv)

    if args.cmd == "import":
        t = import_tasks(args.todo, args.db)
        msg = f"refreshed {t} tasks from {args.todo}"
        if args.md is not None:
            n = import_md(args.md, args.db)
            msg += f" + (re)loaded {n} experiments from {args.md}"
        else:
            msg += ". Experiments are native to the DB — add via 'experiments add' (then commit experiments.duckdb)."
        print(msg)
        return 0

    if args.cmd == "tasks":
        con = connect(args.db)
        try:
            where, params = "", []
            if args.status:
                where, params = " WHERE status = ?", [args.status]
            rows = con.execute(
                f"SELECT id, status, substr(section, 1, 24), substr(task, 1, 72) FROM tasks{where} "
                f"ORDER BY id LIMIT {int(args.limit)}",
                params,
            ).fetchall()
        finally:
            con.close()
        _print_rows(rows, ["id", "status", "section", "task"])
        return 0

    if args.cmd == "list":
        con = connect(args.db)
        try:
            where, params = [], []
            if args.status:
                where.append("status = ?")
                params.append(args.status)
            if args.tag:
                where.append("tags = ?")
                params.append(args.tag)
            if args.since:
                where.append("date >= ?")
                params.append(args.since)
            clause = (" WHERE " + " AND ".join(where)) if where else ""
            rows = con.execute(
                f"SELECT id, date, status, tags, substr(idea, 1, 70) FROM experiments{clause} "
                f"ORDER BY date DESC, id DESC LIMIT {int(args.limit)}",
                params,
            ).fetchall()
        finally:
            con.close()
        _print_rows(rows, ["id", "date", "status", "tag", "idea"])
        return 0

    if args.cmd == "query":
        con = connect(args.db)
        try:
            cur = con.execute(args.sql)
            rows = cur.fetchall()
            headers = [d[0] for d in cur.description] if cur.description else []
        finally:
            con.close()
        _print_rows(rows, headers)
        return 0

    if args.cmd == "stats":
        con = connect(args.db)
        try:
            (total,) = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
            print(f"total: {total}")
            print("\nby status:")
            _print_rows(
                con.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status ORDER BY 2 DESC").fetchall(),
                ["status", "n"],
            )
            print("\nby section:")
            _print_rows(
                con.execute("SELECT section, COUNT(*) FROM experiments GROUP BY section ORDER BY 2 DESC").fetchall(),
                ["section", "n"],
            )
            print("\ntop tags:")
            _print_rows(
                con.execute(
                    "SELECT tags, COUNT(*) FROM experiments WHERE tags <> '' GROUP BY tags ORDER BY 2 DESC LIMIT 12"
                ).fetchall(),
                ["tag", "n"],
            )
            (n_tasks,) = con.execute("SELECT COUNT(*) FROM tasks").fetchone()
            print(f"\ntasks (todo.md): {n_tasks}")
            _print_rows(
                con.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY 2 DESC").fetchall(),
                ["status", "n"],
            )
        finally:
            con.close()
        return 0

    if args.cmd == "add":
        new_id = add_experiment(
            args.db,
            date=args.date,
            idea=args.idea,
            command=args.command,
            metric_before=args.metric_before,
            metric_after=args.metric_after,
            result=args.result,
            decision=args.decision,
            status=args.status,
            tags=args.tags,
            section=args.section,
        )
        print(f"added experiment id={new_id} (commit experiments.duckdb to version it)")
        return 0

    if args.cmd == "export":
        target = export_md(args.db, args.out)
        print(f"exported markdown view -> {target}")
        return 0

    if args.cmd == "report":
        report = report_md(args.db, args.out)
        if args.out:
            print(f"wrote results report -> {args.out}")
        else:
            print(report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
