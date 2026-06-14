"""DuckDB registry — record each candidate and select parents (top-N by fitness).

Records via ``python.lab.experiments.add_experiment`` (the only sanctioned,
non-destructive write into ``experiments.duckdb``). We tag every row ``ARL`` (Auto
Research Loop) so the loop's own rows are queryable in isolation, and we store the
fitness scalar at the FRONT of ``result`` (``fitness=<x> | ...``) so a parent
selector can parse it back with a simple SQL ``LIKE`` + split, without needing a
new column in the shared schema.

status = 'applied' if the candidate beats its parent's fitness, else 'rejected'
(matches the project's add_experiment status convention).
"""
from __future__ import annotations

import re
from pathlib import Path

from python.lab.experiments import add_experiment, connect
from scripts.research_loop.genome import deserialize, fitness

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "experiments.duckdb"
TAG = "ARL"
_FITNESS_RE = re.compile(r"fitness=(-?\d+(?:\.\d+)?)")


def record(genome: dict, metrics: dict, fit: float, parent_fit: float | None,
           *, date: str, db_path: Path = DEFAULT_DB, generation: int = 0, cand: int = 0) -> int:
    """Insert one candidate row. Returns the new experiment id."""
    from scripts.research_loop.genome import serialize

    beats = parent_fit is None or fit > parent_fit
    result = (
        f"fitness={fit:.4f} | death={metrics['death_rate']:.3f} "
        f"margin={metrics['mean_margin']:+.3f} planets={metrics['mean_final_planets']:.2f} "
        f"| pool={','.join(metrics['pool'])} seeds={metrics['seeds']} steps={metrics['steps']} "
        f"| parent_fitness={'na' if parent_fit is None else f'{parent_fit:.4f}'}"
    )
    decision = "promovido (bate o parent)" if beats else "rejeitado (não bate o parent)"
    idea = f"ARL g{generation}c{cand} genome={serialize(genome)}"
    return add_experiment(
        db_path=db_path,
        date=date,
        idea=idea,
        command=(
            "python -m scripts.research_loop.runner "
            f"(eval: h9_4p_gate.run_config pool={','.join(metrics['pool'])} "
            f"seeds={metrics['seeds']} steps={metrics['steps']})"
        ),
        result=result,
        decision=decision,
        status="applied" if beats else "rejected",
        tags=TAG,
    )


def select_parents(top_n: int, *, db_path: Path = DEFAULT_DB) -> list[tuple[dict, float]]:
    """Return up to ``top_n`` (genome, fitness) pairs, best fitness first.

    Reads only ARL rows (tags='ARL'), parses fitness from ``result`` and the genome
    JSON from ``idea``. Used to seed the next generation's parent pool.
    """
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT idea, result FROM experiments WHERE tags = ? ORDER BY id DESC",
            [TAG],
        ).fetchall()
    finally:
        con.close()

    scored: list[tuple[dict, float]] = []
    for idea, result in rows:
        m = _FITNESS_RE.search(result or "")
        gm = re.search(r"genome=(\{.*\})\s*$", idea or "")
        if not m or not gm:
            continue
        try:
            genome = deserialize(gm.group(1))
        except Exception:
            continue
        scored.append((genome, float(m.group(1))))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_n]
