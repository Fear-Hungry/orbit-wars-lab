"""Auto-research loop (FunSearch/AlphaEvolve-style MVP) over PGS strategy knobs.

Proves the cycle CLOSES and fitness DISCRIMINATES: generate genome variants,
evaluate each vs a diverse opponent pool (h9 gate), record in experiments.duckdb,
select the best as next-gen parent, repeat. No Kaggle submission.

Entry point: ``python -m scripts.research_loop.runner``.
"""
