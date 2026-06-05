"""orbit_lite — pure-Python engine that is actually SUBMITTED to Kaggle.

This is the deliverable side of the boundary (DECISIONS.md D10/D11): a
single-game (no batch axis), Kaggle-ready port of the speed-first flow-diff
producer, with NO Rust dependency. Used by ``bots/`` and the submission
packaging. Do NOT import ``orbit_wars_core``/``orbit_wars_py`` here.

Contrast with ``python/orbit_wars_gym``, the Rust-backed gym used only for
local TRAINING/eval. Everything this agent needs lives in this package; it has
no dependencies beyond ``torch`` and the standard library.
"""
