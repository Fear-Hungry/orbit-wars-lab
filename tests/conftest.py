"""Pytest configuration.

Skip the official-parity test modules cleanly when the optional ``kaggle`` extra
is absent, instead of aborting the whole suite with a collection-time ImportError.

These modules need the real Kaggle interpreter (``kaggle_environments``) to compare
the Rust simulator against ground truth. The validation/CI environment must install
it (``uv sync --extra kaggle``); for casual local runs without the extra they are
skipped with a loud warning so the fidelity gap is explicit, never silent.
"""

from __future__ import annotations

import importlib.util
import warnings

_KAGGLE_AVAILABLE = importlib.util.find_spec("kaggle_environments") is not None

# Modules that require kaggle_environments (top-level import or via snapshots.py).
# test_parity_actions.py self-guards with pytest.importorskip, so it is not listed.
_KAGGLE_DEPENDENT = (
    "test_official_spec.py",
    "test_official_snapshots.py",
    "test_parity_tolerances.py",
    "test_training_generator_distribution.py",
    "test_submission_pipeline.py",
)

collect_ignore: list[str] = []
if not _KAGGLE_AVAILABLE:
    collect_ignore = list(_KAGGLE_DEPENDENT)
    warnings.warn(
        "kaggle_environments is not installed: skipping the official-parity tests "
        f"({', '.join(_KAGGLE_DEPENDENT)}). Install the extra with "
        "`uv sync --extra kaggle` to run the simulator-fidelity gate.",
        stacklevel=1,
    )
