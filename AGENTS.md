# Repository Guidelines

## Project Structure & Module Organization
`crates/` contains the Rust simulator and PyO3 binding. `python/orbit_wars_gym/` holds wrappers, encoding, and action decoding. `python/agents/` contains heuristic and policy agents. `python/league/` implements Elo, matchmaking, hall-of-fame, and MAP-Elites. `python/train/` contains training, evaluation, final selection, and validation entry points. Use `configs/` for experiment settings, `scripts/` for smoke/export/benchmark helpers, `tests/` for regression coverage, and `artifacts/` for generated submissions and reports.

## Build, Test, and Development Commands
Run `make install` to install dependencies and `make build` to compile the Rust extension with `maturin`. Use `make smoke` for a quick sanity check, `make test` for the Python suite, and `make bench` for simulator throughput. Before pushing, run `make fmt` and `make lint`. For containers, use `make docker-build`, `make docker-test`, or `make docker-train`.

## Coding Style & Naming Conventions
Python follows Ruff formatting with a `100` character line length; prefer 4-space indentation and explicit, type-aware code. Keep module names `snake_case`, classes `PascalCase`, and config files descriptive, for example `configs/objective_validation.yaml`. Submission templates belong in `python/submission/` and should stay self-contained and minimal.

## Testing Guidelines
Tests use `pytest` under `tests/`. Name files `test_<feature>.py` and keep each test focused on one contract: simulator parity, heuristic behavior, export validity, or objective gates. Run targeted checks during iteration, such as `pytest -q tests/test_submission_pipeline.py`, then finish with `pytest -q`. Changes to exported agents or evaluation logic must add or update regression tests.

## Anti-Overfitting Rules
Do not tune behavior to specific seeds. Validate candidates with `python/train/objective_validation.py` using `configs/objective_validation.yaml`, not ad hoc spot checks. A candidate is not ready unless it passes retained-seed selection and separate `holdout_seeds` against `hall_of_fame`, with acceptable `worst_decile_score_margin`, low selection-vs-holdout gap, per-opponent robustness, and balanced 2p results as both `player_0` and `player_1`. Prefer strategy-level changes over seed-shaped routing.

## Docker-Only Training & Experiments
All training runs, experiment loops, population evaluation, objective validation, benchmarks, and GPU checks must run inside Docker, not directly on the host. Use the provided constrained targets such as `make docker-check`, `make docker-smoke`, `make docker-train`, `make docker-gpu-check`, and `make docker-gpu-train`; for custom commands, run them through `docker compose run --rm lab ...` or `docker compose --profile gpu run --rm lab-gpu ...`. Do not start long-running training or experimental jobs outside the container limits.

## Failure Handling
Never add fallback behavior under any circumstances. If a model, heuristic, export path, benchmark, config, dependency, or any other required component cannot perform the requested operation, fail immediately, stop the workflow, and surface the original error log or traceback with enough context to debug it. Do not silently substitute another policy, skip invalid work, downgrade behavior, retry with a different path, or mask exceptions with default behavior. When touching legacy code that already has fallback paths, remove or harden them so they fail fast and expose the real error instead of expanding them.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects, for example `Add Docker lab and benchmarked Kaggle submission`. Follow that pattern: one-line imperative summary, then details in the body when needed. PRs should explain the competitive impact, list changed configs/scripts, mention validation results, and link any related issue or experiment note. Include screenshots only when the change affects rendered match outputs or docs.

## Security & Experiment Hygiene
Do not commit Kaggle credentials, local tokens, or bulky run artifacts. Keep generated reports in `artifacts/` only when needed for reproducibility. When changing evaluation or selection logic, update the matching config and test files together so exported submissions, local validation, and documentation stay aligned.
