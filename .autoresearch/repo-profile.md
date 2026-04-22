# Repo Profile

- repo: `/home/marcusvinicius/Repositorios/orbit-wars-lab`
- suggested mode: `knowledge-harvest`
- rationale: Multiple toolchains and several plausible edit surfaces suggest harvesting first.
- files scanned: 90

## Git
- branch: `main`
- head: `89c7288`
- dirty: yes

## Detected repo kinds
- `python`
- `rust`

## Markers
- `python`: `pyproject.toml`, `requirements.txt`
- `rust`: `Cargo.toml`

## Languages
- Python: 52
- YAML: 9
- Rust: 9
- Markdown: 7
- TOML: 4
- Shell: 2
- JSON: 1

## Harness candidates
- `make bench`: Makefile target (high)
- `cargo test`: Cargo project (high)
- `make smoke`: Makefile target (high)
- `make test`: Makefile target (high)
- `pytest`: tests directory detected (medium)
- `make docker-gpu-check`: Makefile target (high)
- `make docker-smoke`: Makefile target (high)
- `make docker-test`: Makefile target (high)
- `make lint`: Makefile target (high)

## Mutable surface candidates
- `crates/orbit_wars_core/src/batch.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/combat.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/config.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/generator.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/geometry.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/lib.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/step.rs`: lives under `crates/` (score 20)
- `crates/orbit_wars_core/src/types.rs`: lives under `crates/` (score 20)

## Docs, tests, and CI
- docs: `README.md`, `docs/BLUEPRINT.md`, `docs/PARITY.md`, `docs/SUBMISSION.md`, `docs/TRAINING.md`
- tests: `tests/test_action_decoder.py`, `tests/test_backend_smoke.py`, `tests/test_elo.py`, `tests/test_encoding.py`, `tests/test_evaluate_population.py`, `tests/test_final_selection.py`, `tests/test_heuristics.py`, `tests/test_league_loop.py`, `tests/test_objective_validation.py`, `tests/test_official_snapshots.py`
- ci: none

## Top-level entries
- `.dockerignore`
- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
- `DECISIONS.md`
- `Dockerfile`
- `EXPERIMENTS.md`
- `Makefile`
- `README.md`
- `artifacts`
- `compose.yaml`
- `configs`
- `crates`
- `docs`
- `orbit_wars_gym`
- `pyproject.toml`
- `python`
- `requirements.txt`
- `scripts`
- `tests`
