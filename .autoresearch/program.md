# Autoresearch Program

Target repo: `/home/marcusvinicius/Repositorios/orbit-wars-lab`

## Goal
- Export, validate, and submit a Kaggle-ready `submission.py` for Orbit Wars, using one stable local verify command and one primary metric while keeping edits constrained to the exported heuristic.

## Scope
- docs: `README.md`, `docs/SUBMISSION.md`
- source directories: `python/submission/`, `scripts/`
- tests and benchmarks: `tests/test_submission_pipeline.py`, `scripts/smoke_test.py`, `scripts/benchmark_submission.py`
- configs and CI: local Docker harness via `Dockerfile` and `compose.yaml`; no CI detected

## Questions to answer
1. Does the exported `submission.py` pass the self-contained Kaggle safety checks?
2. Does the exported heuristic hold a positive local baseline in 2p and 4p against the built-in heuristic pool?
3. Is Kaggle submission available from this machine without additional auth repair?
4. If more tuning is needed, can it stay confined to `python/submission/submission_template.py`?

## Required outputs
- one baseline row in `results.tsv`
- at least one note in `findings/`
- one run note in `runs/`
- a recorded submit status, including external blockers if submission cannot complete

## Evidence rules
- cite file paths
- separate facts from hypotheses
- record unknowns explicitly

## Control Plane

1. Goal:
Produce a Docker-validated `submission.py` and submit it to Kaggle once auth works.

2. Verify command:
`docker compose run --rm lab python scripts/benchmark_submission.py --submission artifacts/submission.py --seeds 3 --out artifacts/submission_benchmark.json`

3. Primary metric:
`4p_win_rate` from [artifacts/submission_benchmark.json](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission_benchmark.json:1)

4. Mutable surface:
`python/submission/submission_template.py`

5. Read-only by default:
`tests/`, Rust engine under `crates/`, benchmark harness under `scripts/benchmark_submission.py`

6. Acceptance rule:
Keep only changes that preserve container validation and do not reduce the primary metric. Submit the resulting artifact once Kaggle auth succeeds.
