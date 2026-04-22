# Run 001: Submit Attempt

## Scope

- validate the exported submission in Docker
- confirm the Kaggle competition can be reached from this machine
- submit the current artifact if auth is functional

## Actions

- Ran `docker compose run --rm lab python scripts/smoke_test.py`
- Ran `docker compose run --rm lab pytest -q tests/test_submission_pipeline.py`
- Exported [artifacts/submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission.py:1)
- Benchmarked the artifact with [scripts/benchmark_submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/scripts/benchmark_submission.py:1)
- Installed the host `kaggle` Python package so the local CLI wrapper could run
- Attempted Kaggle API access and browser-based fallback

## Results

- Local validation succeeded.
- Kaggle CLI auth failed with `401 Unauthorized`.
- Browser fallback reached Kaggle login and Google OAuth, but there is no ready Kaggle session and Google OAuth still requires interactive email entry.

## Decision

- Keep the current exported artifact as the ready-to-submit candidate.
- Pause only on external authentication, not on repository readiness.

## Follow-up

1. Refresh the Kaggle API token or log in to Kaggle in the host browser.
2. Resume submission immediately after auth is restored.
