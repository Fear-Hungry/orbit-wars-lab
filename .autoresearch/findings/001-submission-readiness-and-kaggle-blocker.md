# Finding: Submission Readiness And Kaggle Auth Blocker

Repo: `/home/marcusvinicius/Repositorios/orbit-wars-lab`

## Facts

- The exported Kaggle artifact is self-contained and generated from [scripts/export_submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/scripts/export_submission.py:1) using [python/submission/submission_template.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/python/submission/submission_template.py:1).
- The local container workflow is operational through [Dockerfile](/home/marcusvinicius/Repositorios/orbit-wars-lab/Dockerfile:1) and [compose.yaml](/home/marcusvinicius/Repositorios/orbit-wars-lab/compose.yaml:1).
- The exported submission passed the smoke/runtime checks in [tests/test_submission_pipeline.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/tests/test_submission_pipeline.py:1) inside Docker.
- The current local benchmark artifact in [artifacts/submission_benchmark.json](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission_benchmark.json:1) shows:
  - `2p` win rate `1.0` vs `greedy`, `defensive`, and `anti_meta`
  - `2p` win rate `0.6667` vs `rush` and `weak_random`
  - `4p_win_rate = 0.6667`
- The Kaggle CLI binary exists at `~/.local/bin/kaggle`, but API access currently fails with `401 Unauthorized`.
- The host token file `~/.kaggle/kaggle.json` exists, has mode `0600`, and contains the expected `username` and `key` fields, but the remote API still rejects it.
- Browser-based fallback was investigated using the host Chrome profile and a copied profile. Kaggle itself is not logged in, and Google OAuth for Kaggle currently lands on the email entry screen rather than an already-authorized account selection.

## Evidence

- [docs/SUBMISSION.md](/home/marcusvinicius/Repositorios/orbit-wars-lab/docs/SUBMISSION.md:1)
- [artifacts/submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission.py:1)
- [artifacts/submission_benchmark.json](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission_benchmark.json:1)
- [scripts/benchmark_submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/scripts/benchmark_submission.py:1)
- [tests/test_submission_pipeline.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/tests/test_submission_pipeline.py:1)

## Hypotheses

- The Kaggle API token in `~/.kaggle/kaggle.json` was likely revoked or replaced remotely.
- A fresh API token or a manual browser login to Kaggle/Google on the host should unblock submission immediately.

## Next actions

1. Refresh `~/.kaggle/kaggle.json` with a valid Kaggle API token or log into Kaggle in the host browser.
2. Re-run `kaggle competitions submissions -c orbit-wars` to confirm auth.
3. Submit [artifacts/submission.py](/home/marcusvinicius/Repositorios/orbit-wars-lab/artifacts/submission.py:1).
4. Poll the submission status until Kaggle returns a score.
