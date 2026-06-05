# Producer

Status: benchmark opponent and submission floor.

Source: local vendored version of Slawek Biel's public Producer agent, adjusted to use the shared `orbit_lite` package.

Role:

- Default local opponent named `producer`.
- Seed/baseline policy for OEP experiments through its public `agent(obs)` entrypoint.

Packaging:

- Use `scripts.package_producer_submission` to build a Kaggle-style tarball with `main.py`, `_upstream.py`, and shared `orbit_lite`.
