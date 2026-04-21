#!/usr/bin/env bash
set -euo pipefail
maturin develop --release -m crates/orbit_wars_py/Cargo.toml
