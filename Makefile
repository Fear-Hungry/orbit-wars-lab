.PHONY: install build test smoke bench fmt lint clean

install:
	pip install -U pip
	pip install -r requirements.txt

build:
	maturin develop --release -m crates/orbit_wars_py/Cargo.toml

smoke:
	python scripts/smoke_test.py

test:
	pytest -q

bench:
	python scripts/benchmark_sim.py --num-envs 1024 --steps 500

fmt:
	cargo fmt --all || true
	python -m ruff format python tests scripts || true

lint:
	cargo clippy --workspace --all-targets -- -D warnings || true
	python -m ruff check python tests scripts || true

clean:
	rm -rf target .pytest_cache .ruff_cache __pycache__ python/**/__pycache__ tests/**/__pycache__
