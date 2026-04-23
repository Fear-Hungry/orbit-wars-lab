DOCKER_MEMORY ?= 16g
DOCKER_SHM_SIZE ?= 2g
DOCKER_CPUS ?= 4.0
TORCH_COMPUTE_PLATFORM ?= cu128
DOCKER_ENV = TRAIN_MEMORY=$(DOCKER_MEMORY) TRAIN_SHM_SIZE=$(DOCKER_SHM_SIZE) TRAIN_CPUS=$(DOCKER_CPUS)

.PHONY: install build test smoke bench fmt lint clean docker-build docker-build-all docker-check docker-shell docker-shell-16g docker-codex docker-smoke docker-test docker-train docker-train-16g docker-gpu-build docker-gpu-shell docker-gpu-codex docker-gpu-check docker-gpu-train docker-gpu-train-16g

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
	cargo fmt --all
	python -m ruff format python tests scripts

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	python -m ruff check python tests scripts

clean:
	rm -rf target .pytest_cache .ruff_cache __pycache__ python/**/__pycache__ tests/**/__pycache__

docker-build:
	$(DOCKER_ENV) docker compose build lab

docker-build-all: docker-build docker-gpu-build

docker-check:
	$(DOCKER_ENV) docker compose run --rm lab python scripts/docker_healthcheck.py

docker-shell:
	$(DOCKER_ENV) docker compose run --rm lab

docker-shell-16g:
	$(MAKE) DOCKER_MEMORY=16g DOCKER_SHM_SIZE=2g docker-shell

docker-codex:
	$(DOCKER_ENV) docker compose run --rm lab codex

docker-smoke:
	$(DOCKER_ENV) docker compose run --rm lab python scripts/smoke_test.py

docker-test:
	$(DOCKER_ENV) docker compose run --rm lab pytest -q

docker-train:
	$(DOCKER_ENV) docker compose run --rm lab python -m python.train.train_league --config configs/league.yaml

docker-train-16g:
	$(MAKE) DOCKER_MEMORY=16g DOCKER_SHM_SIZE=2g docker-train

docker-gpu-build:
	$(DOCKER_ENV) TORCH_COMPUTE_PLATFORM=$(TORCH_COMPUTE_PLATFORM) docker compose --profile gpu build lab-gpu

docker-gpu-shell:
	$(DOCKER_ENV) docker compose --profile gpu run --rm lab-gpu

docker-gpu-codex:
	$(DOCKER_ENV) docker compose --profile gpu run --rm lab-gpu codex

docker-gpu-check:
	$(DOCKER_ENV) docker compose --profile gpu run --rm lab-gpu python scripts/docker_healthcheck.py --require-cuda

docker-gpu-train:
	$(DOCKER_ENV) docker compose --profile gpu run --rm lab-gpu python -m python.train.train_league --config configs/league.yaml

docker-gpu-train-16g:
	$(MAKE) DOCKER_MEMORY=16g DOCKER_SHM_SIZE=2g docker-gpu-train
