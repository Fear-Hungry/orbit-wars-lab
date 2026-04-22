.PHONY: install build test smoke bench fmt lint clean docker-build docker-shell docker-codex docker-smoke docker-test docker-train docker-gpu-build docker-gpu-shell docker-gpu-codex docker-gpu-check docker-gpu-train

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

docker-build:
	docker compose build lab

docker-shell:
	docker compose run --rm lab

docker-codex:
	docker compose run --rm lab codex

docker-smoke:
	docker compose run --rm lab python scripts/smoke_test.py

docker-test:
	docker compose run --rm lab pytest -q

docker-train:
	docker compose run --rm lab python -m python.train.train_league --config configs/league.yaml

docker-gpu-build:
	docker compose --profile gpu build lab-gpu

docker-gpu-shell:
	docker compose --profile gpu run --rm lab-gpu

docker-gpu-codex:
	docker compose --profile gpu run --rm lab-gpu codex

docker-gpu-check:
	docker compose --profile gpu run --rm lab-gpu python -c "import torch; print({'cuda': torch.cuda.is_available(), 'devices': torch.cuda.device_count()})"

docker-gpu-train:
	docker compose --profile gpu run --rm lab-gpu python -m python.train.train_league --config configs/league.yaml
