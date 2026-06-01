.PHONY: install build test smoke bench fmt lint clean lab-doctor lab-heuristics lab-quick lab-eval lab-league lab-submission docker-build docker-shell docker-codex docker-smoke docker-test docker-train docker-gpu-build docker-gpu-shell docker-gpu-codex docker-gpu-check docker-gpu-train

install:
	pip install -U pip
	pip install -r requirements.txt

build:
	maturin develop --release -m crates/orbit_wars_py/Cargo.toml

smoke:
	python -m scripts.smoke_test

test:
	pytest -q

bench:
	python -m scripts.benchmark_sim --num-envs 1024 --steps 500

lab-doctor:
	python -m python.lab.cli doctor

lab-heuristics:
	python -m python.lab.cli heuristics

lab-quick:
	python -m python.lab.cli quick

lab-eval:
	python -m python.lab.cli eval

lab-league:
	python -m python.lab.cli league

lab-submission:
	python -m python.lab.cli export
	python -m python.lab.cli bench-submission

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
	docker compose run --rm lab python -m scripts.smoke_test

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
