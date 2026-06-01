.PHONY: install build test smoke bench fmt lint clean lab-doctor lab-heuristics lab-quick lab-eval lab-league lab-submission gate-check gate-check-final ppo-train-targeted ppo-select ppo-select-targeted ppo-bench-exported ppo-select-exported docker-build docker-shell docker-codex docker-smoke docker-test docker-train docker-gpu-build docker-gpu-shell docker-gpu-codex docker-gpu-check docker-gpu-train

PPO_SEED ?= 11
PPO_TIMESTEPS ?= 32768
PPO_LEARNING_RATE ?= 1e-4
PPO_SHIP_MARGIN_SCALE ?= 0.0
PPO_CHECKPOINT_IN ?= artifacts/ppo/phase0_seed1_65536_resume_seed4_65536.pt
PPO_RUN_NAME ?= targeted_seed$(PPO_SEED)
PPO_CHECKPOINT_OUT ?= artifacts/ppo/phase0_$(PPO_RUN_NAME)_$(PPO_TIMESTEPS).pt
PPO_TARGETED_OPPONENTS ?= weak_random,rush,anti_meta
PPO_DECODER_MAX_MOVES ?= 4
PPO_DECODER_MIN_SHIPS ?= 2
PPO_DECODER_RESERVE ?= 8
PPO_EXPORT_BENCH_SEEDS ?= 1

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

gate-check:
	uv run --extra dev python -m scripts.gate_check

gate-check-final:
	uv run --extra dev python -m scripts.gate_check --include-final

ppo-train-targeted:
	uv run --extra dev python -m python.train.train_ppo --seed $(PPO_SEED) --training-track phase0_2p --num-players 2 --opponents $(PPO_TARGETED_OPPONENTS) --total-timesteps $(PPO_TIMESTEPS) --rollout-steps 256 --update-epochs 4 --minibatch-size 256 --learning-rate $(PPO_LEARNING_RATE) --ship-margin-scale $(PPO_SHIP_MARGIN_SCALE) --checkpoint-in $(PPO_CHECKPOINT_IN) --checkpoint-out $(PPO_CHECKPOINT_OUT) --decoder-max-moves-per-turn $(PPO_DECODER_MAX_MOVES) --decoder-min-ships-to-launch $(PPO_DECODER_MIN_SHIPS) --decoder-reserve-home-ships $(PPO_DECODER_RESERVE)

ppo-select:
	python -m scripts.select_ppo_checkpoint 'artifacts/ppo/*.pt'

ppo-select-targeted:
	uv run --extra dev python -m scripts.select_ppo_checkpoint $(PPO_CHECKPOINT_IN) $(PPO_CHECKPOINT_OUT) --config configs/eval_quick.yaml --output artifacts/ppo/$(PPO_RUN_NAME)_selection.json

ppo-bench-exported:
	uv run --extra dev python -m scripts.benchmark_ppo_submission --checkpoint $(PPO_CHECKPOINT_IN) --submission-out artifacts/ppo/$(PPO_RUN_NAME)_submission.py --out artifacts/ppo/$(PPO_RUN_NAME)_submission_benchmark.json --seeds $(PPO_EXPORT_BENCH_SEEDS) --opponents $(PPO_TARGETED_OPPONENTS)

ppo-select-exported:
	uv run --extra dev python -m scripts.select_ppo_submission $(PPO_CHECKPOINT_IN) $(PPO_CHECKPOINT_OUT) --out-dir artifacts/ppo/$(PPO_RUN_NAME)_exported_selection --output artifacts/ppo/$(PPO_RUN_NAME)_exported_selection.json --seeds $(PPO_EXPORT_BENCH_SEEDS) --opponents $(PPO_TARGETED_OPPONENTS) --skip-4p

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
