.PHONY: install build sync-binding verify-binding test smoke bench fmt lint clean lab-doctor lab-heuristics lab-quick lab-eval lab-league lab-submission experiments-import experiments-report experiments-stats gate-check gate-check-final oep-promotion-gate ppo-train-targeted ppo-train-mov2 ppo-select ppo-select-targeted ppo-bench-exported ppo-select-exported docker-build docker-shell docker-codex docker-smoke docker-test docker-train docker-gpu-build docker-gpu-shell docker-gpu-codex docker-gpu-check docker-gpu-train

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
BENCH_JOBS ?= 8
OEP_CANDIDATE_REPORT ?= artifacts/gates/oep/candidate_vs_producer_96seed.json
OEP_BASELINE_REPORT ?= artifacts/gates/producer_fix_gates/g2_champion_vs_corrected_producer_96seed.json
OEP_PROMOTION_OUT ?= artifacts/gates/oep/promotion_gate.json

# `uv run` (WITH sync) is used only to (re)build the native binding, because it can
# install build deps. It is the canonical "install/refresh" path.
UV_DEV ?= $(shell command -v uv >/dev/null 2>&1 && printf 'uv run --extra dev')
# Everything that RUNS code uses --no-sync. Plain `uv run` triggers an auto-sync that
# reinstalls orbit-wars-lab from a CACHED wheel and reverts a freshly-built Rust binding
# (.so) — so tests/training/eval silently run the OLD engine. --no-sync uses the venv
# as-is. See docs/PARITY.md "Frescor do binding". Falls back to bare python without uv.
UV_RUN ?= $(shell command -v uv >/dev/null 2>&1 && printf 'uv run --no-sync')

install:
	pip install -U pip
	pip install -r requirements.txt

build:
	$(UV_DEV) maturin develop --release -m crates/orbit_wars_py/Cargo.toml
	@$(MAKE) --no-print-directory sync-binding

# Force-copy the freshly compiled cdylib into the venv. Guards against uv/maturin
# leaving a stale .so installed in site-packages (see docs/PARITY.md). Safe no-op
# warning if the build artifact or installed module is missing.
sync-binding:
	@src=target/release/liborbit_wars_rs.so; \
	if [ ! -f "$$src" ]; then echo "[sync-binding] WARN: $$src missing; run 'make build' first"; exit 0; fi; \
	found=0; \
	for so in .venv/lib/python*/site-packages/orbit_wars_rs/orbit_wars_rs.*.so; do \
		[ -e "$$so" ] || continue; \
		cp -f "$$src" "$$so" && echo "[sync-binding] fresh Rust binding -> $$so"; found=1; \
	done; \
	[ "$$found" = 1 ] || echo "[sync-binding] WARN: no installed orbit_wars_rs .so in venv; run '$(UV_DEV) maturin develop --release -m crates/orbit_wars_py/Cargo.toml'"

# Sanity: print which binding Python loads + assert it matches the build artifact.
verify-binding:
	@$(UV_RUN) python -c "import orbit_wars_rs, pathlib; p=pathlib.Path(orbit_wars_rs.__file__).resolve(); print('loaded:', p)"

smoke:
	$(UV_RUN) python -m scripts.smoke_test

test:
	$(UV_RUN) python -m pytest -q

bench:
	$(UV_RUN) python -m scripts.benchmark_sim --num-envs 1024 --steps 500

lab-doctor:
	$(UV_RUN) python -m python.lab.cli doctor

lab-heuristics:
	$(UV_RUN) python -m python.lab.cli heuristics

lab-quick:
	$(UV_RUN) python -m python.lab.cli quick

lab-eval:
	$(UV_RUN) python -m python.lab.cli eval

lab-league:
	$(UV_RUN) python -m python.lab.cli league

lab-submission:
	$(UV_RUN) python -m python.lab.cli export
	$(UV_RUN) python -m python.lab.cli bench-submission

# Experiment tracker (DuckDB). Needs the `lab` extra: uv sync --extra lab.
experiments-import:
	$(UV_RUN) python -m python.lab.experiments import

experiments-report:
	$(UV_RUN) python -m python.lab.experiments report --out docs/EXPERIMENTS_REPORT.md

experiments-stats:
	$(UV_RUN) python -m python.lab.experiments stats

gate-check:
	$(UV_RUN) python -m scripts.gate_check --jobs $(BENCH_JOBS)

gate-check-final:
	$(UV_RUN) python -m scripts.gate_check --include-final --jobs $(BENCH_JOBS)

oep-promotion-gate:
	$(UV_RUN) python -m scripts.oep_promotion_gate --baseline $(OEP_BASELINE_REPORT) --candidate $(OEP_CANDIDATE_REPORT) --out $(OEP_PROMOTION_OUT)

ppo-train-targeted:
	$(UV_RUN) python -m python.train.train_ppo --seed $(PPO_SEED) --training-track phase0_2p --num-players 2 --opponents $(PPO_TARGETED_OPPONENTS) --total-timesteps $(PPO_TIMESTEPS) --rollout-steps 256 --update-epochs 4 --minibatch-size 256 --learning-rate $(PPO_LEARNING_RATE) --ship-margin-scale $(PPO_SHIP_MARGIN_SCALE) --checkpoint-in $(PPO_CHECKPOINT_IN) --checkpoint-out $(PPO_CHECKPOINT_OUT) --decoder-max-moves-per-turn $(PPO_DECODER_MAX_MOVES) --decoder-min-ships-to-launch $(PPO_DECODER_MIN_SHIPS) --decoder-reserve-home-ships $(PPO_DECODER_RESERVE)

# Movement 2 campaign: de-anchored reward (no production shaping) + KL-to-BC anchor
# + eval-gating (keep-best by paired margin, early-stop on drift). Warm-start from a
# BC/entity checkpoint (PPO_CHECKPOINT_IN, also the KL reference). GPU; fresh engine.
PPO_KL_COEF ?= 0.05
ppo-train-mov2:
	$(UV_RUN) python -m python.train.train_ppo --seed $(PPO_SEED) --policy-arch entity --num-players 2 \
	  --opponents producer,greedy,oep --total-timesteps $(PPO_TIMESTEPS) --rollout-steps 256 --rollout-num-envs 16 \
	  --device cuda --checkpoint-in $(PPO_CHECKPOINT_IN) --checkpoint-out $(PPO_CHECKPOINT_OUT) \
	  --shaping-potential none --kl-to-ref-coef $(PPO_KL_COEF) \
	  --eval-every-updates 20 --eval-seeds 8 --eval-opponent producer --early-stop-patience 8

ppo-select:
	$(UV_RUN) python -m scripts.select_ppo_checkpoint 'artifacts/ppo/*.pt'

ppo-select-targeted:
	$(UV_RUN) python -m scripts.select_ppo_checkpoint $(PPO_CHECKPOINT_IN) $(PPO_CHECKPOINT_OUT) --config configs/eval_quick.yaml --output artifacts/ppo/$(PPO_RUN_NAME)_selection.json

ppo-bench-exported:
	$(UV_RUN) python -m scripts.benchmark_ppo_submission --checkpoint $(PPO_CHECKPOINT_IN) --submission-out artifacts/ppo/$(PPO_RUN_NAME)_submission.py --out artifacts/ppo/$(PPO_RUN_NAME)_submission_benchmark.json --seeds $(PPO_EXPORT_BENCH_SEEDS) --jobs $(BENCH_JOBS) --opponents $(PPO_TARGETED_OPPONENTS)

ppo-select-exported:
	$(UV_RUN) python -m scripts.select_ppo_submission $(PPO_CHECKPOINT_IN) $(PPO_CHECKPOINT_OUT) --out-dir artifacts/ppo/$(PPO_RUN_NAME)_exported_selection --output artifacts/ppo/$(PPO_RUN_NAME)_exported_selection.json --seeds $(PPO_EXPORT_BENCH_SEEDS) --opponents $(PPO_TARGETED_OPPONENTS) --skip-4p

fmt:
	cargo fmt --all || true
	python -m ruff format python tests scripts || true

lint:
	cargo clippy --workspace --all-targets -- -D warnings || true
	$(UV_RUN) python -m ruff check

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
