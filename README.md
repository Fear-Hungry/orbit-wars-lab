# Orbit Wars Lab

Laboratório local para desenvolver agentes competitivos para a competição Kaggle **Orbit Wars**.

O objetivo deste repositório é separar três problemas que não devem ser misturados:

1. **Simulação rápida**: motor Rust para rodar milhares de partidas por geração.
2. **Aprendizado e seleção**: Python para PPO, self-play, liga, PBT/GA e análise.
3. **Submissão Kaggle**: agente Python leve, com inferência NumPy/PyTorch e fallback heurístico.

A regra operacional é simples: o motor Rust é infraestrutura de treino; o `submission.py` final deve ser pequeno, estável e validado no `kaggle-environments`.

## Fontes oficiais usadas como alvo de compatibilidade

- `orbit_wars.json`, versão 1.0.9: define `agents=[2,4]`, `episodeSteps=500`, `actTimeout=1`, `shipSpeed=6.0`, `cometSpeed=4.0`, schema de observação e ação.
- `orbit_wars.py`: define constantes físicas, geração de planetas, cometas, lançamento de frotas, produção, movimento, colisão contínua, rotação planetária, sweep collision e combate.

URLs de referência:

- https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/orbit_wars/orbit_wars.json
- https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/orbit_wars/orbit_wars.py

## Arquitetura

```text
orbit-wars-lab/
  crates/
    orbit_wars_core/       # motor Rust puro: estado, geometria, step, combate, batch
    orbit_wars_py/         # binding PyO3 para Python

  python/
    orbit_wars_gym/        # Gymnasium/PettingZoo wrappers, encoder e decoder
    agents/                # heurísticas, política neural, adaptador de submissão
    league/                # Elo, matchmaking, hall-of-fame, PBT, MAP-Elites
    train/                 # scripts PPO, liga, evolução e avaliação
    submission/            # template para Kaggle

  tests/                   # testes Python de integração e regras
  configs/                 # parâmetros de ambiente, PPO, liga e evolução
  scripts/                 # build, benchmark, paridade e exportação
  docs/                    # blueprint e decisões arquiteturais
```

## Estado atual do scaffold

Este é um repositório inicial para desenvolvimento. Ele já contém:

- motor Rust inicial com física, lançamento, produção, rotação, colisão contínua e combate;
- estrutura de batch simulator;
- binding PyO3 de debug;
- wrappers Python para Gymnasium e API paralela;
- heurísticas baseline;
- Elo, matchmaking, PBT e archive de diversidade;
- configs de treino;
- scripts de benchmark, smoke test, paridade e exportação;
- documentação de decisões.

Ainda exige trabalho antes de treino pesado:

- completar paridade bit-a-bit contra o ambiente oficial;
- substituir a API PyO3 de debug por API NumPy zero-copy para alto throughput;
- validar geração de cometas contra o oficial;
- integrar PPO real com coleta de trajetórias em batch;
- calibrar reward shaping e action decoder por replays.

## Instalação local

Requisitos:

- Python 3.10+
- Rust estável via `rustup`
- `maturin`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# compilar binding Rust -> Python
maturin develop --release -m crates/orbit_wars_py/Cargo.toml

# smoke test
python scripts/smoke_test.py
```

## Docker para treino limitado

O repositório agora inclui um ambiente de container com:

- Python 3.11 em imagem `slim`;
- toolchain Rust para recompilar o binding PyO3 quando necessário;
- binding `orbit_wars_rs` já compilado no build da imagem;
- `git`, `ripgrep` e `tini`;
- `node` no container para executar o `codex` já instalado no host.

Há duas variantes:

- `lab`: imagem menor, com `torch` CPU;
- `lab-gpu`: imagem com `torch` CUDA para treino em GPU NVIDIA.

O `compose.yaml` aplica limites conservadores por padrão para evitar oversubscription de CPU/RAM durante treino local:

- `TRAIN_CPUS=4.0`
- `TRAIN_MEMORY=8g`
- `TRAIN_PIDS=512`
- `TRAIN_SHM_SIZE=1g`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `RAYON_NUM_THREADS=4`

Build e shell:

```bash
docker compose build lab
docker compose run --rm lab
```

Build e shell com GPU:

```bash
docker compose --profile gpu build lab-gpu
docker compose --profile gpu run --rm lab-gpu
```

Atalhos via `Makefile`:

```bash
make docker-build
make docker-shell
make docker-smoke
make docker-test
make docker-train
make docker-codex
make docker-gpu-build
make docker-gpu-shell
make docker-gpu-check
make docker-gpu-train
make docker-gpu-codex
```

Para usar o Codex CLI dentro do container, basta manter sua configuração local em `${HOME}/.codex`.
O `compose.yaml` também monta `${HOME}/.local/bin` e `${HOME}/.local/lib/node_modules`, então o container reutiliza exatamente o `codex` já instalado no host. Na prática, ele compartilha o mesmo `auth.json`, histórico e configuração do notebook/ambiente local, sem exigir novo login nem nova exposição de `OPENAI_API_KEY` a cada container.

Para usar GPU no Docker, o host precisa ter GPU NVIDIA disponível e suporte a GPU no Docker/Compose.

Exemplos:

```bash
docker compose run --rm lab codex
TRAIN_CPUS=6 TRAIN_MEMORY=12g docker compose run --rm lab python -m python.train.train_league --config configs/league.yaml
docker compose run --rm lab maturin develop --release -m crates/orbit_wars_py/Cargo.toml
docker compose --profile gpu run --rm lab-gpu codex
GPU_COUNT=1 docker compose --profile gpu run --rm lab-gpu python -c "import torch; print(torch.cuda.is_available())"
TRAIN_CPUS=8 TRAIN_MEMORY=16g docker compose --profile gpu run --rm lab-gpu python -m python.train.train_league --config configs/league.yaml
```

Por padrão, a variante GPU usa `TORCH_COMPUTE_PLATFORM=cu118` para maximizar compatibilidade. Se o seu host suportar outra variante CUDA do PyTorch, você pode sobrescrever no build:

```bash
TORCH_COMPUTE_PLATFORM=cu126 docker compose --profile gpu build lab-gpu
```

## Caminho de treino recomendado

```bash
# 1. Rodar smoke test e benchmark
python scripts/smoke_test.py
python scripts/benchmark_sim.py --num-envs 1024 --steps 500

# 2. Rodar testes de lógica local
pytest -q

# 3. Gerar probes de paridade contra Kaggle
python scripts/parity_probe.py --episodes 32 --steps 500

# 4. Treinar currículo simples
python -m python.train.train_league --config configs/league.yaml

# 5. Avaliar candidatos finais
python -m python.train.evaluate_population --config configs/eval_final.yaml

# 6. Exportar submissão
python scripts/export_submission.py --checkpoint runs/best.pt --out submission.py
```

## Princípio competitivo

Não tente fazer o modelo aprender toda a geometria crua desde zero. A política deve escolher intenção estratégica; o decoder resolve detalhes táticos. O treino pesado deve selecionar políticas robustas contra uma liga diversa, não apenas maximizar vitória contra o oponente médio da geração atual.
