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
- `torch` CPU ou CUDA instalado durante o build, sem fallback em runtime;
- libs de sistema usadas por PyTorch/SciPy/renderizacao simples (`libgomp1`, `libgl1`, `libglib2.0-0`, `libjpeg`, `libpng`);
- `git`, `ripgrep` e `tini`;
- `node` no container para executar o `codex` já instalado no host.

Há duas variantes:

- `lab`: imagem menor, com `torch` CPU;
- `lab-gpu`: imagem com `torch` CUDA para treino em GPU NVIDIA, exigindo CUDA visivel no container.

O `compose.yaml` aplica limites por padrao para evitar oversubscription de CPU/RAM durante treino local:

- `TRAIN_CPUS=4.0`
- `TRAIN_MEMORY=16g`
- `TRAIN_MEMORY` tambem define `memswap_limit`, entao o container nao ganha memoria extra via swap
- `TRAIN_PIDS=512`
- `TRAIN_SHM_SIZE=2g`
- `/tmp` em `tmpfs` limitado a 1GB
- `NVIDIA_VISIBLE_DEVICES=0` no servico GPU, expondo somente a primeira GPU por padrao
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `RAYON_NUM_THREADS=4`

O entrypoint recusa iniciar se o cgroup de memoria estiver sem limite, acima de 16GB, ou com swap extra liberado. No servico `lab-gpu`, ele tambem falha imediatamente se PyTorch nao enxergar CUDA e pelo menos uma GPU.

Build e shell:

```bash
make docker-build
make docker-check
make docker-shell
```

Build e shell com GPU:

```bash
make docker-gpu-build
make docker-gpu-check
make docker-gpu-shell
```

Atalhos via `Makefile`:

```bash
make docker-build
make docker-build-all
make docker-check
make docker-shell
make docker-shell-16g
make docker-smoke
make docker-test
make docker-train
make docker-train-16g
make docker-codex
make docker-gpu-build
make docker-gpu-shell
make docker-gpu-check
make docker-gpu-train
make docker-gpu-train-16g
make docker-gpu-codex
```

Para usar o Codex CLI dentro do container, basta manter sua configuração local em `${HOME}/.codex`.
O `compose.yaml` também monta `${HOME}/.local/bin` e `${HOME}/.local/lib/node_modules`, então o container reutiliza exatamente o `codex` já instalado no host. Na prática, ele compartilha o mesmo `auth.json`, histórico e configuração do notebook/ambiente local, sem exigir novo login nem nova exposição de `OPENAI_API_KEY` a cada container.

Para usar GPU no Docker, o host precisa ter GPU NVIDIA disponível e suporte a GPU no Docker/Compose.

Exemplos:

```bash
make docker-codex
make docker-train
make docker-gpu-check
make docker-gpu-train
DOCKER_CPUS=6 DOCKER_MEMORY=12g make docker-train
```

Se o processo ultrapassar o limite de memoria configurado, o kernel deve encerrar o processo dentro do container com OOM, normalmente aparecendo como `Killed`, `OOMKilled` ou exit code `137`. Isso evita que o treino use mais RAM que o teto definido pelo Docker. O entrypoint tambem impede rodar com limite acima de 16GB, mesmo que `TRAIN_MEMORY` seja sobrescrito por engano.

Por padrao, a variante GPU usa `TORCH_COMPUTE_PLATFORM=cu128`, alinhada com os wheels CUDA atuais do PyTorch e com GPUs NVIDIA recentes. Se precisar trocar a variante suportada pelo PyTorch, sobrescreva no build:

```bash
TORCH_COMPUTE_PLATFORM=cu126 make docker-gpu-build
```

O Docker nao instala o driver NVIDIA no host. Antes de usar `lab-gpu`, o host precisa ter driver NVIDIA funcionando e NVIDIA Container Toolkit habilitado para Docker. O comando `make docker-gpu-check` valida isso dentro do container e falha se CUDA nao estiver operacional.

Referencias externas usadas para esta configuracao:

- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/>

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
