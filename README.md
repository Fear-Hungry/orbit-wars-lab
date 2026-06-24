# Orbit Wars Lab

Pós-mortem e portfólio técnico de um laboratório local para agentes competitivos
da competição Kaggle **Orbit Wars**, um jogo de estratégia orbital em tempo real
(2 e 4 jogadores).

O projeto foi desenhado em torno de uma ideia: **separar três problemas que não
devem se misturar**.

1. **Simulação rápida** — motor Rust para rodar milhares de partidas por geração.
2. **Aprendizado e seleção** — Python para PPO, self-play, liga (Elo), PBT/GA e análise.
3. **Submissão Kaggle** — agente Python leve, validado no `kaggle-environments`, sem dependência do Rust.

A regra que sustentou tudo: **Rust só simula; Python decide e é o que se submete.**
A plataforma executa um `agent(obs)` Python e o gargalo é a qualidade do modelo,
não a velocidade — então compilar o agente para Rust não é nem possível nem útil.
O detalhe completo está em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) e
[`docs/DECISIONS.md`](docs/DECISIONS.md) (D11).

## Leitura rápida

Para uma visão de portfólio, comece por
[`docs/PORTFOLIO_POST.md`](docs/PORTFOLIO_POST.md). O post resume o problema,
as heurísticas, a validação local, as submissões, as ablações e as lições técnicas
de competição adversarial.

Resumo do que ficou:

- melhor resultado documentado: PGS holdwave, submissão `53537753`, score público
  `1228.8`;
- arquitetura local com Rust para simulação e Python puro para submissão;
- gates de paridade, tarball, liga local seat-rotacionada e top-5 proxy;
- registro de ablações em `experiments.duckdb` e relatório em
  [`docs/EXPERIMENTS_REPORT.md`](docs/EXPERIMENTS_REPORT.md);
- dados pesados locais (`data/`, sweeps em `artifacts/`) ficam fora do Git.

## Arquitetura

```text
orbit-wars-lab/
  crates/            # motor Rust (orbit_wars_core) + binding PyO3 (orbit_wars_py)
  python/            # gym (backend Rust), agentes, liga, treino, CLI
  orbit_lite/        # engine pura-Python usada pela submissão e pelos bots
  bots/              # agentes versionados (producer, oep)
  configs/           # parâmetros de ambiente, PPO, liga e avaliação
  scripts/           # build, benchmark, paridade, gates e exportação
  tests/             # paridade, regras oficiais e regressão
  docs/              # documentação (comece por docs/README.md)
```

## Documentação

A documentação curada vive em [`docs/`](docs/README.md), com uma fonte de verdade
por tópico:

- [`docs/PORTFOLIO_POST.md`](docs/PORTFOLIO_POST.md) — post técnico curto e pós-mortem.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — o modelo de três camadas.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — as decisões de arquitetura (D1–D11) e o porquê.
- [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) — o método competitivo: ciclo, fitness, promoção.
- [`docs/PARITY.md`](docs/PARITY.md) — fidelidade ao ambiente oficial do Kaggle (constantes e testes).
- [`docs/COMPETITIVE_INTEL.md`](docs/COMPETITIVE_INTEL.md) — pesquisa externa que orienta hipóteses.
- [`docs/TRAINING.md`](docs/TRAINING.md) — plano de treino e estado do PPO.
- [`docs/SUBMISSION.md`](docs/SUBMISSION.md) — regra e checklist da submissão.
- [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md) — comandos operacionais para reproduzir tudo.

## Instalação local

Requisitos: Python 3.10+, Rust estável (`rustup`), `maturin`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# compilar binding Rust -> Python
maturin develop --release -m crates/orbit_wars_py/Cargo.toml

# smoke test
python -m scripts.smoke_test
```

Fluxo leve com `uv` e a CLI unificada:

```bash
uv run python -m python.lab.cli doctor
uv run python -m python.lab.cli quick
```

Extras para treino/Kaggle completos: `uv sync --extra train --extra kaggle --extra dev`.

## Docker

Há um ambiente de container com Python 3.11, toolchain Rust e o binding já
compilado, em duas variantes (`lab` com torch CPU, `lab-gpu` com CUDA). Limites de
CPU/RAM são conservadores por padrão para evitar oversubscription em treino local
(parametrizáveis via `TRAIN_CPUS`, `TRAIN_MEMORY`, etc. no `compose.yaml`).

```bash
make docker-build && make docker-shell      # CPU
make docker-gpu-build && make docker-gpu-shell   # GPU NVIDIA
```

Os alvos de `make` cobrem build, smoke, test e train (ver `Makefile`).

## Princípio competitivo

Durante a competição, a aposta técnica foi não tentar fazer o modelo aprender
toda a geometria crua desde zero. A política escolhe **intenção estratégica**; o
decoder resolve detalhes táticos. O treino pesado selecionaria políticas robustas
contra uma liga diversa, não apenas a vitória contra o oponente médio da geração
atual. A meta era terminar no **top 5**; bater o Producer público era entrada, não
destino final (ver [`docs/COMPETITIVE_INTEL.md`](docs/COMPETITIVE_INTEL.md)).
