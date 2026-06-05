# Arquitetura

O repositório separa três problemas que não devem ser misturados. Misturá-los é
a origem da maioria dos bugs de competição (treinar em um universo falso, ou
submeter algo que não roda no Kaggle).

## Modelo de três camadas

| Camada | Onde vive | Responsabilidade | Restrição |
| --- | --- | --- | --- |
| **Simulação** | `crates/orbit_wars_core` (Rust) + `crates/orbit_wars_py` (binding PyO3) | Rodar milhares de partidas por geração: estado, geometria, step, combate, batch. | Infraestrutura de treino **local**. Nunca entra na submissão. |
| **Aprendizado e seleção** | `python/` | PPO, self-play, liga (Elo), PBT/GA, MAP-Elites, avaliação e exportação. | Decide *o que* submeter. |
| **Submissão Kaggle** | `bots/`, `artifacts/submission.py`, `orbit_lite/` | Agente Python leve, com inferência NumPy/PyTorch ou heurística pura. | Pequeno, estável, validado em `kaggle-environments`. |

```text
orbit-wars-lab/
  crates/
    orbit_wars_core/       # motor Rust puro: estado, geometria, step, combate, batch
    orbit_wars_py/         # binding PyO3 para Python

  python/
    orbit_wars_gym/        # Gymnasium/PettingZoo wrappers, encoder e decoder (backend Rust)
    agents/                # heurísticas, política neural, adaptador de submissão
    league/                # Elo, matchmaking, hall-of-fame, PBT, MAP-Elites
    train/                 # PPO, liga, evolução e avaliação
    lab/                   # CLI unificada (doctor, quick, eval, league, test)
    submission/            # template para Kaggle

  orbit_lite/              # engine pura-Python (sem Rust) usada pela submissão e bots
  bots/                    # agentes versionados (producer, oep) e seus cards
  configs/                 # parâmetros de ambiente, PPO, liga e avaliação
  scripts/                 # build, benchmark, paridade, gates e exportação
  tests/                   # paridade, regras oficiais e regressão
  docs/                    # esta documentação
```

## A fronteira que sustenta tudo: Rust simula, Python decide

A regra operacional mais importante do projeto:

> **Rust só simula; Python decide e é o que se submete.**

O Kaggle Orbit Wars executa um `agent(obs)` **Python** via `kaggle-environments`;
não há como submeter um binário ou extensão Rust. E velocidade não é o gargalo: o
agente roda em 4–290 ms/step contra um `actTimeout` de **1 s** — o limite é a
qualidade do modelo de estado/valor, não throughput de compute.

Por isso há **dois pacotes de simulação que parecem sobrepostos mas não são**:

- `python/orbit_wars_gym/` — gym com **backend Rust** (`orbit_wars_core` via PyO3). Usado só para treino/avaliação em massa. Rápido, mas não submissível.
- `orbit_lite/` — engine equivalente em **Python puro**. Usada pela submissão e pelos bots em `bots/`. Lenta o suficiente para treino, mas portável para o Kaggle.

Nenhum bot em `bots/` nem submissão em `artifacts/` pode importar
`orbit_wars_core`/`orbit_wars_py`. A justificativa completa e a guarda recomendada
estão em [`DECISIONS.md`](DECISIONS.md) (D10/D11).

## Por que essa separação

- **Paridade vs. throughput** ([`PARITY.md`](PARITY.md)): o modo paridade carrega snapshots oficiais e compara step a step; o modo treino gera mapas próprios com distribuição fiel e maximiza velocidade. Sem essa separação, treina-se em um universo falso.
- **Risco de submissão**: um `.so` que não carrega no Kaggle = submissão morta, sem ganho. Python puro elimina essa classe de falha.
