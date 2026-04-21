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
