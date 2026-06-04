# Playbook de experimentos

Use este arquivo como entrada operacional. A arquitetura detalhada continua em `docs/BLUEPRINT.md`; aqui ficam os comandos curtos para testar ideias.

## Primeiro check

```bash
python -m python.lab.cli doctor
python -m python.lab.cli heuristics
```

## Testar uma mudança de heurística ou decoder

Edite a estratégia em `python/agents/heuristics.py` ou o decoder em `python/orbit_wars_gym/action_decoder.py`.
Se adicionar uma heurística nova, registre-a em `python/agents/registry.py`.

Depois rode:

```bash
python -m python.lab.cli quick
```

Esse comando:

1. exporta `artifacts/submission.py`;
2. roda um benchmark curto 2p/4p;
3. salva o relatório em `artifacts/submission_benchmark.json`.

Para uma comparação um pouco mais estável:

```bash
python -m python.lab.cli bench-submission --seeds 8 --episode-steps 500
```

## Re-adjudicar ideia estrutural

Benchmarks de 16 seeds (`32` partidas 2p por oponente) servem para smoke e regressão grosseira. Eles não têm potência estatística para matar ideias estruturais com queda aparente perto de `0.05` de win rate.

Antes de rejeitar mudanças de abertura, hammer, reserva, lookahead ou fase, rode uma amostra maior:

```bash
rtk .venv/bin/python -m scripts.benchmark_submission \
  --submission artifacts/submission_candidate.py \
  --opponents artifacts/submission_v_old.py greedy rush \
  --seeds 256 --episode-steps 500 --jobs 8 \
  --out artifacts/candidate_256seed.json

rtk .venv/bin/python -m scripts.compare_benchmark_significance \
  --baseline artifacts/baseline_256seed.json \
  --candidate artifacts/candidate_256seed.json \
  --min-games 128 --min-effect 0.05
```

Interprete `underpowered` como "amostra insuficiente", não como regressão real. Só trate como regressão quando o comparador marcar `significant_regression` e o resultado fizer sentido no breakdown por oponente.

## Avaliar população de candidatos

Para iteração rápida:

```bash
python -m python.lab.cli eval
```

Isso usa:

- manifesto: `configs/final_candidate_pool.yaml`;
- config rápida: `configs/eval_quick.yaml`;
- saída: `artifacts/evaluation_report.json`.

Para a avaliação final pesada:

```bash
python -m python.lab.cli eval --config configs/eval_final.yaml
```

## Rodar uma iteração de liga

Depois de gerar `artifacts/evaluation_report.json`:

```bash
python -m python.lab.cli league
```

Estados persistidos:

- `artifacts/hall_of_fame.json`;
- `artifacts/map_elites.json`.

## Validação

```bash
python -m python.lab.cli test
python -m python.lab.cli test --group parity
pytest -q
```

Use `--dry-run` em qualquer comando do CLI para ver o que ele chamaria sem executar.
