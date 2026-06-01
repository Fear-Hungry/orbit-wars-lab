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
