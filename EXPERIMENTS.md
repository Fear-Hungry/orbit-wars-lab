# Experimentos

Registro curto para comparar ideias de heurística, decoder, modelo e liga.

## Como registrar

Use uma linha por hipótese testada:

```text
YYYY-MM-DD | ideia | comando | métrica principal | resultado | decisão
```

Métricas úteis:

- `2p_win_rate` por oponente;
- `4p_win_rate`;
- `mean_score_margin`;
- `invalid_action_rate`;
- `timeout_rate`;
- `crash_rate`;
- pior seed ou lineup ruim.

## Baseline atual

Artefato: `artifacts/submission.py`

Benchmark salvo: `artifacts/submission_benchmark.json`

Resumo do benchmark local conhecido:

- 2p vs `greedy`: `1.0`;
- 2p vs `defensive`: `1.0`;
- 2p vs `anti_meta`: `1.0`;
- 2p vs `rush`: `0.6667`;
- 2p vs `weak_random`: `0.6667`;
- 4p: `0.6667`;
- crashes/timeouts/ações inválidas: `0.0`.

## Próximas hipóteses

```text
2026-05-31 | reduzir perdas contra rush | python -m python.lab.cli quick | win_rate vs rush | pendente | testar
2026-05-31 | melhorar decisão 4p quando anti_meta+defensive aparecem juntos | python -m python.lab.cli bench-submission --seeds 8 --episode-steps 500 | 4p_win_rate | pendente | testar
```
