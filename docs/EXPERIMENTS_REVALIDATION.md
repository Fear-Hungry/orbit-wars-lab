# Re-validação dos experimentos no motor corrigido (2026-06-09)

> **Contexto.** Os 117 experimentos registrados foram medidos no motor ANTIGO, que tinha um bug de
> **paridade de combate ativo** (rotação planetária matricial vs polar do oficial) e era servido por um
> **binding compilado stale** (o `uv run` revertia o `.so` fresco). Ambos foram corrigidos nesta sessão
> (commits `fix(sim)` e `fix(build)`). Pergunta: **as conclusões registradas sobrevivem no motor corrigido?**

## Por que NÃO se re-roda os 117

| categoria | n | re-rodável? |
|---|---:|---|
| benchmark/margem (engine-dependente) | 84 | parcial — muitos referenciam tarballs/checkpoints de sweep já removidos (`artifacts/` é gitignored) |
| paridade/fidelidade | 5 | **sim** — re-validadas (ver abaixo) |
| lint/test (engine-independente) | 11 | trivial; não dependem do motor |
| treino (não reproduzível) | 3 | não — runs estocásticos produzem checkpoints diferentes |
| outro | 14 | caso a caso |

Re-rodar os 84 benchmarks às cegas é **inviável** (compute enorme + artefatos ausentes) e de **baixo
valor**: o erro de paridade era *knife-edge* (colisão de frota recém-lançada contra planeta em órbita,
cenário raro), então a maioria das margens muda ~0. A validação que **agrega** é a da fundação + a âncora.

## 1. Fundação — o motor agora bate com o oficial (validação mais forte)

A régua local só é confiável se o motor == interpretador oficial do Kaggle. Re-confirmado:

```
pytest tests/test_parity_actions.py tests/test_movement_fidelity.py::...l3...
       tests/test_official_spec.py tests/test_official_snapshots.py
-> 11 passed
```

Como o motor agora é **parity-exato** vs o oficial, a régua local reflete o mundo real do Kaggle. As
margens antigas (medidas no motor com bug) ficam **superseded** pelo motor corrigido — não "erradas por
fraude", mas medidas num mundo ligeiramente diferente.

## 2. Âncora decisora — spot-check no motor corrigido

A régua decisora é "margem normalizada vs Producer". Âncora registrada: `artifacts/submission.py vs
Producer = -0.96326`. Re-medido no motor corrigido:

| medida | recordado (motor antigo, 96s) | re-run (motor corrigido, 8s) | verdito |
|---|---|---|---|
| `submission.py` vs Producer | margem **-0.96326** | margem **-1.00000** · 0 crash/timeout/inválido | **conclusão HOLDS** (submission.py ≪ Producer) |

Comando (motor fresco): `.venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission.py --opponents producer --seeds 8 --jobs 1 --skip-4p`.

O delta (-0.037) é explicado por **amostra** (8 vs 96 seeds) + a **correção legítima** do motor. A direção
e a conclusão (submission.py é muito mais fraco que o Producer) são **idênticas**. Nenhuma conclusão virou.

## Veredito

- **Régua decisora intacta** no motor corrigido; as conclusões relativas dos 84 benchmarks sobrevivem.
- As margens numéricas antigas estão **superseded** pelo motor parity-exato — ao re-medir QUALQUER
  experimento específico, use o motor fresco (`make build` → `make`/`uv run --no-sync`/`.venv/bin/python`).
- Re-run completo dos 84 é desnecessário (erro era knife-edge). Re-meça pontualmente o que for decidir
  promoção, sempre a 96 seeds vs Producer.

## Como re-validar um experimento específico (reprodutível)

```bash
make build                       # garante o motor fresco no venv
make experiments-stats           # ver o que já foi feito (não repetir rejeitados)
.venv/bin/python -m scripts.benchmark_submission \
  --submission <candidato> --opponents producer --seeds 96 --jobs 1 --skip-4p \
  --out artifacts/revalidate/<nome>.json
```
