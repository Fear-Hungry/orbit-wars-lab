# Plano de treinamento

Fonte de verdade do **estado atual** do treino. O *qual algoritmo* e o *porquê*
ficam em D4 de [`DECISIONS.md`](DECISIONS.md); as fases abaixo são o *como*.

## Decisão atual

PPO/self-play fica **deferido** enquanto o caminho OEP (planner com busca sobre o
Producer) ainda for o candidato ativo. A ativação do PPO só volta para a fila
quando uma destas condições for verdadeira:

- OEP passa o gate de promoção contra Producer e vira baseline a ser batido por aprendizado;
- OEP esgota o ganho mensurável contra Producer, com experimentos registrados no store `experiments.duckdb` (`make experiments-report`);
- surge oponente externo forte que exija diversidade de política em vez de busca sobre o planner.

Quando ativado, PPO deve treinar contra Producer/heurístico e só gerar candidato
promovível se o checkpoint exportado tiver margem contra Producer ≥ baseline OEP,
com crash/timeout/fallback igual a zero. Runs smoke de PPO podem existir para
infraestrutura, mas não são caminho de submissão.

## Pipeline robusto — comando canônico erro-zero (2026-06-08)

Provado end-to-end sem erros (smoke 2 chunks: `errors.jsonl` ausente, sem `FAILED`/`Traceback`,
crash/invalid=0, `campaign_report.json` escrito). As 3 fontes de erro que quebravam o treino
estão corrigidas e **travadas por regressão** em `tests/test_strong_opponents_registry.py`:

| Erro que quebrava o treino | Causa-raiz | Fix |
|---|---|---|
| `Out of memory` (OOM) no chunk | box compartilhado em overcommit (não o `train_ppo`, que usa ~1.6GB) | `--min-free-gb` em `run_campaign` (espera se RAM baixa) + `.wslconfig` memory=24GB |
| `unknown phase-0 opponents: producer_h*` | `PHASE0_OPPONENTS` em `train_ppo` não listava os handicapados | adicionados a `PHASE0_OPPONENTS` |
| `'producer_h30' is stateless` | isolação batched (`registry._make_isolated_policy`) só conhecia producer/oep | isolação constrói Producer fresco + aplica o handicap |

**Pré-voo (antes de qualquer run longo):**
```bash
uv run --extra dev python -m pytest -q tests/test_strong_opponents_registry.py tests/test_training_phase0.py
```

**Comando canônico** (campanha eval-gated, treino cumulativo, anti-drift, anti-OOM):
```bash
nohup uv run --extra dev python -m scripts.run_campaign \
  --init <warm_start.pt> --out-dir artifacts/ppo/<run_name> \
  --opponents "<currículo>" --eval-opponents "<régua dessaturada>" \
  --chunks 30 --chunk-timesteps 200000 --rollout-num-envs 16 \
  --ent-coef 0.003 --min-free-gb 4.0 --regress-reset 0.05 --floor-margin -0.99 \
  > artifacts/ppo/<run_name>/run.log 2>&1 & disown
```

**Saúde durante o run** (sem erro): sem `errors.jsonl`, sem `FAILED` no `run.log`,
`crash/timeout/invalid=0` por chunk no `campaign_log.jsonl`. **Ressalva (não é erro de
runtime, é de aprendizado):** entropia subindo por chunk = divergência de RL (vista com
warm-start inflado) → é instabilidade a tratar em T5/representação, não um knob de `ent`.

## Fases (quando o aprendizado for ativado)

### Fase 0 — baseline funcional
- ambiente 2p, sem cometas
- PPO contra greedy/defensive/rush
- reward shaping leve
- medir captura de neutros e sobrevivência inicial

### Fase 1 — órbitas
- ativar rotação
- decoder prevê posição futura
- penalizar perda para sol e borda

### Fase 2 — self-play simples
- política atual contra snapshots anteriores
- Elo local; hall-of-fame pequeno

### Fase 3 — liga completa
- população PPO
- PBT em hiperparâmetros
- heurísticas especializadas
- MAP-Elites

### Fase 4 — cometas
- ativar cometas
- reward auxiliar temporário para custo-benefício
- remover dependência excessiva do shaping no final

### Fase 5 — 4p
- treinar política separada (ver D9 em [`DECISIONS.md`](DECISIONS.md))
- aumentar importância de vulnerabilidade e terceiro jogador

### Fase 6 — seleção final
- seeds retidas; round-robin massivo
- pior decil de score margin
- análise de replays ruins
- exportação de 2 submissões candidatas
