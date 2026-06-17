# T0 — pool proxy top-5 (oponentes públicos do LB)

Agentes públicos baixados de notebooks da competição (benchmark **704095**,
`kaggle kernels pull`, 2026-06-10), código extraído da célula `%%writefile` para
`<dir>/agent.py`. Assinatura `agent(obs, config=None)`, obs no formato oficial (dict).

## Régua T0 reiniciada em 2026-06-14

A semântica 4p do motor mudou depois que a eval v5 começou (audit 2026-06-11), então
**todos os parciais 4p antigos são incomparáveis** — a régua longa recomeça do zero.
Config canônico: [`configs/eval_top5_proxy.yaml`](../../../configs/eval_top5_proxy.yaml).
Baseline Producer/OEP: [`artifacts/top5_proxy/baseline_producer_oep.json`](../../top5_proxy/baseline_producer_oep.json).

Regra de gate: **bater só o Producer e apanhar desta pool ≠ candidato top 5.** Todo
candidato novo é medido contra a MESMA pool (2p e 4p separados, seeds fixas) via
`scripts/eval_top5_proxy.py`.

## Pool ativa (6 agentes) — smoke 2026-06-14 (4 seeds, 500 steps vs Producer, thread-pinned)

| dir | kernel ref | LB declarado | linhagem | deps | smoke |
|---|---|---|---|---|---|
| orbit-star-wars-lb-max-1224 | romantamrazov/orbit-star-wars-lb-max-1224 | 1224 | build_world (Tamrazov) | stdlib | ✅ crash=0 timeout=0 |
| orbit-wars-heuristic-lb-1110 | vickimar/orbit-wars-heuristic-lb-1110 | 1110 | plan_moves (sim heurística) | stdlib | ✅ crash=0 timeout=0 |
| distance-prioritized-agent-lb-max-score-1100 | ykhnkf/distance-prioritized-agent-lb-max-score-1100 | 1100 | build_world (Tamrazov) | stdlib | ✅ crash=0 timeout=0 |
| lb-1050-heuristic-simulation-agent-test-3 | stealthtechnologies/lb-1050-heuristic-simulation-agent-test-3 | 1050 | plan_moves (sim heurística) | stdlib | ✅ crash=0 timeout=0 (4p invalid_action≈0.0015, clipado pelo motor) |
| hellburner | slug `hellburner` (owner TBD — benchmark 704095) | n/a | rollout sim via kaggle_environments | stdlib, kaggle_environments | ✅ crash=0 timeout=0 |
| orbit-wars-rule-base-ml-shot-validator-hybrid | slug = dir (owner TBD — benchmark 704095) | n/a | rule-base + validador de tiro ML | stdlib, numpy | ✅ crash=0 timeout=0 |

`hellburner` e `rule-base-ml` ficam sem LB declarado (não registrado na coleta) mas entram
pela diversidade de linhagem; smoke limpo. `lb-1050` também é o agente da liga `ext_lb1050`;
`hellburner` é `ext_hellburner`.

## Excluídos (não entram na pool)

| dir | motivo |
|---|---|
| orbit-wars-i-m-stronger | NÃO USÁVEL — torch + pesos ausentes |
| orbit-wars-i-m-better | sem `agent.py` (só notebook) |
| simplified-orbit-wars-agent | sem `agent.py` (só notebook) |
| orbit-wars-exp48 | smoke FALHA — crash total (torch/pesos: `AttributeError NoneType.__dict__`) |
| orbitbotnext | smoke FALHA — crash_rate 0.07 (2p) / 0.05 (4p) |

## Uso

```
.venv/bin/python -m scripts.eval_top5_proxy                                  # baseline Producer+OEP
.venv/bin/python -m scripts.eval_top5_proxy --candidate <agent.py> --label X --out /tmp/X.json
```

O runner fixa threads (OMP/MKL=1, modelo CPU do Kaggle) p/ `jobs=N`=N cores limpos — evita
timeouts falsos de contenção. Isolamento: 1 instância de módulo por (env, assento).

Motivação (DB id=129): gate local vs Producer NÃO prediz LB; rush/greedy/anti_meta do
registry são fracos demais (aniquilados 32/32 por floor E hold). Esta pool é a régua de
robustez de campo (H-P4/T0).
