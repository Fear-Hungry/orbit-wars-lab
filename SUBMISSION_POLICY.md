# SUBMISSION POLICY

> Política **mecânica** de submissão. O *operacional* (estrutura do `agent`, checklist
> de invariantes) é dono de [`docs/SUBMISSION.md`](docs/SUBMISSION.md); a régua decisora
> e o piso de promoção vivem em [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md) e [`todo.md`](todo.md).
> Esta página só define os **portões determinísticos** que o
> [`scripts/auto_submit_gate.py`](scripts/auto_submit_gate.py) aplica — o **único** caminho de submissão.

Orbit Wars é competição de **agente** (não tabular): a submissão é um `agent(obs)`
CPU-only (invariantes D10/D11), e a métrica decisora é a **margem normalizada pareada
vs o Producer local** — não há `submission.csv`, colunas ou linhas a validar.

## Auto-submit só acontece se TODAS forem verdadeiras

1. **Regras (D10/D11):** a submissão não importa `orbit_wars_core`/`orbit_wars_py`, roda
   CPU-only e não excede `actTimeout`. Proxies no benchmark: `invalid_action_rate == 0`,
   `crash_rate == 0`, `timeout_rate == 0` (e `fallback_rate == 0` se presente). O conjunto
   completo de invariantes é `make gate-check-final` → opcional via `--gate-report`.
2. **Régua decisora:** margem medida em **≥ `MIN_SEEDS` (default 96)** seeds. Triagem de
   16 seeds **nunca** auto-submete.
3. **Piso de promoção:** `mean_score_margin > PROMOTION_FLOOR + PROMOTION_FOLGA`
   (default `-0.7491 + 0.02`) — bater o Producer **com folga**, não empatar.
4. **Melhora real:** margem ≥ melhor candidato atual `+ MIN_MARGIN_DELTA` (default `0.01`).
   Public LB **nunca** é a única evidência.
5. **Orçamento:** `MAX_AUTO_SUBMISSIONS_PER_DAY` (default **1**) para o caminho automático,
   E o time deve estar abaixo do limite Kaggle de **5/dia** (consultado ao vivo; se
   indeterminável, **fail-closed**).
6. **Sem duplicata:** o fingerprint SHA-256 do arquivo não foi submetido hoje.
7. **Reprodutível:** o candidato carrega config + seed (registrado no benchmark/EXPERIMENTS.md).

## Defaults (env-overridable)

| Variável | Default | Significado |
|---|---|---|
| `AUTO_SUBMIT` | `0` | `0` = dry-run (aprova/reprova, **nunca** submete). `1` = submissão ao vivo. |
| `KAGGLE_COMPETITION` | — | slug da competição; obrigatório para submeter ao vivo. |
| `MAX_AUTO_SUBMISSIONS_PER_DAY` | `1` | teto automático; o resto das 5/dia fica para decisão humana. |
| `PROMOTION_FLOOR` | `-0.7491` | margem do Producer a superar. |
| `PROMOTION_FOLGA` | `0.02` | folga exigida acima do piso. |
| `MIN_MARGIN_DELTA` | `0.01` | melhora mínima vs o melhor atual. |
| `MIN_SEEDS` | `96` | seeds mínimos da régua decisora. |

## Estado atual

**Nível 2.5 (auto-submit *ligável*), mas hoje efetivamente Nível 2:** `AUTO_SUBMIT=0`
por padrão. O PPO atual está em `-1.0`; nenhum candidato chega perto de
`-0.7491 + folga` a 96 seeds, então o gate **reprova por desenho** até existir candidato
provado. Ligar é um flip consciente de `AUTO_SUBMIT=1`.
