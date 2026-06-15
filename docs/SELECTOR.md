# Seletor de submissão calibrado

Dona deste tópico. Como a liga local escolhe (ou se recusa a escolher) um
candidato de submissão. Contexto: a liga foi **falsificada como gate de
promoção** em 2026-06-10 (Spearman local×LB = 0.0; viés de população) — ela só
recupera o direito de escolher provando calibração contra âncoras de LB reais.

## Pipeline (3 camadas, 3 artefatos)

```
scripts/league_submit_ruler.py         → report.json              (VETO + features; nunca decide)
scripts/league_selector_calibration.py → calibration.json         (valida vs LB_ANCHORS)
scripts/league_submission_selector.py  → submission_decision.json (único que diz SUBMIT_CANDIDATE)
```

- O ruler emite `local_veto_passes`, scores split (`score_2p_fixed` /
  `score_2p_peer` / `score_4p_fixed`), vantagens normalizadas (`adv_2p=2wr-1`,
  `adv_4p=(wr-0.25)/0.75`, `field_advantage`), buckets de estilo, latência e
  `risk_penalty`. `selection_status` fica `VETO_ONLY` e `promotion_order_valid`
  fica `false` SEMPRE no report cru.
- **Independência do painel**: tasks fixed de um candidato são função pura de
  (candidato, painel fixo, seeds) — peers só se encontram em tasks `peer_2p`
  diagnósticas, e lineups 4p completam APENAS de `DEFAULT_4P_FILLERS`.
  Travado por `test_selector_panel_independence`.
- A calibração roda as 7 âncoras (`LB_ANCHORS`) como candidatas e exige:
  Spearman ≥ 0.60, zero inversão grave (>75 pts LB), allscripts reprovado,
  holdwave topo-ou-empate, pgs_hold ≤ producer (o falso positivo histórico),
  scoring isolado por candidato. `calibration_valid=false` = liga segue
  veto-only (resultado honesto).
- O motor de decisão aplica as 14 regras em ordem (preflight do pool →
  calibração válida+hash → report é selector/holdout → gates técnicos →
  `CHOICE_GATES_4P` (0.28/0.20/0.18) → latência ≤500ms → buckets →
  P(>incumbente) ≥ 0.80 → P(>2º) ≥ 0.70 → tiebreak P ≥ 0.75) e pode responder:
  `SUBMIT_CANDIDATE`, `KEEP_INCUMBENT`, `RUN_MORE_GAMES`, `CALIBRATION_FAILED`,
  `NO_TECHNICALLY_VALID_CANDIDATE`, `INVALID_REFERENCE_POOL`.
  **Empate favorece SEMPRE o incumbente.**

## Funil e perfis

`quick` (4 seeds, smoke) → `standard` (8, dev) → `strong` (24, veto sério,
reduz a top 2-3) → `selector` (48 seeds, escolha de submissão). Tiebreak:
top-2, 96 seeds, refs críticas + H2H direto.

## Seed splits (regra de burn)

| split      | seed_base | uso                                            |
|------------|-----------|------------------------------------------------|
| dev        | 70_000    | desenvolvimento, queima livre                   |
| validation | 170_000   | veto e calibração oficial                       |
| selector   | 270_000   | SÓ a decisão final de submissão (holdout)       |

Bot alterado depois de ver resultado do holdout `selector` espera novo holdout
(reservar 370_000 como `selector_v2` quando queimar). `--seed-base` e
`--seed-split` são mutuamente exclusivos.

## Rotina operacional

1. Desenvolvimento: `quick`/`standard` (crash, invalid move, regressão óbvia).
2. Pré-seleção: `strong` → top 2-3 (passar gates; não perder p/ producer nem
   incumbente; não colapsar 4p; não falhar bucket crítico).
3. Calibração (se código/painel mudou): `league_selector_calibration` no split
   `validation`; refresh de `LB_ANCHORS` via Kaggle CLI ANTES.
4. Decisão: ruler `--profile selector --seed-split selector` nos top 2-3 →
   `league_submission_selector --report ... --calibration ...`.
5. Tiebreak se top-1/top-2 inseparáveis; ainda empatado → incumbente.
6. Pós-submissão: LB estabilizado entra em `LB_ANCHORS` (melhora a próxima
   calibração).

## Don'ts (cada um tem guard-test em tests/test_selector_dont_rules.py)

Sem `recommended_candidate`; sem `overall_score` cru; sem misturar winrate
2p/4p sem normalizar; peers nunca como filler 4p; seeds de dev nunca decidem
submissão; challenger empatado nunca substitui o incumbente.

## Baseline congelado

`selector_baseline_2026_06_12` — ver docs/SELECTOR_BASELINE.md. Toda mudança de
scoring re-pontua os MESMOS jogos antes/depois.
