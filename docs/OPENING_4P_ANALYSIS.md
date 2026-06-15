# G3.1a — Abertura 4p: investigação e resultado NEGATIVO

> Ataca `bad_opening_4p` (124 derrotas, 33.8% — maior balde único da
> [[docs/LOSS_TAXONOMY.md]]). **Conclusão: o slice NÃO cede a intervenção de
> política de abertura.** Duas hipóteses de lever testadas e refutadas. Evidência
> forte. Registrado no DB id=250 (REJECTED).

Ferramentas (reutilizáveis): `scripts/replay_mining/opening_detector.py` (detector
offline sobre replays reais) e `scripts/replay_mining/opening_rule_eval.py`
(A/B 4p no motor Rust). Dados: `artifacts/replay_mining/opening_{bad,win}.csv`,
`eval_*.json`.

## Fase 1 — Detector: o "tell" NÃO é o que a hipótese dizia

Comparei o comportamento de abertura (primeiros 50 steps) em **124 derrotas
bad_opening_4p** vs **124 vitórias 4p** (controle). Alvo de fleet resolvido por
ground-truth (planeta mais próximo na chegada → contorna a aritmética de aim em
mapa rotativo).

**🐛 descartado:** `exp_success_rate` 1.0 e `mean_arrival_dist ≈ raio` nos DOIS
grupos → mira/ETA funciona; abertura ruim não é bug de targeting.

**Hipótese original (suprimir PvP cedo) REFUTADA:** comportamento de abertura é
estatisticamente IGUAL em vitória vs derrota — `pvp_share` 0.38 vs 0.36,
`neutral_share` 0.24 vs 0.25, `eta_first_pvp` 27 vs 26. E DENTRO das derrotas,
MAIS PvP cedo correlaciona com estado MELHOR (owned +0.30, captures +0.32, morte
mais tarde +0.11) — PvP e expansão andam juntos. Suprimir PvP teria piorado.

**Tell real isolado — "captura mas não SEGURA":**
- `capture_hold_rate` (frac. de neutros capturados no opening ainda nossos ao step 50): **0.71 derrota vs 1.00 vitória**.
- Perde ≥1 captura do opening em **58% das derrotas vs 31% das vitórias**.
- Dose-resposta DENTRO das derrotas: `corr(capture_hold_rate, elim_step) = +0.40` — segurar capturas ↔ sobreviver mais.

## Fase 2 — Regra implementada (limpa, 4p-only, off por padrão)

Margem de "capturar-e-segurar" no opening, no `plan_lite_waves` compartilhado
(Producer/PGS/OEP), via o hook `reinforcement` já existente do `capture_floor`:
para alvo NEUTRO no opening, exige naves p/ limpar defensores **+ pressão inimiga
projetada** (`cheap_enemy_pressure × margin`). Se não dá pra capturar-e-segurar, a
captura contestável é vetada. Knobs `opening_hold_margin` (0=off) /
`opening_hold_until_step` em `CONFIG_4P` (só 4p → **2p intocado por construção**).
26 testes de planner passam; off por padrão = zero mudança de comportamento.

## Fase 3 — A/B 96 seeds: a regra FUNCIONA mecanicamente mas NÃO move a margem

Producer no seat 0 (4p) vs 3 oponentes baseline, 500 steps, A/B pareado.

| oponente | margin | mean_margin | win | hold_50 | veredito |
|----------|--------|-------------|-----|---------|----------|
| producer | 0.0 | −0.5127 | 0.167 | 0.803 | base |
| producer | 0.3 | −0.5336 | 0.167 | 0.813 | margem PIOR |
| producer | 0.6 | −0.5127 | 0.177 | 0.822 | margem IGUAL (hold sobe) |
| rush | 0.0→0.6 | 0.999 | 1.00 | 1.0 | já dominamos; nada a corrigir |
| greedy | 0.0→0.6 | 0.9998 | 1.00 | 1.0 | idem |

A regra eleva `capture_hold_rate` (0.803→0.822) **como projetado**, mas isso **não
converte em margem 4p** em nenhum regime. Vs fracos já vencemos 100%; vs Producer
segurar mais capturas não muda o resultado.

## Fase 2/3 LITERAL — a regra exata do goal (suprimir PvP) também REPROVA

Para executar o goal como escrito (não só o redirect), implementei a regra literal
da Fase 2 — "suprimir ataque a jogador enquanto houver neutro seguro no raio"
(`opening_suppress_pvp` em `plan_lite_waves`, gate de candidatos PvP no opening) — e
rodei o gate de 96 seeds vs Producer:

| config (96s vs Producer) | mean_margin | death_rate | owned_50 |
|--------------------------|-------------|-----------|----------|
| baseline | −0.5127 | 0.635 | 3.70 |
| hold-margin 0.6 (redirect) | −0.5127 (igual) | 0.646 | 3.68 |
| **suprimir-PvP (regra literal do goal)** | **−0.5673 (PIOR)** | **0.719 (PIOR)** | 3.07 |

A regra literal PIORA margem e morte — confirma exatamente a previsão da Fase 1
(dentro das derrotas, MAIS PvP cedo → estado MELHOR). Suprimir PvP nos torna
passivos e somos atropelados. **Os dois levers (literal + redirect) reprovam o
critério "margem 4p melhora".**

## Conclusão (evidência forte)

`bad_opening_4p`, apesar de ser o maior slice de derrota, **não cede a política de
abertura**. Os dois levers plausíveis (suprimir-PvP, segurar-captura) foram
refutados: o primeiro pela ausência de diferença comportamental, o segundo por
A/B causal. `capture_hold_rate` é **sintoma** de posição contestada/perdedora, não
**causa** corrigível — consistente com bad_opening_4p ser majoritariamente
**posicional/dirigido-pelo-oponente** (nasce em vizinhança 4p contestada e perde a
corrida de expansão 3:1), não um erro de política nossa.

**Recomendação:** repriorizar para um slice com lever controlável —
`even_attrition_2p` (33%, decisividade de end-game; 2p é 44% do campo e o problema
é nosso, não posicional) ou `kingmaker_4p`+`overextension` (119/246 derrotas 4p =
"tínhamos posição e perdemos" — defender vantagem, mais acionável que abertura).

Código da regra mantido OFF (reprodutibilidade do A/B); não enviar.
