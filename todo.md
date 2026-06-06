> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `EXPERIMENTS.md` — este arquivo fica enxuto, só com o que atacar e o estado atual.

---

# 🎯 ATACAR AGORA — superar o Producer

Producer = a melhor que temos (o piso). Meta: candidato com **margin > 0 vs Producer a 96 seeds,
crash/timeout/invalid = 0**. Ordem fixa (motor→backtest→bot; corrigir CORREÇÃO antes de otimizar).
⚠️ Decidir **só a 96 seeds** — 16/4-seed é ruído (vimos +0.31 vs −0.21 no MESMO agente).

## A. [#1 — PRÉ-REQUISITO] Corrigir o world-model do orbit_lite (sair do xfail)
Regra do próprio repo: *"orbit_lite vermelho invalida tudo acima"* — L3/L5a estão xfail. O OEP otimiza
um fitness calculado por esse world-model; com colisão-fantasma (planeta rotacionando) e timing de
cometa errados, a busca persegue alvo torto. Embasamento **forte** (correção vs spec oficial; o fix
análogo já existe no Rust: `geometry::swept_pair_hit`).
- [x] A1. FEITO: trocada a colisão de 2 passadas (`direct`+`sweep`, posição velha → colisão-fantasma) por `_swept_pair_hit_mask` único em `_estimate_new_fleet_arrivals` (`movement.py`), planeta antes de bounds/sol. Corrigiu L3 E L5a (a captura de cometa do L5a era colisão frota×cometa).
  - [x] verificar: `xfail` de `test_movement_l3_*` e `test_movement_l5a_*` removidos → `test_movement_fidelity.py` 9 passed.
- [x] A2. NÃO era bug: o orbit_lite já usa a condição oficial de expiração de cometa (`future_idx < path_len` em `_apply_comet_paths`), não o off-by-one do Rust. (Minha suposição inicial estava errada.)
- [x] A3. FEITO: OEP vs Producer **96 seeds** com world-model fiel = **margin=−0.12810, win=0.432** (timeout 0.001). Progressão real: −0.21137 → −0.12810 (+0.083). Mas ainda PERDE (< 0) — o fix ajudou, não bastou. O OEP escolhe o Producer ~82% e desvia ~18% net-negativo: a busca atual PREJUDICA. → item B.

## B. [#2] Diagnosticar POR QUE o OEP perde (−0.21) no stack corrigido
Histórico 2b já achou que *"não é só o OEP nunca ser escolhido"* — atacar **calibração do fitness** ou
**composição de candidatos**, com análise POR-PARTIDA. Agora, com sim+world-model fiéis, é confiável.
- [ ] B1. `scripts.profile_oep_step` + `scripts.trace_submission_actions` em 96 seeds; isolar as partidas PERDIDAS e nomear a causa dominante (fitness mal calibrado? candidato OEP pior? timeout residual?).
  - [ ] verificar: relatório por-partida das derrotas com causa identificada por número.

## C. [#3 — O SALTO] Busca real OU melhoria dirigida pelo diagnóstico (só após A+B)
Escolher o ramo pelo que B apontar:
- [ ] C1 (candidato fraco): busca sobre população de genomas multi-lançamento (ex-Thread 1c). Embasamento **forte**: Justesen, Mahlmann & Togelius 2016 (rolling-horizon EA). Semear pop com guloso + Producer + incumbente.
- [ ] C2 (fitness): risco de timeline curta — penalizar launches bons-por-produção que abrem colapso local em 7–20 turnos. Embasamento **parcial** (intel do fórum).
- [ ] C3 (alternativa do topo): redistribuição por dominância como candidato de plano. Embasamento **fraco** (1 notebook).
  - [ ] verificar (qualquer C): **margin > 0 vs Producer a 96 seeds, crash/timeout/invalid = 0** ← GATE: SUPEROU O PRODUCER.

## D. [PARALELO — ganho barato] Ligar o lookahead em 4p (ex-F1)
Leaderboard pontua 2p E 4p; hoje o OEP é guloso em 4p (`_opponent_id` → None, sem 1-ply). Princípio
**forte** (modelar oponente — Justesen 2016); escolha do oponente 4p é fraca (começar por 1).
- [ ] verificar: em 4p `opponent_entries` ≠ None; win 4p ≥ baseline OEP-4p-off, sem regressão de timeout.

---

# Estado atual (2026-06-05, pós-fixes de fidelidade)

- **Simulador FIEL ao Kaggle** (combate/cometa/gerador/obs corrigidos) e régua confiável — travado por `parity_probe_actions` + `tests/test_parity_actions.py` (~40k steps, 0 divergências).
- **OEP vs Producer 96 seeds = margin −0.21137, win 0.391**, timeout≈0.0005 → ainda **perde**. (O 16-seed +0.31 era ruído.)
- As margens antigas do roadmap foram medidas em **régua infiel** (combate buggado + obs dict) — re-validar SEMPRE a 96 seeds antes de confiar.
- **Débito aberto:** world-model do `orbit_lite` ainda diverge (xfail L3/L5a) → é o item **A** acima.

# Feito nesta passada (detalhe no git/EXPERIMENTS/tests)

- Fidelidade do simulador Rust: 3 bugs de combate (ordem de colisão / swept-pair / timing de cometa) + gerador de treino (naves infladas) + formato de obs da régua (dict→lista oficial).
- Reavaliação do roadmap: margens OEP eram suspeitas; re-validadas a 96 seeds (−0.21).
- Infra/portfólio: `conftest.py` (parity não-silencioso), scripts de diagnóstico versionados, reorganização de `docs/`, docstrings de papel.
- OEP base (ex-1a/1b): seeding por incumbente, torneio único OEP-vs-Producer, deadline temporal removido — **números de margem precisam re-validar a 96 seeds** (eram em sim buggy).
- Perf (ex-2a/2c): hot-path = as 2 chamadas Producer (~51%); cache de garrison incremental. Cortes de custo da 2b (cheap/inline/tensor/shared/min_advantage/max_sources): **re-medir** — eram em sim buggy.
- Hygiene (ex-5/6): gate no-silent-fallback, guard no-native-import (D11), parity probe real.

# Roadmap remanescente (condensado)

- **2º oponente-régua** (ex-ETAPA 1 / 3b): adicionar um oponente forte do fórum (Producer~1200 / timeline-sim) ao gate, para não overfitar só o Producer. Obrigatório para top 5.
- **PPO/self-play** (ex-ETAPA 4 / F2): DEFERIDO. Reabre só se o OEP esgotar ganho contra o Producer ou surgir oponente externo forte. Critério em `docs/TRAINING.md`.
- **Marcar `EXPERIMENTS.md`** que experimentos anteriores aos fixes foram em régua infiel. (Deixei p/ você: seu log tem mudanças não commitadas.)
- **Screenshots `artifacts/kaggle_*.png`**: decidir se entram no portfólio ou saem (`git rm --cached`). Decisão sua.

---

# MANUAL — arquitetura e invariantes (não apagar)

## 3 camadas (nunca misturar)
1. **MOTOR / verdade física** — `crates/orbit_wars_core` (Rust) + binding; `orbit_lite/` é o sim Python leve usado DENTRO do lookahead. Fidelidade: `parity_probe_actions` vs `kaggle-environments` + `test_movement_fidelity`.
2. **BACKTEST / régua** — `benchmark_submission`, `compare_benchmark_significance`, `oep_promotion_gate`, `gate_check`. Mede candidato vs oponente; NÃO decide física nem estratégia.
3. **BOT / decisão** — `bots/oep/{planner,agent}.py`, `bots/producer/`, `python/agents/`, submissão `python/submission/submission_template.py` → `artifacts/submission.py`.

## Ordem de conserto INEGOCIÁVEL: motor → backtest → bot
Bot errado? → confirmar régua fiel. Régua errada? → confirmar motor no parity. Otimizar bot sobre
régua/world-model infiel = perseguir ruído (corrompe a correlação local↔leaderboard). Foi exatamente
o que aconteceu antes destes fixes.

## Invariantes
- **D10/D11** — submissão é Python puro/leve; nenhum `bots/`/`artifacts/` importa o crate Rust. Travado por `test_no_native_in_submission`.
- **Sem fallback silencioso** — falhar barulhento e medido; nunca degradar em silêncio.
- **Régua de qualidade = Producer** (+ 2º oponente futuro). `submission_v_old`/`greedy`/`rush` = só sanity de crash/legalidade, NÃO promovem.
- **Decidir só a 96 seeds.** **Score Kaggle**: esperar ~1h de estabilização.
- **EXPERIMENTS.md**: toda mudança no agente registra margem antes/depois vs Producer, ANTES do commit.

## Ciclo de desenvolvimento (toda mudança)
1. Mexeu em motor/`orbit_lite`? → `parity_probe_actions` + `test_movement_fidelity` ANTES de qualquer benchmark. Vermelho aqui invalida tudo acima.
2. Mexeu em régua/gate? → `bool(checks)` e seeds fixas; não afrouxar limiar/oponente.
3. Mexeu no bot? → hipótese em EXPERIMENTS.md → 96 seeds vs Producer (`make oep-promotion-gate`).
4. Antes de submeter → `scripts.gate_check` → `scripts.export_submission` → submeter → esperar ~1h.

## Parado / não fazer
- Micro-tuning de heurística (reservas/aberturas/hammer) — saturado contra bots locais (exp. 73–99).
- GPU — agente roda CPU-only com `actTimeout=1s` e problema minúsculo; gargalo é qualidade do modelo, não compute (EXPERIMENTS.md l.61). ROI seria paralelismo de ambiente em CPU.
