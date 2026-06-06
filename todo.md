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
- [x] B1. FEITO (diag faithful, 8 seeds): `mean_fitness_delta_oep_minus_producer=+3.74` (o OEP ACHA seu plano melhor) e desvia 14% (`min_advantage`=0 default → desvia em QUALQUER delta>0), mas a 96 seeds PERDE (−0.128). **Causa: o fitness SUPERESTIMA os desvios** — como o world-model agora é fiel, a superestimação vem do **lookahead 1-ply** (best-response a uma resposta prevista do Producer, que na verdade re-planeja todo turno). Desvios de delta pequeno = ruído.
  - [x] C-exp1: `OEP_MIN_ADVANTAGE` (96 seeds, jobs=8/OMP=1): `0→−0.128`, **`15→−0.045 (ÓTIMO)`**, `40→−0.108`. Curva NÃO-monotônica → sweet spot ~15 (desvios de delta 15–40 são bons; <15 é ruído). Melhor OEP até agora = **min_advantage≈15 (−0.045)**, partindo de −0.21. Mas não cruza 0: seleção sozinha satura perto do empate.
  - [x] C-exp2: config WIDER (12/12/4/6) + min_advantage=15 = −0.080 (PIOR que narrow −0.045). Mais candidatos não ajudam.
  - [x] ANÁLISE das curvas (96 seeds): delta∈(0,15] net −0.083 (ruído); delta∈(15,40] net **+0.063 (BONS)**; delta>40 net −0.108 (over-confiança — as maiores vantagens projetadas são as mais ilusórias). Os desvios bons estão numa BANDA média.
  - [x] C-exp3: filtro de BANDA (`max_advantage`, novo knob) 15<delta<40 = **−0.274 (MUITO PIOR)**. Hipótese de banda FALHOU: as contribuições dos desvios são não-lineares/interagentes (tirar delta>40 piorou drástico). Não dá para isolar bandas.
  - [x] C-exp4: horizon=28 + min_advantage=15 = **−0.271 (MUITO PIOR)**. Hipótese do intel (horizonte longo) NÃO vale no OEP — mais projeção → mais superestimação acumulada. Revertido para 18.
  - **CONCLUSÃO OEP (6 evals 96 seeds):** melhor = horizon=18 + min_advantage=15 (**−0.045**, de −0.21). TODO o resto regrediu (min_adv 0/40, wider, banda, horizon=28). **O OEP NÃO bate o Producer por tweak** — satura em −0.045 (empate quase). Gargalo = o FITNESS de 1-ply SUPERESTIMA e é RUIDOSO (não-monotônico ⇒ não é viés thresholdável; é imprecisão). Mais candidatos PIORAM (wider/banda). Logo busca real (C1) provavelmente também piora. O fix de raiz é um fitness mais preciso: **2-ply** (oponente responde ao plano do OEP — Justesen 2016) ou rebuild do scoring. Ambos são builds substanciais com payoff incerto (provável teto ≈ empate, a menos que existam planos robustamente melhores que o Producer perde — e existem, pois o topo do leaderboard > 1228).
  - OEP NÃO submetido (perde o Producer). MAS: o fix do world-model do orbit_lite corrige o modelo que o PRÓPRIO Producer usa para planejar (o Producer 1228 planejava sobre o modelo buggy). Risco mínimo (Kaggle mantém a melhor submissão).
  - [x] SUBMETIDO `53408639` → **ERROR no Kaggle** (inofensivo; 1231.9 segue como melhor). PREMISSA ERRADA: o tarball 1231.9 JÁ tinha o swept-pair (`movement.py` idêntico ao meu fix). Meu fix corrigiu uma REGRESSÃO no REPO (WIP reverteu p/ 2-passadas), NÃO a submissão. A submissão errou porque o `main.py` do 1231.9 é AUTOCONTIDO (422 linhas ≠ do `bots/producer/agent.py` que carrega `_upstream.py` via importlib — falha no Kaggle). ⚠️ **`scripts/package_producer_submission.py` gera submissão QUEBRADA** (estrutura agent.py+_upstream vs main.py autocontido do 1231.9).
  - **CONCLUSÃO:** nada melhor que 1231.9 foi encontrado. O Producer já é o melhor E já tem o fix de colisão. Bater 1231.9 exige build substancial (busca real / 2-ply), payoff incerto. Espaço barato esgotado.
  - [x] CONSERTADO: a submissão errava por `NameError: __file__`. O Kaggle roda o agente via `compile()+exec()` com globals VAZIO (`kaggle_environments/agent.py:48,57`) — `__file__` é indefinido — e adiciona o dir do tarball ao `sys.path`. O `bots/producer/agent.py` usava `Path(__file__)` no nível do módulo → crash. Fix: `_load_upstream` tenta `import _upstream` primeiro (Kaggle-safe; o dir está no path), com fallback `__file__` (repo). Validado pelo MECANISMO exato do Kaggle (`get_last_callable`): carrega + roda. test_oep_agent 12 passed.
  - [ ] (follow-up menor) Packaging do OEP: `bots/oep/agent.py` faz `from bots.oep.planner import agent` — num tarball flat precisa empacotar `bots/` + `orbit_lite/` na raiz (o `get_last_callable` põe o dir no path). Não bloqueia nada hoje (OEP perde do Producer).

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
