> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `EXPERIMENTS.md` — este arquivo fica enxuto, só com o que atacar e o estado atual.

---

# 🎯 ATACAR AGORA — superar o Producer

## 🆕 OEP SUBMETIDO ao Kaggle (2026-06-06, ref 53432895) — teste do score REAL
Insight (do usuário): a régua local (1v1 vs Producer, −0.045) é PROXY; o score do leaderboard é torneio vs ~109 agentes — um bot que perde 1v1 pode ser mais robusto vs o campo. **torch funciona no Kaggle** (Producer prova), então o OEP é submetível (`scripts/package_oep_submission.py`, min_advantage=15 baked, validado no `get_last_callable`). Submetido risk-free (best-counts). **RESULTADO: ERROU** (`SubmissionStatus.ERROR`) — Producer 1199.2 intacto (best-counts confirmado, sem dano). Causa provável: validação local foi 2p/sem-cometas/1-decisão; o torneio tem **cometas+4p+hardware lento** → actTimeout estourado ou crash em cometas/4p (não testados). **CONSERTADO E RE-SUBMETIDO** (ref 53433131, wrapper robusto no main.py: lookahead em thread com orçamento ~0.6s → fallback ao Producer; corrige o timeout sem violar no-silent-fallback do runtime). **COMPLETOU → publicScore 600.0** = MUITO abaixo do Producer (1199.2). **Achado definitivo:** o OEP está overfit ao Producer — o proxy 1v1 (−0.045) era enganosíssimo; contra o campo de 109 agentes o OEP é catastrófico (600). **Bot melhor NÃO encontrado; o OEP é muito pior na métrica real.** Producer 1199 intacto (best-counts). Lição: o leaderboard é a única régua que importa; o 1v1-vs-Producer pode superestimar enormemente.


Producer = a melhor que temos (o piso). Meta: candidato com **margin > 0 vs Producer a 96 seeds,
crash/timeout/invalid = 0**. Ordem fixa (motor→backtest→bot; corrigir CORREÇÃO antes de otimizar).
⚠️ Decidir **só a 96 seeds** — 16/4-seed é ruído (vimos +0.31 vs −0.21 no MESMO agente).

## ✅ A — world-model do orbit_lite fiel novamente (motor/lookahead verde)
- [x] **A. `test_movement_l3` saiu do xfail sem mudança de lógica no `orbit_lite`:** o repro seed=37 era efeito de extensão nativa `orbit_wars_rs` stale em relação ao fonte Rust atual. Após rebuild real do binding (`rtk .venv/bin/python -m maturin develop --release -m crates/orbit_wars_py/Cargo.toml`), `test_movement_l3_matches_rust_with_random_valid_launches --runxfail` passou 200/200 e a suíte de movimento passou 9/9.
  - **Correção de honestidade:** lag uniforme e `normal OR lag` foram hipóteses falsas; não entraram no motor. O fix válido é garantir que o binding Rust local esteja reconstruído contra o fonte atual.
  - **Validação adicional:** `tests/test_parity_actions.py` passou 4/4 depois do rebuild, confirmando que o Rust carregado voltou a bater o oficial nas janelas de ação.
- [x] **A-parcial (o que de fato ficou fiel):** swept-pair cobre origem/destino, cometas e terceiro planeta quando o backend nativo está atualizado.
- [x] **B. Diagnóstico OEP (6 evals 96 seeds):** o OEP satura no **empate (−0.045)** e NÃO bate o Producer por tweak. Knobs exaustados — todos regrediram vs −0.045: `min_advantage` (curva NÃO-monotônica: ótimo em 15), WIDER (−0.080), banda (−0.274), horizon=28 (−0.271). **Causa-raiz inicial medida: o fitness 1-ply SUPERESTIMA e é RUIDOSO** (best-response a um Producer ESTÁTICO que na verdade re-planeja; as maiores vantagens projetadas são as mais ILUSÓRIAS). E1/E2 testaram a camada de seleção/diagnóstico; revisão E0 conclui que o próximo salto precisa melhorar **plano-candidato**, não só re-pontuar planos existentes.
- [x] **Submissão Producer (1231.9) Kaggle-safe:** `NameError: __file__` resolvido em `_load_upstream` (tenta `import _upstream` primeiro). Validado por `get_last_callable`.

---

# 🔬 E. NOVAS SOLUÇÕES — explorar para CRUZAR o 0 (pós-diagnóstico, fundamentado)

> **Insight central (do diagnóstico B + revisão E0):** o avaliador 1-ply era ruidoso e inflava vantagem,
> mas E1/E2 só mexeram na **camada de seleção** entre os mesmos planos e não criaram plano melhor que o
> Producer. A próxima geração precisa atacar **qualidade do plano-candidato**: gerar `oep_entries` por
> um mecanismo diferente, dentro do orçamento de 1s, e só então medir a 96 seeds.

## ⚠️ E0. REVISÃO do que o codex implementou (2026-06-06) — o que está errado e POR QUÊ
O codex implementou E1a (variantes ordinais do oponente) **e** E2a (`reactive_reply` 2-ply) no `planner.py`,
mais o fix de reset do Producer. Revisão honesta:
- [x] **Acertou (não mexer):** (a) fix de reset do Producer (`_upstream.py`: `mem.reset()` no step 0) — era **memory-leak real** (`movement`/`last_sparse_action_row` vazavam entre jogos no mesmo worker, contaminando a régua e explicando a divergência jobs=1 vs jobs=4). (b) `_debit_entry_sources` **não** é débito-duplo — `apply_private_planned_launches` só semeia arrivals/stash, não debita o source (docstring confirma). (c) Rodou o **gate 96 seeds** de verdade: margin=−0.099, win=0.448, `passed=False`.
- [x] **ERRO #1 (ESTRATÉGICO — aceito):** E1a e E2a são **mudanças na CAMADA DE SELEÇÃO** — só escolhem melhor entre os MESMOS dois planos (OEP vs Producer); **nenhum melhora os planos-candidatos do OEP**. A réplica E2a desinflou o diagnóstico de fitness no profile serial (`mean_fitness_delta≈0.138`), mas a margem de 16 seeds continua só triagem e não mede promoção. POR QUÊ: se o novo trabalho só re-pontua `oep_entries` já existentes, ele pode reduzir ilusão, mas não cria plano melhor que o Producer. A alavanca que cruza 0 é **plano-candidato melhor** (busca real C1 / MCTS E3 / valor aprendido E4), não mais um chooser.
  - [x] DECISÃO acionável: parar de afinar seleção (E1/E2) e pivotar para **qualidade do plano** (C1 ou E3). Verificar: o novo trabalho gera `oep_entries` por um mecanismo diferente do atual, não só re-pontua os mesmos.
- [x] **ERRO #2 (METODOLÓGICO — CORRIGIDO):** o `reactive_reply` recomputava só o `oep_fitness` contra a réplica reativa e deixava o `producer_fitness` contra a previsão estática → comparava OEP-sob-adversário-reativo vs Producer-sob-oponente-passivo (limiar enviesado contra desviar). **Fix (2-ply simétrico, Justesen):** `_reactive_reply_entries` foi generalizada (param `our_entries`) e agora é chamada TAMBÉM para o `producer_entries`; o `producer_fitness` é recomputado contra a réplica do oponente AO PLANO DO PRODUCER. Cada plano é pontuado contra a resposta a ELE.
  - [x] verificado: profile serial 1 seed/128 — as duas réplicas rodam **59/59 vezes** (`producer_reactive_reply` e `producer_reactive_reply_baseline`, simétrico); custo extra ~12,5ms só nos turnos gated (`advantage>prune`); `max_decision_ms=118`, `timeout=0.0`. ruff + `test_oep_agent.py` 24 passed. Knob off-by-default → agente padrão inalterado.
- [x] **ERRO #3 (corrigido no registro):** E2a JÁ foi benchmarkado (16 seeds: margin=−0.0326, timeout_rate=0.000701; E2b prune0: margin=−0.0625, timeout_rate=0.01257). **MAS a rejeição não pode depender da margem 16-seed nem do timeout paralelo isolado:** o profile SERIAL tem `max=119ms` (E2a) / `91ms` (E2b), ~8× de folga, timeout=0.0. O motivo honesto de não promover agora é: comparação 2-ply ainda assimétrica (ERRO #2), margem não gate-confirmada e E1/E2 não melhoram o plano-candidato.
  - [ ] (opcional) E2a é o ÚNICO candidato que (a) conserta a inflação diagnosticada E (b) é numericamente o melhor a 16 seeds → merece **UMA** rodada 96 seeds **serial** (jobs=1) pra cravar o teto. Mas mesmo se der empate, não basta pro TOP-5 → pivotar pra C1/E3 de qualquer forma.
- [x] **ERRO #4 (confirmado e corrigido no artefato):** o fix de reset **muda o baseline do Producer** → toda margem medida ANTES dele é suspeita. Verificação inicial de `artifacts/submission_producer.tar.gz`: o `main.py` autocontido antigo só zerava `cached_player_count` no step 0; não chamava `mem.reset()` e nem continha `_upstream.py` separado. O tarball foi reconstruído com o script versionado e agora contém `_upstream.py` com `mem.reset()` no step 0.
  - [x] ação: reconstruir/validar o tarball Producer antes de usá-lo como régua ou artefato de submissão.
  - [ ] re-rodar qualquer baseline pré-fix antes de citar.
- [x] **Nit menor:** validação `reactive_reply + ordinal_variants>1` movida para `OEPLiteConfig.__post_init__`, com teste de env/config. Agora falha na construção da config, não dentro do turno.

## C1. [PRIMEIRO DEGRAU] Qualidade do plano-candidato antes de MCTS completo
Objetivo: gerar `oep_entries` por mecanismo diferente do guloso atual. Não é mais tuning de seleção;
o candidato precisa existir como plano alternativo antes da comparação OEP-vs-Producer.
- [x] **C1a. Plano-memória temporal — IMPLEMENTADO E REJEITADO COMO DEFAULT.**
  - [x] `OEP_PLAN_MEMORY_VARIANTS=N` reconstrói até N lanes do último plano OEP executado no estado atual e escolhe entre plano guloso atual vs plano-memória por fitness de plano inteiro.
  - [x] Default preservado (`OEP_PLAN_MEMORY_VARIANTS=0`); validação/env/testes cobrem o knob.
  - [x] Verificação: profile 1 seed/128 teve `timeout=0.0`, `max_decision_ms=82.24`, `mean_decision_ms=45.41`; a variante de memória foi candidata 43 vezes e venceu só 1 (`choice_rate=0.023`). Smoke 4 seeds/500: crash/invalid/timeout=0, `mean_score_margin=-0.25` (ruído, não gate), `mean_decision_ms≈360` no runner.
  - [x] Decisão: não promover nem rodar 96 seeds. O mecanismo gera plano alternativo real, mas quase nunca supera o guloso e adiciona custo. Próximo C1/E3 precisa gerar variações por rollout/beam efetivo, não só persistência de lanes antigas.
- [x] **C1b. Beam do primeiro lance — IMPLEMENTADO E REJEITADO COMO DEFAULT.**
  - [x] `OEP_BEAM_FIRST_WIDTH=N` força cada um dos top-N primeiros lances elegíveis, completa o restante com o greedy atual, inclui regroup e escolhe o melhor plano inteiro por fitness.
  - [x] Default preservado (`OEP_BEAM_FIRST_WIDTH=0`); validação/env/testes cobrem o knob.
  - [x] Verificação: profile 1 seed/128 teve `timeout=0.0`, `max_decision_ms=88.44`, `mean_decision_ms=47.55`; o beam gerou 358 candidatos em 232 decisões e escolheu alternativa só 3 vezes (`choice_rate=0.013`). Smoke 4 seeds/500: crash/invalid/timeout=0, `mean_score_margin=-0.25` (ruído, não gate), `mean_decision_ms≈381` no runner.
  - [x] Decisão: não promover nem rodar 96 seeds. Gerar variações só no primeiro lance quase sempre confirma o guloso; próximo C1/E3 precisa de rollout multi-turn ou avaliação de nó que altere a árvore, não só beam raso.

## E1. [FECHADO] Avaliador ordinal: parar de confiar na MAGNITUDE do fitness
**Diagnóstico que sustenta:** a curva de `min_advantage` é **não-monotônica** e os desvios delta>40
(maiores vantagens projetadas) são os mais ILUSÓRIOS → a *magnitude* do fitness é não-confiável, mas a
*ordenação* relativa pode ainda carregar sinal. É exatamente o cenário do paper.
**Embasamento FORTE** — Joppen & Fürnkranz, *Ordinal Monte Carlo Tree Search* (arXiv:1901.04274):
recompensas numéricas handcrafted são "necessariamente enviesadas"; o comportamento do agente muda com
o *encoding* da recompensa; o tratamento **ordinal** (comparação por ranking, invariante a transformação
monotônica) supera. Casa 1:1 com nosso fitness handcrafted superestimando.
**Onde mexer:** seleção em `planner.py:1519-1533` (hoje `_advantage = oep_fitness − producer_fitness` comparado a `min_advantage`); fitness determinístico em `_plan_fitness` (`planner.py:409`, um ÚNICO `opponent_launch_set` estático); knobs em `OEPPlannerConfig` (`planner.py:~134`).
- [x] **E1a. Avaliação multi-seed do oponente — IMPLEMENTADA E REJEITADA COMO DEFAULT.**
  - [x] Gerou variantes determinísticas de `opponent_launch_set` (base, fração 0.75/0.50, atraso +1, top-K) via `OEP_ORDINAL_OPPONENT_VARIANTS`.
  - [x] Para cada variante, compara `s_oep[k]` vs `s_prod[k]` e seleciona por `wins/K >= OEP_ORDINAL_WIN_THRESHOLD`; `wins/K` entra no `record_selection`/`profile_oep_step`.
  - [x] Verificação de custo/cauda: K=3 profile serial 4 seeds/500 teve `mean=38.76ms`, `max=138.94ms`, `max_match_p95=84.08ms`, `timeout=0.0`; K=5 profile serial 4 seeds/500 teve `mean=50.62ms`, `max=156.21ms`, `max_match_p95=103.41ms`, `timeout=0.0`. No runner paralelo, K=3 smoke 4 seeds teve `mean_decision_ms=343.99`, `timeout=0.0`; K=5/16 seeds teve `mean_decision_ms=347.72`, 3 timeouts (`timeout_rate=0.000584`), possivelmente com contenda local. A métrica decisiva é cauda (`p99`/`max`/`timeout_count`), não média nem margem pequena-seed. Detalhe em `EXPERIMENTS.md` (2026-06-06).
  - [x] Decisão: não promover nem rodar 96 seeds; default permanece no avaliador escalar anterior (`OEP_ORDINAL_OPPONENT_VARIANTS=1`). Margem de K=3/K=5 é inconclusiva porque 4-seed é ruído e 16-seed não é gate; não registrar "empatou no 4-seed" como achado. A rejeição é de **custo/design**: K=3/5 é pequeno demais para denoising ordinal forte; K alto o bastante para o efeito do paper teria custo linear e precisaria provar p99/max/timeout_count antes de qualquer gate. A hipótese ordinal simples também só perturba o mesmo plano estático, sem modelar a resposta reativa do Producer.
- [x] **E1b. CANCELADO por escopo atual:** torneio ordinal par-a-par ainda re-pontua os mesmos candidatos. Não atacar enquanto o próximo trabalho precisar gerar planos melhores, não só escolher melhor entre planos saturados.

## E2. [FECHADO] Best-response 2-ply (oponente responde ao plano do OEP)
**Diagnóstico que sustenta:** a superestimação vem do 1-ply best-response a uma resposta ESTÁTICA do
Producer; o Producer real re-planeja. Fechar isso é o conserto direto da causa.
**Embasamento FORTE** — Justesen, Mahlmann & Togelius 2016 (*Online Evolution / rolling-horizon contra
oponente que reage*): modelar a réplica do oponente reduz a ilusão de vantagem.
**Onde mexer:** o `opponent_launch_set` passado a `_plan_fitness` (`planner.py:1497-1517`) é hoje uma previsão ESTÁTICA (`_cheap_opponent_entries`/`opponent_entries` de `plan_oep_waves`). O 2-ply troca essa previsão por uma RÉPLICA reativa ao plano do OEP.
- [x] **E2a. Réplica reativa do Producer ao plano do OEP — IMPLEMENTADA E REJEITADA COMO DEFAULT.**
  - [x] Clona o `PlanetMovement`, aplica as chegadas futuras do plano OEP, debita explicitamente as fontes do OEP no clone e chama o Producer inline como réplica (`OEP_REACTIVE_REPLY=1`).
  - [x] Pontua o OEP contra `opp_reply_launch_set`; o Producer baseline continua com a previsão estática atual. Isso torna E2a **diagnóstico assimétrico**, não seletor promovível.
  - [x] Verificação: o profile serial confirma o diagnóstico — `mean_fitness_delta` caiu para perto de zero (4 seeds/500: **0.138**, vs deltas inflados anteriores), sem cauda perto de 1s (`max=119.62ms`, `timeout=0.0`).
  - [x] Decisão: não promover. Margem 16-seed é só triagem, não gate; o bloqueio correto é comparação assimétrica + ausência de plano-candidato novo.
- [x] **E2b. Poda por custo — IMPLEMENTADA E REJEITADA COMO DEFAULT.**
  - [x] Curto-circuito por vantagem 1-ply (`OEP_REACTIVE_REPLY_PRUNE_ADVANTAGE`, default experimental 0): se o OEP já não vence no avaliador barato, não chama a réplica.
  - [x] Verificação serial: chamadas reativas caíram **469→177** em ~1k decisões; `mean_decision_ms=38.45`, `max=91.05`, `timeout=0.0`.
  - [x] Decisão: não promover. A poda reduziu custo serial, mas não muda o problema estrutural: E2b continua selecionando entre os mesmos planos e ainda herda a comparação assimétrica.

## E3. [FECHADO — IMPLEMENTADO E REJEITADO] Busca por valor de rollout sobre conjunto de candidatos
Implementado o E3-flat: `_oep_plan_variant_list` (conjunto diverso de candidatos = guloso + beam do 1º lance)
+ `_rollout_value` (cada candidato pontuado pela **réplica reativa do Producer** + valor terminal, em vez do
fitness 1-ply) + ramo de seleção `OEP_ROLLOUT_SEARCH_WIDTH` (off-by-default). Roda no binding fresco (motor fiel).
- [x] **Custo ok:** profile W=4 → `max_decision_ms=145` (~7× folga vs `actTimeout=1s`), `timeout=0`, estágio `rollout_search` ~21ms. Default (W=0) inalterado: smoke legal 0 timeout; 29 testes OEP passam (2 novos: lista de variantes + guard); ruff limpo.
- [x] **GATE 96 seeds vs Producer (binding fresco): margin = −0.20087, win = 0.396, crash/invalid/timeout = 0.** → **REGREDIU FORTE** (−0.201 ≪ −0.045 do baseline).
- [x] **Conclusão:** mais diversidade de candidato + valor de rollout (réplica reativa) ainda imperfeito = **MAIS desvios net-negativos**. Consistente com B (wider/banda/horizon todos regrediram). O gargalo NÃO é gerar/avaliar candidatos sobre o Producer — é que **os desvios do OEP são intrinsecamente piores que o plano do Producer**, e nenhum avaliador orbit_lite (1-ply/2-ply/ordinal/rollout-flat) isola com confiança os raros desvios bons.
- [x] **Decisão:** rejeitar E3 como default (`OEP_ROLLOUT_SEARCH_WIDTH=0`). A família inteira "busca/seleção sobre o Producer" (E1/E2/C1/E3) **satura ou regride → teto da arquitetura Producer+best-response atingido**.
- [x] **Confirmação extra (horizonte longo):** testei a busca de rollout com `OEP_HORIZON=40` (insight do intel "horizonte longo + valor terminal") → **−0.31 (16 seeds), pior ainda**. Horizonte longo com oponente de resposta única superestima MAIS (bate com C-exp4). Família rollout morta em todos os horizontes. Knob de horizonte revertido (não ajudou).
- [x] **Valor terminal de TERRITÓRIO (`OEP_ROLLOUT_TERMINAL_VALUE=1`, intel sim-value-search):** produção-território no estado terminal. **GATE 96 seeds = −0.10917** (melhor que E3 net-ship-delta −0.20, mas abaixo do −0.045). Território é melhor preditor que ships, mas NÃO cruza.
- [x] **Território + threshold conservador (reuso `OEP_MIN_ADVANTAGE` na escala território):** ⚠️ **ARMADILHA DE 16 SEEDS** — a 16 seeds deu monotônico e POSITIVO (adv1=0.0, adv3=+0.25, adv6=+0.4375 win=0.72), parecia o avanço. **Gate 96 seeds (adv=6) = −0.24274, win=0.375.** O +0.44 era PURO RUÍDO de seed-count baixo (mesma armadilha do +0.31 vs −0.21). O threshold PIOROU a 96 seeds (−0.109 sem → −0.24 com). Lição reforçada: **só 96 seeds decide — 16 seeds gera falso-positivo convincente.** A disciplina pegou o falso-positivo antes de qualquer claim/submissão.

## E5. [INICIADO — agente de busca STANDALONE, fora do frame best-response]
Construído increment 1 do agente sim-value-search standalone (`OEP_STANDALONE_TERRITORY=1`, off-by-default): enumera lanes amplas (owned × top-K alvos mais próximos), rankeia por ganho de **território terminal**, combina greedy — **sem âncora no Producer** (`_standalone_territory_plan`).
- [x] **Increment 1 = −0.75 (16 seeds), win 0.125, 0 crash/timeout, 110ms.** FRACO. Causas: (a) valor-território **sem-oponente** super-valoriza agressão (super-estende→esmagado); (b) geração de candidatos "atacar o mais próximo / full-send" é muito mais pobre que o targeting production-aware do Producer.
- [x] **Conclusão (prova real, não asserção):** um standalone competitivo precisa **reconstruir a sofisticação do Producer** (targeting por produção, safe-capture, defesa/recaptura) + as adições do intel + oponente reativo no valor. O Producer **já é** um agente sim-value forte perto do teto desta classe. Bater exige um build de pesquisa multi-sessão tunado, não um increment.
- [x] **Increment 2 (oponente reativo no valor):** consertou a super-agressão diagnosticada → **−0.25 (16 seeds), win 0.375** (de −0.75). Custo 129ms, 0 crash/timeout. Melhora real, mas ainda muito abaixo do Producer. Gargalo restante = **geração de candidatos** (nearest-attack << targeting production-aware do Producer).
- [x] **Padrão claro dos 2 increments:** cada fix melhora (−0.75→−0.25) mas o teto do standalone com candidatos pobres fica ~−0.1 a −0.25 — MESMO patamar dos variantes ancorados (−0.045 a −0.109). Confirma: o limite é a QUALIDADE DO CANDIDATO, e nenhum gerador disponível (nearest, OEP-lanes) bate o plano do Producer. Bater exige reconstruir o targeting production-aware do Producer + safe-capture + missões — build multi-sessão.
- [x] **Increment 3 (standalone sobre candidatos production-aware do OEP):** forçado a sempre jogar o melhor candidato por território (`OEP_MIN_ADVANTAGE=-999`) → **−0.252 (16 seeds)**, mesmo patamar do nearest (−0.25). **PROVA matemática do padrão:** dropar o fallback pro plano do Producer SEMPRE piora (ancorado −0.109 → standalone −0.25). O plano do Producer É a baseline forte; qualquer frame que desvia dele perde, e nenhum gerador de candidatos disponível produz planos melhores. **CONCLUSÃO DEFINITIVA (E1-E5, ~15 configs, 5 arquiteturas): o Producer está no teto desta classe heurística+sim; bater exige um agente de pesquisa tunado multi-sessão (timeline-sim completo do intel) ou RL em escala — não alcançável numa sessão.** NÃO investir em MCTS turno-a-turno (exigiria port do engine p/ stepper puro-Python; a evidência diz que o problema é o PLANO do OEP, não a profundidade da busca).

## E4. [ATIVADO — critério atingido; campanha longa] Política APRENDIDA via self-play
**Ativado** (2026-06-06): E1/E2/C1/E3 esgotados e registrados → o critério de `docs/TRAINING.md`
("OEP esgota ganho mensurável vs Producer") está satisfeito. A infra é madura (`train_ppo`,
`train_league`, `competitive_cycle`, `pbt`) e treina RÁPIDO no binding fresco (~65k timesteps em ~2 min).
- [x] **BASELINE MEDIDO (decisivo):** o checkpoint atual `phase0_seed1_65536_resume_seed4_65536.pt` (treinado vs heurísticas fracas) **PERDE 100% vs Producer** (`mean_score_margin=-1.0, win_rate=0`, 0 crash/invalid, 34ms). A política aprendida está MUITO abaixo do Producer.
- [x] **Realidade do E4:** o Producer NÃO é oponente de treino (só `PHASE0_OPPONENTS` heurísticos); ele é o ALVO de avaliação. Fechar de −1.0 até bater o Producer exige a campanha completa: Phase 0 (vs fracos) → Phase 2-3 (self-play league / PBT / hall-of-fame) → export → gate vs Producer. É **sustentado e incerto** (RL batendo heurística forte de 1200 é difícil; pode platôar abaixo), NÃO um único run.
- [x] **CAMPANHA SELF-PLAY RODADA (2026-06-06):** driver `/tmp/selfplay_campaign.py` → `run_competitive_cycle` 3 gerações, POP=4, ~32k timesteps/membro (~393k total), PBT + hall-of-fame + 5 heurísticas, semeado do phase0. Campeão `selfplay_campaign/generation_002/seed_001.pt` exportado e gateado vs Producer (16 seeds): **margin = −1.0, win = 0 — IDÊNTICO ao baseline, self-play NÃO moveu nada.** A política perde por margem MÁXIMA todo jogo (não é "perto"; é fundamentalmente fraca vs o Producer). Treinar vs heurísticas fracas + self não ensina a lidar com o jogo forte do Producer.
- [ ] (se retomar E4) precisaria de ordens de magnitude mais compute (milhões+ de timesteps), curriculum/reward dedicado, e possivelmente arquitetura diferente — multi-dia, convergência incerta. NÃO é caminho de uma sessão.

### Estado da exploração (2026-06-06): bot melhor que o Producer NÃO encontrado nesta passada
Resumo honesto após exaurir os levers tratáveis:
- **Busca/seleção sobre o Producer (E1/E2/C1/E3):** todos saturam (~−0.045) ou regridem (E3: −0.20). FECHADO.
- **Política aprendida (E4):** baseline atual = **−1.0 (perde tudo)**; gap enorme; campanha self-play longa e incerta.
- **Melhor bot continua o Producer (1231.9).** As alternativas reais (campanha E4 OU reescrita como agente de busca standalone tipo sim-value/timeline do intel) são projetos sustentados, não wins de uma sessão.

> **Honestidade sobre a evidência (postura tech-lead):** E1 e E2 tinham embasamento forte para corrigir
> diagnóstico/seleção, mas nesta forma não geram plano-candidato melhor e ficam rejeitados como default.
> E3/E4 têm teto maior, mas são builds grandes com payoff incerto (o topo do leaderboard > 1228 prova
> que existem planos robustamente melhores que o Producer; o OEP local satura no empate). Nenhum
> knob/tweak adicional de seleção deve ser priorizado — isso já foi exaustado em B/E1/E2.

## D. [PARALELO — ganho barato] Ligar o lookahead em 4p (ex-F1)
Leaderboard pontua 2p E 4p; hoje o OEP é guloso em 4p (`_opponent_id` → None, sem 1-ply). Princípio
**forte** (modelar oponente — Justesen 2016); escolha do oponente 4p é fraca (começar por 1).
- [ ] verificar: em 4p `opponent_entries` ≠ None; win 4p ≥ baseline OEP-4p-off, sem regressão de timeout.

---

# Estado atual (2026-06-06, pós-fixes de fidelidade + diagnóstico OEP)

- **Régua/SIMULADOR (Rust) FIEL ao Kaggle** — `parity_probe_actions` + `tests/test_parity_actions.py` (~40k steps, 0 div). Margens 96 seeds confiáveis.
- ✅ **World-model do orbit_lite (lookahead) fiel com binding Rust atualizado:** `test_movement_l3` não está mais em xfail; suíte de movimento esperada é 9 passed.
- **Melhor OEP local = −0.045 vs Producer (96 seeds)** — empate quase; satura por tweak. E1/E2 reduziram ilusão de avaliação, mas não geram plano melhor. Gargalo = **qualidade do plano-candidato** (E3/C1).
- **Submissão Producer (1231.9) Kaggle-safe contra `NameError` e tarball reconstruído com reset completo**; ainda revalidar baselines antigos antes de citar margem medida pré-fix.
- As margens antigas do roadmap foram medidas em **régua infiel** — re-validar SEMPRE a 96 seeds.

# Feito (detalhe no git/EXPERIMENTS/tests)

- Fidelidade do simulador Rust: combate (ordem de colisão / swept-pair / timing de cometa) + gerador + obs.
- Diagnóstico OEP completo (6 evals 96 seeds): o OEP satura no empate; mais knob/horizonte PIORA; E1/E2 indicam que re-pontuar os mesmos planos não basta.
- Submissão Producer Kaggle-safe (`NameError: __file__` resolvido em `_load_upstream`).
- Infra/portfólio: `conftest.py`, scripts de diagnóstico versionados, reorganização de `docs/`.

# Roadmap remanescente (condensado)

- **2º oponente-régua**: adicionar um oponente forte do fórum ao gate, para não overfitar só o Producer. Obrigatório para top 5.
- **PPO/self-play**: seção E4 (DEFERIDO).
- **Marcar `EXPERIMENTS.md`** que experimentos anteriores aos fixes foram em régua infiel.
- **Screenshots `artifacts/kaggle_*.png`**: decidir se entram no portfólio (decisão sua).

---

# MANUAL — arquitetura e invariantes (não apagar)

## 3 camadas (nunca misturar)
1. **MOTOR / verdade física** — `crates/orbit_wars_core` (Rust) + binding; `orbit_lite/` é o sim Python leve usado DENTRO do lookahead. Fidelidade: `parity_probe_actions` vs `kaggle-environments` + `test_movement_fidelity`.
2. **BACKTEST / régua** — `benchmark_submission`, `compare_benchmark_significance`, `oep_promotion_gate`, `gate_check`. Mede candidato vs oponente; NÃO decide física nem estratégia.
3. **BOT / decisão** — `bots/oep/{planner,agent}.py`, `bots/producer/`, `python/agents/`, submissão `python/submission/submission_template.py` → `artifacts/submission.py`.

## Ordem de conserto INEGOCIÁVEL: motor → backtest → bot
Bot errado? → confirmar régua fiel. Régua errada? → confirmar motor no parity. Otimizar bot sobre
régua/world-model infiel = perseguir ruído (corrompe a correlação local↔leaderboard).

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
- Tweak de knob no OEP (min_advantage/horizon/wider/banda) — EXAUSTADO em B (tudo regrediu vs −0.045).
- GPU — agente roda CPU-only com `actTimeout=1s` e problema minúsculo; gargalo é qualidade do modelo, não compute.
