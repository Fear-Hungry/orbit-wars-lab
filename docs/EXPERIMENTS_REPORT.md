# Relatório de Experimentos

> Gerado de `experiments.duckdb` · **117** experimentos · período **2026-05-31** → **2026-06-07**. Fonte editável: `EXPERIMENTS.md` (importável).

## Resumo

| status | n | % |
|---|---:|---:|
| ✅ aplicados | 14 | 12% |
| ❌ rejeitados | 54 | 46% |
| 📋 logados | 46 | 39% |
| ⏳ pendentes | 3 | 3% |

## ✅ Aplicados (14) — o que entrou

| data | tag | ideia | antes → depois | decisão |
|---|---|---|---|---|
| 2026-06-07 | Paraleliz… | Paralelizar opponent (threads E processos) — AMBOS FALHAM | sequencial 250-275 SPS → THREADS (ThreadPoolExec… | default revertido p/ sequencial (oppone… |
| 2026-06-07 | T4 | T4: entity encoder (masked pooling) vs FlatActorCritic — comparação c… | flat colapsa (launchF1 … → entity: launchF1=0.60, … | aceitar entity como caminho; pendente: … |
| 2026-06-06 | E4 | E4 campanha self-play league rodada (genuína tentativa de bater o Pro… | hipótese: self-play lea… → campanha rodou end-to-e… | manter Producer 1231.9 como melhor bot.… |
| 2026-06-06 | E4 | E4 ATIVADO + baseline da política aprendida vs Producer | E1/E2/C1/E3 esgotados →… → treino roda no binding … | manter Producer 1231.9 como melhor bot.… |
| 2026-06-05 | Parity | Parity probe oficial↔Rust implementado | `scripts/parity_probe.p… → backend `reset_from_sta… | aceitar 5b como sonda implementada; abr… |
| 2026-06-05 | OEP | OEP 3a: gate de promoção contra Producer formalizado | regra estava documentad… → smoke 16 seeds/32 jogos… | aceitar infraestrutura; 3a ainda exige … |
| 2026-06-05 | OEP | OEP 1b: profile de cauda por decisão | profiler só reportava m… → 500 steps/1 seed: mean_… | manter 1b aberto; próxima alavanca cont… |
| 2026-06-05 | F2 | F2: PPO/self-play deferido ate OEP resolver Producer | infra PPO batched e che… → docs atualizados: PPO f… | aceitar fechamento F2 sem treino; mante… |
| 2026-06-05 | OEP | OEP 2b: sonda de fidelidade dos modelos de oponente | 2b tinha tentativas por… → 1 seed/128 steps/256 am… | manter 2b aberto; proximo modelo barato… |
| 2026-06-05 | OEP | OEP 2b: sonda de exemplos com Producer isolado por jogador | a primeira sonda chamav… → com runtimes isolados, … | corrigir a interpretacao anterior; prox… |
| 2026-06-05 | OEP | OEP 1b-iii diagnostico real: trace via arquivo/env default vs max_sou… | diagnosticos com runtim… → trace serial real: defa… | manter max_sources=5 como candidato dia… |
| 2026-06-04 |  | baseline decisor + âncora Kaggle | régua antiga 16 seeds s… → vs old=0.58333, margin=… | manter como primeira âncora local↔LB, m… |
| 2026-06-04 | G1 | G1 Producer corrigido vs Producer bugado | bugado mirror: win=0.50… → corrigido vs bugado: wi… | G1 seguro; adotar Producer corrigido co… |
| 2026-06-03 |  | régua honesta da heurística atual | baseline anterior não r… → vs old=0.46875, greedy=… | usar como bloqueio antes de novos commi… |

## ❌ Rejeitados (54) — becos sem saída (não repetir)

| data | tag | ideia | resultado | decisão |
|---|---|---|---|---|
| 2026-06-07 | P3 | P3: extensão +500k (620k total) entity PPO REGRIDE | escala cega de PPO REGRIDE: treino "sau… | melhor checkpoint = ppo_entity 120k… |
| 2026-06-07 | P3 | P3: campanha limitada entity-init PPO (120k timesteps) — VA… | imitação->PPO com currículo forte FUNCI… | P3 validado em corrida curta; campa… |
| 2026-06-07 | T4 | T4 cont.: entity export (Python puro) + online BC | accuracy offline de BC != margin online… | manter entity como arquitetura de i… |
| 2026-06-07 | P2 | P2: behavioral cloning launch-aware (bc_producer/oep/mix) | BC SAIU do colapso (-1.0) e é warm-star… | aceitar P2: gate margin>-1.0 atingi… |
| 2026-06-06 | OEP | OEP E1a tentativa: avaliador ordinal multi-variante do opon… | K=3 e K=5 são pequenos demais para dar … | rejeitar E1a nesta forma como defau… |
| 2026-06-06 | OEP | OEP E3: busca por valor de rollout sobre conjunto de candid… | REGREDIU FORTE: -0.201 << -0.045. Mais … | rejeitar E3 como default (`OEP_ROLL… |
| 2026-06-06 | OEP | OEP E2a/E2b tentativa: réplica reativa 2-ply do Producer | a réplica reativa desinfla o diagnóstic… | rejeitar E2a/E2b como default; mant… |
| 2026-06-06 | OEP | OEP C1a tentativa: plano-memória como candidato alternativo… | gera plano-candidato novo, mas quase nu… | rejeitar C1a como default; manter k… |
| 2026-06-06 | OEP | OEP C1b tentativa: beam do primeiro lance como candidato al… | gera plano-candidato novo, mas o beam r… | rejeitar C1b como default; manter k… |
| 2026-06-05 | OEP | OEP 2b tentativa: resposta adversarial barata | resposta barata corta custo mas cega o … | rejeitar 2b atual; precisa modelo b… |
| 2026-06-05 | OEP | OEP 1b tentativa: deadline por estágios | corte por estágio é legal e configuráve… | rejeitar; `OEP_TIME_BUDGET_MS` remo… |
| 2026-06-05 | OEP | OEP 1b-iii: superfície de knobs e corte 4×4 rejeitado | corte forte reduz custo mas perde quali… | rejeitar 4×4 como default; aceitar … |
| 2026-06-05 | OEP | OEP 2b tentativa: Producer inline em vez de chamadas `agent… | evitar `agent()` completo reduz custo, … | rejeitar como default; manter `OEP_… |
| 2026-06-05 | OEP | OEP 2b tentativa: Producer tensor runtime | chamar `ProducerLiteRuntime.tensor_acti… | rejeitar como default; manter `OEP_… |
| 2026-06-05 | OEP | OEP 1b-iii: cortes 5x5/6x5/5x6 rejeitados | cortar alvo ofensivo para 5 quebra cedo… | rejeitar 5x5/6x5/5x6 como default; … |
| 2026-06-05 | OEP | OEP 2b tentativa: baratear apenas o 2o Producer | manter seed oficial nao basta: atalhos … | rejeitar `producer_inline`/`produce… |
| 2026-06-05 | OEP | OEP 2b tentativa: filtrar top3 do oponente inline/tensor | filtrar volume por ships nao recupera a… | rejeitar top3 como default; manter … |
| 2026-06-05 | OEP | OEP 2b tentativa: tensor com memoria compartilhada seed+opo… | memoria compartilhada importa e recuper… | rejeitar como default; evidencia ap… |
| 2026-06-05 | OEP | OEP 2b tentativa: shared tensor sem shadow por seed sparse | `last_sparse_action_row` nao basta para… | rejeitar sem benchmark 4 seeds; pro… |
| 2026-06-05 | OEP | OEP 2b tentativa: sincronizar movement compartilhado sem re… | sincronizar movement por fora nao repro… | rejeitar sem benchmark 4 seeds; con… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: desligar regroup por config | desligar regroup piora qualidade e cust… | rejeitar como default; manter `OEP_… |
| 2026-06-05 | OEP | OEP 2b tentativa: remover resposta adversarial do lookahead | remover oponente e barato no smoke pequ… | rejeitar como default; 2b precisa d… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: fração única full-send | full-send unico remove a opcao de half-… | rejeitar sem 16 seeds; manter fract… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: reduzir apenas max_waves_per_turn par… | reduzir W isoladamente nao corta o cust… | rejeitar sem 16 seeds; manter `OEP_… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: reduzir apenas defensive_targets para… | cortar defesa isoladamente nao reduz cu… | rejeitar sem 16 seeds; manter `OEP_… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: max_sources=5 apenas após step 100 | corte tardio preserva o smoke pequeno e… | rejeitar como default; manter os en… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: max_sources=5 apenas após step 200 | atrasar para step 200 nao resolve custo… | rejeitar sem 16 seeds; cortes tardi… |
| 2026-06-05 | OEP | OEP 1b-iii tentativa: selecao conservadora por min_advantage | escolher quase sempre Producer foi fals… | rejeitar como default; precisa melh… |
| 2026-06-05 | OEP | OEP tentativa: Producer seed/opponent com runtimes independ… | separar runtimes Producer dentro do OEP… | rejeitar; nao alterar default OEP p… |
| 2026-06-04 |  | hammer força alvo do planeta líder quando coordenação forte | forçar alvo do líder mesmo com limiar a… | rejeitar |
| 2026-06-04 |  | pressure por aggression_ratio material sem perfil de alvo | pressão por ratio sem saber alvo melhor… | rejeitar |
| 2026-06-04 |  | avaliação local 2-ply por estado após melhor recaptura inim… | penalidade 2-ply explícita fica mais le… | rejeitar |
| 2026-06-04 |  | hammer target com bônus para alvo avaliado pelo planeta líd… | viés explícito do líder piora self-play… | rejeitar |
| 2026-06-04 |  | reserva 7 na abertura 0-10 quando há neutro seguro | reduzir reserva mesmo pouco melhora gre… | rejeitar |
| 2026-06-04 |  | abertura 0-10 estrita: não capturar neutro inseguro mesmo s… | regra fiel ao diagnóstico trava expansã… | rejeitar |
| 2026-06-04 |  | reserva 5 restrita à abertura adaptativa 15-80 | liberar reserva em ADAPTIVE_OPENING pio… | rejeitar |
| 2026-06-04 |  | abertura 0-10 estrita e reserva 5 condicionada a neutro seg… | abertura estrita reduz agressividade co… | rejeitar por enquanto |
| 2026-06-04 |  | pressão por ratio 2p com to_me>=0.70/profile>=20 e 0.75/pro… | pressão por ratio ainda antecipa defesa… | rejeitar |
| 2026-06-04 |  | reutilizar alvo de hammer sempre que já usado | reaproveitamento amplo causa overcommit… | rejeitar |
| 2026-06-04 |  | reserva 5/6 na abertura 0-10 quando há neutro seguro | liberar reserva cedo melhora bots simpl… | rejeitar |
| 2026-06-04 |  | filtro de neutros seguros na abertura adaptativa 15-80 | filtro adaptativo reduz demais opções e… | rejeitar |
| 2026-06-04 |  | pressão por ratio 2p com to_me>=0.58 | detecta pressão cedo demais e derruba r… | rejeitar |
| 2026-06-04 |  | reserva 15 mid-game ampla 2p sem expansão/pressão/TOTAL_WAR | reserva ampla prende produção e piora s… | rejeitar |
| 2026-06-04 |  | foco orbital 10-13 quando orbital seguro existe | foco orbital cria regressão rápida cont… | rejeitar |
| 2026-06-04 |  | throttle reduzido em fases urgentes/oportunistas | melhora âncoras simples mas não recuper… | rejeitar |
| 2026-06-04 |  | penalidade depth-2 suave para neutro recapturável | penalidade atrapalha corrida de abertur… | rejeitar |
| 2026-06-04 | Producer | Producer-wave: aumentar seletivamente tamanho da frota por … | inflar wave size sem replanejar orçamen… | rejeitar e reverter; implementar Pr… |
| 2026-06-04 |  | beam search 2p sobre candidatos atuais com avaliação global… | gastar compute com avaliação global fra… | rejeitar e reverter; próxima busca … |
| 2026-06-04 |  | planner 2p simplificado por produção projetada + orçamento … | objetivo simplificado não reproduz Prod… | rejeitar e reverter; precisa planne… |
| 2026-06-04 | OEP | OEP-1ply experimental sobre Producer: resposta única + fraç… | scaffold roda legalmente e adiciona res… | não promover; próxima etapa é reduz… |
| 2026-06-04 | OEP | OEP-1ply: seleção por torneio de fitness, sem fallbacks sil… | remove `except Exception`, remove seleç… | manter como scaffold honesto, não p… |
| 2026-06-04 |  | refactor de layout dos bots: `orbit_lite` compartilhado, `b… | organização apenas: Producer é fixture … | aceitar como infraestrutura; não pr… |
| 2026-06-03 |  | abertura 0-10 estrita: neutro só se for seguro | fiel ao diagnóstico, mas perde produção… | rejeitar |
| 2026-06-03 |  | reserva 5 apenas na abertura/adaptativa 2p sem ameaça | regressão rápida contra rush; reserva m… | rejeitar |

## ⏳ Pendentes (3) — ainda não feito

| data | tag | ideia |
|---|---|---|
| 2026-06-04 |  | fórum Kaggle como alvo externo |
| 2026-05-31 |  | melhorar decisão 4p quando anti_meta+defensive aparecem juntos |
| 2026-05-31 |  | reduzir perdas contra rush |

## Por família (tag)

| tag | total | ✅ | ❌ | ⏳ |
|---|---:|---:|---:|---:|
| OEP | 41 | 5 | 27 | 0 |
| P3 | 4 | 0 | 2 | 0 |
| E4 | 2 | 2 | 0 | 0 |
| P5 | 2 | 0 | 0 | 0 |
| Parity | 2 | 1 | 0 | 0 |
| Producer | 2 | 0 | 1 | 0 |
| T4 | 2 | 1 | 1 | 0 |
| F2 | 1 | 1 | 0 | 0 |
| F3 | 1 | 0 | 0 | 0 |
| F4 | 1 | 0 | 0 | 0 |
| Fase | 1 | 0 | 0 | 0 |
| G1 | 1 | 1 | 0 | 0 |
| G2 | 1 | 0 | 0 | 0 |
| P0 | 1 | 0 | 0 | 0 |
| P1 | 1 | 0 | 0 | 0 |
| P2 | 1 | 0 | 1 | 0 |
| PPO | 1 | 0 | 0 | 0 |
| Paralelizar | 1 | 1 | 0 | 0 |
| Thread | 1 | 0 | 0 | 0 |
| UNLOCK | 1 | 0 | 0 | 0 |

