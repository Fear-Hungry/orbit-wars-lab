# todo.md — Orbit Wars Lab

Foco atual: **gerar candidatos que batam o Producer** (régua real). Micro-tuning está
saturado (old=0.656 / greedy=0.937 / rush=0.937 em todos os experimentos 73–99) — parado.
Tudo abaixo serve à tese: tornar a versão ampla do OEP, que já vence (win=0.5625,
margin=+0.125), **legal sob orçamento de tempo**.

## Thread 1 — Tornar o OEP forte legal sob orçamento (PRIORIDADE)

- [x] 1a. Adicionar seeding por incumbente ao `bots/oep/planner.py` (versão fiel do shift
      buffer; lit. do conceito forte: Gaina et al. 2017, arXiv:1704.06942 — adaptação ao
      planner guloso é inferência, não 1:1 do paper).
      CONTEXTO: o genoma é um conjunto de waves POR TURNO (gulosa, `plan_oep_waves` L666;
      `_greedy_select`), não uma sequência evoluída por população; lançamentos são one-shot
      (já debitados em `apply_private_planned_launches` L710). Logo persiste-se a INTENÇÃO
      (lanes origem→alvo vencedoras), não a linha de lançamento literal.
  - [x] 1a-i. DECIDIR o mecanismo (sub-item de decisão, embasamento da adaptação é fraco):
        (A) incumbente-como-candidato — recomputa a wave-set das lanes do turno anterior e
            exige que a busca nova só substitua se `fit(novo) > fit(incumbente)`; mais SEGURO,
            estabiliza decisões e evita flip-flop, mas não corta custo sozinho.
        (B) seeding-de-shortlist — usa as lanes do turno anterior para PODAR a enumeração
            (explorar só um delta em volta), permitindo rodar shortlist menor com qualidade
            da config ampla; é o que de fato corta custo, porém mais arriscado.
        Começar por (A); só ir a (B) se 1b sozinho não fechar o orçamento.
        DECISÃO: implementado (A) incumbente-como-candidato; (B) fica deferido.
  - [x] 1a-ii. Persistir o estado: adicionar campo `last_lanes` (ou `last_plan_entries:
        LaunchEntries | None`) em `OEPLiteMemory` (classe L563) e zerá-lo em
        `OEPLiteMemory.reset()` (L569). Reset por-episódio já acontece no `step==0` (L617-618).
  - [x] 1a-iii. Gravar o incumbente: após escolher `chosen` (L697-701), extrair as lanes
        (source,target,fração) e guardar em `mem.last_lanes` junto de `mem.last_sparse_action_row`
        (L717), descontando o que já foi lançado (one-shot).
  - [x] 1a-iv. Consumir o incumbente: no início de `tensor_action`, reprojetar as lanes de
        `mem.last_lanes` sobre o estado atual (origens/alvos podem ter mudado de dono → filtrar
        lanes inválidas) e injetar como candidato/semente na chamada de `plan_oep_waves` (L666).
  - [x] verificar: com seeding ligado e o MESMO orçamento por step que hoje dá win=0.375,
        rodar vs Producer 16+ seeds → margin média ≥ 0.0 e timeout_rate = 0.0
        RESULTADO: 16 seeds/32 jogos vs Producer, margin=0.00000, timeout_rate=0.0.
  - [x] verificar: sem regressão de legalidade — crash=0, invalid_action_rate=0 (lanes
        inválidas filtradas corretamente após mudança de dono de planeta)
        RESULTADO: 16 seeds/32 jogos vs Producer, crash=0.0, invalid_action_rate=0.0.
- [ ] 1b. Adicionar corte anytime por ESTÁGIOS com fallback legal (lit. do conceito forte:
      Gaina et al. 2017, arXiv:1704.07075 — adaptação: a busca aqui é vetorizada S×T×G em
      `_build_fraction_candidates` L501, NÃO um loop interrompível; logo o corte é por
      fronteira de estágio, não por candidato).
      DESIGN: o plano do Producer (`producer_entries`, L640) já é computado primeiro e é
      legal/barato — ele é o best-so-far/fallback. O refino OEP é o custo caro a ser gated.
  - [ ] 1b-i. Marcar deadline no início de `tensor_action` (L613): `deadline = perf_counter()
        + budget`. Budget ~150ms (folga sobre actTimeout=1000ms; ler do obs se exposto, senão
        constante). Usar `time.perf_counter` (monotônico).
  - [ ] 1b-ii. Gatear o rollout do oponente (L648-655) e `plan_oep_waves` (L666): se
        `perf_counter() > deadline` antes de cada um, PULAR e manter `chosen = producer_entries`.
  - [ ] 1b-iii. Tornar a seleção de waves incremental e interrompível: dentro de
        `plan_oep_waves`/`_greedy_select`, checar o deadline ANTES de cada wave extra
        (`max_waves_per_turn`, W) — parar de adicionar waves quando o tempo acabar e usar o
        conjunto já selecionado (que é legal por construção). Esse é o checkpoint mais fino
        disponível sem quebrar o batch S×T×G.
  - [ ] 1b-iv. Gatear o estágio de regroup (`enable_regroup`) atrás do mesmo deadline — é
        refino opcional, primeiro a ser cortado sob pressão de tempo.
  - [ ] verificar: config ampla (a que deu win=0.5625) rodar vs Producer 96 seeds →
        timeout_rate=0, crash=0, invalid=0, mean_ms dentro do orçamento, margin ≥ 0 preservada
  - [ ] verificar: o corte NUNCA produz ação ilegal — sob deadline apertado força fallback
        para `producer_entries`; rodar com budget artificialmente baixo (ex. 20ms) e conferir
        crash=0, invalid=0 (degrada para Producer, não quebra)
  - [ ] NOTA (acoplamento): 1b corta a VARIÂNCIA (o episódio ocasional de 251ms), não a média.
        Se um único batch S×T×G já estoura o budget, 1b não salva — aí o gargalo é a Thread 2
        (modelo mais barato) ou shortlist menor (1c). Medir a distribuição de mean_ms, não só
        a média, para saber se 1b basta ou se Thread 2 é obrigatória.
- [ ] 1c. Trocar o candidato único guloso por BUSCA sobre população de genomas multi-lançamento
      (OEP de verdade). (Lit. forte: Justesen, Mahlmann & Togelius 2016)
      ACHADO: hoje há UM só candidato OEP, montado guloso em `_greedy_select` (L697), comparado
      como conjunto contra UM candidato Producer via `_plan_fitness`. Nunca se busca conjuntos
      alternativos — é o baseline guloso que o OEP supera avaliando o fitness CONJUNTO.
  - [ ] 1c-i. Definir o genoma = tupla de `LaneIntent` (source,target,fraction) — o tipo já
        existe e já há conversor `_entries_from_lane_intents` (L398) e extrator
        `_lane_intents_from_entries` (L358). Reusar; não criar representação nova.
  - [ ] 1c-ii. Operadores de mutação sobre a tupla de lanes: add-lane (de um candidato do
        shortlist S×T já calculado em `_build_fraction_candidates`), drop-lane, swap-target,
        e perturbar-fraction (±1 passo do conjunto `fractions`). Crossover opcional (1-ponto
        entre dois genomas) — começar só com mutação.
  - [ ] 1c-iii. Loop de busca: semear a população com (a) o resultado guloso atual
        (garante OEP ≥ guloso por construção), (b) `producer_entries`, (c) o incumbente do 1a;
        avaliar cada genoma com o `_plan_fitness` CONJUNTO já existente vs `opponent_launch_set`;
        manter o argmax. Tamanho de população e nº de gerações GATEADOS pelo deadline do 1b.
  - [ ] 1c-iv. Garantir legalidade de cada genoma reusando o reparo já existente em
        `_entries_from_lane_intents` (filtra dono/alive, repara orçamento de origem L478-489).
  - [ ] verificar: genoma multi-lançamento (busca) vs candidato guloso atual head-to-head →
        margin_delta > 0 com paired_p significativo em scripts.compare_benchmark_significance
  - [ ] verificar: vs Producer ≥16 seeds → margin ≥ versão sem busca, crash/timeout/invalid=0
  - [ ] NOTA (custo): a busca multiplica as avaliações de fitness por N (pop×gerações) — acopla
        DIRETO com 1b (deadline) e Thread 2 (modelo barato). Sem 1b, 1c estoura o orçamento na
        hora. Fazer 1c SÓ depois de 1b estar no lugar.

## Thread 2 — Baratear o forward model (habilita 1b/1c)

Embasamento de literatura FRACO/NA — isto é engenharia de perf. Regra que manda: diagnose.md
fase perf = MEDIR antes de consertar. Os candidatos abaixo são suspeitas a confirmar pelo profile.

- [ ] 2a. Profilar o step do OEP com `perf_counter` por ESTÁGIO (não "logar tudo"): cronometrar
      separadamente (i) as 2 chamadas do Producer (L812 seed, L822 opponent), (ii)
      `ensure_planet_movement` (L801), (iii) `build_distance_cache` (L807), (iv)
      `_build_fraction_candidates`/`plan_oep_waves` (L848), (v) os 2 `_plan_fitness`.
  - [ ] verificar: breakdown em ms por estágio impresso, somando ~o mean_ms total; hot path
        identificado por número (não por palpite)
- [ ] 2b. (suspeito #1) Eliminar o 2º Producer completo: o OEP paga DOIS rollouts Producer por
      step (seed L812 + oponente L822) antes de buscar. O seed é necessário (é o genoma base);
      o OPONENTE pode usar um modelo barato. Trocar `opponent_policy` por uma resposta heurística
      leve (ex. reusar `cheap_enemy_pressure`, L717) e reservar o Producer completo só pro seed.
  - [ ] verificar: mean_ms cai materialmente (profile 2a antes/depois) E margin vs Producer
        ≥16 seeds não regride (o modelo de oponente barato não pode cegar o lookahead)
- [ ] 2c. (suspeito #2) `_fill_garrison_trajectory` (L899): loop Python `for k in range(...)`
      (L1070) sobre o horizonte. Confirmar se a projeção é reconstruída do zero a cada step ou
      se o cache incremental (`_roll_garrison_projection` L1150, `_mark_garrison_dirty` L1210)
      está sendo de fato aproveitado; vetorizar a recorrência se o profile apontar custo aqui.
  - [ ] verificar: custo da projeção cai (microbench) E gates L1–L5a verdes
        (tests/test_movement_fidelity.py sem regressão — fidelidade é inegociável)
- [ ] NOTA (prioridade): se 2a confirmar que os 2 Producers dominam o tempo, 2b pode ser o item
      de MAIOR alavancagem do plano inteiro — derruba a média o bastante para 1b/1c caberem no
      orçamento. Nesse caso, fazer 2a+2b ANTES de 1c (talvez até antes de 1a).

## Thread 3 — Régua de promoção honesta (parar de decidir por greedy/rush)

- [ ] 3a. Formalizar o gate de promoção do OEP: margin ≥ 0 vs Producer em ≥96 seeds,
      timeout/crash/invalid = 0, via scripts.compare_benchmark_significance.
  - [ ] verificar: gate documentado (EXPERIMENTS.md/DECISIONS.md) e 1 run de promoção produz
        veredito paired ≥ baseline G2
- [ ] 3b. (opcional) Adicionar um 2º oponente-régua (variante do Producer ou notebook forte do
      fórum) para não overfitar um único oponente.
  - [ ] verificar: candidato promovido não regride contra o 2º oponente

## Fases anteriores — pendências não terminadas

- [ ] F1. OEP em 4p — o lookahead adversarial está DESLIGADO em 4p, além de nunca ter sido medido.
      ACHADO: `_opponent_id` (L140-143) retorna `None` quando `player_count != 2`, então
      `opponent_entries = None` e o OEP degenera para pontuar só o próprio estado (vira o
      guloso estilo Producer, sem 1-ply). `CONFIG_4P` (L63) só ajusta shortlists/horizonte.
      Toda validação foi `--skip-4p`. (D9: 4p é dinâmica diferente — overextension vira
      vulnerabilidade para terceiros.)
  - [ ] F1-i. DECIDIR o modelo de oponente em 4p (embasamento da escolha específica é fraco;
        o princípio "modelar oponente, não só estado próprio" é forte — Justesen 2016):
        (A) modelar 1 oponente mais relevante (líder por produção OU mais próximo por ETA) como
            a resposta de 1-ply — mais barato, fiel ao espírito do OEP 2p; RECOMENDADO começar aqui.
        (B) modelar os 3 oponentes (resposta agregada) — mais fiel mas ~3× o custo do rollout
            adversarial; só se (A) não bastar.
        CUIDADO 4p: best-response contra UM oponente pode induzir overextension que um TERCEIRO
        pune — por isso (A) deve favorecer cautela (CONFIG_4P já reduz agressão; manter).
  - [ ] F1-ii. Estender `_opponent_id` (ou criar `_opponent_ids`) para devolver alvo(s) em 4p
        conforme a decisão F1-i; ligar o ramo de `opponent_entries`/`opponent_launch_set`
        (hoje gated por `opp_id is not None`, L820+) para o caso 4p.
  - [ ] F1-iii. Criar um gate 4p no benchmark (rodar SEM `--skip-4p`) e registrar baseline:
        OEP-atual-4p (oponente off) vs Producer, para medir o ganho do lookahead 4p contra ele.
  - [ ] verificar: em 4p, `opponent_entries` deixa de ser None (o lookahead realmente engaja)
  - [ ] verificar: OEP-com-oponente-4p vs Producer ≥16 seeds → win 4p ≥ baseline OEP-4p-off,
        crash/timeout/invalid=0, sem regressão de timeout (CONFIG_4P já é mais barato)
- [ ] F2. PPO / self-play (D4) — infra de rollout batched foi construída (commits a887216,
      814eb04, 30f6d22) mas NENHUM run de treino produziu candidato. É o caminho pesado de
      longo prazo. Decisão docs/SUBMISSION.md: usar Producer/heurístico como oponente de PPO.
      DECIDIR: vale ativar agora ou só depois de o OEP bater o Producer? (Recomendo depois —
      OEP já tem sinal positivo; PPO é aposta de maior custo/prazo.)
  - [ ] verificar (se ativado): 1 run de PPO contra Producer produz checkpoint com win vs
        Producer ≥ baseline OEP, sem crash/timeout
- [ ] F3. Minerar notebooks públicos fortes do fórum (exp. 105): 704095 (benchmark 109
      agentes), 704113 (Producer ~1200). Extrair ideias de produção projetada / redistribuição
      / lookahead ainda não incorporadas. É inteligência competitiva, não código.
  - [ ] verificar: lista de 2-3 ideias acionáveis extraídas, cada uma com hipótese testável
        registrada em EXPERIMENTS.md
- [ ] F4. Fechar pendências de micro-tuning local (exp. 106-107): "reduzir perdas vs rush" e
      "decisão 4p com anti_meta+defensive juntos". ATENÇÃO: provavelmente SUPERADAS pelo pivô
      (régua vs-old/rush está desqualificada como promotor). NÃO ressuscitar cegamente —
      primeiro DECIDIR se ainda fazem sentido contra a régua Producer.
  - [ ] verificar: decisão registrada (manter parado OU re-escopar contra Producer)

## Thread 4 — Risco deferido (só com repro concreto)

- [ ] 4a. L5b / fidelity de fixtures set_state arbitrárias: validar que o sim interno do
      lookahead não diverge do motor real em estados de borda. Investir SÓ se aparecer um
      caso em que o fitness do planner contradiz o resultado real da partida.
  - [ ] verificar: repro de divergência sim-interno vs motor, ou decisão explícita de manter deferido

## Parado / não fazer agora

- [x] Micro-tuning de heurística (reservas, aberturas, hammer, pressão por ratio) — saturado
      contra bots locais; não separa mais candidatos. Ver experimentos 73–99.
