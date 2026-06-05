# todo.md — Orbit Wars Lab

Foco atual: **gerar candidatos que batam o Producer** (régua real). Micro-tuning está
saturado (old=0.656 / greedy=0.937 / rush=0.937 em todos os experimentos 73–99) — parado.
Tudo abaixo serve à tese: tornar a versão ampla do OEP, que já vence (win=0.5625,
margin=+0.125), **legal sob orçamento de tempo sem fallback silencioso**.

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
      OBS 2026-06-05: o incumbente como terceiro candidato foi removido da seleção corrente
      para cumprir o contrato novo: torneio estrito OEP vs Producer. O histórico 1a fica como
      experimento registrado, não como comportamento atual.
- [ ] 1b. Dimensionar o custo do OEP sem fallback temporal. A busca atual é um passe vetorizado
      S×T×G em `_build_fraction_candidates`, não uma EA iterativa interrompível; portanto o
      conserto correto é reduzir/parametrizar C e falhar no gate se o pior caso estourar o teto.
      DESIGN: o plano do Producer (`producer_entries`) é candidato de seleção por fitness, não
      fallback. O runtime não deve pular `plan_oep_waves`, não deve retornar Producer por timeout
      e não deve usar `OEP_TIME_BUDGET_MS`.
  - [x] 1b-i. Remover o deadline/fallback temporal do runtime OEP.
        RESULTADO: `OEP_TIME_BUDGET_MS`, `_deadline*` e `should_stop` removidos em 2026-06-05.
  - [x] 1b-ii. Tornar a seleção um torneio único de plano inteiro:
        `chosen = oep_entries if fit(oep) > fit(producer)+min_advantage else producer_entries`.
        RESULTADO: 16 seeds/32 jogos vs Producer, win=0.50000, margin=0.00000,
        mean_ms=287.48, crash=0.0, timeout=0.0, invalid=0.0.
  - [ ] 1b-iii. Reduzir C por configuração (shortlist/frações/waves/regroup) até o pior caso
        medido ficar sob o teto com margem, sem alterar a regra de seleção por fitness.
  - [ ] 1b-iv. Se um dia virar EA iterativa real, reintroduzir anytime como output legítimo
        best-so-far da busca, nunca como fallback por exceção/timeout.
  - [ ] verificar: config ampla (a que deu win=0.5625) rodar vs Producer 96 seeds →
        timeout_rate=0, crash=0, invalid=0, mean_ms dentro do orçamento, margin ≥ 0 preservada
  - [ ] verificar: o corte NUNCA produz ação ilegal — sob deadline apertado força fallback
        para `producer_entries`; rodar com budget artificialmente baixo (ex. 20ms) e conferir
        crash=0, invalid=0 (degrada para Producer, não quebra)
  - [ ] NOTA (acoplamento): 1b corta a VARIÂNCIA (o episódio ocasional de 251ms), não a média.
        Se um único batch S×T×G já estoura o budget, 1b não salva — aí o gargalo é a Thread 2
        (modelo mais barato) ou shortlist menor (1c). Medir a distribuição de mean_ms, não só
        a média, para saber se 1b basta ou se Thread 2 é obrigatória.
        TENTATIVA 2026-06-05: deadline por estágios é legal (`OEP_TIME_BUDGET_MS=20`
        teve crash/timeout/invalid=0), mas `150ms` em 16 seeds regrediu margin
        0.00000→-0.18750; rejeitado e removido para cumprir a regra sem fallback.
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

- [x] 2a. Profilar o step do OEP com `perf_counter` por ESTÁGIO (não "logar tudo"): cronometrar
      separadamente (i) as 2 chamadas do Producer (L812 seed, L822 opponent), (ii)
      `ensure_planet_movement` (L801), (iii) `build_distance_cache` (L807), (iv)
      `_build_fraction_candidates`/`plan_oep_waves` (L848), (v) os 2 `_plan_fitness`.
  - [x] verificar: breakdown em ms por estágio impresso, somando ~o mean_ms total; hot path
        identificado por número (não por palpite)
        RESULTADO: `artifacts/gates/oep/profile_1seed_128steps.json`, 223 decisões,
        mean_decision_ms=35.78, coverage=99.42%; hot path: Producer seed 26.71%,
        Producer opponent 24.79%, `plan_oep_waves` 19.43%, fitnesses somadas ~14.08%.
- [ ] 2b. (suspeito #1) Eliminar o 2º Producer completo: o OEP paga DOIS rollouts Producer por
      step (seed L812 + oponente L822) antes de buscar. O seed é necessário (é o genoma base);
      o OPONENTE pode usar um modelo barato. Trocar `opponent_policy` por uma resposta heurística
      leve (ex. reusar `cheap_enemy_pressure`, L717) e reservar o Producer completo só pro seed.
  - [ ] verificar: mean_ms cai materialmente (profile 2a antes/depois) E margin vs Producer
        ≥16 seeds não regride (o modelo de oponente barato não pode cegar o lookahead)
        TENTATIVA 2026-06-05: `opponent_response_mode=cheap` reduziu 16-seed mean_ms
        289.99→205.53, mas regrediu margin 0.00000→-0.12500; rejeitado como default.
- [ ] 2c. (suspeito #2) `_fill_garrison_trajectory` (L899): loop Python `for k in range(...)`
      (L1070) sobre o horizonte. Confirmar se a projeção é reconstruída do zero a cada step ou
      se o cache incremental (`_roll_garrison_projection` L1150, `_mark_garrison_dirty` L1210)
      está sendo de fato aproveitado; vetorizar a recorrência se o profile apontar custo aqui.
  - [ ] verificar: custo da projeção cai (microbench) E gates L1–L5a verdes
        (tests/test_movement_fidelity.py sem regressão — fidelidade é inegociável)
- [x] NOTA (prioridade): se 2a confirmar que os 2 Producers dominam o tempo, 2b pode ser o item
      de MAIOR alavancagem do plano inteiro — derruba a média o bastante para 1b/1c caberem no
      orçamento. Nesse caso, fazer 2a+2b ANTES de 1c (talvez até antes de 1a).
      DECISÃO: confirmado; as duas chamadas Producer somam ~51.5% do tempo perfilado.

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

## Thread 5 — Correções de higiene encontradas na auditoria (2026-06-05)

- [x] 5a. Fallback silencioso no agente submetido viola a regra "Forbid silent fallbacks"
      (commit b94819c). Em `python/submission/submission_template.py:1442`, `agent()` envolve
      tudo em `except Exception: return fallback_greedy(obs)`; em `:1429` o próprio
      `fallback_greedy` engole erro e retorna `[]`. Se a política degradar todo step, o agente
      vira greedy sem sinal — e isso corrompe a correlação local↔leaderboard. NÃO remover a
      rede de segurança (crash no Kaggle = 0); torná-la BARULHENTA.
  - [x] Instrumentar: contar `fallback_rate` / `illegal_move_rate` por episódio na avaliação local.
        RESULTADO: `SUBMISSION_STATS` no template/export neural e agregação em
        `scripts/benchmark_submission.py` como `fallback_rate`, `policy_illegal_move_rate`
        e `fallback_error_rate`.
  - [x] Falhar o gate de submissão se `fallback_rate > 0` em seeds técnicas.
        RESULTADO: `scripts/gate_check.py` adiciona `gate_1b_no_silent_fallbacks`.
  - [x] verificar: rodar 16+ seeds vs Producer e confirmar `fallback_rate == 0.0` na régua atual;
        se >0, o número aparece no relatório do gate em vez de passar silencioso.
        RESULTADO: `artifacts/gates/fallback_metrics/template_vs_producer_16seed.json`,
        32 jogos vs Producer, `fallback_rate=0.0`, `policy_illegal_move_rate=0.0`,
        `fallback_error_rate=0.0`, crash/timeout/invalid=0.0.
- [x] 5b. `scripts/parity_probe.py:15` é um stub (`print("TODO: ...")`) — a paridade
      Rust↔motor oficial NÃO é checada. Como toda a régua de backtest depende da fidelidade do
      motor (modelo de 3 camadas), implementar a sonda real OU registrar decisão explícita de
      deferir com justificativa.
  - [x] verificar: probe instancia `kaggle_environments.make('orbit_wars')`, dá step do Rust a
        partir do snapshot oficial e compara planets/fleets/comets dentro de tolerância — ou
        EXPERIMENTS.md registra por que fica deferido.
        RESULTADO: `scripts/parity_probe.py` agora carrega o snapshot oficial via
        `reset_from_states`, stepa os dois motores com ações vazias e compara estado completo.
        A sonda é real e falha hoje em `seed=0 step=1 planet=12 x`:
        oficial não rotaciona o planeta, Rust rotaciona. Ver 5d.
- [x] 5d. Corrigir divergência Rust↔motor oficial revelada pelo parity probe.
      Primeiro mismatch reproduzível: `rtk .venv/bin/python -m scripts.parity_probe --episodes 1
      --steps 8 --disable-comets` retorna `[PARITY-PLANETS] seed=0 step=1 id=12 field=x
      official=73.44162950284988 rust=72.32234662124957`. Antes de confiar no backtest
      Rust como régua absoluta, decidir se a rotação orbital do Rust está errada, se o oficial
      expõe posições pré-rotação, ou se a comparação precisa alinhar subfase.
      RESULTADO: Rust alinhado à semântica oficial de rotação orbital (`step=1` ainda
      mostra fase inicial; fase orbital pública é `max(step-1, 0)`) e à expiração oficial
      dos cometas (removidos antes de avançar para fora do último ponto válido). O
      `PlanetMovement` usa a mesma fase orbital.
  - [x] verificar: parity probe passa em janela sem spawn futuro e com cometas ativos.
        RESULTADO: `rtk .venv/bin/python -m scripts.parity_probe --episodes 2 --steps 49`
        passa (`checked_steps=98`), cobrindo a janela inicial antes do spawn oculto do
        step 50; `rtk .venv/bin/python -m scripts.parity_probe --episodes 2 --start-step
        55 --steps 64` passa (`checked_steps=128`), cobrindo cometas já presentes até
        antes do próximo spawn. A sonda agora falha explicitamente se a janela cruza um
        spawn futuro (`50/150/250/350/450`), porque o motor oficial usa seed interna não
        recuperável pelo snapshot visível; isso preserva o invariante do OEP de não
        pontuar além da próxima fronteira de spawn.
- [x] 5c. Blindar a fronteira Rust/Python (invariante D11 em DECISIONS.md): teste de arquitetura
      `test_no_native_in_submission` que falha se `artifacts/submission.py` (e o tarball de
      submissão) importarem `orbit_wars_core` / `orbit_wars_py`. Impede regressão silenciosa que
      quebraria a submissão no Kaggle.
  - [x] verificar: o teste passa hoje (nenhum import nativo) e falha de propósito ao injetar um
        `import orbit_wars_py` no template de submissão.
        RESULTADO: `test_no_native_imports_in_submission_artifacts_and_producer_tarball`
        checa o template renderizado, `artifacts/submission.py` quando existe e o tarball
        Producer gerado em `tmp_path`; `test_native_runtime_import_detector_rejects_rust_boundary_crossing`
        prova que `import orbit_wars_py`/`from orbit_wars_rs import ...` seriam detectados.
        `rtk .venv/bin/python -m pytest -q tests/test_submission_pipeline.py`: 41 passed.

## Parado / não fazer agora

- [x] Micro-tuning de heurística (reservas, aberturas, hammer, pressão por ratio) — saturado
      contra bots locais; não separa mais candidatos. Ver experimentos 73–99.
- [x] Aceleração por GPU (avaliado 2026-06-05) — NÃO fazer. (1) Agente submetido roda CPU-only
      com `actTimeout=1s` (test_official_spec.py:26) e problema minúsculo (24–52 planetas,
      A=2); planner já tensorizado roda 177–290ms/step na CPU — GPU seria mais lenta
      (small-batch underutilization). (2) Treino: perfil `lab-gpu` já existe, mas a MLP é
      pequena; gargalo provável é o rollout em CPU, não a rede — perfilar antes. (3) Sim
      batcheado em GPU exigiria reescrever o motor Rust: esforço enorme p/ jogo 2p. Veredito do
      próprio EXPERIMENTS.md (l.61): gargalo é modelo de estado/valor, não orçamento de compute.
      Lit. forte: Isaac Gym (arXiv:2108.10470), EnvPool (arXiv:2206.10558) — ganho de GPU exige
      milhares de envs paralelos; o ROI aqui é paralelismo de ambiente em CPU.
