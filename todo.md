> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `EXPERIMENTS.md` — este arquivo fica enxuto, só com o que atacar e o estado atual.

---

# 🎯 FOCO ATUAL (2026-06-09) — desancorar a recompensa do Producer (Alavanca A)

> Causa-raiz do teto vs Producer, com endereço no código. Embasamento: Ng/Harada/Russell 1999 (PBRS
> invariância só na forma de diferença); Devlin & Kudenko 2012 (dynamic PBRS / Φ = valor do crítico);
> Wu 2024 (Q-shaping). Bug-hunt + DuckDB ficaram PARKED (fim deste arquivo) — esforço pesado demais.

- [x] **Mov.1 — corrigir a FORMA do shaping para diferença de potencial** ✅ 2026-06-09.
  `gym_env.py`: extraído `_state_potential(s)`=Φ; `_base_shaping_reward` agora retorna `γ·Φ(s') − Φ(s)` com
  zeragem terminal (`Φ(terminal)≡0`); `step()` passa `done`; novo param `shaping_gamma`. `train_ppo.py`: path
  batched passa `done=done` (mesmo método reusado → cobre single-env E batched/GPU); `shaping_gamma=training_cfg.gamma`
  nos 2 call-sites (PBRS no mesmo γ do GAE). Embasamento: Ng/Harada/Russell 1999.
  - [x] verificar: teste de telescoping discounted `Σ_t γ^t·F_t == γ^N·Φ(s_N) − Φ(s_0)` — `tests/test_reward_shaping.py` (3 casos) PASS
  - [x] verificar: `test_training_phase0` PASS (15) + lint limpo nos arquivos mudados
- [ ] **Mov.1 — medir o efeito competitivo** (a ablação diagnóstica): rodar mesmo-compute do baseline e ler margem vs Producer a 96 seeds.
  - [ ] verificar: margem **> 0** → recompensa amarrava (seguir Mov.2); margem **~0** → é representação (Alavanca B, Tavakoli 2017)

- [x] **Consertar as 5 falhas de paridade (combate ativo)** ✅ 2026-06-09 (`/diagnose`). Causa: `compute_planet_paths`
  em `step.rs` usava rotação MATRICIAL; o oficial usa POLAR (`r·cos(atan2+ωt)`) — float-diferente (~1e-10), virava
  colisão knife-edge de frota recém-lançada. Fix: replicar o caminho polar exato. Verificado: `test_parity_actions` 4/4
  + `test_movement_fidelity` + suíte **222 passed** (com `.so` fresco) + `cargo test` 18. Detalhe: [[project_simulator_parity]].
- [x] **CRÍTICO — consertar o build do binding Rust** ✅ 2026-06-09. Causa: `uv run` (auto-sync) reinstala
  `orbit-wars-lab` de um wheel STALE em cache e reverte o `.so` fresco → motor VELHO silenciosamente. Fix no `Makefile`:
  novo `UV_RUN = uv run --no-sync` em TODOS os alvos que rodam código; `make build` agora chama `sync-binding`
  (force-copy de `target/release/liborbit_wars_rs.so` → venv); novo `make verify-binding`. Documentado em
  `docs/PARITY.md` "Frescor do binding". Detalhe: [[build_uv_reverts_fresh_so]]. Verificado: `make build` sincroniza,
  `uv run --no-sync` preserva (paridade passa). 
  - [ ] **AÇÃO SUA:** seu eval em background (`scripts.eval_brep_direct` no `c00.pt`) foi iniciado com `uv run` puro →
    está rodando no **motor STALE (pré-fix de paridade)**. Matar e relançar via `make`/`--no-sync` para usar o motor correto.
  - [ ] opcional: `uv cache clean orbit-wars-lab` (remove o wheel stale; exige nenhum processo uv segurando o lock)
- [ ] **Mov.2 — re-fontear Φ para desancorar a guia** (sob compute finito Φ ainda guia para o estilo Producer).
  Escolher: (a) Φ = valor do crítico `V(s)` [recomendado]; (b) recozer `base_shaping_scale` a **0** (hoje para em 0.15,
  `train_ppo.py:79-80`) + apoiar no reward esparso `gym_env.py:119` e tunar GAE-λ; (c) só potencial terminal.
  - [ ] decidir: qual opção (a/b/c) — decisão de arquitetura, embasar antes de codar
- [ ] **Critério decisor (vira `/goal`):** a mesmo compute do baseline, **margem normalizada vs Producer a 96 seeds > 0**
  (`make oep-promotion-gate` / `scripts.benchmark_ppo_submission`); check barato: dist. de ações no início do treino
  deixa de colapsar na do Producer (diagnósticos de `test_map_bias_invariance`).

- [x] **Migração DuckDB dos experimentos** ✅ 2026-06-09 (/goal). `python/lab/experiments.py`: parser do
  `EXPERIMENTS.md` (folda pipes internos de comando) → `experiments.duckdb`; status derivado por seção/decisão
  (feito/rejeitado/logado/**todo**); CLI `import|list|query|stats|add|export|**report**`; `make experiments-{import,report,stats}`.
  117 importados, export round-trip 117=117, **relatório** em `docs/EXPERIMENTS_REPORT.md` (14 aplicados/54 rejeitados/3 pendentes).
  `duckdb` no `pyproject` (extra `lab`); `.duckdb` gitignored. Teste: `tests/test_experiments_db.py` 8 passed.

## 🅿️ PARKED (sessão 2026-06-09, esforço pesado — retomar leve se quiser)
- [ ] **Bug-hunt** (Workflow A): 20 reviews já rodaram (cache em `subagents/workflows/wf_0ad11f4e-7db`). Se retomar:
  triar à mão a partir do cache, sem novo fan-out de verificação.

---

# 🎯 ATACAR AGORA — objetivo top 5

Estado operacional curto:

- **Producer é a melhor submissão operacional atual** (~1200 LB). Congelar como default até existir candidato provado.
- **OEP é útil como adversário/professor** (~1100 LB), mas não como linha de tuning: knobs/overlays OEP já foram atacados e não devem voltar sem hipótese nova.
- **PPO atual ainda é fraco** (`-1.0` vs Producer nos registros antigos). O próximo ataque é imitação + currículo forte, não PPO do zero contra heurísticas fracas.
- **Histórico detalhado fechado vive em `EXPERIMENTS.md`**. Este arquivo deve ficar só com o que ainda vamos atacar.
- **Decidir só com evidência pareada suficiente**: 16 seeds = triagem; 96 seeds decide. Score Kaggle precisa estabilizar antes de conclusão.

## 🧭 SEGUIR AGORA — lição do tópico Kaggle 704741 ("Lessons learned so far")
O tópico do Radek muda a prioridade: PPO competitivo não é "mais reward shaping"; é **sistema de treino**
com features corretas, métricas de PPO, currículo/opponent pool e avaliação por checkpoints. O nosso erro
atual é ficar entre dois mundos: OEP/Producer overlay já saturou, enquanto PPO ainda é baseline fraco
treinado contra heurísticas fracas.

- [ ] **Congelar o default competitivo no Producer até existir candidato provado.** Não gastar mais sessão com
  knobs de seleção OEP (`min_advantage`, horizon, ordinal, reactive, rollout) sem gerador de plano novo ou
  evidência externa forte. OEP local e OEP Kaggle já provaram overfit ao Producer/proxy.
- [x] **Antes de novo run PPO longo, consertar instrumentação:** ✅ 2026-06-07. `explained_variance` no summary/checkpoint
  (`last_explained_variance`) e série por update em `summary["update_series"]` (`approx_kl`, `clipfrac`, entropy,
  `value_loss`, `policy_loss`, explained_variance, timesteps). Script de comparação vs Producer a seeds fixas já existe
  (`scripts.benchmark_ppo_submission`). **Pendente (campanha):** eval periódico DENTRO do loop com win/margin por
  oponente + captura de neutros por checkpoint (parte do loop de campanha do P3, não pré-req de código).
- [~] **Auditar map bias / features assimétricas:** ✅ auditoria FEITA 2026-06-07 (`scripts/audit_map_bias.py`,
  `symmetry.py`, `tests/test_map_bias_invariance.py`); baseline em `artifacts/map_bias/invariance_report.json`.
  **Achado (BC policy, 192 estados):** player-swap gap = **0.0** (encoder já correto em perspectiva) mas
  rotate_180 gap = **2.5–5.6 logits** e reflect_x = **1.7–3.6** → forte viés ESPACIAL de `x/y` absolutos + `planet_id`.
  - [x] **Correção ESPACIAL aplicada (data-augmentation):** ✅ 2026-06-07. `collect_imitation_dataset --augment` adiciona
    cópias rotate_180/reflect_x ao split de train com o MESMO label (índices invariantes). Resultado (bc_producer_aug):
    gap rotate_180 2.5–5.6 → **0.24–1.07**, reflect_x 1.7–3.6 → **0.22–0.89** (~5–10×); launch F1 0.16–0.45 → **0.62**;
    **margem online −0.915 → −0.880** (melhorou), crash/timeout/invalid=0. Critério "reduz sensibilidade sem piorar
    benchmark" ✅. Canonicalização não foi necessária (augmentation já atingiu o alvo). **Target head segue fraco (0.09)
    mesmo com aug → é representação, não simetria → T4.**
- [x] **Trocar o currículo fraco por oponente forte:** ✅ 2026-06-07. `PHASE0_OPPONENTS` inclui Producer e OEP (P0);
  currículo ponderado é expressável por repetição em `--opponents` (ex.: `producer×7,greedy×2,oep` → 70/20/10,
  confirmado em opponent_segments); warm-start por imitação pronto (P2 BC + `--checkpoint-in` carrega o BC). Falta só
  rodar a campanha (abaixo).
- [ ] **Rodar campanha PPO só depois dos três itens acima:** milhões+ de timesteps, checkpoints periódicos,
  tournament local por checkpoint, promoção só com 96 seeds vs Producer + 2º oponente-régua, crash/timeout/invalid=0.
  Se não houver budget para isso, declarar Producer como submissão operacional.
- [ ] **Alternativa não-RL:** abandonar overlay no Producer e construir planner standalone de timeline/sim-value
  estilo fórum (missões safe-capture, rescue, recapture, reinforce, snipe, redistribuição por dominância). Isso é
  build multi-sessão; não é MCTS turno-a-turno sobre os candidatos atuais.

## 🧪 EXPERIMENTOS A FAZER — PPO usando Producer/OEP como professores e adversários
Hipótese: Producer (~1200 LB) e OEP (~1100 LB) são fortes o bastante para servir como **currículo**.
PPO direto do zero contra eles provavelmente perde tudo; a rota testável é **imitação → PPO contra pool forte
→ self-play**. Cada experimento precisa salvar artefato, métrica por checkpoint e comparação pareada.

- [x] **P0. Registrar adversários fortes no treino.** ✅ feito e verificado 2026-06-06.
  - [x] **Implementar registry:** `producer` adicionado a `PHASE0_OPPONENTS` em `python/train/train_ppo.py` reaproveitando `python/agents/registry.py` (sem loader paralelo).
  - [x] **Implementar OEP policy:** `oep_agent(state, player)` em `python/agents/registry.py` converte estado→observação oficial e reusa `bots/oep/agent.py` (lazy import). Registrado em `get_heuristic_policies()` e `HEURISTIC_NAMES`.
  - [x] **Garantir reset determinístico:** ambos resetam memória em `step == 0` por construção — Producer em `_upstream.py:378`, OEP em `planner.py:2257` (`tensor_action`). Coberto por `test_producer_runtime_resets_cached_movement_on_step_zero`. **Subescopo extra:** como ambos usam runtime singleton de módulo, adicionei guard fail-loud em `train_phase0` que recusa batched rollout (`rollout_num_envs > 1`) com `producer`/`oep` (contaminação entre envs interleaved); single-env é seguro (jogos sequenciais). Set `STATEFUL_SINGLETON_OPPONENTS` no registry.
  - [x] **Teste unitário:** `tests/test_strong_opponents_registry.py` chama `producer`/`oep` em estados 2p e 4p (todos os players) e valida `moves_are_legal`.
  - [x] **Smoke de treino:** rodado; terminou ok.
  - [x] **Correto quando:** ✅ smoke sem crash; `summary["opponents"] == ["producer","oep"]`; `opponent_segments == {producer:8, oep:8}`; checkpoint carregável por `scripts.benchmark_ppo_submission` (crash/timeout/invalid_action_rate = 0). Suíte `test_training_phase0 + test_oep_agent + test_strong_opponents_registry` = 46 passed.

- [x] **P1. Dataset de imitação Producer/OEP.** ✅ feito e verificado 2026-06-06 — **revelou 2 blocantes para P2 (ver `P1.5` abaixo).**
  - [x] **Criar coletor:** `scripts/collect_imitation_dataset.py` roda jogos locais com experts e salva `.npz` + `.meta.json` por dataset em `artifacts/imitation/`.
  - [x] **Conteúdo mínimo por exemplo:** `obs`, `player`, `step`, `expert_id`, `raw_moves` (CSR), `action` (4-tupla PPO), `seed`, `split_id`, `quant_error`, `matched_moves`, `num_expert_moves`, `is_no_op`, `is_hard`, `legal`. (Não há `legal_mask` no stack; gravado `legal` por exemplo.)
  - [x] **Resolver ação contínua -> discreta:** `python/orbit_wars_gym/action_inverse.py` — busca em grade + argmin da `move_set_distance` (métrica documentada: angle rad + |Δships|/source_ships, miss/extra penalties). Round-trip exato (erro 0) coberto por teste.
  - [x] **Datasets separados:** `producer_only`, `oep_only`, `producer_oep_mix`, `hard_states` (este só grava estados onde as ações projetadas de Producer e OEP divergem; 2 labels por estado).
  - [x] **Split fixo:** `split_for_seed(seed)` (`seed%5` → train/val/test); por seed, não por linha. Testado.
  - [x] **Relatório:** `artifacts/imitation/dataset_report.json` com n, por-expert, por-step, max_bucket_share por cabeça, no_op_rate, legal_action_rate, hard_rate, quant_error mean/p50/p95/max, content_hash.
  - [x] **Correto quando:** ✅ `--self-check` reproduz content_hash (determinismo); `legal_action_rate == 1.0`; `is_no_op` == (0 expert moves) sem descartes silenciosos; quant_error p95 documentado (~4.2 nos datasets principais). **Ressalva:** `max_bucket_share.source` = 0.84–0.96 (> 0.90 em hard_states) — colapso COM explicação: experts lançam quase sempre do planeta mais forte → `source_rank 0`. Vira input de `P1.5`.

- [x] **P1.5. Cabeça `launch?` binária + move condicional** (decisão do usuário; fork resolvido). ✅ a–f completos e verificados (a–e 2026-06-06, f masking 2026-06-07).
  - Evidência que motivou: Producer/OEP **seguram ~81% dos turnos** (`num_expert_moves==0`), mas o decoder SEMPRE lançava. Solução escolhida: `action = [launch, source, target, frac, offset]`, `P(pass)=P(launch=0)`, `P(move)=P(launch=1)*∏heads`.
  - [x] **(a) policy.py:** `self.launch=Linear(256,2)`; `forward` retorna launch; `get_action_and_value` emite ação (B,5) com `logprob = launch_lp + is_launch*move_lp` e `entropy = launch_entropy + p_launch*move_entropy`.
  - [x] **(b) decoder:** aceita `[launch,s,t,f,o]`, `launch==0 -> []`; compat 4-campos (legado/inverse) marcada.
  - [x] **(c) inverse + dataset:** `InverseResult.launch`/`.action5`; collector salva `action` (N,5); report ganha `launch_rate` e heads em cols 1..4.
  - [x] **(d) PPO:** `gym_env.action_space = MultiDiscrete([2,16,32,4,5])`; rollout/update já genéricos quanto à largura; smoke shape-5 OK (entropy 5.79).
  - [x] **(e) export:** `export_submission` exige `launch.weight/bias`, `_neural_action` prepõe argmax de launch, `_neural_decode` trata `launch==0->[]`; benchmark do export = crash/timeout/invalid 0.
  - [x] **(f) masking** ✅ feito 2026-06-07. `python/orbit_wars_gym/action_masks.py` (builder flat `MASK_DIM=50` launch|source|target + `split_masks`); `policy.get_action_and_value(..., masks=)` aplica `masked_fill`; ligado no rollout single-env **e** batched, `_concat_segments`, `_ppo_update` (mesma mask no sampling E no update — testado em `test_same_mask_reproduces_logprob_for_ppo_ratio`); export `_neural_action` usa argmax mascarado p/ paridade. 5 testes em `tests/test_action_masks.py`; PPO smoke com mask saudável (kl 0.0025, clipfrac 0.035). Regra: launch=0 sempre válido; launch=1 só se há planeta lançável; source até len(own_launchable); target até planet_count-1; head sem entrada válida cai p/ all-valid (evita NaN, ignorado pois launch=0).
  - [x] **Correto (parcial):** ✅ 198 testes passam; `decode([0,...])==[]`; no-op vira `launch=0` (não (0,0,0,0) ativo); `logprob(pass)` independe de source/target/frac/offset; PPO smoke shape-5; export inclui launch e roda sem ação ilegal. Registrado em `EXPERIMENTS.md`. **Pendente:** BC report com `no_op_rate≈0.81` + métricas separadas de launch e heads ativos (vem no P2); masking (P3).

- [x] **P2. Behavioral cloning antes do PPO.** ✅ feito e verificado 2026-06-07 (gate atingido; BC fraco mas saiu do colapso).
  - [x] **Criar treino BC:** `python/train/train_bc.py` com loss launch-aware (`CE(launch) + 1[launch==1]*(CE(source)+CE(target)+CE(frac)+CE(offset))`) e métricas por expert. Testado (`tests/test_train_bc.py`).
  - [x] **Checkpoints:** `artifacts/bc/bc_producer.pt`, `bc_oep.pt`, `bc_mix.pt` treinados no dataset 0-11 (1024 train cada); salvam `summary` (decoder), `config`, `val_metrics`.
  - [x] **Validação offline:** report com launch P/R/F1, predicted vs expert pass rate, active-head top-1 acc (só launch=1), loss por expert. (top-3/erro de moves reconstruídos não feitos — top-1 cobre o gate.)
  - [x] **Benchmark online:** vs `producer,oep` 16 seeds 256 steps: bc_producer margin=-0.9150 (win 0.016), bc_oep=-0.9674 (win 0.0), bc_mix colapsou p/ pass. crash/timeout/invalid=0.
  - [x] **Teste de não-regressão:** `tests/test_train_bc.py` confirma carga via `export_submission._load_checkpoint_payload` (com `launch.*`); export roda sem ação ilegal.
  - [x] **Correto quando:** ✅ `bc_producer mean_score_margin=-0.915 > -1.0` vs Producer/OEP, `crash/timeout/invalid=0`, aprendeu além de no-op (source acc~0.70, launch calibrado quando não colapsa). **NÃO promover BC como submissão** (perde quase tudo vs Producer; é warm-start p/ P3).
  - **Limitações medidas (input p/ P3/T4/T5):** (1) target head fraco (acc~0.07-0.12) — gargalo da quantização do target_rank; (2) launch head às vezes colapsa p/ classe majoritária (pass ~60%) → P3 deve usar class-weight/entropy no launch; (3) instabilidade entre seeds/tamanhos de dataset.

- [~] **P3. PPO fine-tune com currículo forte.** **INFRA PRONTA 2026-06-07; falta a CAMPANHA (compute longo) + decisão de promoção.**
  - [x] **Inicialização (arch-aware):** `train_ppo` seleciona a arquitetura pelo `summary["arch"]` do checkpoint (ou `--policy-arch`). **entity-init funciona ponta a ponta**: carrega bc_entity → PPO entity → salva/exporta arch-aware (smokes: arch=entity auto-detectado, flat-init compat, export crash/timeout/invalid=0). `_build_policy` + `_POLICY_ARCHS` em train_ppo.
  - [x] **Currículo inicial:** `70% Producer / 20% heurísticas / 10% OEP` via repetição em `--opponents` (ex.: `producer×7,greedy×2,oep`). Verificado em opponent_segments.
  - [x] **Instrumentação obrigatória:** `explained_variance` + `update_series` (approx_kl, clipfrac, entropy, value_loss, policy_loss) no summary/checkpoint. (reward médio/length/neutral captures já em `_aggregate_episode_metrics`.)
  - [x] **Corrida limitada validada (120k timesteps, entity-init):** ✅ 2026-06-07. SPS 85.9 (~23 min); `artifacts/ppo/ppo_entity.pt`.
    Curvas: EV −0.14→+0.79→estável **+0.90–0.94**; KL 0.005–0.02; clipfrac 0.07–0.15; entropy 1.35→3.0 (sem colapso); value_loss 0.55→0.03. Treino SADIO.
    **Benchmark vs Producer/OEP 16 seeds: margin −0.7491 (win 0.094)** — **melhorou +0.20 sobre o BC entity (−0.9535)**; crash/timeout/invalid=0.
  - [x] **Correto quando:** ✅ (na corrida limitada) curva pareada **melhora sobre o BC** (−0.95→−0.75), `explained_variance` materialmente > 0 (0.79–0.94), e PPO **não destrói o BC** (EV positivo já em u5). **Funil imitação→PPO validado.**
  - [x] **Extensão +500k (620k total):** margin −0.749 (120k) → −0.848 (620k), win 0.094→0.047. **REENQUADRAR (input do usuário):** o post de referência treinou PPO **~12–15h em GPU potente** (provável 10–50M timesteps). Meus runs CPU (120k/620k @ ~85 SPS) são **escala de VALIDAÇÃO**, 1–2 ordens de grandeza abaixo. Logo a "regressão" é mais provável **sub-treino/instabilidade inicial** do que conclusão convergida. O que está SÓLIDO: (1) o funil BC→PPO melhora cedo (−0.95→−0.75) e (2) treino é sadio (EV ~0.94). Melhor checkpoint local = `ppo_entity` (120k, −0.75).
  - **Evidência externa decisiva (usuário):** o PPO do post atingiu **~1300 de score — ACIMA do Producer (~1228)**, com **~12–15h de GPU**. Ou seja: **PPO é caminho PROVADO para bater o Producer / TOP-5**, e o gargalo é COMPUTE, não a abordagem. Meu pipeline (imitação→PPO + masking + entity + eval-gating) é o harness certo; falta o budget.
  - **Nota de compute:** a regra do CLAUDE.md "sem GPU" é sobre **inferência** (agente roda CPU-only, actTimeout=1s). **Treino** é outra história. Gargalo de throughput = env-stepping (sim Rust + planners Producer/OEP em CPU); ~85 SPS → 15M timesteps ≈ ~49h CPU. Campanha competitiva = **GPU + envs vetorizados** = fronteira de compute do usuário.
  - [x] **UNLOCK de throughput — isolamento por-env dos opponents** ✅ 2026-06-07. `make_runtime`/`make_agent` em bots/oep + bots/producer (runtime fresco por instância); `registry.get_isolated_opponents(name, count)` (pool cacheado de instâncias isoladas; reset por step==0); rollout batched usa uma instância por env; guard de P0 removido (agora seguro). Provado sem contaminação cruzada (`tests/test_opponent_isolation.py` — producer+oep) + 217 testes verdes.
    - **Medido:** single-env CPU **147 SPS** → 16-env GPU (RTX 5060 Ti) **265 SPS (~1.8×)**. 10M timesteps ≈ ~10.5h (faixa do fórum).
    - **Novo gargalo:** planner do opponent (Producer/OEP em Python, sequencial por env) — vetorização batcha sim Rust + policy GPU, mas o opponent é linear no nº de envs.
  - [x] **Paralelizar opponents — TENTADO (threads + processos), AMBOS FALHAM** 2026-06-07. Threads (ThreadPoolExecutor): 12 SPS (~20× pior) — planners GIL-bound. Processos (`ProcessOpponentPool`, spawn, env→worker fixo): 74.5 SPS (~3.3× pior) — IPC/pickle do estado por step > custo do planner (~3.6ms). Correção verificada nos dois. **Default revertido p/ sequencial** (`opponent_workers=1`); pool fica opt-in experimental. Conclusão: paralelizar a *chamada* não compensa (planner barato vs transporte).
  - **Lever real de throughput** (se quiser >1.8×): reduzir o CUSTO do opponent — currículo com mais greedy/producer (baratos) e menos OEP (caro), ou um opponent mais rápido. O unlock 1.8× (~250–275 SPS) já dá campanha overnight (10M ≈ ~10h).
  - [ ] **NECESSÁRIO (provado pela regressão): eval periódico + seleção por margem pareada + early-stop.** Gancho no train loop que, a cada N updates, exporta + benchmarka vs Producer/OEP e guarda o melhor; parar se a margem piorar 3 evals seguidos ou entropy subir demais. **Sem isso, treino longo regride.**
  - [ ] **Tuning anti-drift:** reduzir ent_coef / ancorar no BC (KL ao BC ou ent menor); revisar shaping (pode estar desalinhado da margem final). Re-rodar campanha COM eval-gating.
  - [ ] **CAMPANHA LONGA + promoção:** só depois do eval-gating; gate de 96 seeds = decisão do usuário. `ppo_entity` (−0.75) ainda NÃO bate Producer → não promover.

- [ ] **P4. Hall-of-fame/self-play só depois de sair do buraco.**
  - [ ] **Pré-condição:** só iniciar quando melhor PPO/BC tiver `mean_score_margin > -1.0` vs Producer e alguma vitória/empate real em triagem.
  - [ ] **Pool:** manter sempre Producer e OEP na pool; adicionar checkpoints próprios promovidos por margem pareada, não por reward de treino.
  - [ ] **Sampling:** limitar self-play para não virar monocultura; cada batch precisa conter uma fração mínima de Producer/OEP.
  - [ ] **Antiesquecimento:** rodar eval contra snapshots antigos antes de substituir o campeão.
  - [ ] **Correto quando:** novo checkpoint melhora contra versões próprias sem regredir materialmente contra Producer/OEP; se regredir, volta para currículo com mais expert.

- [x] **P5. Ablation de map bias/features.** ✅ auditoria + tratamento (augmentation) feitos e verificados 2026-06-07.
  - [x] **Teste de invariância:** `tests/test_map_bias_invariance.py` + `python/orbit_wars_gym/symmetry.py` (rotate_180/reflect_x/swap_players, involuções testadas) + `scripts/audit_map_bias.py` (gap de logits por head/transform).
  - [x] **Baseline:** `artifacts/map_bias/invariance_report.json` — perspectiva gap 0.0; rotate_180 2.5–5.6; reflect_x 1.7–3.6 (viés espacial de x/y absolutos + planet_id).
  - [x] **Tratamento (data augmentation):** `--augment` no collector; `artifacts/map_bias/invariance_report_aug.json` — gap caiu ~5–10×, margem online melhorou (−0.915→−0.880). Ver SEGUIR AGORA acima.
  - [~] **Tratamento alternativo (canonicalização) / entity-arch:** não feito — augmentation já atingiu o critério; canonicalização e entity encoder ficam em **T4** (target head ainda fraco = problema de representação).
  - [ ] **Tratamentos:** testar pelo menos duas opções: canonicalização relativa ao jogador e data augmentation no rollout/dataset.
  - [ ] **Benchmark BC:** treinar BC pequeno com encoder atual vs variante e comparar loss/accuracy/margem online.
  - [ ] **Correto quando:** variante reduz sensibilidade artificial a simetrias sem piorar benchmark pareado contra Producer/OEP; se loss melhora mas jogo piora, não promover.

- [ ] **P6. Gate de promoção PPO.**
  - [ ] **Gate 1 triagem:** 16 seeds vs Producer/OEP + sanity contra `greedy,rush,anti_meta`; usar só para decidir se vale 96 seeds.
  - [ ] **Gate 2 decisor:** 96 seeds pareadas vs Producer e OEP, com cometas ligados e registro 2p/4p separado.
  - [ ] **Gate 3 top-5 proxy:** incluir pelo menos um agente público forte do benchmark comunitário antes de submeter.
  - [ ] **Métricas obrigatórias:** `mean_score_margin`, win rate, paired delta, worst decile, crash/timeout/invalid, mean/p95/max decision ms, 2p e 4p separados.
  - [ ] **Registro:** adicionar linha em `EXPERIMENTS.md` com baseline, candidato, comandos, artefatos, margem antes/depois e decisão.
  - [ ] **Correto quando:** `crash/timeout/invalid=0`, margem pareada não-negativa vs Producer/OEP, sem regressão clara em 4p, e top-5 proxy não contradiz a promoção.

## 🧨 MAIS OPÇÕES DE EXPERIMENTOS — objetivo TOP 5
Base prática: tópicos Kaggle `704095` (109-agent tournament), `704741` (PPO que treina), `704849`
(bulk replays), `704817` (lookup/precompute), `704777` (2p/4p) e score real das nossas submissões
(`Producer≈1200`, `OEP≈1100`). Base de literatura via `paper-search`: transfer/imitation em deep RL
(Zhu et al. 2023; Zhu et al. 2018), PPO/sistemas escaláveis em jogos complexos (Ye et al. 2020),
multi-agent RL/opponent modelling (Wong et al. 2022) e desafios de RL real/simulado (Dulac-Arnold et al. 2021).
Força da evidência: **forte** para imitação+RL e opponent pool; **parcial** para a forma exata em Orbit Wars.

- [ ] **T0. Régua top-5 local antes de qualquer nova submissão.**
  - [ ] **Selecionar pool:** baixar/empacotar agentes públicos fortes do benchmark `704095`: `exp30`, `exp29/27`, `sim-value-search`, `advanced-timeline`, `dominance redistribution` e 1-2 forks top.
  - [ ] **Isolar fixtures:** colocar cada agente em `artifacts/opponents/top5_proxy/<id>/` com README curto de origem, commit/notebook, dependências e comando de smoke.
  - [ ] **Smoke individual:** cada oponente precisa rodar 2p e, se aplicável, 4p por poucos seeds contra Producer sem crash/timeout.
  - [ ] **Configurar gate:** criar `configs/eval_top5_proxy.yaml` com Producer, OEP e pool pública forte, separando 2p/4p e seeds fixas.
  - [ ] **Relatório:** salvar `artifacts/top5_proxy/baseline_producer_oep.json` para saber onde Producer/OEP ficam nessa régua.
  - [ ] **Correto quando:** qualquer candidato novo é comparado contra a mesma pool; se ele só bate Producer mas apanha da pool top proxy, não é candidato top 5.

- [ ] **T1. Replay mining das nossas derrotas reais.**
  - [ ] **Coleta:** usar fluxo do tópico `704849` ou script próprio para baixar replays das submissões Producer/OEP e organizar por `submission_ref/outcome/opponent`.
  - [ ] **Parser:** extrair por replay: formato 2p/4p, seed, scores por step, ownership, fleets, comets, ações, timeouts/crashes.
  - [ ] **Classificador de derrota:** marcar causas prováveis: abertura ruim, comet, 4p/kingmaker, overextension, falta de defesa/recapture, redistribuição tardia, timeout/crash.
  - [ ] **Amostras visuais:** gerar links/HTML para 5-10 derrotas representativas por classe.
  - [ ] **Tabela decisora:** produzir `artifacts/replay_mining/loss_taxonomy.csv` e resumo com impacto estimado por classe.
  - [ ] **Correto quando:** temos top-3 padrões de derrota com replays exemplares; cada novo experimento cita qual classe de perda está atacando.

- [ ] **T2. DAgger / imitação iterativa com Producer+OEP.**
  - [ ] **Pré-condição:** P1/P2 concluídos e uma policy BC exportável funcionando.
  - [ ] **Coleta on-policy:** rodar a policy BC/PPO em seeds fixas; detectar estados ruins por perda de margem, ação inválida/sem efeito, divergência grande de expert ou queda futura.
  - [ ] **Relabel:** consultar Producer/OEP nesses estados e adicionar labels ao dataset com peso maior para estados críticos.
  - [ ] **Ciclos:** repetir 3-5 vezes `policy -> estados ruins -> expert labels -> BC resume -> benchmark`.
  - [ ] **Controle:** manter dataset test fixo fora dos ciclos para medir generalização, não memorização.
  - [ ] **Correto quando:** cada ciclo melhora ou mantém margem contra Producer/OEP em seeds fixas; se dois ciclos seguidos piorarem, parar e auditar quantização/labels.

- [ ] **T3. Distilação de valor/critic a partir dos experts.**
  - [ ] **Gerar targets:** rollouts de Producer/OEP com retorno final, score-margin normalizado, produção final, sobrevivência e flags de vitória.
  - [ ] **Treinar critic:** pré-treinar value head para predizer retorno/score-margin a partir do encoder; salvar `critic_pretrain.pt`.
  - [ ] **Ablation:** comparar PPO inicializado com actor BC apenas vs actor BC + critic pré-treinado.
  - [ ] **Métricas:** `explained_variance`, value_loss inicial, estabilidade de KL/clipfrac, margem em eval curto.
  - [ ] **Correto quando:** critic pré-treinado aumenta `explained_variance` e não causa regressão online nos primeiros updates; se só melhora loss offline, não promover.

- [~] **T4. Entity encoder / attention em vez de MLP flat.** **OFFLINE VENCEU (decisivo) 2026-06-07; export + online pendentes.**
  - [x] **Design mínimo:** `EntityActorCritic` (`python/agents/policy.py`) — reshape da MESMA obs flat em planetas(96×14)/frotas(256×10), MLP por-entidade + masked mean-pool (flag de presença), trunk + heads compatíveis; mesma interface `get_action_and_value(masks)`. Invariante a ordem de slot (testado). Sem mudar dataset/decoder/ação.
  - [x] **Comparação controlada (mesmo dataset/seed/épocas):** entity >> flat em quase todo head: launch F1 0.16→**0.60**, **target 0.065→0.185 (~3×)**, offset 0.16→0.50, frac 0.52→0.72, source ~empate; flat colapsou (over-pass), entity calibrado.
  - [x] **Métricas (offline):** accuracy por cabeça (acima), legal rate ok. **Falta:** runtime por decisão + margem online + tamanho de submissão.
  - [x] **Exportabilidade:** ✅ export arch-aware (`export_submission` detecta `summary["arch"]`); forward entity em Python puro (reshape + MLP por-entidade + masked pool). Roda legal, **18ms/decisão** (vs ~25ms flat), crash/timeout/invalid=0. 4 testes em `tests/test_entity_policy.py` (inclui invariância a permutação).
  - [x] **Online (BC):** bc_entity vs Producer/OEP 16 seeds = **margin −0.9535, win 0.0** — NÃO melhor que flat (−0.915) nem aug (−0.880). **Achado: accuracy offline de BC NÃO se traduz em margin online** (BC sozinho, qualquer arch, é warm-start fraco que perde quase tudo).
  - [x] **Correto quando (avaliado):** entity é melhor offline + exportável + rápido, mas **não melhor online via BC** → pelo critério estrito, não promover por BC. **Conclusão:** entity é a melhor ARQUITETURA p/ inicializar o PPO (target head + launch calibrado + exportável + 18ms); a decisão flat-vs-entity deve ser settled pelo **PPO online (P3)**, não pelo BC. Próximo: P3 fine-tune com entity-init.

- [ ] **T5. Action masking e decoder supervisionado.**
  - [ ] **Máscaras óbvias:** origem sem naves, planeta inexistente, alvo próprio quando modo ataque, fração que viola reserva mínima, offset que leva a sol/borda quando detectável.
  - [ ] **Auxiliary head:** treinar `intent` (`no-op`, `capture`, `attack`, `reinforce`, `regroup`, `evacuate/comet`) para estabilizar representação.
  - [ ] **Integração PPO:** aplicar máscara antes de sample/logprob para manter cálculo correto de PPO; não mascarar só no decode final.
  - [ ] **Testes:** unidade para logprob/entropy com máscara e para não gerar ação impossível em estados sintéticos.
  - [ ] **Correto quando:** invalid/sem-efeito cai, entropy não colapsa prematuramente, e benchmark melhora ou mantém margem contra Producer/OEP.

- [ ] **T6. Política/treino 4p separado.**
  - [ ] **Track separado:** criar config/checkpoint `phase5_4p` com dataset/rollouts 4p e features de terceiro jogador/vulnerabilidade.
  - [ ] **Experts 4p:** usar Producer/OEP se rodarem em 4p com segurança; senão, criar pool mista com heurísticas 4p estáveis.
  - [ ] **Mixture por modo:** selecionar policy 2p para 2p e policy 4p para 4p no wrapper de submissão.
  - [ ] **Gate:** avaliar 2p e 4p separadamente; não aceitar ganho 4p que destrói 2p.
  - [ ] **Correto quando:** 4p melhora com `crash/timeout/invalid=0`, p95 decision ms seguro e 2p mantém margem contra Producer/OEP.

- [ ] **T7. Especialista de cometas.**
  - [ ] **Diagnóstico antes:** usar replay mining para confirmar que cometas explicam perda material; se não aparecem no top-3, adiar.
  - [ ] **Features:** adicionar janela até próximo spawn (`50/150/250/350/450`), planeta comet, distância/ETA, risco de overcommit e evacuação.
  - [ ] **Política especialista:** testar heurística ou cabeça auxiliar para contestar, ignorar ou evacuar cometa.
  - [ ] **Ablation:** Producer puro vs Producer+comet specialist vs PPO/BC com feature de spawn window.
  - [ ] **Correto quando:** melhora em seeds com cometa decisivo sem piorar seeds sem disputa de cometa; se só ajuda casos raros e piora geral, não promover.

- [ ] **T8. Planner standalone timeline-sim completo.**
  - [ ] **Escopo novo:** não reutilizar frame OEP de “desviar do Producer”; construir planner com missões explícitas.
  - [ ] **Missões mínimas:** `safe_capture`, `rescue`, `recapture`, `reinforce`, `snipe`, `hammer/multiprong`, `dominance redistribution`.
  - [ ] **Timeline:** simular arrivals/produção/recapture por planeta em horizonte suficiente, com risco de resposta de oponente.
  - [ ] **Valor terminal:** território/produção/risco, não só delta de ships.
  - [ ] **Critério duro:** precisa bater Producer/OEP em 96 seeds; se só empata ou ganha em 16 seeds, não chamar de progresso.
  - [ ] **Correto quando:** melhora pareada robusta, tempo dentro de `actTimeout`, e replays mostram missões novas úteis em derrotas antes classificadas.

- [ ] **T9. Mixture-of-experts / seletor de política.**
  - [ ] **Experts elegíveis:** Producer, OEP, PPO-BC, PPO-finetune, especialista 4p, especialista cometa; cada expert precisa ter benchmark próprio.
  - [ ] **Features do seletor:** formato 2p/4p, step, vantagem de produção, pressão inimiga, spawn de cometa, vulnerabilidade, fase de jogo.
  - [ ] **Baseline simples:** seletor heurístico por regras antes de treinar seletor aprendido.
  - [ ] **Auditoria:** logar qual expert foi escolhido por turno/fase e resultado posterior.
  - [ ] **Correto quando:** seletor supera o melhor expert individual em benchmark pareado; se só alterna e empata, complexidade não vale.

- [ ] **T10. Population-based training real.**
  - [ ] **Pré-condição:** PPO/BC já saiu do `-1.0` vs Producer; antes disso PBT só multiplica runs ruins.
  - [ ] **Mutáveis:** LR, entropy, clip, reward scales, decoder fractions, opponent mix, BC loss coefficient.
  - [ ] **Exploit/explore:** promover por margem pareada em eval fixo, não por reward médio de treino.
  - [ ] **Lineage:** todo checkpoint precisa registrar pai, hparams, dataset hash, opponent mix e evals.
  - [ ] **Correto quando:** pelo menos um braço melhora contra Producer/OEP e não esquece hall-of-fame; se população converge para heurística fraca, resetar curriculum.

  ### DESENHO CONCRETO — PBT-sobre-PPO (workflow `pbt-over-ppo-design`, 2026-06-09)
  > Lit: PBT (Jaderberg 2017, arXiv:1711.09846) + sensibilidade hparam PPO (Andrychowicz 2020, arXiv:2006.05990; Engstrom 2020, arXiv:2005.12729). **Honestidade: força FRACA no "vai cruzar 0"** — RL batendo heurística 1228 é projeto multi-dia (EXPERIMENTS l.82); orçar **3-5 runs overnight**, esperar muitas gerações VERMELHAS. Cross-ref [[ppo_two_structural_ceilings]] (PBRS Φ = objetivo do Producer; genes de shaping são suspeitos).
  > **Schedule medido (RTX 5060 Ti, 8 cores):** 1 membro 273 SPS; 2 concorrentes 438 agregado (joelho); **3+ REGRIDE** (CPU env-step binda, não VRAM). → pop **6**, time-share em **3 ondas de 2**, chunk **60k ts**, **12 gerações** ≈ **8-9h overnight**. Warm-start: **TODOS de `artifacts/ppo/ppo_entity.pt`** (−0.75); `bc_entity.pt` = referência KL congelada (papel diferente).
  > **Genoma (atenção):** `lr`[5e-5,1e-3] >> `beta_kl`(KL-ao-BC) > `gae_lambda`[.9,.99] ~ `gamma` ~ `update_epochs`[2,10] > `vf_coef`~`clip_coef`~`ent_coef`(floor .003) > shaping/decoder. Exploit = bottom-2 copiam ckpt+hparams de top-2; explore = perturbar ×{0.8,1.25}.

  - [ ] **PRÉ-CONDIÇÃO 1 — patch `benchmark_submission.py` (gate inverificável sem isto):** em `_run_match` coletar wall-clock por decisão → `summary['p95_decision_ms']=percentile(95)`; em `_summary_from_records` adicionar `seat0_margin`/`seat1_margin` (split por assento, não só média).
    - [ ] verificar: `benchmark_ppo_submission` em `ppo_entity.pt` a 4 seeds mostra `p95_decision_ms`, `seat0_margin`, `seat1_margin` no JSON.
  - [ ] **PRÉ-CONDIÇÃO 2 — patch `train_ppo.py` (anti-drift + WSL):** flags `--beta-kl` + `--bc-ref-checkpoint`; carregar π_BC congelado de `bc_entity.pt`; somar `+beta_kl*KL(π‖π_BC)` à loss (P3 l.111); clampar `ent_coef` annealed em floor 0.003; `os._exit(0)` após o `torch.save` final.
    - [ ] verificar: smoke 4k-ts `--beta-kl 0.05` completa, **sai limpo (sem hang)** e loga termo `kl_to_bc`.
  - [ ] **PRÉ-CONDIÇÃO 3 — generalizar `ppo_campaign.py` p/ população (orquestrador PBT):** estado `members[]` {ckpt, hparams, seed, best_margin, history}; treino via `subprocess.Popen` em ondas de 2 (NUNCA ProcessPool — trava no WSL); eval **serial `jobs=1`** (forçar; `_margin()` usa jobs=8 hoje); exploit (bottom-2←top-2) + explore (`_perturb` por gene); patience na top member; `campaign_report.json`/geração c/ seat-split + pareto. `population_size=1` = comportamento atual (backward-compat).
    - [ ] verificar: smoke `tests/` pop=2 gen=2 chunk 4k 2 seeds `--device cuda` roda ponta-a-ponta, sem ProcessPool, gera `campaign_report.json` com margem+seat por membro e exploit copia checkpoint.
  - [ ] **3 seed-sets disjuntos (assert pairwise-empty):** train=`range(0,N)`, fitness=`range(1000,1016)` (16), frozen=`range(9000,9064)` (64). Frozen tocado **uma vez** no vencedor (anti seed-overfit, Cobbe 2018).
  - [ ] **LANÇAR run overnight:** `ppo_campaign --init artifacts/ppo/ppo_entity.pt --bc-ref-checkpoint artifacts/bc/bc_entity.pt --out-dir artifacts/pbt --population-size 6 --generations 12 --chunk-timesteps 60000 --concurrency 2 --device cuda --eval-opponents producer --eval-episode-steps 500 --fitness-seeds 1000..1015 --policy-arch entity`.
    - [ ] verificar (2 primeiras gerações): sem hang; margens logadas; `nvidia-smi` mostra 2 procs ~800MiB.
  - [ ] **GATE FROZEN (vencedor, /goal):** `benchmark_ppo_submission --checkpoint artifacts/pbt/best.pt --opponents producer --episode-steps 500 --jobs 1 --seeds 9000..9063` (2p **E** 4p).
    - [ ] verificar (condição /goal): `seat0_margin>0 AND seat1_margin>0 AND mean_score_margin>0 AND crash=timeout=invalid=0 AND p95_decision_ms<700`; 4p sem crash/timeout/invalid e margin≥0. **PASS frozen é necessário, não suficiente** (gate humano 96 seeds é a palavra final).

- [ ] **T11. Precompute/lookup para liberar orçamento de planner.**
  - [ ] **Medição baseline:** antes de otimizar, medir p50/p95/max decision ms e hotspots do planner/decoder atual.
  - [ ] **Cache alvo:** intercept angle/ETA por pares de planetas por janela de 50 turns, inspirado no tópico `704817`, mas em Python submetível.
  - [ ] **Validação geométrica:** comparar lookup contra cálculo direto em amostra grande; registrar erro por step/par.
  - [ ] **Uso controlado:** se lookup tiver erro ±1, ele só pode ser usado quando não muda decisão crítica ou com fallback explícito medido.
  - [ ] **Correto quando:** p95/max cai materialmente, decisões não divergem em casos sensíveis e qualidade não piora no gate.

- [ ] **T12. Critério de corte para matar linha cedo.**
  - [ ] **Definir antes de rodar:** cada família precisa declarar métrica-alvo, custo máximo, seeds de triagem, gate decisor e condição de parada.
  - [ ] **Kill criteria padrão:** 3 ciclos sem melhora pareada, `crash/timeout/invalid>0`, regressão 4p grave, `explained_variance` ruim persistente ou custo acima do orçamento.
  - [ ] **Registro:** toda morte de linha entra em `EXPERIMENTS.md` com motivo técnico, para não ressuscitar tuning morto.
  - [ ] **Reabertura:** só reabrir família se houver evidência nova: oponente novo, replay mining novo, bug corrigido ou hipótese diferente.
  - [ ] **Correto quando:** o backlog fica enxuto e nenhuma linha morta volta por intuição; o próximo experimento sempre aponta para um padrão medido.

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
- ~~GPU~~ — **REVISTO 2026-06-07: GPU LIBERADA para TREINO** (a máquina tem GPU local; ver CLAUDE.md "Compute / GPU"). A regra antiga ("compute não é o gargalo") valia para micro-tuning de heurística, mas o caminho PPO competitivo (~1300 > Producer com ~12–15h GPU) é justamente limitado por compute. **Inferência/submissão segue CPU-only** (invariante D10/D11): treinar na GPU, exportar modelo que roda em CPU.
