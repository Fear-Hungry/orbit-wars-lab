> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `EXPERIMENTS.md` — este arquivo fica enxuto, só com o que atacar e o estado atual.

---

# 🎯 ATACAR AGORA — objetivo top 5

Estado operacional curto:

- **Producer é a melhor submissão operacional atual** (~1200 LB). Congelar como default até existir candidato provado.
- **OEP é o 2º arquétipo forte** (busca), **não morto**: LB convergiu p/ **1171.5**, ~empata Producer (1174.9); local 1v1 **−0.045 @96** (correção 2026-06-07 — o score 600 foi leitura precoce, pré-convergência). Útil como adversário/professor e gate mínimo, mas **não como linha de tuning**: knobs/overlays OEP já saturaram e não voltam sem hipótese nova.
- **PPO atual ainda é fraco** (`-1.0` vs Producer nos registros antigos). O próximo ataque é imitação + currículo forte, não PPO do zero contra heurísticas fracas.
- **Histórico detalhado fechado vive em `EXPERIMENTS.md`**. Este arquivo deve ficar só com o que ainda vamos atacar.
- **Decidir só com evidência pareada suficiente**: 16 seeds = triagem; 96 seeds decide. Score Kaggle precisa estabilizar antes de conclusão.

## 📄 DOC — model cards de implementação (pedido 2026-06-07)
Motivo: `EXPERIMENTS.md` é audit trail (hipótese→margem→decisão), **não** descreve COMO os
modelos são construídos nem consolida "o que deu certo neles". A implementação mora no fonte
(`bots/oep/planner.py` 110KB, `bots/producer/_upstream.py`, entity encoder em `python/`).

- [ ] **Decidir o par a documentar.** `OEP + ppo_entity` (suas duas linhas) ou `Producer + OEP`
  (produção + melhor busca). Honestidade: nenhum modelo SEU bate o Producer (OEP −0.045, ppo_entity −0.665).
  - [ ] verificar: par escolhido escrito no topo do doc.
- [ ] **Escolher o destino:** novo `docs/MODELS.md` (1 dono/tópico) ou expandir os `bots/*/card.md` (hoje rasos, ~5 linhas).
  - [ ] verificar: `docs/README.md` lista o doc se for `docs/MODELS.md`.
- [ ] **Documentar cada modelo a partir do FONTE** (não do log): entrada/obs, encoder/representação,
  algoritmo de decisão (FSM+scoring do Producer / gerar-plano+fitness+`min_advantage` do OEP /
  forward+heads do ppo_entity), espaço de ação, e "por que funciona" ancorado em `arquivo:linha` ou data do log.
  - [ ] verificar: toda afirmação de design aponta para `arquivo:linha` do fonte OU uma data de `EXPERIMENTS.md`.
- [ ] **Seção "o que deu certo vs o que matou"** consolidando os `submit?=aceitar`/`rejeitar` por modelo.
  - [ ] verificar: toda linha `aceitar` daquele modelo aparece resumida.

## 🧭 SEGUIR AGORA — lição do tópico Kaggle 704741 ("Lessons learned so far")
O tópico do Radek muda a prioridade: PPO competitivo não é "mais reward shaping"; é **sistema de treino**
com features corretas, métricas de PPO, currículo/opponent pool e avaliação por checkpoints. O nosso erro
atual é ficar entre dois mundos: OEP/Producer overlay já saturou, enquanto PPO ainda é baseline fraco
treinado contra heurísticas fracas.

- [ ] **Congelar o default competitivo no Producer até existir candidato provado.** Não gastar mais sessão com
  knobs de seleção OEP (`min_advantage`, horizon, ordinal, reactive, rollout) sem gerador de plano novo ou
  evidência externa forte. (Correção 2026-06-07: a família de **seleção** OEP saturou (−0.045 local; E1–E3 regrediram), mas a premissa antiga de "OEP overfit/morto no LB" está **obsoleta** — OEP convergiu p/ 1171.5, ~empata Producer.)
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
  - **Sinal externo forte (usuário), prova local PENDENTE:** o PPO do post atingiu **~1300 de score — ACIMA do Producer (~1228)**, com **~12–15h de GPU**. Ou seja: **PPO é caminho plausível e externamente sinalizado para bater Producer/TOP-5**, com o COMPUTE como provável gargalo. **Mas no repositório isso NÃO foi reproduzido.** Provado localmente: o funil BC→PPO melhora cedo (−0.95→−0.75). Não provado localmente: PPO bate Producer. Meu pipeline (imitação→PPO + masking + entity + eval-gating) é o harness certo; falta budget E reprodução.
  - **Nota de compute:** a regra do CLAUDE.md "sem GPU" é sobre **inferência** (agente roda CPU-only, actTimeout=1s). **Treino** é outra história. Gargalo de throughput = env-stepping (sim Rust + planners Producer/OEP em CPU); ~85 SPS → 15M timesteps ≈ ~49h CPU. Campanha competitiva = **GPU + envs vetorizados** = fronteira de compute do usuário.
  - [x] **UNLOCK de throughput — isolamento por-env dos opponents** ✅ 2026-06-07. `make_runtime`/`make_agent` em bots/oep + bots/producer (runtime fresco por instância); `registry.get_isolated_opponents(name, count)` (pool cacheado de instâncias isoladas; reset por step==0); rollout batched usa uma instância por env; guard de P0 removido (agora seguro). Provado sem contaminação cruzada (`tests/test_opponent_isolation.py` — producer+oep) + 217 testes verdes.
    - **Medido:** single-env CPU **147 SPS** → 16-env GPU (RTX 5060 Ti) **265 SPS (~1.8×)**. 10M timesteps ≈ ~10.5h (faixa do fórum).
    - **Novo gargalo:** planner do opponent (Producer/OEP em Python, sequencial por env) — vetorização batcha sim Rust + policy GPU, mas o opponent é linear no nº de envs.
  - [x] **Paralelizar opponents — TENTADO (threads + processos), AMBOS FALHAM** 2026-06-07. Threads (ThreadPoolExecutor): 12 SPS (~20× pior) — planners GIL-bound. Processos (`ProcessOpponentPool`, spawn, env→worker fixo): 74.5 SPS (~3.3× pior) — IPC/pickle do estado por step > custo do planner (~3.6ms). Correção verificada nos dois. **Default revertido p/ sequencial** (`opponent_workers=1`); pool fica opt-in experimental. Conclusão: paralelizar a *chamada* não compensa (planner barato vs transporte).
  - **Lever real de throughput** (se quiser >1.8×): reduzir o CUSTO do opponent — currículo com mais greedy/producer (baratos) e menos OEP (caro), ou um opponent mais rápido. O unlock 1.8× (~250–275 SPS) já dá campanha overnight (10M ≈ ~10h).
  - **Ordem inegociável: P3.1 (eval-gating) → P3.2 (anti-drift) → T5 (target/action redesign).** Sem eval hook, treino longo só produz outro checkpoint pior (provado pelo +500k).
  - [x] **P3.1 — eval-gating automático** ✅ 2026-06-07 via `scripts/run_campaign.py` (orquestrador EXTERNO por subprocessos, melhor que hook in-loop: train-CUDA e benchmark-CPU em processos separados → evita o deadlock fork-after-CUDA). A cada bloco (200k–500k) exporta+benchmarka 16 seeds vs Producer/OEP, guarda `best.pt` por margem pareada, early-stop por `--patience`. Smoke validado (2 chunks, eval jobs paralelos sem deadlock). **Nota:** `--patience 3` para a regra "piorar em 3 avaliações"; lancei a 1ª campanha com patience 5 (mais folga).
  - [ ] **P3.2 — tuning anti-drift (depois do hook):** reduzir `ent_coef` / ancorar no BC (KL ao BC ou ent menor); revisar shaping (pode estar desalinhado da margem final). Re-rodar a campanha em blocos COM eval-gating ligado.
  - [~] **CAMPANHA LONGA — 1ª rodada RODANDO (overnight, GPU) 2026-06-07.** `run_campaign.py` detached (`nohup`+`disown`, PID 1488136; o `run_in_background` do harness sumia com subprocessos aninhados). Config: entity-init (`bc_entity.pt`), 16 envs GPU, ent 0.01, currículo 70/20/10, até 30×200k=6M, eval producer,oep 16s/256, keep-best, patience 5. Monitorar `artifacts/ppo/campaign/{campaign_log.jsonl,best.pt,campaign_report.json}`. **1ª rodada usa ent 0.01 (não anti-drift) de propósito: o eval-gating guarda o melhor antes de qualquer drift, e o resultado decide se P3.2 (ent menor/KL-BC) é necessário.**
    - **Critério de promoção (do usuário):** só escalar a 96 seeds quando um checkpoint superar **−0.7491 com FOLGA** em triagem 16 seeds; antes disso é desperdício. 96 seeds + submissão = decisão do usuário. `ppo_entity` (−0.75) ainda NÃO bate Producer → não promover.
    - ⚠️ **CRASHOU por OOM ~2026-06-07.** Só chunk 0 completou. `dmesg`: `Killed process (python) anon-rss:13GB`, `coredns` co-morreu (assinatura de pressão GLOBAL).
    - ✅ **DIAGNOSTICADO 2026-06-07 (skill /diagnose, com medição).** Causa-raiz = **overcommit da máquina, NÃO o `train_ppo`.** Medido: train_ppo na config idêntica da campanha (`rust_batch`, 16 envs, 2p, oep) usa **~1.6GB flat** (pico 1627MB através de múltiplos updates, sem inclinação → leak refutado). Box compartilhado: Docker (kairos-postgres+minio+desktop+coredns), 4×`claude`, zotero-mcp, `local_run/dtw_build.py` (~1GB), contra teto `.wslconfig` `memory=16GB`. O OOM foi spike transitório de algo a 13GB (run/eval antigo?), não o treino. **O sub-task antigo de baixar `--rollout-num-envs 16→8` está REFUTADO** — treino já cabe folgado; reduzir envs seria workaround cego.
    - [x] **Subir teto do WSL2** ✅ 2026-06-08. `free -h` = `Mem total 23Gi` / `Swap 16Gi` (alvo 24/16 atingido; WSL reporta ~1Gi a menos).
      - [x] verificar: `free -h` mostra `Mem total ~24Gi` após restart. ✅ 23Gi.
    - [x] **Fix de raiz no código: guarda de memória em `run_campaign.py`** ✅ 2026-06-08. `_wait_for_memory(min_free_gb, chunk)` + `--min-free-gb` (default 4.0) bloqueiam antes de cada chunk; `_mem_available_gb` lê `MemAvailable`, `inf` se ilegível (nunca trava em parse miss). Coberto por `tests/test_campaign_robustness.py` (25 passed).
      - [~] verificar: unidade ✅ (testes passam); integração ao vivo (`dtw_build.py`+docker → loga "low mem") NÃO exercitada — o caminho de espera é testado, mas não houve cenário de baixa memória real.
    - [x] **Campanha h* relançada (corrigida pelas lições do log)** ✅ 2026-06-08, PID 9364, `artifacts/ppo/campaign_h/`. **NÃO** resumida da `campaign/` quebrada (report `-Inf`, init colapsado). Config corrigida vs `campaign_report.json`: init=`ppo_entity_200k.pt` (best −0.6653, não `ppo_cliff_1M` colapsado), `ent 0.003` (não 0.01 que colapsa em treino continuado — EXP:241), currículo gradiente `producer_h30×2,h50×2,h70,greedy,producer,oep`, eval producer,oep 16s, 24×200k, keep-best, patience 5, min-free 4GB. Crash de parse anterior RESOLVIDO (h* registrados; 25 testes verdes). Monitorar `artifacts/ppo/campaign_h/{campaign_log.jsonl,best.pt,campaign_report.json}`.
      - [x] verificar: FALHOU ✅ avaliado 2026-06-08. chunk0 e chunk1 = margin −1.0, win 0.0 (400k ts). Campanha PARADA no chunk 1 (critério duro: não avançar p/ chunk 2). 3/4 critérios de desalinhamento disparam (margin/win, neutral_captures 2.84, EV 0.93 desacoplado). **Veredito: PPO current reward is misaligned** (registrado em `EXPERIMENTS.md` 2026-06-08). Producer segue como submissão operacional.
## 🔱 PÓS-MISALIGNMENT (decisão do usuário 2026-06-08) — DUAS FRENTES paralelas
> **Veto duro:** PPO bruto atual está MORTO. Não relançar `train_ppo` com a mesma ação (MultiDiscrete cru) nem o mesmo reward. Toda run nova passa por item `[ready]` com hipótese + verificação. Régua: 16→96 seeds, `crash/timeout/invalid=0`, early-stop por **margem/AUC externa, NUNCA por EV/reward interno**.

### Fundação comum (de-risca as duas frentes) — FAZER PRIMEIRO
- [x] **F0. Matriz Producer × OEP por seed (2p).** ✅ 2026-06-08 via `scripts/p4_matrix.py` (eval-only, instâncias isoladas, `_shares`+`normalized_margin`). Artefato `artifacts/p4/validate_2p_256_comets.json` (256 steps + comets, 16 seeds).
  - [x] verificar: ✅ win/margem por seed; regime OEP>Producer existe (5/16 seeds); Producer ganha overall (+0.282, 9/16); invalid=0. **Achado-chave:** o "OEP>>Producer" só aparecia em config curta (128/sem comets) — artefato; régua validada a 256/comets. Registrado em `EXPERIMENTS.md`.
  - [ ] **(opcional) 4p:** `p4_matrix` é 2p-only no loop; estender p/ N players quando a Frente A precisar de regime 4p. Não bloqueia B.

### FRENTE A — competitiva P4 (não-RL, piso rápido)
- [ ] **A1. Selector conservador Producer/OEP** com features por mapa/seed (da F0). Default = Producer (piso ~1200 LB); só troca p/ OEP em regime onde a matriz mostra vantagem clara.
  - [ ] verificar: selector ≥ Producer em 16 seeds (não regride no regime Producer), ganha no regime OEP; gate 16→96.
- [ ] **A2. Overlays pequenos e reversíveis** sobre o selector (cada overlay tem flag de off e é medido isolado).
  - [ ] verificar: cada overlay melhora margem pareada sem regressão; reversível por flag; crash/timeout/invalid=0.
- [ ] **A3. Gate 16→96 + 4p separado** antes de qualquer submissão.
  - [ ] verificar: margem ≥ 0 vs Producer a 96 seeds, sem regressão 4p, crash/timeout/invalid=0.

### FRENTE B — PPO REDESENHADO (selector de candidatos, não MultiDiscrete)

> **ESTADO (2026-06-08): linha CONCLUÍDA na PARIDADE com o Producer. Reward tuning ESGOTADO.**
> B1+B3+B4 levaram de −1.0 (colapso) a **0.00 seat-neutral vs Producer** (paridade), **+0.17 vs OEP**
> (bate), held-out **+1.0** (generaliza) → anti-overfitting CUMPRIDO. **Surpassar (>0) NÃO atingido.**
> Causa provada (3 rewards: PBRS, +ent, +terminal15×, todos teto na paridade): always-producer é
> ótimo-local robusto; não é reward/exploração/timesteps. Detalhe em `EXPERIMENTS.md`. Checkpoints:
> `artifacts/ppo/frente_b/{candidate_b1b3_150k,campaign_b4ent/best}.pt`. Evals honestos: `eval_candidate_seats` (seat-neutral); `eval_candidate_selector` é ENVIESADO (player-0), não usar.
>
> **RETOMAR POR UMA DAS DUAS FRENTES (multi-sessão, escolha do usuário — NÃO mais reward tuning):**
> - [ ] **Self-play / liga** (AlphaStar/PSRO): sinal de vitória vs oponentes EVOLUINDO + Producer/OEP sempre na pool + anti-esquecimento. Continua a linha PPO. verificar: novo campeão > paridade vs Producer a 96 seeds sem regredir vs pool.
> - [ ] **Busca / lookahead** (T8): bot que PLANEJA timeline/sim-value, não seleciona plano de expert. verificar: protótipo dentro de actTimeout bate paridade vs Producer a 96 seeds.

- [~] **B1. Espaço de ação = índice de candidato (com mask).** Candidatos por `producer/oep/greedy/defensive/rush/no-op`; a policy escolhe O ÍNDICE, não a 5-tupla crua. Substitui a `MultiDiscrete([2,16,32,4,5])` morta.
  - [x] **Candidate factory (keystone)** ✅ 2026-06-08. `python/agents/candidate_factory.py` (`CandidateFactory`, 6 candidatos, no_op no índice 0, fail-safe-to-pass, fresh isolated por env). Bug de isolamento pego por teste e corrigido: `registry.make_isolated_opponent` (instância fresca/chamada vs pool cacheado compartilhado). 5 testes em `tests/test_candidate_factory.py`; 26 verdes (factory+isolation+registry). (Nota: registry não tem `reinforce`; usei `defensive` como defend.)
  - [x] **Env mode** ✅ 2026-06-08. `OrbitWarsGymEnv(action_mode="candidate")`: `action_space=Discrete(6)`, `step` decodifica índice→moves do candidato via factory; `raw` (MultiDiscrete) intacto. 5 testes em `tests/test_env_candidate_mode.py` (episódios limpos por índice, no-op passa, raw inalterado, modo inválido levanta). 10 verdes c/ factory.
  - [x] **Selector head na policy** ✅ 2026-06-08. `CandidateSelectorActorCritic` em `policy.py` (encoder entity reusado + cabeça única `Discrete(K)` + value; Categorical mascarada; sem a lógica launch-gated dos 5 heads; `policy.py` segue sem importar registry/bots). 5 testes em `tests/test_candidate_selector_policy.py` (mask exclui, entropia ≤ log(allowed), logprob reproduzível p/ ratio PPO).
  - [x] **arch registrada** ✅ 2026-06-08: `candidate_selector` em `_POLICY_ARCHS`; `_build_policy('candidate_selector', obs_dim)` instancia (112.6k params). 40 testes verdes (factory+env+policy+isolation+registry+entity).
  - [x] **train_ppo rollout/update FIADO** ✅ 2026-06-08 (4 pontos, branch por `isinstance(model, CandidateSelectorActorCritic)`): (1) dispatcher força single-env p/ candidate; (2) `build_phase0_env` aceita `action_mode`; (3) single-env rollout usa mask all-True `(K,)` + ação escalar; (4) `_ppo_update` usa `{"candidate": mask}` em vez de `split_masks`; label `rollout_backend` corrigido. 2 testes de plumbing (`tests/test_train_candidate_integration.py`) + 15 de phase0 (regressão flat/entity intacta) = 17 verdes. **Frente B roda ponta a ponta em escala de smoke.**
  - [x] **SMOKE RUN B1+B3** ✅ 2026-06-08. candidate_selector + dense_potential, producer,greedy, 50k ts CPU. Eval greedy 8s/256: **vs Producer −0.25 (win 0.375), vs defensive +0.996, vs rush +0.999** (held-out, generaliza). NÃO colapsou; expande (12 neutros); EV/entropy sadios. **Supera o teto histórico −0.6653 a 1/4 dos ts. Frente B validada.** (Registrado em EXPERIMENTS.)

- [~] **B-EXT. Estender o treino B1+B3 para cruzar margem 0 vs Producer E OEP.**
  - [~] **+100k producer,greedy (150k cumulativo)** RODANDO (PID 72250, ~80 min). Warm-start do 50k (−0.25). Single-run (sem checkpoint intermediário — por isso o chunked abaixo).
  - [x] **Driver chunked com OS 3 OPONENTES** ✅ 2026-06-08: `scripts/run_candidate_campaign.py` (decisão do usuário: treinar vs `producer,oep,greedy` — OEP no nível do Producer, estilo distinto → anti-overfitting). Train chunk (subprocess) → eval in-process vs `producer,oep,defensive,rush` → **keep-best por `min(margem_producer, margem_oep)`** (campeão tem que bater OS DOIS) → continua do best. Fecha o gap de checkpoint (por-chunk) + dá trajetória. `--help` OK; compõe peças já validadas (train candidate+dense, eval in-process). **NÃO lançado ainda (competiria CPU com o 150k).**
  - [x] **150k avaliado** ✅ 2026-06-08: **vs OEP +0.079 (win 0.5) — BATE o 2º melhor!**; vs Producer −0.25 (estagnou, igual 50k); held-out +0.98/+0.99. EV colapsou 0.76→−0.03 → single-run cego é frágil (não melhorou Producer). **Metade do goal (OEP) feita; Producer é a barra dura.**
- [ready] **B-CAMP. Campanha chunked 3-oponentes p/ cruzar o Producer.** `run_candidate_campaign --init candidate_b1b3_150k.pt` (treina producer,oep,greedy; eval producer,oep,held-out; keep-best por `min(prod,oep)`; early-stop patience 3). Dry-run validado antes do real.
  - [ ] verificar: margem vs Producer melhora sobre −0.25 sem regredir vs OEP (que já é +0.08); se `min(prod,oep)≥0` com folga → 96 seeds (decisão do usuário).
  - [ ] se Producer seguir estagnado: **B4 (BC-anchor)** — o EV colapsado (−0.03) motiva âncora ao BC p/ estabilizar value/policy.
- [ ] **B2. Features = entity/canonical + features de candidato** (valor projetado, ETA, alvo, custo de fonte por candidato).
  - [ ] verificar: features de candidato entram na policy; permutation-invariance dos candidatos testada.
- [x] **B3. Reward DENSO auditável (potential-based)** ✅ 2026-06-08. `OrbitWarsGymEnv(reward_mode="dense_potential")`: `F = γ·Φ(s')−Φ(s)` (Ng/Harada/Russell 1999, policy-invariant); Φ = 0.4·prod_share + 0.3·ship_share + 0.3·planet_share (contestadas, ∈[0,1]); colapso (0 planetas) → Φ=0; fleets in-flight no ship_share; componentes logados em `info["dense_potential"]`; terminal_win segue como reward terminal. `reward_mode` fiado em `Phase0TrainingConfig`→`build_phase0_env`→single-env rollout. 6 testes (`tests/test_dense_reward.py`: PBRS exato, Φ∈[0,1]~0.5 no start, colapso→0, legacy intacto). **B1+B3 combinados treinam ponta a ponta: `mean_return=−0.014` (~empate) vs −9.1 do reward antigo** — sinal são, bounded (smoke 32 ts, não conclusivo de margem).
  - [ ] verificar (precisa de run): correlação componente↔margem externa num treino real.
- [ ] **B4. BC-anchor durante o PPO** (CE/KL contra expert) p/ preservar competência imitada.
  - [ ] verificar: ablação mostra que o anchor evita o colapso −1.0 visto no campaign_h (margem não despenca em 200k).
- [ ] **B5. Currículo adaptativo h30-first.** h30 até foothold forte, depois h50, h70; full Producer só como probe.
  - [ ] verificar: progressão de degrau só com foothold medido (margem por degrau), não por timesteps.
- [ ] **B6. Early-stop por margem/AUC externa, NUNCA por EV/reward interno.**
  - [ ] verificar: gate bloqueia promoção com base em margem pareada; EV/reward interno proibidos como critério de parada.

- [ ] **P4. Hall-of-fame/self-play só depois de sair do buraco.**
  - [ ] **Pré-condição:** só iniciar quando melhor PPO/BC tiver `mean_score_margin > -1.0` vs Producer e alguma vitória/empate real em triagem.
  - [ ] **Pool:** manter sempre Producer e OEP na pool; adicionar checkpoints próprios promovidos por margem pareada, não por reward de treino.
  - [ ] **Sampling:** limitar self-play para não virar monocultura; cada batch precisa conter uma fração mínima de Producer/OEP.
  - [ ] **Antiesquecimento:** rodar eval contra snapshots antigos antes de substituir o campeão.
  - [ ] **Correto quando:** novo checkpoint melhora contra versões próprias sem regredir materialmente contra Producer/OEP; se regredir, volta para currículo com mais expert.

- [x] **P5. Ablation de map bias/features.** ✅ **CONCLUÍDO 2026-06-07** — auditoria + tratamento (augmentation) feitos, verificados e fechados. Critério atingido; **não há trabalho morto pendente aqui** (o gargalo restante é representação/decoder → migrado para T5).
  - [x] **Teste de invariância:** `tests/test_map_bias_invariance.py` + `python/orbit_wars_gym/symmetry.py` (rotate_180/reflect_x/swap_players, involuções testadas) + `scripts/audit_map_bias.py` (gap de logits por head/transform).
  - [x] **Baseline:** `artifacts/map_bias/invariance_report.json` — perspectiva gap 0.0; rotate_180 2.5–5.6; reflect_x 1.7–3.6 (viés espacial de x/y absolutos + planet_id).
  - [x] **Tratamento escolhido (data augmentation):** `--augment` no collector; `artifacts/map_bias/invariance_report_aug.json` — gap caiu ~5–10×, margem online melhorou (−0.915→−0.880). Ver SEGUIR AGORA acima.
  - [x] **Decisão sobre as duas opções (augmentation vs canonicalização):** augmentation atingiu o critério → **canonicalização dispensada** (não vira trabalho a fazer). Documentado; reabrir só com evidência nova.
  - [x] **Benchmark BC da variante:** bc_producer (baseline) vs bc_producer_aug comparados em loss/F1/margem online (aug melhor: launchF1 0.16→0.62, margem −0.915→−0.880).
  - [x] **Correto quando:** ✅ variante (aug) reduziu sensibilidade a simetrias **sem piorar** o benchmark pareado (na verdade melhorou). **Resíduo NÃO é simetria:** target head segue fraco (0.09) mesmo com aug → problema de representação/decoder → **endereçado em T5**, não em P5.

- [ ] **P6. Gate de promoção PPO.**
  - [ ] **Gate 1 triagem:** 16 seeds vs Producer/OEP + sanity contra `greedy,rush,anti_meta`; usar só para decidir se vale 96 seeds.
  - [ ] **Gate 2 decisor:** 96 seeds pareadas vs Producer e OEP, com cometas ligados e registro 2p/4p separado. **Pré-condição de custo: só rodar 96 seeds quando o candidato superar `-0.7491` com FOLGA na triagem de 16 seeds** (antes disso é desperdício de compute).
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

- [~] **T5. Redesign do target/action head** (gargalo de representação/decoder). **Escopo separado:** masking BÁSICO já está FEITO em P1.5f; T5 é o que falta além dele.
  - [x] **Masking básico (FEITO em P1.5f):** launch/source/target mascarados (launch=0 sempre válido; launch=1 só se há lançável; aplicado no sampling E no PPO update, paridade no export). Não refazer.
  - [ ] **Máscaras semânticas (NOVO):** além das óbvias — alvo próprio em modo ataque, fração que viola reserva mínima, offset que leva a sol/borda quando detectável.
  - [ ] **Target por pointer/entidade válida (NOVO, núcleo do T5):** trocar o `target_rank` quantizado por pointer sobre planetas válidos; é o gargalo medido (target acc `0.07–0.12` flat, `0.185` entity, ainda baixo).
  - [ ] **Action space top-k por ETA/valor (NOVO):** restringir candidatos a poucos alvos de alto valor antes do head decidir.
  - [ ] **Cabeça autoregressiva (NOVO, opcional):** source→target→frac→offset, condicionando cada head no anterior.
  - [ ] **Auxiliary head `intent`:** (`no-op`, `capture`, `attack`, `reinforce`, `regroup`, `evacuate/comet`) para estabilizar representação.
  - [ ] **Testes:** logprob/entropy com máscara e não-geração de ação impossível em estados sintéticos.
  - [ ] **Correto quando:** **target acc ONLINE** melhora vs entity atual (`0.185`), invalid/sem-efeito cai, entropy não colapsa, e benchmark pareado melhora ou mantém margem vs Producer/OEP.

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

## 🧬 FAMÍLIA H — CandidateFactory + hiper-heurística (nova linha heurística, 2026-06-07)

> **Distinção crucial (correção de enquadramento):** o que saturou foi a linha ESTREITA
> *"Producer/OEP + seleção/rollout sobre os mesmos tipos de plano"* (E1/E2/C1/E3). **NÃO** saturou
> o espaço de heurísticas / metaheurísticas / hiper-heurísticas. O gargalo medido é o **gerador de
> candidatos pobre**, não o seletor: E3 (candidatos diversos + rollout, 96 seeds) caiu p/ `-0.201`
> < OEP-best `-0.045`; C1b escolheu alternativa em 3/232 decisões (`choice_rate=0.013`), C1a em 1/43
> (`0.023`) — perturbação mínima do guloso, não exploração real. **Próximo ganho heurístico = melhorar
> o GERADOR, não o seletor** ("o seletor está escolhendo entre facas quase iguais; é preciso pôr armas
> diferentes na mesa"). Força da evidência: **forte** (painel 96-seed + `choice_rate` medido).
> Literatura: hyper-heuristics buscam no espaço de COMPONENTES heurísticos, não nas soluções cruas
> ([JORS survey][h1]); RHEA evolui sequências curtas de ação por tick e executa a 1ª da melhor
> sequência (Gaina et al., *Rolling Horizon Evolutionary Algorithms*); abstrações de ação reduzem o
> espaço a subconjuntos promissores em jogos multiagente de ação grande ([Marino et al.][h3]). Força
> p/ a forma exata em Orbit Wars: **parcial**.
>
> **Relação com o backlog existente:** H6 + H2–H5 são a versão sistematizada-com-gate de `T8` (planner
> timeline standalone); H7 é a de `T9` (seletor/MoE). Rodar a família H, **não** T8/T9 em paralelo.
> Registrar como família NOVA no `EXPERIMENTS.md` (`H1`, `H2`…), não como continuação de E/C.

- [ ] **H1. CandidateFactory + oracle de candidatos (PRIMEIRA PEÇA — pré-req de toda a família).**
  - [ ] **Interface:** `CandidateFactory` retorna vários `PlanCandidate{família, ações, custo_ms, features}`. Producer e OEP viram DOIS candidatos entre muitos, não o centro do universo.
  - [ ] **Oracle diagnóstico (gate da família):** medir `oracle_best_margin` = margem do MELHOR candidato por estado (escolha perfeita) vs `selected_margin` (seletor real).
    - [ ] verificar (regra de decisão): `oracle_best_margin <= baseline` → **problema é o GERADOR** (candidatos novos não prestam; seletor não salva); `oracle_best_margin > baseline` E `selected_margin <= baseline` → **problema é o SELETOR/avaliador**. Sem essa medição, metaheurística vira roleta cara.
  - [ ] **Correto quando:** o oracle separa as duas hipóteses em 16 seeds vs Producer; só seguir se houver "sangue" nos candidatos (`oracle_best_margin > -0.045` com folga). Caso contrário, é decorar o mesmo cadáver.
- [ ] **H2. Família `production_projected_attack`.** Alvo avaliado no ETA: produção acumulada, ships esperados, defesa provável, valor pós-captura.
  - [ ] verificar: entra no oracle e aparece entre os top por `oracle_best_margin` em ≥1 fase de jogo.
- [ ] **H3. Família `timeline_risk` (fitness).** Mesmo alvo + penalidade por cometa, colisão, chegada tardia, overkill, perda de fonte e janela orbital ruim.
  - [ ] verificar: timeline-risk melhora `selected_margin` vs o fitness 1-ply atual em 16 seeds, sem subir invalid/timeout.
- [ ] **H4. Macro-genes `hammer` / `multiprong`.** hammer = fontes sincronizadas no mesmo alvo (reduz defesa incremental); multiprong = duas ameaças simultâneas forçando divisão de resposta.
  - [ ] verificar: gera candidato não-guloso escolhido com `choice_rate` materialmente > 0.05 (vs 0.013/0.023 do beam/memória) E margem ≥ baseline.
- [ ] **H5. Famílias `regroup_dominance` + `bait_and_flip` (+ `anti_leader_4p` / `third_party_4p`).** regroup = reforçar fronteira dominante em vez de atacar; bait = ataque pequeno p/ induzir resposta + captura secundária; 4p = atacar líder de produção/território e capturar planeta enfraquecido pós-conflito alheio.
  - [ ] verificar: aparecem no oracle e reduzem derrotas por overextension/contra-ataque (cruzar com replay-mining `T1` quando existir).
- [ ] **H6. RHEA sobre MACRO-ações (não MCTS cru).** `pop 8–24`, `horizon 3–6 macro-decisões`, `budget 80–180ms`; seed inicial = Producer + OEP + melhor genoma anterior *shifted*; mutação troca família/alvo/fração/delay/fonte/sync; fitness = timeline-risk + produção projetada + segurança de fonte + domínio territorial.
  - [ ] verificar: roda dentro de `actTimeout=1s` (p95/max medidos, folga ≥3×) E supera o `oracle_best_margin` do conjunto ESTÁTICO (prova que MELHORA candidatos, não só escolhe entre os antigos).
- [ ] **H7. Hiper-heurística contextual (offline-otimizada, runtime barato 2p/4p).** Tabela/árvore simples: features (`phase`, `player_count`, `ship/production/planet balance`, `enemy_frontier_pressure`, `comet_risk_near_owned`, `nearest_capture_eta`, `leader_gap_4p`, `volatility`) → output (pesos+budget por família, threshold de agressão, modo `attack/regroup/defend/opportunistic`). **Otimizar OFFLINE** (random search agressivo / NTBEA / racing / bandit por família) sobre o genoma (`attack_weight`, `production_eta_weight`, `source_safety`, `overkill_penalty`, `comet_penalty`, `regroup_trigger`, `hammer_sync_window`, `multiprong_split_ratio`, `early_aggression`, `late_regroup_bias`, `leader_attack_4p`, `rhea_population`, `rhea_horizon`, `mutation_rate`). Rodar BARATO na submissão.
  - [ ] verificar: o seletor decide QUAL gerador merece compute por estado (não só "OEP ou Producer"); o preset sobrevive à escada e bate o gate (abaixo).

**Escada de avaliação da família (régua inalterada):** `4 seeds` = legalidade/perfil; `16` = triagem; `32` = pré-gate (CI, não ponto); `96` = promoção (`margin ≥ 0 vs Producer`, sem regressão vs OEP, gate 4p separado antes de confiança TOP-5). Producer = piso; OEP = 2º arquétipo.

**Não fazer dentro de H (já matado — não reabrir):** continuar E1/E2 como seletor (E2a `-0.033`, assimétrico/diagnóstico); só aumentar o beam do 1º lance (confirma guloso + custo); usar 4 seeds como sinal de QUALIDADE; otimizar contra greedy/rush como meta (são sanity técnico, não régua).

[h1]: https://link.springer.com/article/10.1057/jors.2013.71 "Hyper-heuristics: a survey of the state of the art (JORS, 2013)"
[h3]: https://webdocs.cs.ualberta.ca/~santanad/papers/2019/marinoMTL19.pdf "Evolving Action Abstractions for Real-Time Planning in Extensive-Form Games (Marino et al., 2019)"

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

# 🧭 DIREÇÃO 2026-06-07 — OEP esgotado; P3 anti-drift é a frente viva

> Síntese de doc de 2026-06-07. **Estado** e **Parado** são contexto (não viram task);
> **Eixos de ataque** são checklist ordenado por ganho líquido, com critério de verificação.
> `mean_score_margin` pareado é a moeda; métrica interna de PPO (EV/KL/clipfrac) não é.

## Estado real (não é task)
- OEP como "melhor resposta sobre Producer" está **esgotado**: E1/E2/C1/E3 saturaram ou regrediram.
- Self-play fraco (3 ger, ~393k ts) **não transferiu**: `margin=-1.0` vs Producer.
- Melhor avanço recente = **BC → entity PPO curto**, não mais busca OEP.
- **Melhor checkpoint vivo:** `ppo_entity_200k.pt` → `margin=-0.6653` (best por margem pareada; o 120k era −0.7491). Anti-drift deve **continuar do BEST (−0.6653)**, não do 120k nem do último — a continuação +200k com ent 0.01 COLAPSOU p/ −1.0. Ainda **não** bate Producer.
- **+500k ts REGREDIU** `-0.749 → -0.848` com EV/KL/clipfrac "saudáveis" → prova: seleção externa por jogos pareados é obrigatória ([arXiv][1]).
- Infra pronta (masking, BC-init, instrumentação, currículo, export arch-aware); falta **disciplina de seleção**, não plumbing.

## 🚫 Parado / não explorar — adendo 2026-06-07 (salvo como diagnóstico)
- Thresholds OEP, "baratear" 2º Producer, tensor/inline Producer, beam raso de 1º lance, plano-memória, rollout search raso, ordinal K=3/K=5, cortes por deadline, cortes fixos de fontes/alvos.
- **Treino PPO simplesmente mais longo** (matado pela regressão +500k).
- **Paralelizar o planner do opponent em Python** — GIL mata threads, IPC mata processos (já matado no registro).

## 🎯 Eixos de ataque (ordenados por ganho líquido)
Ganho líquido máximo agora = **P3 anti-drift + redesign do target/action head**. OEP incremental virou areia movediça: consome tempo, produz falsos positivos e não cruza Producer.

- [ ] **1. P3.1 — PPO anti-drift a partir do `ppo_entity 120k` (MAIOR ganho).** Treino em blocos curtos, avaliar a cada bloco, salvar o melhor por margem pareada, early-stop no primeiro drift. Campanha **não** é "mais timesteps".
  - Evidência: +500k regrediu `-0.749 → -0.848` apesar de curvas sadias → métrica interna ≠ vitória ([arXiv][1]).
  - [ ] verificar: melhor checkpoint (selecionado por margem pareada) supera `-0.7491` em triagem 16 seeds, sem crash/timeout/invalid; early-stop dispara quando a margem piora.
- [ ] **2. P3.2 — preservar a competência imitada.** Ablar `ent_coef` menor, KL-to-BC e loss auxiliar de imitação; revisar shaping para alinhar com `mean_score_margin`.
  - [ ] verificar: ablação registrada mostra qual config mantém margem ≥ baseline E reduz o drift visto no +500k.
- [ ] **3. T5 — reformar o target/action head (gargalo de representação/decoder).** Target por pointer/entidade válida, máscara agressiva de planetas inviáveis, action space top-k por ETA/valor, eventual cabeça autoregressiva source→target→frac→offset.
  - Evidência: target fica `0.07–0.12` no flat, `0.185` no entity (ainda baixo); augmentation melhorou margem BC p/ `-0.8803` mas **não** corrigiu target; mais epochs de BC não resolve.
  - [ ] verificar: target acc **online** melhora vs entity atual (`0.185`) E a margem pareada não regride.
- [ ] **4. Campanha longa SÓ com eval-gating.** Usar batching já aceito (16 envs RTX 5060 Ti ~265 SPS, 1.8× CPU; 10M ts ≈ ~10.5h); reduzir custo do opponent por currículo; checkpoints + eval em janelas curtas. **Não** tentar paralelizar o planner do opponent.
  - [ ] verificar: campanha overnight roda com eval periódico + seleção/early-stop ligados; nenhum checkpoint promovido sem margem pareada medida.
- [ ] **5. Busca standalone sim-value/timeline (rota não-RL).** Reescrever o bot como busca própria: timeline, produção projetada, risco de snipe, redistribuição/dominância, genes `hammer/multiprong`, 2–4 ações de alto valor. **Não** herdar a seleção OEP saturada. Orbit Wars é 1v1 ou 4p FFA ([Kaggle][4]).
  - [ ] verificar: protótipo roda dentro de `actTimeout=1s` e bate o `mean_score_margin` do OEP em triagem 16 seeds vs Producer.
- [ ] **6. Liga/PBT real (não self-play pequeno), depois de sair do buraco.** Modelo tipo AlphaStar: imitação → liga com competidores congelados + exploiters + diversidade anti-forgetting ([DeepMind][2]); PBT explora/explora hiperparâmetros copiando pesos de melhores e mutando ([DeepMind][3]).
  - [ ] verificar: liga mantém Producer/OEP sempre na pool e novo campeão melhora vs versões próprias sem regredir contra Producer/OEP.
- [ ] **7. Gate 4p real (só depois dos acima).** Hoje 4p é sanity (`--skip-4p`). Como a competição tem 2p e 4p ([Kaggle][4]), candidato que passa 2p precisa de gate 4p separado: lineups mistos, anti-snowball, comportamento contra dois inimigos no líder, sobrevivência/produção por fase.
  - [ ] verificar: gate 4p separado existe e o candidato que passa 2p não regride material em 4p.

## 🔧 Disciplina de processo (pré-requisito dos eixos acima)
- [ ] **Transformar P3 em loop automático:** treino curto + benchmark pareado + seleção de checkpoint.
  - [ ] verificar: a cada bloco de N updates, exporta + benchmarka vs Producer/OEP e persiste o melhor por margem pareada.
- [ ] **Registrar curva de margem por checkpoint** (não só `last_*`).
  - [ ] verificar: artefato com `mean_score_margin` por checkpoint legível/plotável.
- [ ] **Bloquear promoção se `timeout/fallback/invalid > 0`** e nunca aceitar "curvas saudáveis" sem margem.
  - [ ] verificar: gate falha barulhento quando qualquer um desses > 0, independente de EV/KL.

[1]: https://arxiv.org/abs/1707.06347 "[1707.06347] Proximal Policy Optimization Algorithms"
[2]: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning — Google DeepMind"
[3]: https://deepmind.google/blog/population-based-training-of-neural-networks/ "Population based training of neural networks — Google DeepMind"
[4]: https://www.kaggle.com/competitions/orbit-wars?utm_source=chatgpt.com "Orbit Wars"
