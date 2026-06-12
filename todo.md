> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `EXPERIMENTS.md` — este arquivo fica enxuto, só com o que atacar e o estado atual.

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

## ⚠️ CORREÇÃO METODOLÓGICA (2026-06-08) — avaliar a 500 steps + ambos assentos
Usuário perguntou se confirmei vs o que o Kaggle usa. **Eu não tinha.** Corrigido:
- [x] **Paridade do motor CONFIRMADA:** instalei extra `kaggle`; `parity_probe_actions` (2p/4p/comets/janela tardia)
  + suíte oficial = **19 passed, exato**. RustBatchBackend é fiel ao Kaggle. Mira do bot transfere.
- [x] **Comprimento estava ERRADO:** medi H a 96/128 steps; **oficial = 500**. A 500 (ambos assentos, n=12):
  **OEP vs Producer = 0.0 (EMPATA, não +0.07)**, eval greedy = −1.0. `producer-mirror` seat0 = +0.50 (viés de
  assento grande → obrigatório mediar 2 assentos). **Conclusão real: Producer é o melhor bot @500** (bate com
  LB 1228>1100). Anchors a 96 steps eram artefato de jogo curto. Memória [[kaggle_500_step_eval_required]].
- [ ] **NOVA RÉGUA p/ qualquer claim de força:** `--episode-steps 500` + ambos assentos. 96 steps só p/ smoke.

## 🅗 FAMÍLIA H — B→A: primitiva de DEFESA + tuning metaheurístico da eval-function
**Trilha heurística/metaheurística deste worktree (`feat/family-h-candidate-factory`).** Reabre a linha H
NUM QUADRANTE QUE A SATURAÇÃO NÃO COBRIU. A saturação (EXPERIMENTS.md 77-82) esgotou *seleção
hiper-heurística em runtime sobre primitivas ofensivas fracas*; a causa DOMINANTE diagnosticada é
**fraqueza estratégica — sem defesa/recapture** (commit c30b5d3). Literatura que sustenta a rota:
Gaina, Devlin, Lucas, Perez-Liebana 2020 (arXiv:2003.12331) — o ganho de SOTA em jogos em tempo real
vem de **otimizar OFFLINE os parâmetros** de UM agente rico (N-Tuple Bandit EA), não de selecionar
heurísticas em runtime (**FORTE**); Martin & Collins 2026 (arXiv:2601.09594, AS-CMA) — CMA-ES com
amostragem adaptativa para **fitness ruidosa e cara** (**FORTE** para o método). Reenquadramento
honesto: a eval-function tunada por evolução É a *value function aprendida* que a conclusão PPO pediu
para "realizar o +0.048" — mesmo alvo, sem RL. Risco principal a vigiar: custo de fitness (N seeds ×
jogos por avaliação CMA-ES) — multi-sessão; mitigado por amostragem adaptativa/EA amostra-eficiente.

### Lever B — primitiva de DEFESA/recapture/reforço-sob-ameaça (pré-requisito; ataca a causa dominante)
Toda família H atual é OFENSIVA. `_projected_defense` (family_h.py:94) só é usado para *atacar*
(quanto preciso para capturar), nunca para *defender o que é meu*. Construir e MEDIR isolada.
- [x] **B1. Família `defensive_reinforce` em `bots/oep/family_h.py`** ✅ 2026-06-08. 3 missões (reforço-sob-ameaça
  via proxy de ameaça `_incoming_threat` + frotas in-flight; recapture de inimigos fracos no `RECAPTURE_RANGE`;
  redistribuição de interior calmo p/ fronteira). Toda mira pelo `Aimer` engine-accurate. Registrada.
  - [x] verificar: ✅ `pytest tests/test_family_h.py` = 16 passed (legalidade 2p/4p + board vazio).
- [x] **B2. Benchmark `defensive_reinforce` como BOT COMPLETO** ✅ 2026-06-08. `b2_defensive_reinforce.json`:
  margin vs producer+oep = **−0.9975** (crash/timeout/invalid=0). Trace: emite moves só sob pressão real
  (10/128 turnos vs Producer); 0/128 vs oponente passivo.
  - [x] verificar (CRITÉRIO DURO de B): **NÃO cruzou.** Defesa isolada segue −1.0 → **defesa pura é degenerada**
    (nunca captura neutros, não expande, é moída). Resultado ESPERADO pelo plano: o valor só aparece COMBINADA →
    a primitiva de defesa vira TERMO da eval-function (A). Registrar em EXPERIMENTS.md.

### Lever A — UMA eval-function parametrizada, pesos tunados OFFLINE vs Producer (CMA-ES / N-Tuple Bandit)
Substitui o seletor-de-famílias por UMA política de scoring sobre candidatos, com ~10-20 termos pesados
(valor de ataque, defesa/reforço, recapture, economia/produção, anti-overcommit, eta/distância). Os PESOS
são o genoma; o otimizador metaheurístico os ajusta direto contra o Producer. NÃO começar A antes de B.
- [x] **A1. Eval-function parametrizada `make_eval_policy(weights)` em `bots/oep/family_h.py`** ✅ 2026-06-08.
  Genoma de 12 pesos (`EVAL_WEIGHT_NAMES`): ofensiva (prod/defense/eta/enemy_denial/comet/overkill/overextend/
  consolidate) + defesa do B (reinforce) + params (reserve/capture_margin/min_score). Alocador greedy 1-shot.
  Registrada como `eval_function`. `EVAL_DEFAULT_WEIGHTS` = baseline a mão.
  - [x] verificar: ✅ legal 2p/4p (19 passed), benchmark crash/timeout/invalid=0. **Baseline default: vs greedy
    +0.90, vs rush +0.92 (ESMAGA os fracos — muito > H8 best_by_value +0.02/+0.28), vs Producer −0.87** (perde:
    greedy de 1 turno não tem lookahead do planejador). Bot FUNCIONAL, gargalo = só os fortes.
- [x] **A2. Harness de tuning offline `scripts/tune_eval_weights.py`** ✅ 2026-06-08. ES separável `(μ/μ_w,λ)`
  dependency-free (cma não instalado; evitei nova dep — decisão sua); fitness = `normalized_margin` vs Producer
  (mesma métrica do gate), 2 assentos por simetria; holdout disjunto p/ anti-overfit; `content_hash`. **jobs=1
  sequencial** (você pediu sem paralelizar — pool causaria contenção).
  - [x] verificar (smoke 3 gen/3 seeds): ✅ MELHORA o default — train −0.897→−0.830, holdout −1.0→−0.985.
    ES sobe a cada geração. Deflação train↔holdout 0.154 (3 seeds = ruído; A3 usa mais seeds). crash/timeout/invalid=0.
  - [~] **Run sério EM ANDAMENTO** (background, 1 core): pop=8, gen=10, train 0-3, holdout 100-103, steps=96
    → `a2_producer_seq.json`. Estabelece o teto REAL do greedy-1-turno vs Producer (smoke sugere ~−0.83).
- [x] **A3. Validação OOS dos pesos tunados** ✅ 2026-06-08 — **CRITÉRIO NÃO CRUZADO (teto estrutural).**
  Holdout disjunto: tuned melhora train (-0.923→-0.818) mas holdout fica -1.0 (deflação +0.182 = overfit).
  Varredura de configs em 8 holdout seeds vs Producer: default -0.971, tuned -0.983, defensivo -0.993,
  balanceado -0.992 → TODO o espaço de pesos colapsa em ~-0.97..-0.99. **Não é overfit/tuning, é TETO da
  classe greedy-1-turno.** Registrado em EXPERIMENTS.md.
- [x] **A4. Decisão de promoção** ✅ **NÃO promover** — eval_function não bate Producer (teto estrutural).
  É forte vs fracos (+0.90) mas isso não basta p/ Top-5. Melhor bot heurístico segue OEP (+0.071 vs Producer).
- [ ] **DECISÃO DE ARQUITETURA (usuário):** o único lever heurístico que bate Producer é LOOKAHEAD/rollout
  (eval_function como leaf-value de um rollout que modela a resposta do oponente). Isso ≈ reconstruir OEP
  (que já existe e já bate Producer +0.071). Opções: (a) aceitar OEP como teto heurístico e voltar à linha
  PPO/critic; (b) investir multi-sessão num rollout-bot novo com a eval_function como leaf. `tune_eval_weights`
  fica pronto p/ tunar o leaf-value se (b).
  - [ ] verificar: usuário decide (a) ou (b); não é tuning autônomo.

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
