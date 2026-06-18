# Loop autônomo de treino→decisão (literatura-fundamentado)

Loop: **treinar → diagnosticar erros que passaram → corrigir UMA alavanca → retreinar → convergir**.
Fundamentado em duas coleções Zotero curadas (2026-06-17):
- **"Orbit Wars — Competitive Game AI"** (`3J3XKXM2`) → backbone de decisão (abaixo).
- **"Kaggle/Orbit Wars"** (`LQSSTX8I`) → eixo P: predição de trajetória/interceptação (tracking/órbita).

Força da evidência marcada **[forte/parcial/fraca]** para ESTE problema (jogo competitivo, sim exato, actTimeout 1s, campo privado 54% 4p). Base: abstracts+tags+conhecimento canônico (PDFs não sincronizados no storage local desta máquina em 2026-06-17).

---

## Eixo M — a meta-régua do loop (como treinar/avaliar/decidir)

### 1. Avaliação — o fix nº1 (a régua não prediz o LB)
- **Balduzzi, "Re-evaluating Evaluation" / Nash averaging** (`HJJAI6Q6`) **[forte, on-point]**.
  Rankear por média-vs-pool é enviesado por agentes redundantes/fracos — é LITERALMENTE o nosso
  Spearman −0.6 ([[local_league_is_submission_gate]]). **Fix:** agregar a régua pela **Nash averaging**
  da matriz payoff agente-vs-agente (maxent Nash da matriz antissimétrica de win-rate) → rating
  invariante a redundância producer-lineage. Implementação: `scripts/nash_eval.py` (núcleo puro +
  testes); requer round-robin referência-vs-referência (cacheável, refs fixas) + linha/coluna do candidato.
  **Convergência do loop passa a ser definida pelo rating Nash, não pela média vs holdwave.**

### 2. Pool de oponentes / currículo — não overfitar
- **PSRO (Lanctot et al.)** (`Z8M6IE9Z`) **[forte]**: InRL overfita oponentes (joint-policy correlation).
  Best-response a uma MISTURA meta-Nash da população, crescendo o pool (double-oracle).
  **Regra do loop:** cada iteração ADICIONA o melhor checkpoint anterior ao pool e amostra oponentes
  pela meta-Nash (≈ `--pfsp` da campanha).
- **NFSP (Heinrich & Silver)** (`KJ7W3D3I`) **[parcial — imperfect-info, princípio transfere]**:
  misturar best-response + política média evita perseguir/ciclar em não-transitividade.
- **AlphaStar (Vinyals et al.)** (`CISW4DJH`) **[forte — análogo mais próximo]**: SL-init→RL + liga com
  **main agents + main exploiters + league exploiters** + PFSP. **Regra:** pool com 3 papéis —
  incumbentes (producer/holdwave), **exploiters** (rusher/bigwave/proxies elite), e os próprios bests
  passados. Casa com [[manual_exploiters_insufficient]] (exploiter precisa ser do pool, não ad-hoc).

### 3. BC + RL ancorado — o double-bind do KL
- **InstructGPT (Ouyang)** (`AAV4UFF2`) + **Stiennon "summarize from HF"** (`IZRPPQN3`) **[forte no mecanismo]**:
  KL→referência é o controle anti-drift/anti-reward-hack, mas é **double-bind**: forte demais → colapsa na
  referência (Producer/BC, sem ganho — exatamente [[ppo_two_structural_ceilings]]); fraco demais → drift e
  reward-hack (faults). **Diagnóstico→fix:** estagnou ≈BC/Producer (KL baixo, margem sem subir) →
  **reduzir** `kl_to_ref_coef`/`bc_anchor_coef`; drift com faults/ilegal (KL alto) → **aumentar**.
  0.05 é só ponto de partida; tunar NESSE eixo, uma direção por iteração; considerar **schedule** (anneal).
- **DAgger (Ross et al.)** (`9HHTU4Q5`) **[forte]**: BC puro compõe erro O(T²ε) ao longo dos 500 steps →
  BC-init sozinho degrada no late-game. **Assinatura:** falha concentrada em step 300+ / estados fora da
  distribuição dos experts → é erro composto → fix = mais correção on-policy (RL / relabel DAgger dos
  estados visitados pela política, usando producer/holdwave como expert), NÃO mais épocas de BC.
- **GAIL (Ho & Ermon)** (`5W7ZTV8K`) **[parcial]**: se BC+PPO platôa, imitar a OCUPAÇÃO (distribuição)
  adversarialmente em vez de casar ações — alavanca de reserva, maior variância.

### 4. O penhasco reativo-vs-lookahead — diagnóstico do teto
- **Mirrokni et al., "Implications of Lookahead Search"** (`MAC5TDEM`) **[parcial — teórico, outros jogos]**:
  agente reativo (0-lookahead) é sistematicamente explorável por oponentes com lookahead. É a explicação
  teórica do [[drl_reactive_planner_cliff]] (−1.0 vs planejadores). **Teste de diagnóstico:** se a POLÍTICA
  PPO bate reativos (greedy/rush) mas perde p/ PLANEJADORES (producer/oep/holdwave) → é o penhasco →
  nenhum treino de política resolve; **escalar p/ busca+eval (abaixo).**

### 5. Escalação — busca + eval aprendida (a tese central)
- **AlphaZero (Silver)** (`9PRDIRDP`) + **MuZero (Schrittwieser)** (`VI5ZJE4Q`) + **ExIt (Anthony)** (`DHM74V25`)
  + **Bitter Lesson (Sutton)** (`WESHDCHX`) **[forte — tese central declarada]**: a fórmula vencedora é
  **busca guiada por avaliação APRENDIDA**, não rede reativa nem knobs hand-tuned (Bitter Lesson: hand-tuned
  estagna; search+learning escala). ExIt: busca = operador de melhoria de política; a rede generaliza e
  depois guia a busca. MuZero: força ∝ orçamento de busca (temos actTimeout 1s → cada nó precisa contar).
  **Regra de escalação:** se a campanha de política-pura (atual) platôa ABAIXO dos planejadores (penhasco),
  a próxima iteração PIVOTA para **eval aprendida DENTRO do PGS** (ciclo ExIt: busca PGS gera alvos → treina
  value/policy → rede guia o PGS) — é a direção H7 ([[h7_value_net_pipeline_works_quality_insufficient]];
  mean-pool insuficiente → atenção/max-n), NÃO uma política reativa maior.

---

## Eixo P — coleção tracking/órbita (Kaggle/Orbit Wars + Drive) → **PARADO: mismatch de domínio [fraca]**
Tracking/estimação: Kalman (`P6CKQXH4`), UKF (`6T4DZJGZ`), particle filters (`PGDCXMK8`, `HBJMT58V`),
JPDA/MHT (`WRIFEW7Q`, `64UWC7TQ`), KalmanNet (`2I5UPPGC`), PINN/Neural-ODE de propagação orbital, surveys.
PDFs no Google Drive (pasta `1B2hR4t5cRidV_jBOQTHYNUbA9wXhMs70`), lidos via conector.

**Veredito honesto (evidência forte de mismatch):** esta coleção é sobre **estimação de órbita de objetos
FÍSICOS sob INCERTEZA** — debris/satélites reais, arrasto atmosférico, massa/geometria desconhecidas,
sensores ruidosos. O survey Caldas&Soares 2024 (lido: abstract + scan): termos dominantes uncertain(35),
Kalman(25), observation(22), measurement(13), covariance(12), noise(7), radar(5). Mas o JOGO Orbit Wars é
**determinístico e totalmente observável**: o agente vê 96 planetas + 256 frotas exatos (features em
`python/agents/policy.py`), simulador parity-exact, zero fog/ruído/estocástico no env/planner, e o planner
já PREVÊ os lançamentos do oponente exatamente. Logo Kalman/UKF/PF/KalmanNet/orbit-determination resolvem
um problema (estado ruidoso/parcial) que **o jogo não tem** — é **colisão de nome** ("orbit" satélite real
vs o jogo "Orbit Wars").
- Único conceito não-estimação que tocaria o jogo = **geometria de interceptação** de alvo móvel — mas isso
  os papers de FILTRAGEM não fornecem, e o `movement` já propaga/intercepta exato.
- **Decisão:** eixo P **PARADO**. Só reabrir se o jogo revelar info oculta (não revela) ou se medirmos
  propagação/interceptação como gargalo real. Não forço relevância (honestidade > citar). Foco = eixo M.

---

## Prioridade de execução
1. **AGORA (paralelo à campanha, CPU-only):** núcleo do **Nash averaging** (`scripts/nash_eval.py` + testes).
   Round-robin das refs roda quando a CPU liberar (não disputar com os rollouts da campanha).
2. **Campanha terminar:** diagnosticar com double-bind KL (§3) + teste do penhasco (§4) + assinatura DAgger;
   corrigir UMA alavanca; retreinar. Pool cresce PSRO/AlphaStar (best + exploiters).
3. **Se política-pura platôa < planejadores:** escalar p/ eval-aprendida-no-PGS (§5).
4. **Eixo P:** fila, gated em evidência de perda por interceptação/ameaça.

## Critério de convergência ("bot forte")
- ✅ checkpoint com **rating Nash alto** no pool diverso (não média-vs-holdwave) **+ PASS_LOCAL no DRL gate**
  (bate/empata pool forte 2p, sobrevive 4p, zero faults) **+ passa veto top5-proxy**.
- ⏹️ OU rating Nash platôa por 2 iterações com correções hipótese-dirigidas esgotadas → parar, reportar melhor.
- 🧱 Teto duro: 6 campanhas. Disciplina: **uma alavanca por iteração** (não grid). Prova final = LB (decisão humana).
