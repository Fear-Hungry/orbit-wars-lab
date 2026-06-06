> **Log de trabalho interno (não é documentação curada).** Journal append-only do
> roadmap competitivo. A documentação de portfólio, com uma fonte de verdade por
> tópico, está em [`docs/`](docs/README.md). Mantido na raiz para continuidade.

---

# Organização do repositório (portfólio)

Feito nesta passada (reorganização da documentação e do versionamento):

- [x] Consolidar `docs/` com índice navegável (`docs/README.md`) e uma fonte de verdade por tópico
- [x] Mover `DECISIONS.md` → `docs/DECISIONS.md` e corrigir a contradição D4 (PPO base) vs `TRAINING.md` (PPO deferido)
- [x] Criar `docs/ARCHITECTURE.md` consolidando o modelo de três camadas (antes espalhado em README/AGENTS/DECISIONS)
- [x] Deduplicar a regra Rust→Python (canônica em D10/D11; `SUBMISSION.md` só referencia) e "Producer é piso" (canônica em `COMPETITIVE_INTEL.md`)
- [x] Reescrever `README.md` como porta de entrada (remover seção "scaffold inicial" desatualizada e duplicação de comandos)
- [x] Tornar `.gitignore` honesto: `artifacts/*` + negações explícitas dos 7 arquivos versionados; adicionar `.zed/`

Pendente — **decisão/ação sua** (não mexi: é código com risco ou conteúdo seu):

- [x] ~~Remover o shim redundante `orbit_wars_gym/__init__.py`~~ — CORRIGIDO: o shim NÃO é redundante. Ele é load-bearing (faz `import_module("python.orbit_wars_gym")` e permite `import orbit_wars_gym` a partir da raiz sem `python/` no PYTHONPATH — usado por scripts/bots/agents). Mantido e documentado com docstring de papel.
- [ ] Decidir se os 4 screenshots `artifacts/kaggle_*.png` (signin/home/flow) entram no portfólio ou saem (`git rm --cached`); parecem scratch de automação, não asset.
  - [ ] verificar: `git ls-files artifacts/` lista só os deliverables que você quer mostrar
- [x] Versionar os scripts de diagnóstico não rastreados (`scripts/compare_oep_opponent_models.py`, `scripts/trace_submission_actions.py`) — são ferramentas reais referenciadas na Thread 2b (sonda de fidelidade de lanes / trace de ações). Compilam OK; commitados.
  - [x] verificar: `git status --short` não mostra `??` em `scripts/`
- [x] Reduzir ambiguidade de nomes/papéis (sem renomear/apagar): docstrings de papel em `orbit_lite/__init__.py` (Python puro, SUBMETIDO), `python/orbit_wars_gym/__init__.py` (Rust-backed, só TREINO) e no shim da raiz, todas apontando para `docs/ARCHITECTURE.md` + D10/D11. Rename foi descartado: `orbit_lite` é importado pela submissão/bots/tests e o gym pelo shim — renomear quebraria empacotamento e imports.

---

# ⚠️ REAVALIAÇÃO — a régua/simulador eram INFIÉIS quando o roadmap abaixo foi medido (2026-06-05)

Nesta passada descobrimos que a régua de avaliação — que decide TUDO no roadmap abaixo —
NÃO era fiel ao Kaggle em três pontos, todos consertados agora:

1. **Dinâmica de combate do Rust** — o `parity_probe` antigo (itens 5b/5d) só usava ações
   VAZIAS, então combate/movimento de frota NUNCA foram testados. O harness novo
   `scripts/parity_probe_actions.py` achou 3 bugs reais (ordem de colisão; swept-pair vs
   posição única → colisão-fantasma em planeta rotacionando; timing de expiração de cometa).
   ⇒ **O item 5d ("Rust alinhado à semântica oficial") cobria só a dinâmica passiva; estava
   INCOMPLETO.** A régua de combate divergia do Kaggle o tempo todo.
2. **Formato do obs na régua** — `to_official_observation` emitia `dict` em vez do formato
   `list` do Kaggle. (NÃO afetou OEP/Producer: ambos têm `_to_list_observation` privado; só
   quebrava agentes sem esse wrapper, como o tarball empacotado — que crashava 455/455 na régua.)
3. **Gerador de treino** — inflava as naves iniciais (já corrigido).

**CONSEQUÊNCIA** (aplicando a própria regra das linhas ~158-163: nunca otimizar o BOT sobre
régua infiel = perseguir ruído): **todas as margens OEP-vs-Producer medidas no roadmap abaixo
estão SUSPEITAS** — foram obtidas sobre combate buggado. Precisam re-validar no stack corrigido
ANTES de retomar a otimização do OEP:

- [x] Re-medir o baseline da ETAPA 0 (`margin=-0.18750`) — FEITO no sim corrigido. ⚠️ O 16-seed deu +0.31250 (win 0.656) mas era **RUÍDO de amostra pequena**; a **96 seeds (192 jogos, o limiar de decisão) = margin=−0.21137, win=0.391, timeout≈0.00046**. O combate fix corrigiu a FIDELIDADE mas NÃO inverteu a posição competitiva: o OEP AINDA perde para o Producer ao escalar (parecido com o −0.18750 antigo). A premissa do roadmap (OEP < Producer, melhorar) CONTINUA válida. **Meta-lição: 16-seed é perigosamente ruidoso (+0.31 vs −0.21) — não decidir por ele.**
- [x] Re-rodar o promotion gate 3a (`mean_score_margin=-0.099316`, 96 seeds) — FEITO no sim corrigido: margin=−0.21137 < 0 → o OEP **NÃO promove** (mesma conclusão qualitativa do gate antigo; na verdade ligeiramente PIOR que o −0.099 buggy). Saída: `/tmp/oep_revalidate_96seed.json`.
- [ ] Re-testar as conclusões da Thread 2b no sim corrigido: as regressões "corte X regrediu margem" precisam ser re-medidas — MAS sempre a 96 seeds (o 16/4-seed smoke usado lá é ruído, como acabamos de ver). O combate fix mudou os números exatos; o ranking dos cortes pode ter mudado.
- [ ] Marcar em `EXPERIMENTS.md` que experimentos anteriores a este fix foram medidos em régua infiel (combate + obs).
- [x] Verdict sobre a melhor submissão (Producer empacotado) na régua FIEL: roda limpo (0 crash/timeout/invalid) e vence greedy/rush 1.00 (margem +1.0, 4 seeds). Ela própria não regrediu.

**NÃO afetado** (continua válido): D1–D11, o modelo de 3 camadas, a regra no-silent-fallback,
os próprios fixes de paridade/gerador/obs desta passada, e os gates de arquitetura (5a/5c/6a/6b).
Pendência separada: o `orbit_lite` (planner da submissão) ainda tem os 2 bugs de world-model
(ver DÉBITO abaixo) — o Producer roda, mas planeja sobre um modelo levemente errado.

---

# DÉBITO — orbit_lite (engine da submissão) diverge do oficial

Descoberto ao corrigir o simulador Rust (2026-06-05): corrigir o Rust expôs que o
`orbit_lite` (world-model da SUBMISSÃO, `orbit_lite/movement.py` ~1900 linhas torch)
carrega os MESMOS 2 bugs que o Rust tinha — ele estava fixado contra o Rust buggy.
Os 2 testes de fidelidade (`test_movement_l3_*`, `test_movement_l5a_*`) estão
`xfail(strict=True)` até isso ser corrigido (quando passarem, o pytest avisa para
remover o xfail).

- [ ] Alinhar a colisão do `orbit_lite` à oficial: planeta em rotação não pode gerar colisão-fantasma (usar swept-pair contra a motion old→new do planeta, como `geometry::swept_pair_hit` no Rust). Evidência: L3 seed=7 step=5 planeta=12 → oficial/Rust=10, orbit_lite=8.
  - [ ] verificar: remover `xfail` de `test_movement_l3_matches_rust_with_random_valid_launches` e o teste passa
- [ ] Alinhar o timing de expiração de cometa do `orbit_lite` à oficial: cometa expira/é removido após o movimento e antes do combate (condição `path_index >= len`), não um step antes.
  - [ ] verificar: remover `xfail` de `test_movement_l5a_comet_projection` e o teste passa
- Nota: impacto competitivo — o planner da submissão projeta sobre um world-model errado; vale priorizar antes de submeter.

# Paridade do simulador Rust com o oficial (step logic) — 2026-06-05

Auditoria dirigida por harness novo (`scripts/parity_probe_actions.py`): dirige o
env oficial do Kaggle e o `RustBatchBackend` a partir do MESMO estado inicial com
ações idênticas e compara estado a estado. O probe antigo (`parity_probe.py`) só
usava ações vazias, então combate/movimento de frota nunca eram testados — foi
assim que estes bugs passaram. 3 bugs reais encontrados e corrigidos:

- [x] **Ordem dos checks de colisão** (`step.rs` move_fleets): Rust removia frota por out-of-bounds/sol ANTES de checar planetas; o oficial checa planetas primeiro (frota rápida que ultrapassa borda/sol ainda entrega combate no planeta do caminho).
- [x] **Colisão swept-pair** (`geometry.rs::swept_pair_hit` + `step.rs` compute/commit): Rust usava `point_to_segment` contra a posição única (velha) do planeta → planeta em rotação registrava colisão-fantasma onde já não estava. Portado o `swept_pair_hit` oficial (segmento frota × segmento planeta) e reestruturado o step para computar (old→new) antes de mover, mover com swept, e só então commitar. Removida a antiga 2ª passada `sweep_planet_collisions`.
- [x] **Timing de expiração de cometa** (`step.rs`): Rust removia cometa expirado no INÍCIO do step (antes dos lançamentos); o oficial remove após o movimento e antes do combate. Efeito: lançamento a partir de um cometa capturado no step de expiração tinha sucesso no oficial mas não no Rust (`next_fleet_id` divergia). Movida a remoção e corrigida a condição para `path_index >= len`.
  - [x] verificar: `scripts.parity_probe_actions` PASS em todas as janelas inter-spawn 0–499, 2p e 4p, com cometas e launch-prob 0.7 (≈40k steps comparados, 0 divergências)
  - [x] verificar: `cargo test -p orbit_wars_core` (18) e novo `tests/test_parity_actions.py`
  - Infra nova: `scripts/parity_probe_actions.py` + `tests/test_parity_actions.py` travam a paridade de combate/movimento/cometa contra o oficial.

# Correção de paridade — gerador de treino (motor Rust)

Verificado em 2026-06-05 (com `kaggle-environments` instalado): o motor Rust bate
paridade em modo *parity* (snapshots oficiais — `test_official_spec`,
`test_official_snapshots`, `test_parity_tolerances` passam), MAS o gerador de
*training mode* infla as naves iniciais.

- [x] Corrigir o gerador de treino Rust para casar a distribuição oficial de naves (2026-06-05). Causa raiz NÃO era `sample_large_group_ships` (a fórmula já era idêntica ao oficial), e sim **3 divergências do `generate_planets` oficial**: (a) fase extra que injetava um grupo orbitando de naves altas (oficial não tem); (b) `assign_home_planets` preferindo grupo diagonal no 4p (oficial escolhe home aleatório); (c) `group_is_valid` checando overlap intra-grupo (oficial só checa contra pré-existentes). Removidas as três. Validado a 256 seeds: ships |d|=0.35 (2p)/0.23 (4p) vs tol 4.0; todas as 6 métricas OK.
  - [x] verificar: `pytest tests/test_training_generator_distribution.py -q` passa (após `maturin develop --release`)
  - [x] verificar: `cargo test -p orbit_wars_core` passa (18); suíte de paridade `38 passed`
  - Nota: o teste foi de 16→256 seeds. Não dá para casar per-seed (ChaCha8 vs Mersenne Twister — `docs/PARITY.md` proíbe reproduzir o RNG Python); 16 seeds eram subdimensionados p/ comparar dois RNGs a ±4 naves. Tolerâncias mantidas; só o tamanho da amostra subiu (fortalece o gate).
- [x] (Infra) Tornar a paridade não-silenciosa: sem o extra `kaggle`, os testes de paridade davam ImportError de COLETA e abortavam a suíte inteira. Adicionado `tests/conftest.py` que ignora os módulos dependentes do Kaggle (`test_official_spec`, `test_official_snapshots`, `test_parity_tolerances`, `test_training_generator_distribution`, `test_submission_pipeline`) com um `warnings.warn` explícito quando o extra falta — skip barulhento, não erro mudo. No-op quando o extra está presente (CI/validação roda tudo). `test_parity_actions` já se auto-protege com `importorskip`.
  - [x] verificar: `pytest --collect-only tests/test_official_spec.py` coleta sem erro com o extra; sem o extra, conftest emite warning e ignora os módulos em vez de abortar a suíte.

---

# todo.md — Orbit Wars Lab

OBJETIVO DA COMPETIÇÃO: **terminar no TOP 5 do leaderboard** (decisão 2026-06-05; padrão para
toda competição). Calibração (docs/COMPETITIVE_INTEL.md): o Producer é PISO, não teto — o topo
é dominado por heurística + busca limitada + simulação de timeline longa + redistribuição por
dominância. Logo bater o Producer é a ENTRADA, não a meta; a régua real é um oponente forte do
fórum (ver 3b, agora obrigatório).

Foco atual: **gerar candidatos que batam o Producer** (régua de entrada). Micro-tuning está
saturado (old=0.656 / greedy=0.937 / rush=0.937 em todos os experimentos 73–99) — parado.
Tudo abaixo serve à tese: tornar a versão ampla do OEP, que já vence (win=0.5625,
margin=+0.125), **legal sob orçamento de tempo sem fallback silencioso**.

---

# ORDEM DE EXECUÇÃO (roadmap para top 5)

Faça em ordem. Cada etapa só abre quando o gate de saída da anterior fecha. As tags `[Thread Xy]`
apontam para o detalhe técnico e o histórico de tentativas mais abaixo neste arquivo.

## ETAPA 0 — Baratear o step (PRÉ-REQUISITO de tudo) — EM ANDAMENTO
Motivo: 1c, timeline-sim e 4p multiplicam o custo do step. Sem orçamento, nada acima cabe.
As duas chamadas Producer somam ~51.5% do tempo (2a confirmado). Todo corte tentado até agora
regrediu a margem — este é o nó real. ATUALIZAÇÃO 2026-06-05: foi corrigido um vazamento de
memória no Producer (`movement`/`last_sparse_action_row` não eram limpos em `step==0` dentro do
mesmo processo/worker). Baselines anteriores contra o Producer stateful ficam suspeitos; com a
régua corrigida, o OEP default 16 seeds/jobs=4 está em margin=-0.18750, timeout_rate=0.003676.
Antes de otimizar custo, é preciso recuperar margem contra o Producer corrigido.
⚠️ REVISTO 2026-06-05: esse −0.18750 ainda era no combate BUGGADO (só o vazamento de memória
do Producer estava corrigido). No simulador com paridade de COMBATE (parity_probe_actions):
16 seeds deu +0.31250 (RUÍDO), mas **96 seeds (decisão) = margin=−0.21137 / win=0.391 /
timeout≈0.00046** — o OEP AINDA perde ao escalar. A ETAPA 0 ("recuperar margem") CONTINUA
aberta: o combate fix corrigiu a fidelidade, não a margem. ⚠️ 16-seed é ruído — decidir só
a 96 seeds.
- [ ] 0.1 `[Thread 2b]` Baratear o 2º Producer (oponente do lookahead) sem cegar o 1-ply.
  - [ ] verificar: profile 2a antes/depois mostra queda material de mean_ms E margin vs Producer
        16 seeds NÃO regride (todas as tentativas até hoje regrediram — ver histórico 2b;
        `OEP_OPPONENT_RESPONSE_MODE=none` preservou o smoke 4 seeds, mas caiu em 16 seeds para
        margin=-0.43750/win=0.28125, timeout_rate=0.000195).
- [ ] 0.2 `[Thread 1b-iii]` Reduzir C por config (shortlist/frações/waves) sem mudar a regra de
      seleção por fitness.
  - [ ] verificar: config ampla vs Producer 96 seeds → margin ≥ 0, timeout/crash/invalid = 0,
        mean_ms (p95) sob o teto. GATE DE SAÍDA DA ETAPA 0.

## ETAPA 1 — Régua dupla (DESCONGELADO; obrigatório para top 5)
Motivo: o Producer é PISO. Medir só contra ele overfita o piso e não diz nada sobre top 5.
- [ ] 1.1 `[Thread 3b]` Adicionar 2º oponente-régua forte do fórum (notebook 704113 Producer~1200
      / timeline-sim) ao benchmark e ao gate de promoção.
  - [ ] verificar: candidato atual roda contra os DOIS oponentes; baseline de cada um registrado
        em EXPERIMENTS.md. GATE DE SAÍDA DA ETAPA 1.

## ETAPA 2 — Salto de qualidade (busca real + técnicas do topo)
Motivo: empatar com o Producer não é top 5. Aqui está o ganho de nível.
- [ ] 2.1 `[Thread 1c]` Trocar candidato guloso único por BUSCA sobre população de genomas
      (1c-i..iv).
- [ ] 2.2 `[F3 ideia 2]` Fitness com risco de timeline curta (penalidade de defesa/recaptura por
      arrivals nos próximos 7–20 turnos).
- [ ] 2.3 `[F3 ideia 1]` Shortlist top-K por fonte antes de gerar genomas (varrer K∈{3,5,8}).
  - [ ] verificar: bate o Producer no gate 96 seeds (margin > 0) E não regride contra o 2º
        oponente da Etapa 1. GATE DE SAÍDA DA ETAPA 2.

## ETAPA 3 — Cobertura 4p (hoje o lookahead está DESLIGADO em 4p)
Motivo: leaderboard pontua 2p e 4p; D9 diz que são dinâmicas distintas. Hoje `_opponent_id`
retorna None em 4p → OEP vira guloso sem lookahead.
- [ ] 3.1 `[F1-i..iii]` Ligar o oponente 1-ply em 4p (começar por 1 oponente mais relevante) e
      criar gate 4p (sem `--skip-4p`).
  - [ ] verificar: em 4p `opponent_entries` deixa de ser None; win 4p ≥ baseline OEP-4p-off,
        crash/timeout/invalid = 0. GATE DE SAÍDA DA ETAPA 3.

## ETAPA 4 — Aposta pesada (só se 2/3 saturarem abaixo de top 5)
- [ ] 4.1 `[F2]` Reavaliar PPO/self-play contra a régua DUPLA. Gatilho: 1c saturou, margem parou
      de crescer, e ainda não é nível de top 5.
  - [ ] verificar: 1 run PPO produz checkpoint com win ≥ baseline OEP contra os dois oponentes,
        sem crash/timeout.

---

# ARQUITETURA DO REPOSITÓRIO — fluxo a seguir, passo a passo

## As 3 camadas (nunca misturar)
1. **MOTOR / verdade física** — `crates/orbit_wars_core` (Rust) + binding `crates/orbit_wars_py`;
   `orbit_lite/` é o sim Python leve usado DENTRO do lookahead do OEP. Régua de fidelidade:
   `scripts/parity_probe.py` contra `kaggle-environments` + `tests/test_movement_fidelity.py`.
2. **BACKTEST / régua de avaliação** — `scripts/benchmark_submission.py`,
   `scripts/compare_benchmark_significance.py`, `scripts/oep_promotion_gate.py`,
   `scripts/gate_check.py`. Mede candidato vs oponente; NÃO decide física nem estratégia.
3. **BOT / decisão** — `bots/oep/{planner,agent}.py`, `bots/producer/`, `python/agents/`,
   e a submissão final `python/submission/submission_template.py` → `artifacts/submission.py`.

## Regra de fluxo (ordem de conserto INEGOCIÁVEL): motor → backtest → bot
Sempre consertar de BAIXO para CIMA, nunca o contrário.
- Bot parece errado? → 1º confirmar que a RÉGUA (backtest) é fiel.
- Régua parece errada? → 1º confirmar que o MOTOR passa no parity.
- Só mexer no BOT quando motor e régua estão confiáveis. Otimizar bot sobre régua infiel =
  perseguir ruído (corrompe a correlação local↔leaderboard).

## Invariantes que NÃO se quebram (DECISIONS.md + AGENTS.md)
- **D10/D11 — fronteira Rust/Python**: submissão é Python leve/puro; nenhum `bots/` nem
  `artifacts/` importa `orbit_wars_core`/`orbit_wars_py`. Rust SÓ simula treino local.
  Travado por `test_no_native_in_submission`.
- **Sem fallback silencioso**: falhar barulhento e medido; nunca degradar em silêncio
  (corrompe a correlação local↔leaderboard).
- **Régua de qualidade = Producer** (e, a partir da Etapa 1, o 2º oponente). `submission_v_old`,
  `greedy`, `rush` são só sanity de crash/legalidade — NÃO promovem.
- **EXPERIMENTS.md**: toda mudança no agente registra margem normalizada antes/depois vs o
  oponente nomeado, ANTES do commit.
- **Score Kaggle**: nunca julgar pelo score imediato; esperar ~1h de estabilização.

## Ciclo de desenvolvimento (passo a passo, toda mudança)
1. Mexeu em motor/`orbit_lite`? → `parity_probe` + `test_movement_fidelity` ANTES de qualquer
   benchmark. Vermelho aqui invalida tudo acima.
2. Mexeu em régua/gate? → garantir `bool(checks)` e seeds fixas; não afrouxar limiar/oponente.
3. Mexeu no bot? → escrever hipótese em EXPERIMENTS.md → smoke 16 seeds vs Producer → se passar,
   gate estrito 96 seeds (`make oep-promotion-gate`).
4. Antes de submeter → `scripts.gate_check` (no-silent-fallback + no-native-import) →
   `scripts.export_submission` → submeter → esperar ~1h antes de concluir qualquer coisa.

---

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
  - [ ] verificar (⚠️ RE-VALIDAR — régua infiel, ver topo): com seeding ligado e o MESMO orçamento por step que hoje dá win=0.375,
        rodar vs Producer 16+ seeds → margin média ≥ 0.0 e timeout_rate = 0.0
        RESULTADO: 16 seeds/32 jogos vs Producer, margin=0.00000, timeout_rate=0.0.
  - [ ] verificar (⚠️ RE-VALIDAR — régua infiel, ver topo): sem regressão de legalidade — crash=0, invalid_action_rate=0 (lanes
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
  - [ ] 1b-ii. ⚠️ código FEITO; a margem (margin=0.00000) foi medida em régua infiel — RE-VALIDAR (ver topo). Tornar a seleção um torneio único de plano inteiro:
        `chosen = oep_entries if fit(oep) > fit(producer)+min_advantage else producer_entries`.
        RESULTADO: 16 seeds/32 jogos vs Producer, win=0.50000, margin=0.00000,
        mean_ms=287.48, crash=0.0, timeout=0.0, invalid=0.0.
  - [ ] 1b-iii. Reduzir C por configuração (shortlist/frações/waves/regroup) até o pior caso
        medido ficar sob o teto com margem, sem alterar a regra de seleção por fitness.
        PROGRESSO 2026-06-05: `bots/oep/planner.py` agora expõe knobs via env vars
        (`OEP_MAX_SOURCES_PER_LANE`, `OEP_MAX_OFFENSIVE_TARGETS`,
        `OEP_MAX_DEFENSIVE_TARGETS`, `OEP_MAX_WAVES_PER_TURN`, `OEP_FRACTIONS`,
        `OEP_MIN_ADVANTAGE`, `OEP_ENABLE_REGROUP`, `OEP_LATE_CONFIG_STEP`,
        `OEP_LATE_MAX_SOURCES_PER_LANE`) e `scripts/profile_oep_step.py` usa a mesma
        configuração do agente. Tentativa `4×4,waves=3,fractions=0.5,1.0` reduziu custo no smoke
        4 seeds (`mean_ms` 273.99→200.86, timeout=0), mas regrediu margem
        0.00000→-0.50000 e win 0.50000→0.25000; rejeitada como default.
        Tentativas `5×5,waves=4` e `6×5,waves=4` também regrediram em 4 seeds
        (ambas margin=-0.50000). `5×6,waves=4` preservou o smoke 4 seeds
        (margin=0.00000, mean_ms=232.22), mas regrediu em 16 seeds para
        margin=-0.12500/win=0.43750 e mean_ms=300.19; rejeitada como default.
        `OEP_ENABLE_REGROUP=0` também falhou: profile curto ficou mais lento
        (mean_decision_ms=53.85, p95=76.66, max=101.71) e o smoke 4 seeds regrediu para
        margin=-0.25000/win=0.37500, mean_ms=423.85, timeout_rate=0.000883; rejeitada
        como default. `OEP_FRACTIONS=1.0` reduziu o eixo G no profile curto
        (mean_decision_ms=35.67, p95=45.46, max=49.92), mas o smoke 4 seeds regrediu para
        margin=-0.25000/win=0.37500, mean_ms=362.78, timeout_rate=0.000808; rejeitada
        como default. `OEP_MAX_WAVES_PER_TURN=3` preservou margem no smoke 4 seeds
        (margin=0.00000/win=0.50000, timeout=0), mas aumentou o custo para mean_ms=422.72
        e piorou a cauda do profile curto (max=132.78); rejeitada como default.
        `OEP_MAX_DEFENSIVE_TARGETS=1` também preservou margem no smoke 4 seeds
        (margin=0.00000/win=0.50000, timeout=0), mas aumentou custo para mean_ms=444.60
        e piorou o profile curto (mean_decision_ms=55.73, p95=75.97); rejeitada como default.
        Corte tardio `OEP_LATE_CONFIG_STEP=100` + `OEP_LATE_MAX_SOURCES_PER_LANE=5`
        preservou o smoke 4 seeds e reduziu custo nele (margin=0.00000, mean_ms=246.76),
        mas regrediu em 16 seeds para margin=-0.06250/win=0.46875, mean_ms=391.07,
        timeout_rate=0.001172; rejeitado como default. Atrasar o mesmo corte para step 200
        falhou já no smoke 4 seeds por custo/timeout (margin=0.00000, mean_ms=419.03,
        timeout_rate=0.000967); rejeitado como default.
        Diagnóstico posterior: comparando default vs `max_sources=5` na MESMA trajetória
        OEP-default-vs-Producer 16 seeds, nenhuma das 5013 decisões mudou (`different=0`);
        portanto a regressão de `max_sources=5` não é explicada por diferença local simples
        nos estados visitados pelo default, mas por estados/memória visitados quando o corte
        está implantado ou pela semântica stateful das chamadas Producer. Não insistir em
        cortes fixos por source sem sonda de trajetória implantada.
        Nova sonda na trajetória implantada `max_sources=5`, ainda com Producer isolado, também
        deu `different=0` em 5001 decisões; logo este tipo de diagnóstico isolado não reproduz
        a regressão real do benchmark. A próxima sonda precisa logar ações da trajetória real
        via `bots/oep/agent.py`/env em processos separados, preservando a semântica global do
        Producer usada pelo benchmark.
        Sonda real via `scripts.trace_submission_actions` fez isso em modo serial: default e
        `OEP_MAX_SOURCES_PER_LANE=5` tiveram ações idênticas em 5138 decisões (`different=0`);
        ambos ficaram com margin=-0.12500 em `jobs=1`, mas o corte reduziu mean_ms
        38.24→32.37. Portanto `max_sources=5` é promissor no modo serial, mas não promove
        ainda porque a régua decisora `jobs=4` já havia registrado regressão; antes de aceitar
        qualquer corte é preciso padronizar ou explicar a divergência `jobs=1` vs `jobs=4`.
        ACHADO 2026-06-05: a divergência vinha de bug real no Producer fixture: `step==0` não
        limpava `movement`/`last_sparse_action_row`. Corrigido em `bots/producer/_upstream.py`
        e travado por teste. Pós-fix, default jobs=4 ficou margin=-0.18750/win=0.40625,
        timeout_rate=0.003676; `max_sources=5` ficou com a mesma margem/win e timeout=0,
        mas mean_ms=359.81. Nenhum promove; a prioridade volta a recuperar qualidade contra
        o Producer corrigido.
        Tentativa de recuperar qualidade só por seleção conservadora também falhou:
        `OEP_MIN_ADVANTAGE=999` foi bom no smoke 4 seeds (margin=+0.50000, mean_ms=213.72),
        mas em 16 seeds voltou para margin=-0.18750/win=0.40625, mean_ms=386.00,
        timeout_rate=0.002628. Rejeitado como default.
        Tentativa de separar `seed_policy`/`opponent_policy` em dois Producer runtimes
        independentes dentro do OEP também falhou: `min_advantage=999` caiu para
        margin=-0.25000 em 16 seeds e o default independente regrediu para margin=-0.50000
        no smoke 4 seeds; a mudança foi revertida.
        Instrumentação de seleção adicionada em `scripts.profile_oep_step`: profile default
        4 seeds/128 steps vs Producer (`artifacts/gates/oep/profile_selection_default_4seed_128steps.json`)
        teve mean_score_margin=+0.57544 no horizonte curto, crash/timeout/invalid=0,
        `oep_choice_rate=0.18351`, `producer_choice_rate=0.81649` e delta médio
        `fit(oep)-fit(producer)=+9.37863`; ainda assim houve partidas perdidas
        (seed 1/player 1 margin=-1.00000; seed 3/player 1 margin=-0.10316). Diagnóstico:
        o problema não é apenas o OEP "nunca ser escolhido"; a próxima mudança precisa atacar
        qualidade/calibração do fitness ou composição de candidatos com análise por partida.
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
        PROGRESSO 2026-06-05: `scripts/profile_oep_step.py` agora reporta `p50/p95/p99/max`
        por partida e cauda agregada. Profile pós-cache, `1` seed, `500` steps,
        `opponent_response_mode=producer`: `mean_decision_ms=33.44`, `max_decision_ms=59.97`,
        `max_match_p95_decision_ms=42.20`, crash/timeout/invalid=0.0. Neste smoke a cauda
        não é o risco imediato; as duas chamadas Producer ainda dominam (~55.6% do tempo).
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
        TENTATIVA 2026-06-05: `OEP_PRODUCER_PLAN_MODE=inline` evita chamadas completas
        ao `agent()` do Producer e reusa o `movement/cache/status` do OEP; no smoke 4
        seeds reduziu `mean_ms` 273.99→184.33, mas regrediu margem 0.00000→-0.50000
        e win 0.50000→0.25000. Rejeitado como default; provável causa: não reproduz
        a memória/runtime completo do Producer ao longo do episódio.
        TENTATIVA 2026-06-05: `OEP_PRODUCER_PLAN_MODE=tensor` chama
        `ProducerLiteRuntime.tensor_action` diretamente sobre tensores e evita
        dict/list/move roundtrip; após corrigir `torch.no_grad()`, profile curto ficou
        próximo do modo policy (`mean_decision_ms=33.80`, `max_decision_ms=54.23`), mas
        4 seeds vs Producer regrediu `mean_score_margin` 0.00000→-0.50000 e win
        0.50000→0.25000. Rejeitado como default; preservar só como modo experimental.
        TENTATIVA 2026-06-05: isolar o corte só no 2º Producer mantendo o seed oficial
        (`OEP_OPPONENT_RESPONSE_MODE=producer_inline`/`producer_tensor`). `producer_inline`
        reduziu o estágio do oponente no profile curto para 6.02ms, mas 4 seeds regrediu
        margin 0.00000→-0.50000, win 0.50000→0.25000, mean_ms 273.99→257.46.
        `producer_tensor` também regrediu em 4 seeds (margin=-0.50000, mean_ms=244.94).
        Rejeitados como default; ficam apenas como modos experimentais explícitos.
        PROGRESSO 2026-06-05: `scripts.compare_oep_opponent_models` mede fidelidade de lanes
        contra o Producer real antes de benchmarks caros. Em 1 seed/128 steps/256 amostras:
        `cheap` teve lane_recall=0.2073/lane_precision=0.2324 e empty_model=0.668;
        `producer_inline` teve recall=0.7793/precision=0.6393; `producer_tensor` teve
        recall=0.8411/precision=0.6783. Como inline/tensor ainda perdem margem, o erro
        provável é overprediction/estado-memória/volume, não apenas ausência de lanes.
        TENTATIVA 2026-06-05: filtrar `producer_inline`/`producer_tensor` para top3 lanes por
        navios aproximou o volume no diagnóstico (`tensor_top3` model_lane_count=1.996 vs
        real=1.883), mas não recuperou margem: `producer_tensor_top3` 4 seeds teve
        margin=-0.50000, win=0.25000, mean_ms=234.46; `producer_inline_top3` teve
        margin=-0.50000, win=0.25000, mean_ms=288.42. Rejeitado como default.
        CORREÇÃO 2026-06-05: a primeira sonda de lanes contaminava a memória global do
        Producer com chamadas extras. Após usar runtimes Producer isolados por jogador,
        `producer_tensor` reproduz exatamente o Producer na trajetória Producer-vs-Producer
        (recall=1.0000/precision=1.0000 em 1 seed/128 steps), `producer_inline` fica alto
        (recall=0.9199/precision=0.9512) e `cheap` continua fraco
        (recall=0.2847/precision=0.3008). Portanto a regressão do `producer_tensor` no OEP
        não é mismatch simples de lanes contra Producer isolado; o próximo diagnóstico deve
        comparar contra o 2º Producer default dentro do OEP (policy stateful após a chamada seed).
        TENTATIVA 2026-06-05: `OEP_OPPONENT_RESPONSE_MODE=producer_shared_tensor` usa um
        runtime tensor compartilhado seed+oponente e faz uma chamada shadow do seed para
        reproduzir a contaminação de memória. Recuperou parte da regressão
        (`margin=-0.25000` em 4 seeds, contra `-0.50000` de tensor/inline isolados), mas
        ficou mais caro que o default (`mean_ms=292.55` vs 273.99) e ainda regride vs
        `margin=0.00000`; rejeitado como default.
        TENTATIVA 2026-06-05: `producer_seeded_shared_tensor` tentou evitar o shadow caro
        alimentando `last_sparse_action_row` do runtime compartilhado com o seed policy.
        Não preservou a semântica: profile curto voltou a margin=-0.53936 e
        mean_decision_ms=71.94; o oponente tensor subiu para 20.19ms. Conclusão:
        `last_sparse_action_row` não basta; a parte relevante da memória é o `movement`
        interno atualizado pelo Producer.
        TENTATIVA 2026-06-05: `producer_synced_shared_tensor` sincronizou o `movement`
        compartilhado com os entries oficiais do seed sem replanejar o seed. Também falhou:
        profile curto mean_decision_ms=61.65 e margin=-0.59160; sync custou 3.05ms e
        oponente synced tensor 15.16ms. A família shared/seeded/synced não fecha 2b.
- [x] 2c. (suspeito #2) `_fill_garrison_trajectory` (L899): loop Python `for k in range(...)`
      (L1070) sobre o horizonte. Confirmar se a projeção é reconstruída do zero a cada step ou
      se o cache incremental (`_roll_garrison_projection` L1150, `_mark_garrison_dirty` L1210)
      está sendo de fato aproveitado; vetorizar a recorrência se o profile apontar custo aqui.
  - [ ] verificar (⚠️ L3/L5a agora xfail pelo débito orbit_lite — RE-VALIDAR, ver topo): custo da projeção cai (microbench) E gates L1–L5a verdes
        (tests/test_movement_fidelity.py sem regressão — fidelidade é inegociável)
        RESULTADO: `scripts/benchmark_garrison_cache.py` separa update/status e confirma que o
        cache incremental era invalidado desde o passo 1 pelo roll vazio de `fleet_buckets`.
        Após só marcar dirty global quando havia buckets de frota não-zero e preencher a
        trajetória a partir de `dirty_from`, 2 seeds × 128 steps × H=18: sem cometas
        fresh=3.315ms, cached=2.085ms, speedup=1.59×, `dirty_from_before=18.0`;
        com cometas fresh=3.974ms, cached=3.437ms, speedup=1.16×. Fidelity L1–L5a:
        `tests/test_movement_fidelity.py` 9 passed.
- [x] NOTA (prioridade): se 2a confirmar que os 2 Producers dominam o tempo, 2b pode ser o item
      de MAIOR alavancagem do plano inteiro — derruba a média o bastante para 1b/1c caberem no
      orçamento. Nesse caso, fazer 2a+2b ANTES de 1c (talvez até antes de 1a).
      DECISÃO: confirmado; as duas chamadas Producer somam ~51.5% do tempo perfilado.

## Thread 3 — Régua de promoção honesta (parar de decidir por greedy/rush)

- [ ] 3a. ⚠️ infra do gate FEITA; o resultado do candidato (margin=-0.099316, "não promove") foi em régua infiel — RE-VALIDAR (ver topo). Formalizar o gate de promoção do OEP: margin ≥ 0 vs Producer em ≥96 seeds,
      timeout/crash/invalid = 0, via scripts.compare_benchmark_significance.
      PROGRESSO 2026-06-05: `scripts/oep_promotion_gate.py` formaliza a regra sobre
      relatórios JSON de `benchmark_submission`: default exige `192` jogos (`96` seeds × 2
      lados), margem média `>=0.0` contra `2p:producer`, crash/timeout/invalid `0.0`,
      `paired_games>=192`, `paired_margin_delta>=0.0` vs baseline G2 e nenhum veredito de
      regressão significativa. `make oep-promotion-gate` roda o verificador usando o baseline
      G2 default.
      RESULTADO 2026-06-05: run estrito 96 seeds/192 jogos produzido em
      `artifacts/gates/oep/candidate_vs_producer_96seed.json`; `make oep-promotion-gate` gerou
      `artifacts/gates/oep/promotion_gate.json` e FALHOU como esperado para este candidato:
      `mean_score_margin=-0.099316`, `timeout_rate=0.000383`, `win_rate=0.447917`,
      crash/invalid/fallback=0. O gate está formalizado e executado; OEP atual não promove.
  - [ ] verificar (⚠️ RE-VALIDAR — régua infiel, ver topo): gate documentado (EXPERIMENTS.md/DECISIONS.md) e 1 run de promoção produz
        veredito paired ≥ baseline G2
        SMOKE 2026-06-05: usando o relatório OEP 16 seeds já existente e `--min-games 32`,
        o gate passou: `mean_score_margin=0.0`, crash/timeout/invalid=0.0,
        `paired_margin_delta=1.0`, `verdict=margin_significant_improvement`.
        STRICT 2026-06-05: 96 seeds/192 jogos: `paired_margin_delta=0.900684`,
        `verdict=margin_significant_improvement`, mas gate total falhou por margem negativa e
        timeout técnico > 0.
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
- [x] F2. PPO / self-play (D4) — infra de rollout batched foi construída (commits a887216,
      814eb04, 30f6d22) mas NENHUM run de treino produziu candidato. É o caminho pesado de
      longo prazo. Decisão docs/SUBMISSION.md: usar Producer/heurístico como oponente de PPO.
      DECIDIR: vale ativar agora ou só depois de o OEP bater o Producer? (Recomendo depois —
      OEP já tem sinal positivo; PPO é aposta de maior custo/prazo.)
      DECISÃO 2026-06-05: deferir PPO/self-play. Caminho ativo continua OEP vs Producer; PPO só
      reabre depois de OEP passar o gate de promoção contra Producer, esgotar ganho mensurável, ou
      surgir oponente externo forte que exija diversidade de política.
  - [x] verificar (se ativado): 1 run de PPO contra Producer produz checkpoint com win vs
        Producer ≥ baseline OEP, sem crash/timeout
        RESULTADO: não ativado por decisão. Sem run obrigatório neste ciclo; critério de ativação
        registrado em `docs/TRAINING.md` e `docs/SUBMISSION.md`.
- [x] F3. Minerar notebooks públicos fortes do fórum (exp. 105): 704095 (benchmark 109
      agentes), 704113 (Producer ~1200). Extrair ideias de produção projetada / redistribuição
      / lookahead ainda não incorporadas. É inteligência competitiva, não código.
      RESULTADO 2026-06-05: `docs/COMPETITIVE_INTEL.md` registra fontes Kaggle consultadas,
      notebooks puxados para `/tmp/orbit_wars_f3` e quatro hipóteses acionáveis: shortlist top-K
      por fonte, fitness com risco de timeline curta, redistribuição por dominância como candidato
      de plano e genes compostos hammer/multiprong.
  - [x] verificar: lista de 2-3 ideias acionáveis extraídas, cada uma com hipótese testável
        registrada em EXPERIMENTS.md
- [x] F4. Fechar pendências de micro-tuning local (exp. 106-107): "reduzir perdas vs rush" e
      "decisão 4p com anti_meta+defensive juntos". ATENÇÃO: provavelmente SUPERADAS pelo pivô
      (régua vs-old/rush está desqualificada como promotor). NÃO ressuscitar cegamente —
      primeiro DECIDIR se ainda fazem sentido contra a régua Producer.
      DECISÃO 2026-06-05: manter parado. As pendências de 31/05 foram formuladas contra
      `rush`, `anti_meta`, `defensive` e 4p agregado antes do pivô Producer; desde 04/06,
      `submission_v_old.py`, `greedy` e `rush` são sanity técnico, não promoção. Não há
      hipótese Producer/OEP específica nestas pendências antigas. Se voltarem, devem entrar
      como nova hipótese mensurável contra Producer ou como F1 4p, não como micro-tuning local.
  - [x] verificar: decisão registrada (manter parado OU re-escopar contra Producer)

## Thread 4 — Risco deferido (só com repro concreto)

- [x] 4a. L5b / fidelity de fixtures set_state arbitrárias: validar que o sim interno do
      lookahead não diverge do motor real em estados de borda. Investir SÓ se aparecer um
      caso em que o fitness do planner contradiz o resultado real da partida.
      DECISÃO 2026-06-05: manter deferido. Não há repro atual de fitness contradizendo resultado
      real; L1–L5a seguem cobertos por `tests/test_movement_fidelity.py`, `parity_probe` recusa
      janelas que cruzam spawn futuro oculto e 6a travou `_effective_config` para nunca pontuar
      além da próxima fronteira de spawn. Só reabrir com repro nomeado.
  - [x] verificar: repro de divergência sim-interno vs motor, ou decisão explícita de manter deferido

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
      ⚠️ REVISTO 2026-06-05: este alinhamento cobria só a dinâmica PASSIVA (ações vazias do
      `parity_probe`). Com ações reais (`parity_probe_actions`), 3 bugs de combate/colisão
      swept/expiração-no-lançamento ainda divergiam e foram corrigidos depois. Ver a seção
      ⚠️ REAVALIAÇÃO no topo. Não confiar em "Rust = oficial" só por este item.
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

## Thread 6 — Decisões de hardening pós-validação (2026-06-05)

- [x] 6a. PONTO 1 (parity probe não cobre pós-spawn) — DECISÃO: ACEITAR e FECHAR. O vão é
      benigno por construção: o clamp `_effective_config` (`bots/oep/planner.py:147-151`, chamado
      todo step em `:1051`) impede o planner de pontuar além da próxima fronteira de spawn —
      exatamente a região que a probe não verifica. Modelar spawn futuro é impossível (seed
      oculta); adivinhar pioraria o fitness. NÃO investir em mais fidelidade aqui. Único trabalho:
      travar o invariante com teste para refactor futuro não quebrá-lo em silêncio.
      RESULTADO 2026-06-05: invariante travado em `tests/test_oep_agent.py`.
  - [x] verificar: `test_effective_horizon_never_crosses_spawn` — para step em 45–55 (e demais
        vizinhanças de COMET_SPAWN_STEPS), `_effective_config(cfg, step).horizon <=
        min(s - step for s in COMET_SPAWN_STEPS if s > step)`. Passa hoje.
- [x] 6b. PONTO 2 (gate passa vazio) — DECISÃO: CORRIGIR. Em `scripts/gate_check.py` (~L269),
      trocar `"passed": all(...)` por `"passed": bool(checks) and all(...)` em
      `_gate_no_silent_fallbacks`. Integridade do gate = fundação de não queimar submissões.
      RESULTADO 2026-06-05: `scripts/gate_check.py` agora exige `bool(checks)`; teste novo cobre
      relatório vazio.
  - [x] verificar: relatório sem `formats` (ou sem 2p/4p) → gate `passed: False` (hoje True); o
        run real atual com `technical_seeds=[0,1,2,3]` continua `passed: True`.
        RESULTADO: `tests/test_gate_check.py` cobre relatório vazio; o relatório real
        `artifacts/gates/fallback_metrics/template_vs_producer_16seed.json` inclui seeds 0–15
        e continua `gate_1b_no_silent_fallbacks.passed=True`.

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
