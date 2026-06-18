# Pipeline de pesquisa automático

Design de um loop autônomo (estilo AlphaEvolve/FunSearch) que gera, avalia e seleciona
candidatos rumo ao **Top 5** do leaderboard (LB) Kaggle "Orbit Wars". As decisões de
arquitetura subjacentes estão em [`DECISIONS.md`](DECISIONS.md); o método de seleção em
[`BLUEPRINT.md`](BLUEPRINT.md); o estado de treino em [`TRAINING.md`](TRAINING.md).

Este doc é a fonte de verdade para **o desenho do loop e o sinal de fitness**. Não
contém código — descreve componentes, contratos e guardrails.

## 1 — Problema & objetivo

A meta da competição é terminar no **Top 5** do LB. Bater o `Producer` (score ~1228) é
**necessário mas não suficiente** — o topo do LB começa em ~1575+.

O ciclo manual (mutar knob → rodar gate → decidir → submeter) é lento e enviesado pelo
operador: cada iteração custa atenção humana e tende a repetir famílias já testadas.
Automatizar o *gerar→avaliar→selecionar→registrar* libera o humano para o que importa
(calibração de sinal, decisões de arquitetura) e roda 24/7, explorando o espaço de
variantes em paralelo ao caminho PPO de longo prazo.

**Mas automatizar só vale se o sinal otimizado for preditivo.** A próxima seção é o
coração deste doc.

## 2 — A crux: o sinal de fitness (não a orquestração)

> Um loop que otimiza um sinal **não-preditivo** apenas overfitta mais rápido.

A liga/avaliação **local foi FALSIFICADA como preditor do LB**. Evidência (força:
**forte**, é medição direta no nosso campo):

- **Spearman ~0.0** entre ranking da liga local e ranking do LB real; **-0.6** na faixa
  competitiva (ordem *invertida* onde mais importa).
- `hold` e `wave_s100` passaram a liga local com **P ≥ 1.00** (probabilidade de bater o
  campeão) e **floparam no LB** (~1057.6 e ~1036.6 — nível "allscripts", longe de 1228).
- Causas diagnosticadas:
  - **Viés de população**: 8/10 do pool de liga é a mesma linhagem → mede semelhança
    intra-família, não generalização (Balduzzi et al. 2018, "Re-evaluating Evaluation" —
    população enviesada infla rankings; força: **forte**).
  - **Oponentes externos fracos demais** para discriminar o topo.
  - **Topo plano**: CIs de 90% sobrepostos entre os melhores; a "promoção" vira ruído.
- O **LB em si é ruidoso**: resubmit idêntico deu **1169.9 vs 1228.8** (±~60), e é
  **rate-limited** (poucas submissões/dia). Não dá para usar o LB como fitness denso.

**Conclusões operacionais (invariantes do pipeline):**

1. A liga local vale **só como VETO** (descartar candidato claramente quebrado), **nunca
   como promoção**. Nenhum candidato é submetido *porque* ganhou a liga.
2. O **gargalo do pipeline é o sinal de fitness**, não a orquestração. Todo o esforço de
   design vai para tornar o fitness **robusto** (seção 5), não para encadear mais etapas.
3. Calibração contra o LB é **esparsa e cara** — entra como ground-truth ocasional, não
   como loop interno.

Corolário de espaço de busca: **tunar pesos do `eval_function` é teto provado**
(EXPERIMENTS 77–84) — espaço morto, *fora* do escopo de mutação (seção 4).

## 3 — Arquitetura do loop (5 componentes)

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              ▼
   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
   │ GERAR   │──▶│ AVALIAR  │──▶│ SELECIONAR │──▶│ REGISTRAR │──▶│ GOVERNAR │
   │ (mutar) │   │ (gate    │   │ (bater     │   │ (DuckDB)  │   │ (submit  │
   │         │   │  robusto)│   │  parent?)  │   │           │   │  budget) │
   └─────────┘   └──────────┘   └────────────┘   └───────────┘   └──────────┘
        ▲                                                              │
        └──────────────────────── repeat (24/7) ───────────────────────┘
```

| Componente | Papel | Arquivo existente |
|---|---|---|
| **Gerar** | Propõe candidato: muta knobs de estratégia (wave/hoard/threat-value) sobre um parent do DB; ou dispara um treino PPO como candidato | `bots/pgs/planner.py` (`PGSConfig`, `make_runtime`); ramo PPO: `python/train/train_ppo.py` (`Phase0TrainingConfig`) |
| **Avaliar** | Gate de robustez multi-oponente, 500 steps, ambos assentos; devolve `death_rate`, `mean_margin`, `mean_final_planets` | `scripts/h9_4p_gate.py` (CLI `--seeds --steps --opponents --no-comets`); população: `python/train/evaluate_population.py` (`evaluate_population`) |
| **Selecionar** | Aplica critério de promoção (bater o parent no gate) + veto da liga | gate acima + liga `python/train/train_league.py` (`run_league_iteration`) como **veto** |
| **Registrar** | Persiste candidato, config, métricas e status (`todo/applied/rejected/logged`) no DB evolutivo; consulta `rejected` antes de re-rodar | `experiments.duckdb` + `python/lab/experiments.py` (`add_experiment`, `query_experiments`, `report_md`) |
| **Governar** | Empacota/valida o melhor candidato não-submetido e submete sob budget diário | `scripts/export_submission.py`, `scripts/benchmark_submission.py`; orquestração `python/lab/cli.py` (export→bench→eval→league) |

O **pool de oponentes** alimenta Avaliar/Selecionar:
`python/agents/registry.py::get_isolated_opponents(name, count)`, com
`HEURISTIC_NAMES = (producer, oep, greedy, defensive, rush, anti_meta, weak_random)`.
Cada game pede instâncias **isoladas** (ver gotcha em §8).

Referência de método: FunSearch (Romera-Paredes et al. 2024) e AlphaEvolve usam
exatamente este loop *propor (LLM/mutação) → avaliar (fitness automático) → manter no
banco de programas → re-amostrar parents*. Força do mapeamento: **forte** para a forma
do loop; **fraca** para o ganho esperado (depende inteiramente da qualidade do fitness).

## 4 — Espaço de busca

Duas frentes, decididas (decisão de arquitetura 2):

**(a) Variantes de estratégia — MVP.** A mutação propõe **knobs** do planner PGS:
- parâmetros de **wave** (tamanho/limiar de ondas grandes vs. spray),
- **hoard** (acúmulo de navios antes de comprometer),
- **threat-value** (peso da ameaça par-a-par forward — a alavanca que destravou desvio em
  4p; ver `bots/pgs/threat.py`).

Gerador MVP = **mutação/perturbação** sobre parents do DB. Upgrade = **LLM-as-generator**
(seção 7).

**(b) Ramo PPO — upgrade documentado.** O gerador dispara treinos
(`train_ppo.py`/`competitive_cycle.py`) e o **checkpoint CPU-exportável** vira candidato
avaliado pelo mesmo gate. **GPU livre para treino**; inferência/submissão **CPU-only**
(invariante D10/D11). Justificativa: PPO do fórum atingiu ~1300 com 12–15h de GPU — o
gargalo desse caminho é **compute**, não a abordagem (força: **parcial**).

**O que NÃO muta:**
- **Pesos do `eval_function`** — teto provado (EXPERIMENTS 77–84). Espaço morto.
- Qualquer dependência de CUDA em `bots/`/`artifacts/` — quebra o invariante CPU-only.

**Prioridade de busca: 4p.** O campo é **maioria 4p (54%)** e toda a família PGS colapsa
para o *floor* do Producer em 4p (0 desvios). 4p é a **alavanca primária** — o gate
default roda 4p (`get_isolated_opponents` montando mesas de 4).

## 5 — Fitness & promoção

**Fitness = robustez multi-oponente** (decisão de arquitetura 3). Ataca diretamente o
overfit-de-oponente-único que falsificou os gates anteriores (gates validados só vs.
Producer floparam — ver §2 e memória `pgs_hold_beats_producer`).

Contrato do gate (`h9_4p_gate.py`):
- **Oponentes**: pool diverso — no mínimo `producer,oep,rush,greedy` (estilos distintos:
  meta-forte, busca, all-in, ganancioso). `--opponents` aceita lista.
- **Horizonte**: **500 steps** (oficial). Avaliar a 96/128 *engana* — OEP "bate" Producer
  só em jogo curto e empata a 500 (memória `kaggle_500_step_eval_required`; força:
  **forte**).
- **Assentos**: **ambos** (simétrico), para não premiar vantagem posicional.
- **Métricas**: `death_rate` (primária em 4p — sobreviver é pré-condição),
  `mean_margin`, `mean_final_planets`.

**Critério de promoção:** o candidato **bate o parent** no gate de robustez —
`death_rate` não-pior **e** `mean_margin` melhor **agregados sobre o pool inteiro** (não
contra um único oponente). Empate dentro do CI ⇒ não promove (topo plano gera ruído).

**Veto da liga:** rodar a liga (`run_league_iteration`) só para **descartar** candidatos
claramente quebrados; jamais para promover.

**Calibração esparsa vs. LB:** periodicamente (não a cada iteração), submeter o campeão
corrente e registrar o score real no DB. Serve para **detectar deriva** entre o
gate-robusto e o LB. Se o gate começar a divergir do LB (como a liga divergiu),
re-projetar o pool de oponentes. O LB é ground-truth, mas **ruidoso (±60) e
rate-limited** — usar com parcimônia e nunca como fitness denso.

## 6 — Autonomia & governança de submit

Decisão de arquitetura 1: **loop autônomo gerando+avaliando 24/7**, mas **submit
governado**.

Guardrails de submissão:
- (a) candidato passa o **gate de robustez** (§5), **e**
- (b) **budget de 1 submissão/dia** para o **melhor candidato não-submetido** registrado
  no DB.

**Submit 100% livre foi considerado e REJEITADO**: o LB é ruidoso (±60) e rate-limited;
submeter cada candidato que passa o gate (i) estoura o rate-limit, (ii) gasta o
ground-truth caro em ruído, e (iii) tenta otimizar contra um sinal estocástico de baixa
amostragem. O budget diário força o pipeline a **escolher um candidato por dia** —
acoplando o gate-robusto (denso, barato) à calibração-LB (esparsa, cara).

**Pré-submit obrigatório:** empacotar + validar o artefato antes de submeter
(`export_submission.py` → `benchmark_submission.py`). O gate valida **código**; a
validação de tarball valida **artefato** — um módulo novo do bot fora da lista fixa de
`BOT_FILES` vira `ModuleNotFoundError` silencioso → fallback (memória
`package_pgs_submission_bot_files_gotcha`; força: **forte**, já nos mordeu).

**Nota de implementação:** o submit automático ao Kaggle **ainda não existe** no repo —
precisa de um wrapper do `kaggle` CLI + credenciais. Até existir, o passo "Governar"
para na validação do tarball e **sinaliza para revisão humana** (submit manual sob o
mesmo budget de 1/dia).

## 7 — Caminho de upgrade

Em ordem de alavancagem (todos com peça já no repo):

1. **LLM-as-generator** (FunSearch/AlphaEvolve): substituir a mutação aleatória por um LLM
   que propõe knobs/estratégias condicionado nos parents de maior fitness e nos
   `rejected` do DB. Ganho esperado: força **parcial** (depende do fitness ser preditivo).
2. **MAP-Elites / QD**: manter um *arquivo* de elites por nicho comportamental (ex.:
   tamanho-de-onda × hoard) em vez de um único campeão — diversidade que ataca o viés de
   população. Já no repo via `train_league.py` (PBT+MAP-Elites+hall-of-fame). Respaldo:
   **forte** (QD é o anti-veneno conhecido do colapso de população).
3. **Ramo PPO completo**: `competitive_cycle.py` já encadeia train→eval→liga
   multi-iteração; promover o checkpoint CPU-exportável como candidato de primeira classe.
   É o caminho provado para ~1300 (compute-limitado).

## 8 — Riscos & como falsificar

| Risco | Sinal de que aconteceu | Falsificação / mitigação |
|---|---|---|
| **Overfit ao gate** (o gate vira a nova liga falsificada) | Campeão do gate diverge do LB na calibração esparsa | Rotacionar/expandir o pool de oponentes; tratar Spearman(gate, LB) < ~0.3 como **gate quebrado**, não candidato ruim |
| **Custo de compute** (ramo PPO domina a fila) | Throughput de candidatos despenca | Separar filas: estratégia (CPU, barata, MVP) vs. PPO (GPU, lenta, assíncrona) |
| **Ruído do LB** mascara ganho real | Score oscila ±60 sem mudança de código | Nunca decidir por 1 submissão; budget diário acumula evidência; resubmit de controle |
| **Inflação de triagem** (~3–4× com poucos seeds) | Margem "boa" some ao aumentar seeds | Nunca decidir por 12–16 seeds; motor **fresco** (build `uv` reverte `.so`; usar `--no-sync`) |
| **Cross-contaminação de oponente** | Treino/eval silenciosamente corrompido | Usar `make_isolated_opponent` por game; `get_isolated_opponents` devolve instâncias cacheadas — pedir o mesmo nome 2× corrompe a memória per-game |

## 9 — Critério de sucesso do pipeline

Condições checáveis (em ordem de exigência):

1. **Loop fecha sozinho**: gerar→avaliar→selecionar→registrar roda ≥24h sem intervenção,
   produzindo candidatos com métricas no DuckDB e status correto.
2. **Gate é preditivo** (a meta real): em ≥5 pontos de calibração esparsa,
   **Spearman(gate-robusto, LB) > ~0.3** — falsificável, mede se *consertamos* o sinal
   que a liga não tinha (que dava ~0.0 / -0.6).
3. **Promoção bate o LB**: ao menos um candidato promovido pelo gate **supera o parent no
   LB real** (não só no gate) — fechando o laço entre fitness denso e ground-truth.
4. **Meta terminal**: candidato promovido pelo pipeline **bate o Producer no LB** (>1228,
   fora do ruído ±60) — pré-condição do Top 5.

O pipeline só é considerado um sucesso quando **(2)** se sustenta; sem isso, ele é uma
máquina de overfit mais rápida.

## 10 — Resultado da validação (2026-06-13)

Pipeline construído e validado ponta-a-ponta. As peças mecânicas funcionam; **o sinal
de fitness reprovou o critério (2)** — exatamente o risco da seção 8.

**Calibração (`calibrate.py`, 4 anchors PGS, 6 seeds, 500 steps):**

| config | LB real | gate fitness |
|---|---|---|
| pgs_holdwave | 1228.8 | **−0.6667** |
| pgs_wave_s100 | 1146.1 | **−0.6667** |
| pgs_hold | 1057.6 | **−0.6667** |
| pgs_allscripts | 1021.5 | −0.8333 |

Spearman = **+0.77**, mas é **FALSE PASS**: os três anchors competitivos (171 pts de gap
no LB) receberam fitness **idêntica**; o ρ vem só de ranquear o piso (allscripts). O gate
**não ordena o topo** — é o mesmo "topo plano / floor-veto-only" que falsificou a liga
([[local_league_is_submission_gate]]). `calibrate.py` agora detecta isso (`competitive_tied`)
e rebaixa o veredito; `self_research` respeita o flag e marca resultados como exploratórios.

**Run honesto (`runner.py`, 6 candidatos, 500 steps):** todos colapsaram para fitness
**−0.5000 idêntica** (`DISCRIMINATES=False`). A discriminação que o MVP mostrou a 80 steps
era **artefato de horizonte curto**.

**Diagnóstico:** o gate de robustez 4p sobre o pool padrão discrimina **regime**
(H9 −0.5 > holdwave −0.667 > allscripts −0.833) mas é **plano dentro de um regime** — e o
campo 4p é onde toda família PGS colapsa pro floor ([[field_is_majority_4p]]). Buscar knobs
dentro do regime H9 é subir uma colina plana → `self_research` foi **encerrado** (compute
não desperdiçado).

**Próximo lever (decisão pendente do usuário — não auto-executado):** o sinal precisa
discriminar o topo. Candidatos:
1. **Exploiters de estilo no pool** (rusher all-in, pgs_bigwave) — `league_agents.py` os
   adicionou justamente p/ separar a família hold; exige wirá-los no gate (hoje league-only).
2. **Avaliar em 2p** — PGS separa em 2p, colapsa em 4p; o campo é 46% 2p.
3. **Cross-regime na busca** — mutar `scripts`/`threat_value_4p`/`defend_in_4p`, não só knobs.

Sem um desses, o pipeline é uma máquina de overfit mais rápida — então o gate fica como
**VETO** (descarta floors) até (2) se sustentar.

## 11 — Runner MVP `arl.py` (Auto-Research Loop governado, 2026-06-18)

O MVP do loop do `goal.md` vive em **`scripts/research_loop/arl.py`** + o núcleo puro
**`scripts/research_loop/policy.py`**. Ele formaliza o que `runner.py`/`self_research.py`
faziam de forma ad-hoc, adicionando o **contrato de iteração**, o **vocabulário de 5
decisões** e os **modos** exigidos pelo objetivo. Reusa as peças existentes
(`genome`/`evaluator`/`registry`) — não duplica avaliação.

**Modos (1 comando cada):**

```bash
# Modo 0 — dry-run: seleciona parent, monta hipótese/comando, NÃO avalia, NÃO edita,
#          valida que o registro no DB funcionaria. Não precisa do .so.
.venv/bin/python -m scripts.research_loop.arl --dry-run  --iterations 1
# Modo 1 — smoke: avaliação minúscula (2 seeds, 60 steps, producer) só p/ validar wiring.
#          NUNCA promove (o piso de seeds garante no máximo needs_more_seeds).
.venv/bin/python -m scripts.research_loop.arl --smoke    --iterations 1
# Modo 2 — research: orçamento honesto (default 6 seeds/500 steps; use --seeds 24 p/ permitir
#          promoção, pois o piso é 16). Mantém patch só localmente; não submete.
.venv/bin/python -m scripts.research_loop.arl --research --iterations 6 --seeds 24
```

**Contrato de iteração** (em `artifacts/research_loop/arl_report.json` e no DuckDB):
`run_id, parent, hypothesis, candidate, patch, commands, seeds, metrics, faults, decision`
(+ `fitness`, `delta`, `reason`, `mode`, `db_id`). O relatório Markdown
(`arl_report.md`) lidera com a **trust line** da calibração.

**Decisões** (`policy.keep_or_discard`, pura e testada — `tests/test_arl_policy.py`):
`promoted | rejected | inconclusive | needs_more_seeds | technical_fail`. Ordem dos
checks É a garantia:
1. **Falha técnica domina** → `technical_fail` (timeout/invalid/bad_status/fallback/
   exceção/p95>budget). **Nunca** vira `rejected` competitivo — `status_for` mapeia para
   `logged`. Uma avaliação que estoura é capturada como `{"error": ...}` e roteada por aqui.
2. Sem amostra válida → `needs_more_seeds` (abaixo do piso) ou `inconclusive`.
3. Sem bar do parent → `inconclusive`.
4. Abaixo do piso de seeds (`--min-promotion-seeds`, default 16; memória "never decide by
   12-16 seeds") → `needs_more_seeds`. **É isto que impede o smoke de promover.**
5. `delta = fitness − parent` fora da banda de ruído (`--noise-band`, default 0.10) →
   `promoted`/`rejected`; dentro da banda (topo plano) → `inconclusive`.

**Guardrail de honestidade:** se a calibração não for confiável (rho<0.3 ou FALSE PASS),
qualquer `promoted` é **rebaixado para `inconclusive`** com o motivo registrado — coerente
com a §10 (hoje a calibração é FALSE PASS, então o loop é exploratório por design).

**Off-limits respeitados:** o runner não altera gates/seeds/thresholds/critérios/pool de
validação; só lê a calibração e usa o gate existente. Sem submit Kaggle em lugar nenhum.

**Como deixar rodando e interromper sem perder rastreabilidade:** o `--research` é
idempotente por iteração — cada candidato é gravado no DuckDB assim que avaliado (tag
`ARL`), então `Ctrl-C` entre iterações não perde nada já registrado; o parent da próxima
execução é relido do frontier do DB (`registry.select_parents`), de modo que o progresso
**compõe** entre runs. Para o daemon contínuo budget-bounded, use `self_research.py`.

**Iteração smoke executada (2026-06-18, registrada):**
`--smoke --iterations 1` rodou a avaliação real (producer, 2 seeds, 60 steps, 11s),
produziu amostra válida (`death=0.000 margin=−0.500 fitness=−0.4995`, `timeouts=0`),
decidiu **`needs_more_seeds`** (2 < piso 16 — smoke não promove), gravou a linha
**id=253** no `experiments.duckdb` (`status=logged`) e escreveu
`artifacts/research_loop/arl_report.{json,md}`. O caminho `technical_fail` foi validado
em integração (`--pool nonexistent_bot_xyz` → exceção capturada → `technical_fail`, não
`rejected`).

**Modo 3 — handoff de promoção (bridge p/ a régua seat-rotacionada, 2026-06-18).** No fim
de um `--research`, a ARL seleciona os **sobreviventes do veto local** (`select_survivors`:
decisão não-vetada **e** `delta > noise_band` — bate o parent no fitness local, ainda que
NÃO-verificado) e **emite** (não roda) o comando da régua real:

```
.venv/bin/python scripts/league_submit_ruler.py --candidates arl_<runid> ... --profile strong
```

Mecânica: cada genoma sobrevivente é gravado em `artifacts/research_loop/candidates/<name>.json`;
`scripts/league_agents.py::_register_league_artifacts()` faz **auto-registro file-drop**
desses JSON como fábricas `_pgs(**genome)` (mesmo idioma dos tarballs/submissions), então
o `league_submit_ruler --candidates <name>` **resolve e roda de verdade** (provado: FACTORIES
constrói o agente — sem fallback silencioso). O comando também vai p/ `promote_survivors.sh`,
p/ o manifesto consolidado `survivors.json` (lista + comando exato) e p/ o bloco `handoff` do
`arl_report.json`/`.md`. Flags: `--ruler-profile {quick,standard,strong}` (default `strong` =
24 seeds/500 steps), `--no-handoff`, `--survivors-json <path>`, e `--run-ruler` (opt-in
explícito que EXECUTA a régua via subprocess; **default OFF = só emite, nunca roda sozinho**
→ `ruler_executed=false` no manifesto). **A ARL nunca submete** ao Kaggle; o veredito de
promoção continua humano + governado (1/dia). O dir de staging é gitignored e não é auto-podado.

> ⚠ **Realidade do sinal plano (§10):** no `--noise-band` default (0.10) sobre o gate H9, a
> fitness local é PLANA (candidatos colapsam p/ ~−0.5000 idêntico), então `select_survivors`
> tende a devolver **0 survivors** → "nothing to hand off", sem comando. Isso é o sinal
> honesto, não bug: o bridge só pré-filtra quem MERECE a régua cara. Para ver o caminho de
> survivors hoje, force com `--noise-band` baixo; o ganho real virá de **consertar o sinal**
> (eval 2p / exploiters de estilo / cross-regime), não de afrouxar a banda.
