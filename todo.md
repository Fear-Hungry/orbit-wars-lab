> **Log de trabalho interno (não é documentação curada).** A documentação de portfólio,
> com uma fonte de verdade por tópico, está em [`docs/`](docs/README.md). O histórico
> detalhado (experimentos rejeitados, resultados por item) vive no **git** e em
> `experiments.duckdb` (DB) — este arquivo (todo.md) fica enxuto, só com o que atacar e o estado atual.

---

# 🐛 CONFIRMADO (2026-06-11, /diagnose) — seletor pode gastar acima do safe_drain (drenagem dupla)

> Achado do usuário, CONFIRMADO empiricamente. `_greedy_select` (`orbit_lite/planner_core.py:367`)
> checa financiamento contra `source_budget` = `obs.ships` BRUTO; `used_src` só impede alvo↔fonte,
> não reuso da fonte. Repro sintético: 2 candidatos da mesma fonte (ships=100, safe_drain=40) →
> 2 waves, 80 naves (2× o teto seguro). Em jogo REAL é raro mas ocorre: producer 4p+rusher
> (8 seeds × 500): 1 violação 2.0× (gastou 24, drain 12); OEP vs rusher (2 seeds): 1 violação 1.5×
> (gastou 108, drain 72 = frações 1.0+0.5 da mesma fonte). Zero em producer-vs-producer 2p (1000
> chamadas): em posição calma drain≈ships e 2·drain>ships não financia — o bug só dispara com a
> fonte AMEAÇADA (drain≪ships), exatamente o regime de derrota (rushers/4p). Probes (tag
> DEBUG-sd01): `/tmp/repro_safe_drain.py` (sintético) e `/tmp/probe_drain_live2.py` (ao vivo).
> PGS não é afetado (não chama `_greedy_select`); artifacts/ são cópias congeladas.

- [ ] **Fix de dois orçamentos em `_greedy_select`** (`orbit_lite/planner_core.py:367-440`): novo
  param opcional `source_spend_budget` (default `None` → clone do `source_budget`, preserva
  callers antigos); `can_fund` passa a checar o spend budget; ao selecionar, debitar OS DOIS;
  continuar retornando `source_budget` como leftover real (contrato do `_plan_regroup` intacto).
  **Atenção de shape**: `drain` é `[S]` (shortlist) e o budget é `[P]` (por slot de planeta) —
  construir `spend = zeros(P); spend[source_idx] = drain.floor()` e passar isso, NÃO
  `drain.floor()` direto como o parecer original sugeria.
  - [ ] verificar: `pytest tests/test_planner_core_source_budget.py` passa (teste abaixo)
- [ ] **Call sites**: producer `bots/producer/_upstream.py:256-273` (spend = scatter de
  `drain.floor()`); OEP `bots/oep/planner.py:1028-1044` (chave `source_spend_budget` no built),
  `:1080-1107` (repassar ao `_greedy_select`), `:1138-1209` (`_masked_score_after_prefix` mantém
  e debita os dois orçamentos; o can-fund da linha ~1174 checa o spend) e `:1258` (segunda
  chamada recebe os dois pós-prefixo).
  - [ ] verificar: re-rodar `/tmp/probe_drain_live2.py` (producer 4p+rusher 8 seeds E oep vs
    rusher) → 0 violações; suítes producer/oep existentes passam
- [ ] **Teste de regressão** `tests/test_planner_core_source_budget.py`: 2 candidatos de alta
  pontuação da mesma fonte, send=40 cada, `source_budget=100`, `source_spend_budget=40` → no
  máximo 1 selecionado, total da fonte ≤ 40, leftover real = 60. Caso 2 (estilo OEP): sends
  40 (frac 1.0) e 20 (frac 0.5) com spend 40 → só o de 40 dispara. Caso 3: spend default
  (None) reproduz comportamento atual (2 waves) — trava o contrato de compat.
  - [ ] verificar: teste falha no código atual (40→80) e passa com o fix
- [ ] **Gate antes de promover**: o fix MUDA o producer (que é a submissão viva, LB 1228) — no
  producer todo candidato envia o drain inteiro, então o fix ⇒ no máx. 1 wave/fonte/turno.
  Violações são raras (~1/6647 turnos-assento), efeito esperado pequeno, mas medir mesmo assim:
  H2H fixado vs incumbente a 500 steps, ambos assentos, 96 seeds frozen + liga como VETO.
  - [ ] verificar: margem ≥ 0 vs incumbente nos 96 seeds E sem rebaixamento no veto da liga
    antes de embarcar em tarball

# 📋 PARECER (2026-06-11, tarde) — review do pacote de 14 commits (régua v4 + hardening)

> Review tech-lead dos commits f2ecf48..cef5d97. Veredito: hardening de validade INTERNA é
> sólido e bem testado; o risco remanescente é de validade EXTERNA — a pool de referências da
> régua (`DEFAULT_REFERENCES`: 6/8 linhagem própria + 2 ext) reproduz a estrutura de viés de
> população que falsificou a liga como gate (Spearman LB = 0.0; Balduzzi 2018). Detalhe na conversa.

- [ ] **Encodar veto-only na régua v4** (mesmo tratamento que o report.json ganhou com
  `bt_predictive`/`lb_inversions`): o report da régua deve marcar explicitamente que o ranking
  PASS>INCONCLUSIVE>REJECT é veto/sanidade, não ordem de promoção; decisão de submeter deve citar
  evidência voltada ao LB (estilo elite big-wave/hoard, desempenho 4p), não posição na régua.
  - [ ] verificar: report da régua contém campo/aviso explícito de veto-only E a próxima decisão
    de submissão registrada no DB cita critério externo à régua
- [x] **Resetar runtime PGS/OEP após fallback** ✅ 2026-06-11 (validado no review): implementação
  MELHOR que a sugerida — `notify_fallback_applied()` troca o `_RUNTIME` de módulo imediatamente
  (o thread zumbi segue mutando a instância VELHA que ele referencia; a nova nasce limpa), em
  TODO fallback, não só pós-overrun. Já embarcado nos tarballs submetidos (53582859/53582886).
  - [x] verificar: `test_fallback_notifies_oep_runtime_reset` + `reset_count==5` no teste de
    bloqueio-e-retomada; suíte das 9 áreas afetadas = 70 passed
- [x] **Monitorabilidade dos runs em background** ✅ resolvida na v11 (validado no review): runs
  v4–v10 morreram TODOS sem output (run.log 0B, sem task_results) — o checkpoint por chunk
  (eac4c6b/ac6ad0c/42c9d4c) + `--task-results-out` corrigiu: v11 (PID 249495, ~3h) tem 8 tasks
  2p gravadas incrementalmente.
  - [ ] (residual) investigar POR QUE v4–v10 morreram silenciosamente (OOM? WSL? wedge pré-SIGALRM)
    — se foi wedge de agente, o hard timeout cbb4ab7 já cobre; confirmar se v11 fecha inteiro
- [ ] **Instrumentar o floor-por-budget do planner PGS** (gap achado no review de 9ee1a7c): o
  `budget_low() → producer_floor_payload()` é um fallback SILENCIOSO um nível abaixo do wrapper —
  em hardware lento, `SUBMISSION_STATS` mostraria `fallbacks=0` com o PGS jogando Producer puro o
  jogo todo (a classe exata de degradação invisível desta campanha). Contar só os retornos por
  budget_low (floor_in_4p/deviation_max_step são comportamento desenhado, não contam).
  - [ ] verificar: contador (ex.: `budget_floor_returns`) exposto ao wrapper/`SUBMISSION_STATS`,
    reportado pelo `validate_pgs_tarball` e ≈0 na validação local
- [ ] **Commitar os 14 arquivos modificados** (reset pós-fallback, semântica de fault no
  benchmark/objective_validation, half2p league-only): os tarballs VIVOS no Kaggle foram gerados
  deste working tree — sem commit, a evidência hash-bound (bf3dca8) aponta para fonte
  irreproduzível.
  - [ ] verificar: `git status` limpo e sha1 dos tarballs submetidos reproduzível do HEAD
- [ ] **(menor) lineup 4p pode variar com o painel** quando o candidato pertence ao template
  (`_complete_4p_lineup` completa com `ref_order`, que inclui os outros candidatos do comando) —
  documentar ou fixar o preenchimento só com referências para manter o claim de estabilidade.
  - [ ] verificar: mesmo candidato + mesmo template ⇒ mesmo lineup, com qualquer painel

# ✅ RESOLVIDO (2026-06-11) — entrypoint duplicado do PGS: "pgs" podia significar OUTRO bot

> Achado (auditoria do usuário, confirmado): `bots/pgs/planner.py` ainda expunha `agent()` sobre
> `PGSRuntime()` com os DEFAULTS do dataclass = **all-scripts (config rejeitada, LB 1022)**,
> enquanto `bots/pgs/agent.py` pina `SUBMISSION_CONFIG` (hold+w60s150). Divergência REAL em uso:
> `registry.pgs_agent` (heuristic policies, usado em treino/eval) importava do planner —
> `get_heuristic_policies()["pgs"]` e `get_isolated_opponents("pgs")` eram bots DIFERENTES.
> Mesmo mecanismo do incidente da submissão all-scripts acidental (id=129/142).

- [x] **Entrypoint único** ✅ 2026-06-11 (DB id=181): `agent()`+`_RUNTIME` REMOVIDOS do planner
  (nota no lugar explicando o porquê; `make_runtime` fica); `registry.pgs_agent` roteia para
  `bots.pgs.agent`; `pgs_allscripts` da liga segue existindo mas EXPLÍCITO e nomeado (intencional,
  âncora do hard gate).
  - [x] verificar: `test_planner_exposes_no_default_entrypoint` (trava a reintrodução) +
    `test_registry_pgs_routes_to_operational_entrypoint`; 25 testes passam nas 7 suítes
    consumidoras (pgs_bot, isolation, registry, floor_fidelity, parallel, entity, masks)
- [ ] **atenção (não-bloqueante)**: resultados de treino/eval ANTIGOS que usaram
  `get_heuristic_policies()["pgs"]` mediram contra o all-scripts (bot fraco, LB 1022) — se alguma
  conclusão dependeu desse oponente, reler com isso em mente
  - [ ] verificar: grep nos configs/logs de treino por oponente "pgs" não-isolado antes de reusar conclusão

# ✅ RESOLVIDO (2026-06-11) — cache de tarball da liga rodava código VELHO após re-export

> Achado (auditoria do usuário): `league_agents._tarball_agent` só extraía se `cache/main.py` não
> existisse — re-exportar um tarball com o mesmo nome (ex.: novo campeão da worktree B em
> `submission_brep.tar.gz`) continuava rodando o código da PRIMEIRA extração. Clássico "testei o
> bot novo, mas era cache velho" — e os jogos da liga atribuíam o resultado à versão errada.

- [x] **Cache keyed por hash de conteúdo** ✅ 2026-06-11 (DB id=180): dir = `<nome>-<sha1[:12]>`;
  re-export ⇒ extração E overlay de import (`_TARBALL_ISO`) novos; instâncias já construídas
  mantêm a própria versão (sem swap retroativo); tarball ausente agora falha LOUD mesmo com
  cache quente (antes rodava versão desconhecida em silêncio).
  - [x] verificar: `tests/test_league_tarball_isolation.py` 4 passed (re-export invalida cache;
    versão antiga preservada na instância em voo; FileNotFoundError sem o tarball); smoke real
    criou `cache/brep-e55f6dedfbdc/` e o agente carrega
- [ ] **limpeza opcional**: os dirs antigos keyed por nome (`cache/brep/`, `brep_league3/`,
  `*_probe/`) viraram órfãos — nunca mais são usados; deletar quando quiser
  - [ ] verificar: `ls artifacts/league/cache/` só com dirs `<nome>-<hash>` após a limpeza

# ✅ RESOLVIDO (2026-06-11) — report encoda a falsificação do BT (veto-only legível por máquina)

> Achado (auditoria do usuário): spearman_lb=0.036 impresso, mas a tabela seguia ranqueando
> `pgs_wave_s100` ACIMA de `pgs_holdwave` (CI90 até separado!) com o LB real dizendo o contrário
> (1228.8 vs 1138.6, gap 90 > ruído ±60) — o banner avisava, mas o `ranking` do report.json
> continuava consumível como ordem de promoção.

- [x] **`bt_predictive` + `lb_inversions` no report** ✅ 2026-06-11 (DB id=179): `bt_predictive`
  exige spearman ≥ +0.6 com ≥ 5 âncoras (a condição "sustentado ≥10 rounds" segue manual/loop —
  report é stateless); `lb_inversions` lista par-a-par onde o BT inverte o LB com gap acima do
  ruído de resubmissão (±60). Real: **5 inversões** — `pgs_hold` inflado em 4 pares (vs holdwave/
  oep/producer/brep) + `s100 > holdwave` (a do achado). Aviso explícito: "ordem da tabela é
  INTERNA da pool — não usar como ranking de submissão".
  - [x] verificar: 15 testes passam (unidade `lb_inversions` filtra gap ≤ ruído; fim-a-fim exige o
    aviso + campos no JSON); report real imprime as 5 inversões e `bt_predictive=false`
  - [ ] consumidores futuros do `report.json` devem checar `bt_predictive` antes de usar `ranking`
    como ordem de campo. Hoje há UM consumidor automático: `league_run.top5()` usa o ranking p/
    adensar matchmaking (30% dos 2p no topo) — uso INTERNO da pool, legítimo (não é promoção);
    challenger/intransitivity não leem report.json (verificado)

# ✅ RESOLVIDO (2026-06-11) — liga agora APLICA semântica Kaggle (antes só media timeout/invalid)

> Diagnóstico (auditoria do usuário, confirmado no código oficial instalado): a liga contava
> timeout/invalid mas deixava a ação entrar — bot que crashava/estourava tempo continuava jogando e
> podia VENCER localmente, quando na Kaggle estaria morto. Semântica real (core.py/agent.py/
> orbit_wars.py do kaggle_environments): exceção → `ERROR`; estouro de act além do banco
> `remainingOverageTime` (**12s**, checado ANTES do decremento) → `DeadlineExceeded` → `TIMEOUT`;
> nos dois casos o agente NUNCA mais age e recebe `reward=None` (não pode vencer nem com max ships).
> Movimento inválido NÃO penaliza (o `process_moves` oficial só pula a entrada) — regra exata `len==3`.

- [x] **Semântica aplicada no `league_match`** ✅ 2026-06-11 (DB id=178): banco de overage 12s por
  assento; ERROR/TIMEOUT matam o agente até o fim do jogo (planetas seguem produzindo, frotas em
  voo continuam); assento errado é EXCLUÍDO do argmax de vencedor; `agent_status` por assento
  sempre gravado no game dict; check de entrada inválida alinhado ao oficial (`len==3`).
- [x] **Consumidores alinhados à fonte única** ✅: `decisive_winner` do `league_report` respeita
  `agent_status` (ausente = jogos antigos, todos elegíveis — comportamento inalterado);
  `league_challenger` e `league_intransitivity` deixam de ter argmax próprio e importam a regra.
  - [x] verificar: 14 testes passam (kill por ERROR, kill por exaustão do banco, banco absorve
    overrun pequeno, errado não vence, invalid sem penalidade); report real inalterado
    (4997 decisivos no acervo antigo)
- [ ] **ressalva a monitorar**: timing local tem contenção (liga roda jogos em paralelo) — o banco
  de 12s absorve ruído, mas se aparecer TIMEOUT local em bot validado <1s no tarball oficial,
  suspeitar de contenção antes de culpar o bot
  - [ ] verificar: nos próximos rounds, seção FAULTS sem TIMEOUT esporádico em bots com p95 < 200ms

# ✅ RESOLVIDO (2026-06-11) — liga v1 inauditável p/ faults: ausência da chave era lida como "limpo"

> Diagnóstico (auditoria do usuário, confirmado fim-a-fim): **5025/5025 jogos** em `artifacts/league/v1`
> sem a chave `faults`, e `league_report.aggregate_faults` tratava ausência como zero fault —
> crash/timeout/invalid antigos viraram "jogo limpo". Causa-raiz: o contrato omit-when-clean do
> `league_match` (chave omitida quando limpo, "old-JSON compat") tornava jogo PRÉ-instrumentação
> indistinguível de jogo limpo PÓS-fix — por isso até os rounds pós-112 estão sem a chave.

- [x] **Contrato invertido + cobertura de auditoria** ✅ 2026-06-11 (DB id=177): `league_match` agora
  SEMPRE grava `faults` (`{}` = auditado limpo; ausente = pré-instrumentação/NÃO auditado);
  `league_report` ganhou `fault_audit()` + linha `⚠ auditoria de faults: X/N auditados` + campo
  `fault_audit` no `report.json`; docstrings corrigidas (ausência ≠ limpo).
  - [x] verificar: 12 testes passam (`test_league_match_faults` com o contrato novo travado +
    `test_fault_audit_separates_unaudited_from_clean`); report real imprime
    `0/5025 jogos auditados — 5025 pré-instrumentação`
- [ ] **decorrência**: o acervo histórico segue não-auditável retroativamente (faults nunca foram
  medidos lá) — a decisão já aberta de arquivar/re-acumular `cont/` (itens pgs_* e brep pré-fix
  acima) agora tem mais um motivo; novos rounds entram auditados automaticamente
  - [ ] verificar: após os próximos rounds da liga, `report.json["fault_audit"]["audited"]` > 0 e crescendo

# ✅ RESOLVIDO (2026-06-11) — empacotador OEP com fallback SILENCIOSO (mesmo bug já corrigido no PGS)

> Diagnóstico (/diagnose, reproduzido): `scripts/package_oep_submission.py` (MAIN_TEMPLATE) engolia
> exceção com `except Exception: pass` e retornava `_producer(obs)` em timeout/erro **sem
> `SUBMISSION_STATS`**. Repro determinístico: OEP 100% morto → 10/10 jogadas do Producer, episódio
> "completa", e `benchmark_submission` (que detecta fallback SÓ via
> `agent.__globals__["SUBMISSION_STATS"]`) via **0 fallbacks** — Producer rodando escondido no LB.
> Era exatamente o bug travado no PGS em 2026-06-10; o template OEP nunca recebeu o fix (único
> commit: d601511).

- [x] **Portado o template instrumentado do PGS** ✅ 2026-06-11: SUBMISSION_STATS
  (calls/fallbacks/timeouts/fallback_errors) + `box["err"]` no except + kill-switch
  `_MAX_CONSEC_TIMEOUTS=3` (threads daemon estouradas roubam CPU dos steps seguintes);
  preservados o `os.environ.setdefault("OEP_MIN_ADVANTAGE", ...)` e `agent` como ÚLTIMO
  callable (gotcha get_last_callable).
  - [x] verificar: `tests/test_oep_submission_fallback_instrumented.py` (espelho do PGS, 6 testes)
    passa — OEP morto 10x → stats {calls:10, fallbacks:10, fallback_errors:10}; kill-switch para
    de lançar OEP após 3 timeouts; repro original re-rodado pós-fix: fallback agora VISÍVEL.
- [x] **Tarballs STALE re-empacotados** ✅ 2026-06-11: `submission_pgs_hold.tar.gz` e
  `submission_pgs_wave_s100.tar.gz` também tinham o wrapper SEM instrumentação (0 ocorrências de
  SUBMISSION_STATS — fallback invisível; explica underperformance "sem erro" dos refs 53541125/53542864:
  timeout no hardware Kaggle vira Producer silencioso, e o wrapper antigo nem tinha kill-switch →
  espiral de CPU). Re-empacotados com as MESMAS configs pinadas (hold: `scripts="hold"`; wave_s100:
  `scripts="hold", wave_min_ships=60.0, wave_start_step=100`) + `submission_oep.tar.gz` (defaults
  min_advantage=15, budget 0.6); os três embarcam o bots/pgs atual (inclui fix floor fidelity id=171).
  - [x] verificar: `tar -xzOf <t> main.py | grep -c SUBMISSION_STATS` = 2 nos três; configs pinadas
    conferidas; `validate_pgs_tarball` 500 steps = VALIDATION OK nos três (0 fallbacks locais)
  - [ ] **decidir: re-submeter hold/wave_s100 com os tarballs instrumentados?** Ressalva honesta:
    (a) a hipótese "fallback silencioso explica o LB baixo" é PLAUSÍVEL mas não-verificada (local
    valida com 0 fallbacks; só o hardware Kaggle mais lento dispararia timeouts); (b) o Kaggle NÃO
    expõe SUBMISSION_STATS — o ganho real lá é o kill-switch (evita a espiral de timeouts), a
    instrumentação serve aos gates/benchmarks LOCAIS; (c) custa slots de submissão
    - [ ] verificar: se re-submeter, comparar score estabilizado novo vs antigo (mesma config) —
      delta >> ruído ±60 confirmaria a hipótese do fallback
- [ ] **(prevenção) decidir: extrair o wrapper para módulo único** compartilhado pelos empacotadores
  (PGS/OEP/BReP geram o mesmo wrapper por cópia — foi a duplicação que deixou o OEP para trás)
  - [ ] verificar: um único template-fonte; os testes de fallback dos dois empacotadores importam dele

# ✅ RESOLVIDO (2026-06-10, noite) — instrumento da liga consertado; separação fina DEFERIDA a H4/H5

> **Decisão do usuário (2026-06-10):** PARAR a frente de exploiter manual. Instrumento consertado (VETO + regra de
> promoção + coluna H2H + âncoras + footgun). A **retrodição NÃO foi perseguida até o fim de propósito**: (1) parte
> da ordem de campo que ela exige reproduzir é RUÍDO Kaggle (~±60-80, medido no resubmit idêntico do holdwave
> 1228→1151-1189): s100 vs hold (51 pts) e holdwave vs s100 (~80-120) não são estatisticamente separáveis no
> próprio campo; só **holdwave >> hold (170 pts) é real** — e a liga ainda empata os dois porque vs o pool eles
> jogam quase idêntico (holdwave só segura ondas <60 navios até step 150, raramente decisivo vs família Producer).
> (2) Separá-los exigiria um exploiter INTERCEPTADOR (counter-puncher que abate frotas pequenas), não um atacante —
> mas exploiter MANUAL já se mostrou insuficiente (rusher 0-win, bigwave aniquila mas perde). → separação fina
> deferida às hipóteses APRENDIDAS H4 (MAP-Elites, DB 160) e H5 (PSRO+rectified-Nash, DB 161).

## FOCO ANTERIOR (resolvido acima) — gate da liga FALSIFICADO pelo campo

> Diagnóstico (sessão 2026-06-10): `pgs_hold` (liga #2, P≥producer=1.00) → LB **1057.6** e `pgs_wave_s100`
> (liga #1, LB_est ~1225) → LB **1036.6** — ambos ~115-135 pts ABAIXO do producer (1173.1), nível allscripts.
> Spearman recalculado com as âncoras de hoje: **0.0** (7 âncoras) e **-0.6** na faixa competitiva (sem allscripts).
> Empacotamento DESCARTADO (configs pinadas verificadas nos 3 tarballs); ext agents sem crash (probe 0 exceções).
> Causa: rating BT é relativo à população — pool 8/10 producer-lineage, ext bots fracos demais p/ discriminar o topo
> (Balduzzi et al. 2018, arXiv:1806.02643; AlphaStar/PSRO p/ exploiters). Detalhe: s100 foi promovido com CI90
> sobreposto ao holdwave e PERDENDO H2H p/ hold (0.44).

- [x] **Rebaixar o gate da liga a VETO** ✅: `GATE_REFERENCE` agora é "veto floor" + `INCUMBENT="pgs_holdwave"`;
  docstring de `league_report.py` traz a regra de promoção de 3 condições. Liga não promove sozinha.
  - [x] verificar: `scripts/league_agents.py` documenta a regra nova (comentário VETO ONLY + falsificação)
- [x] **Adicionar âncoras de hoje em `LB_ANCHORS`** ✅: `pgs_hold=1057.6`, `pgs_wave_s100=1109.4` (ainda subindo,
  marcado p/ refresh); resubmit holdwave 53542884 anotado como ruído ~±60 (1151.6 hoje), não âncora.
  - [x] verificar: report imprime 7 âncoras (producer/oep/brep/allscripts/holdwave/hold/s100); Spearman caiu p/ +0.1-0.3
- [~] **Injetar exploiters de estilo no pool** — PARCIAL (achado honesto): `pgs_bigwave` (hold/wave100/delay25) entrou
  no pool — em 24 seeds PERDE a família hold na média (win 0.38/0.25/0.23 vs hold/holdwave/s100) MAS ANIQUILA
  23-38% dos jogos (sinal discriminativo real, ≠ rusher 0-win). Útil como sonda, insuficiente sozinho. Mas o
  **RUSHER FALHOU**: v1 (spread/s50) e v2 (focus-fire/f120) deram **0-win vs a pool inteira** → fica FORA (free win =
  viés de população, Balduzzi 2018). Construído em `bots/exploiters/rusher.py` mas não registrado no POOL.
  - [ ] verificar (retrodição) — **NÃO ATINGIDO** com exploiters baratos: a liga AINDA não ranqueia holdwave acima de
    hold/s100 com CI90 separados (s100 1080[1056,1100] ≈ holdwave 1065[1042,1085] ≈ hold 1064[1043,1087], sobrepostos).
    → ESTE é o achado que motiva H4/H5: exploiter manual não reproduz o campo; precisa de exploiter APRENDIDO/QD.
- [x] **Regra estatística de promoção** ✅: docstring do `league_report.py` exige (1) P≥ref≥0.6 (2) CI90 separado do
  incumbente (3) H2H≥0.5 com N≥60; report imprime a coluna `H2Hinc` (cand vs incumbente) + 2 linhas de veredito.
  - [x] verificar: regra escrita no league_report E coluna H2Hinc impressa (ex.: s100 46-41, hold 91-91 vs holdwave)
- [x] **Footgun do report** ✅: default agora é `DEFAULT_GLOBS` (p*+cont+waveround+bl3round); `league_run.py` importa e
  usa o MESMO glob no standings.
  - [x] verificar: `python -m scripts.league_report` sem args agrega 2779 jogos (não mais o subset p*)
- [x] **Registrar no experiments.duckdb** ✅: id=159 (falsificação do gate, status=measured) com diagnóstico completo.

### Liga "redonda" = validador todos-contra-todos do PPO (2026-06-10, noite)

> Objetivo do usuário: liga como validador all-vs-all (como o Kaggle), sem viés de comparar só vs Producer, para
> mostrar que o PPO bate QUALQUER bot anterior. Métrica certa = matriz H2H par-a-par (não o BT agregado, que é
> enviesado pela população — Balduzzi 2018).

- [x] **#1 Challenger report** (`scripts/league_challenger.py`): dado um bot, roda H2H vs CADA membro do pool
  (2p ambos assentos, 500 steps, paralelo), com Wilson CI + coluna LB + veredito `DOMINA / bate N de M`. Validado
  (holdwave bate producer/allscripts, perde p/ hold → reproduz a liga). **Critério p/ PPO:** win ≥ 0.50 vs cada bot
  real, ≥40 jogos decisivos/par.
- [~] **#2 Adicionar elite do MAP-Elites ao pool — PULADO (redundante)**: o "vencedor" w2_c3/w3_c4 tem genoma =
  `pgs_bigwave` (já no pool); o "fit +1.00" foi inflação de amostra pequena (8 jogos; bigwave tem BT 940 na liga).
  MAP-Elites re-achou estilos existentes, não gerou novo → nada a adicionar.
- [~] **#3 Burst nos pares magros** (`/tmp/burst_thin.sh` → cont/burst_*.json): 11 pares <30 jogos (bigwave/
  brep_league3 recém-entrados) + 1 par vazio (allscripts×bigwave). Rodando. **verificar:** após o burst, todos os
  55 pares do pool com ≥30 jogos decisivos.

### Novas hipóteses (pedido do usuário: não chegamos a #1 → pesquisar gap + levantar hipóteses no DB)

> Gap estrutural confirmado: melhor do time = holdwave 1228 (estável); top-5 ≈ 1575; líder 1679 → ~350-450 pts.
> A falha de avaliação de hoje (pool não-diversa, BT enviesado, intransitividade) é exatamente o que QD e PSRO+Nash
> foram desenhados para resolver. Exploiters MANUAIS não bastam (rusher 0-win) → precisa de geração APRENDIDA.

- [~] **H4 (DB id=160, EXECUTADA id=164) — MAP-Elites/QD sobre o espaço do PGS** (`scripts/mapelites_pgs.py`):
  48 evals, 4 workers, descritores medidos do jogo (wave_size × cadence), fitness vs {producer,holdwave}.
  - [x] verificar (local): arquivo cobre ≥4 nichos com ≥3 batendo Producer → **ATINGIDO: 9 nichos, 4 batendo Producer**
    (w1_c3 wave24, w2_c3 wave47 fit+1.0, w2_c4 wave35=s100, w3_c4 wave72=big-wave).
  - [x] verificar (não-hold): **REFUTADO** — única região não-hold (`hold,snipe`) não bate nem Producer (fit −0.75).
    Confirma ablação DB 77-84: hold é o ÚNICO desvio vencedor do PGS. → diversidade vencedora NÃO existe dentro do PGS.
  - [ ] ≥1 elite não-hold cruza 1228 no LB → **impossível** (não há elite não-hold vencedor); candidato hold-family
    `w3_c4` (big-wave, wave_min~100/start~68) pode ser submetido se o usuário quiser — mas é da família hold.
- [~] **H5 (DB id=161, PARCIAL id=163) — Liga open-ended PSRO+rectified-Nash** (`scripts/league_intransitivity.py`):
  - [x] verificar (intransitividade): matriz mostra ciclo real holdwave→bigwave→oep→holdwave; ||cíclico||/||A||=0.729
    (inflado por quase-empates do topo — ressalva honesta, mas o ciclo é real).
  - [x] verificar (meta-Nash): **MEDIDO — NÃO supera puras**: no pool atual `pgs_hold` é Nash PURO não-explorável
    (max ganho de qualquer pura = +0.000). Achado: pool hand-picked tem Nash puro → precisa exploiter do hold p/ Nash misto.
  - [ ] best-response treinado cruza 1228 no LB → **🔒 não-executável nesta sessão** (precisa PPO worktree B + submissão).

---

# 🎯 FOCO ANTERIOR (2026-06-09, tarde) — frente HEURÍSTICA nesta worktree; BReP/DRL fica na worktree B

> Decisão do usuário: BReP roda em paralelo na **worktree B** (sem conflito com esta); esta worktree ataca
> **heurística/metaheurística/hiperheurística**. Linhas MORTAS que NÃO voltam (DB 77–99 + memória): tuning de
> pesos de eval (família H), knobs OEP, e qualquer SELEÇÃO sobre candidate-set que contém o Producer (teto de
> paridade provado — workflow ppo-explore). Literatura: Burke et al. 2013 (Hyper-heuristics: a survey,
> 10.1057/jors.2013.71 — distinção SELEÇÃO vs GERAÇÃO de heurísticas; nossa evidência mata seleção → ir de geração);
> Churchill & Buro 2013 (Portfolio Greedy Search, 10.1109/cig.2013.6633643) e 2015 (Hierarchical Portfolio Search,
> Prismata); Wang et al. 2016 (Portfolio Online Evolution, StarCraft); Gaina et al. 2022 (RHEA, 10.1109/TG.2021.3060282).
> **Força da evidência: FORTE** para portfolio-search em jogos RTS-like multi-unidade; **PARCIAL** para Orbit Wars
> especificamente (sem paper no domínio; o análogo comunitário é o planner timeline/sim-value do fórum ≈ T8).

- [x] **H-P0. Medir o orçamento de simulação por turno** ✅ 2026-06-09 (`scripts/hp0_sim_budget.py` →
  `artifacts/hp0_sim_budget.json`). Worst-case **517 avaliações/turno** a H=50 (setup ~6ms, candidato ~1.3ms,
  single-thread); OEP inteiro usa só 20–35ms/turno. **GO com folga** (gate era ≥20).
- [x] **H-P1. Portfolio de scripts de missão — IMPLEMENTADO** ✅ 2026-06-09 (`bots/pgs/planner.py`): portfolio
  por planeta-fonte {PRODUCER, HOLD, SNIPE, CAPTURE(payback≤20t), REINFORCE(chega antes do flip), EVAC};
  **piso = atribuição all-PRODUCER via o MESMO gerador (`ProducerLiteRuntime`)** — análogo heurístico do KEEP do
  BReP. Scripts enxergam a projeção COM os lançamentos previstos do oponente (determinístico ⇒ plano do turno é
  exatamente previsível — a brecha sound).
  - [x] verificar: floor (max-deviations=0) vs Producer = **margem 0.0 EXATA** (gerador fiel; mapas simétricos)
- [~] **H-P2. Busca PGS por turno — implementada; TUNING em curso** (greedy sob modelo estático + árbitro REATIVO
  conservador, port do `_reactive_reply_entries` do OEP; margem mínima de aceitação `arbiter_margin=25`).
  Aprendizados medidos (1 seed): valor território-dominante → −0.35@60 (overexpand); valor naves-dominante sem
  consciência defensiva → −1.0@120 (passividade); **com consciência defensiva + árbitro: +0.64@120**, mas
  **a 500 steps reverte (4 seeds: −0.5, win 0.25)** — ganha cedo, perde no longo (mesma armadilha do OEP em jogo
  curto). p95 decision 106–160ms (ok < 700ms).
  - [x] diagnóstico da reversão ✅: (a) floor ≈ Producer a menos de RUÍDO DE FLOAT (1ª divergência: ângulo na 6ª
    casa, step 18; `scripts/check_pgs_floor_fidelity.py`) → jogos floor-vs-Producer a 500 steps são cara-ou-coroa
    de aniquilação (caos), margem por seed = ±1; (b) com desvios, o resultado fica DETERMINADO pelo bot (mesmo
    resultado nos 2 assentos): seed 1000 vira W-W (aniquila Producer no step ~200), 1001–1003 viram L-L. Saldo
    4 seeds: −0.5. Gate de fase (≤150) NÃO muda. → desvios decidem jogos; precisa isolar QUAL script ajuda.
  - [x] **ablação por família de script** ✅ (8 seeds × 500, `artifacts/pgs/abl_*.json`): **HOLD +0.375 (win 0.69,
    seat0 7W/1L)**; EVAC 0.0; SNIPE −0.25; CAPTURE −0.5; REINFORCE −0.625; all+árbitro-80 −0.625. Conclusão:
    o desvio que VENCE é **vetar lançamentos ruins do Producer** (sem risco de mira/compromisso); scripts que
    ADICIONAM lançamentos não passam pelo avaliador 1-ply (modelo estático infla). H-P1 formalizado em
    `tests/test_pgs_bot.py` (8 passed: legalidade 2p/4p + floor ≈ Producer). `pgs` registrado no registry
    (isolável; STATEFUL_SINGLETON_OPPONENTS).
  - [x] **triagem 16 seeds ✅ CRITÉRIO BATIDO** (`artifacts/pgs/t16_*.json`): **hold-only (árbitro 25):
    mean +0.334, seat0 +0.554, seat1 +0.114, win 0.656** — ambos assentos > 0; hold-árbitro-10: +0.27 com
    seat1 −0.01 (conservadorismo do árbitro paga); p95 ~106-160ms < 700ms; crash/invalid = 0.
  - [x] **GATE DECISOR (H-P3) ✅ PASSOU** 2026-06-09 (id=122 no DB): 96 seeds FROZEN (9000–9095) vs Producer,
    500 steps, 2 assentos: **mean +0.2181, seat0 +0.2488, seat1 +0.1874, 117W/73L/2T (win 0.609, p≈0.0007)**,
    p95 83.7ms / max 249ms, crash/invalid 0. **PGS hold-only SUPERA o Producer** — primeira linha a quebrar a
    paridade. Suíte completa 241 passed. `artifacts/pgs/gate96_9*.json`.
    - [x] verificar: mean > 0 E seat0 > 0 E seat1 > 0, crash/invalid = 0, p95 < 700ms ✓ todos
  - [x] cross-check vs OEP ✅ (32 seeds frozen, 500 steps): **+0.406** (seat0 +0.375 / seat1 +0.4375, win 0.703)
    — PGS bate as DUAS réguas. `artifacts/pgs/gate_oep32.json`.
  - [x] **empacotado e SUBMETIDO** ✅ 2026-06-09 (id=123 no DB; pedido explícito do usuário). Tarball
    `artifacts/submission_pgs.tar.gz` (`scripts/package_pgs_submission.py`): main.py com budget 0.7s + fallback
    Producer, `agent` último callable (gotcha get_last_callable), bots/pgs vendorizado (`_helpers.py`, sem
    dependência do planner OEP). Validação oficial (`scripts/validate_pgs_tarball.py`): 500 steps DONE/DONE,
    p95 182ms / max 355ms. **Submissão Kaggle ref=53519882 (PENDING)**.
  - [x] **verificar score LB do ref=53519882** ✗ **REPROVADO 2026-06-10 (DB id=129): LB 1001.7 << 1228**.
    Diagnóstico (/diagnose, replays + diff): **NÃO foi erro** (61 eps COMPLETE, 0 timeout/ERROR; diff replay:
    111/114 ações do Kaggle = PGS local). Foi **falha de generalização**: (a) vetos HOLD validados SÓ vs
    Producer (árbitro modela oponente como Producer-like) — em 2p, único regime com vetos, fez 13W/10L vs
    bots ~1000 (esperado ~73% se nível Producer); 0W/11L vs >1100; (b) regime: TODAS as derrotas =
    aniquilação steps 115–238 por rushers/expanders — o gate local mede margem a 500 steps vs Producer,
    regime que não existe no LB. Em 4p (floor puro, sem vetos) 11W/5L vs ≤1050 ≈ esperado. Comparador:
    Producer sub=53366194 segue estável a 1173 no MESMO campo (16W/21L vs opp~1168). Producer continua o default.
- [ ] **H-P4 (NOVO, decorre do id=129): gate de ROBUSTEZ DE CAMPO antes de qualquer nova submissão.**
  O gate "96 seeds vs Producer a 500 steps" provou-se NÃO-preditivo do LB (PGS: +0.218 local → 1001.7 LB).
  Duas correções no protocolo, em ordem de custo:
  - [~] **(a) adicionar oponentes de CAMPO reais à régua local** — EM CURSO 2026-06-10:
    - [x] medido: rush/greedy/anti_meta do registry são FRACOS DEMAIS — floor E hold aniquilam 32/32
      com margem +1.0 (`artifacts/pgs/field/*.json`) → registry não serve de régua de campo
    - [x] **T0 pool baixada**: 4 agentes públicos usáveis (LB 1224/1110/1100/1050, 2 linhagens distintas,
      stdlib-only) em `artifacts/opponents/top5_proxy/` (README com refs) + `scripts/eval_vs_external.py`
      (isolamento por instância, mesmas métricas + annihilated_rate)
    - [x] triagem lida (DB ids 136–137): **TODOS os 7 kernels públicos screenados são aniquilados pelo
      Producer local** (leva 1: 12 matchups 16/16 margem +1.0; leva 2: hellburner/konbu17 8/8, orbitbotnext
      quebrado na fonte). Harness local VALIDADO: interpretador oficial reproduz (Producer aniquila lb1224
      4/4); lb1050 aniquila lb1224 8/8 (títulos de kernel ≠ código publicado)
    - [x] verificar ✓: Producer/PGS >> toda a pool pública; **conclusão: o campo 1500+ é privado** —
      top-5 hoje = ~1575+ (líder 1731); kernels públicos NÃO servem de régua de campo
    - [x] **T1 replay-mining FEITO** (DB id=138): ladder-walk → sub 53402231 (~1710), 3 replays de
      vitórias sobre 1580–1600 minerados + contraste com nossa derrota LB
    - [x] verificar ✓ taxonomia (3 padrões): (1) **disciplina de onda** — elite lança ~0.4/step com
      60–80% ≥50 navios; Producer/PGS = "spray" 1.4–1.6/step mediana 14, 5% ≥50 (perfil dos PERDEDORES
      de elite); (2) **hoard** — 2–5× navios do oponente em t=100–350 mesmo com menos planetas (score =
      MARGEM DE NAVIOS); (3) expansão rápida só até t=50, depois consolida
- [x] **H-P5 v0 (disciplina WAVE incondicional) — REJEITADO** (DB id=139). Knobs `wave_min_ships`/
  `wave_start_step` implementados no `PGSConfig` + métrica `launch_profile` no `eval_pgs_direct`
  (ficam no código, default OFF). O perfil convergiu ao elite (%≥50: 25%→69%) mas a margem caiu
  monotonicamente nas DUAS réguas: vs Producer +0.334→+0.158 (wave30)→+0.007 (wave50); vs OEP
  −0.054→−0.154. Veto incondicional por tamanho perde tempo contra geradores que pegam neutros
  primeiro. **Aprendizado colateral importante**: hold-only vs OEP = −0.054 nos seeds 1000–1015 mas
  +0.406 nos frozen 9000+ — triagem 16s é MUITO ruidosa; decisão só a 96 seeds (regra reforçada).
- [~] **H-P5 v1 (/goal ATIVO): fusão CONDICIONAL de ondas — IMPLEMENTADO, triagem rodando.**
  `_wave_merge_filter` no planner: agrupa lançamentos do floor por ALVO; só grupos de ATAQUE
  (alvo inimigo) abaixo de `wave_min_ships` são retidos, liberando quando o grupo cruza o limiar
  OU envelhece `wave_max_delay` (8) turnos; expansão (neutros) e defesa (próprios) NUNCA filtradas
  (lição do v0, id=139). Estado mínimo entre turnos: `_wave_pending {alvo: 1º step retido}`, reset
  no step 0. Testes `test_pgs_bot.py` 3 passed (floor fidelity intacta — default off); trace
  instrumentado: filtro cirúrgico (11 retenções/250 turnos).
  - [x] triagem lida (DB id=140): w40 → Producer +0.262 / OEP −0.118; w60 → Producer +0.191 /
    OEP −0.065. **Pareado por seed: nenhum sinal positivo** (Producer 7↑/8↓ e 6↑/8↓; OEP ruído;
    16–26 seeds inalterados por par — só flips ±1 levemente negativos em 64 jogos pareados)
  - [x] w40/w60 (start 50): REPROVADOS na triagem (id=140) — reter ataque na fase de EXPANSÃO
    cede tempo; v0 incondicional idem (id=139)
  - [x] **v1.1 PHASE-GATED (w60, start_step=150): PASSOU O GATE FROZEN** ✅ 2026-06-10 (DB id=141,
    condição do /goal CUMPRIDA). Triagem: empate estatístico nas 2 réguas. Frozen pareado por seed:
    vs Producer −0.0124 (9↑/9↓ equilibrados, 169/192 jogos idênticos; mean +0.206, ambos assentos >0);
    vs OEP **+0.0000 (64/64 jogos idênticos**, mean +0.406). Mudança cirúrgica: disciplina de onda
    elite no late game SEM regressão local. Config: `PGSConfig(scripts="hold", wave_min_ships=60,
    wave_start_step=150)`.
  - [x] **SUBMETIDO (autorizado pelo usuário)** ✅ 2026-06-10: **ref=53537753** (DB id=142),
    validação oficial OK (p95 91ms). **ACHADO CRÍTICO no empacotamento** (id=142): a submissão
    anterior (53519882, LB ~1022) rodou a config DEFAULT com os scripts ofensivos REJEITADOS
    (main.py → agent() module-level → PGSConfig() cheio), NÃO o hold-only do gate — o hold-only
    nunca tinha sido testado no campo. Fix permanente: `SUBMISSION_CONFIG` pinada em
    `bots/pgs/agent.py` (hold + wave w60s150). Isso REVISA a leitura do id=129 (parte da
    subperformance 2p pode ser dos scripts ofensivos).
  - [ ] **ler score LB do ref=53537753 após ~1h15** (watcher local armado): comparar com 1022
    (all-scripts) e 1228 (Producer); registrar no DB
    - [ ] verificar: score estabilizado lido e registrado; se > 1228, discutir promoção a default
- [ ] ~~submissão-experimento do wave30+hold (v0)~~ DESCARTADA (recomendação aceita: v0 reprovou
  nas 2 réguas; slot guardado para candidato que passe os gates)
  - [ ] **(b) decidir o destino da linha PGS**: vetos HOLD são neutros/negativos contra o campo — ou o árbitro
    passa a exigir vantagem robusta contra MÚLTIPLOS modelos de oponente (não só Producer-like), ou a linha
    congela e o esforço vai para T0/T8
    - [ ] verificar: PGS revisado > floor vs régua com rusher (16 seeds triagem), antes de re-tocar o gate frozen

# ✅ ENCERRADO (2026-06-09) — desancorar a recompensa do Producer (Alavanca A — REJEITADA, id=120)

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
  - [x] **Motor da worktree B consertado** ✅ 2026-06-09 (esta sessão): cherry-pick dos fixes de build (`3900024`)
    e paridade polar (`105af98`) no branch `frente-b-candidate-selector` + `make build` (sync-binding) +
    **22 testes de paridade/fidelidade PASS** com `.so` fresco. Todo resultado BReP anterior (brep_gpu +0.18,
    brep_seat −0.0155@96) foi medido no motor PRÉ-fix — re-medir antes de confiar.
  - [ ] **AÇÃO SUA (worktree B, antes do run paralelo):** os launchers `scripts/run_brep_*.sh` usam `uv run` PURO →
    auto-sync reverte o `.so` fresco para o wheel stale. Trocar por `uv run --no-sync` nos scripts OU rodar
    `uv cache clean orbit-wars-lab` na B (sem processo uv segurando o lock) antes de lançar.
    - [ ] verificar: `make verify-binding` na B mostra o `.so` do build atual após um `uv run` qualquer
  - [ ] **AVISO (worktree B): checkpoints v1 ≠ código v2.** O `policy.py` sujo da B mudou `N_EDIT` 4→6 e a SEMÂNTICA
    dos códigos de edit; `eval_brep_direct` não carrega mais os ckpts v1 (`brep_gpu/*`, `brep_seat/*` — edit head 64
    vs 96). Para re-medi-los: construir com `n_edit=4` + tabela v1 de scales; senão, medir só a linha v2 (fresh,
    KEEP-init = piso de paridade no motor CORRETO). Sugestão p/ v2: `ent-coef ≤ 0.003` (0.01 com 6 códigos empurra
    a política p/ longe do KEEP — provável causa da regressão 0→−0.12 da brep_v2).
- [x] **Mov.2 — implementado** ✅ 2026-06-09 (`/goal`). A evidência do P3 (DB) reformulou: o gargalo é **drift de
  recompensa** (PPO melhora sobre BC e REGRIDE ao escalar), não representação. 3 knobs compostos em `train_ppo`:
  `--shaping-potential none` (de-anchor: dropa o shaping de produção); `--kl-to-ref-coef/--ref-checkpoint` (âncora KL
  ao BC, anti-drift; `launch_gated_kl` masked-safe); `--eval-every-updates/--early-stop-patience` (eval-gating keep-best).
  Smoke GPU validou os 3 juntos (eval_series pegou o drift e manteve o best). `make ppo-train-mov2`. Experimento id=119 no DB.
- [x] **Campanha Mov.2 rodada e MEDIDA** ✅ 2026-06-09 (id=120 DB, **REJECTED**). 2M ts (de-anchor=none + KL + eval-gating)
  → margem 96s vs Producer = **−0.997 (PIOR que P3 −0.75)**. `--shaping-potential none` STARVA o sinal contra oponente
  forte (perde tudo → reward esparso −1 → gradiente ~0). Levers de recompensa NÃO quebram o teto do Producer.
- [x] **DECISÃO — próximo lever** ✅ 2026-06-09 (usuário): **frentes paralelas** — BReP/DRL na worktree B
  (motor já consertado lá, ver acima); ESTA worktree vira frente heurística/metaheurística/hiperheurística
  (FOCO ATUAL no topo). Mov.2-com-shaping-0.05 descartado (prior baixo, GPU vai pro BReP).
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

# 🏟️ LIGA LOCAL (decisão do usuário 2026-06-10: ela pontua e decide submissões)

> **ATUALIZAÇÃO 2026-06-10 (noite): liga ENXUGADA para só o que funciona.** A liga contínua parou
> no meio do round 102 (~20:01, sem traceback — interrupção externa; state.json consistente em
> round=101, retomável). Calibração colapsou nos rounds 100–101: Spearman BT vs LB caiu de +0.214
> para **+0.036** (≈0 com 6 âncoras; hold LB=1058 e holdwave LB=1229 empatados em BT). Removido do
> `league_report.py` o que estava FALSIFICADO: **µ-kaggle** (réplica do rating online, spearman ~0)
> e **LB_est** (regressão BT→LB). Mantido o que funciona: BT+CI90 bootstrap, hard gate de veto
> (allscripts < cluster), piso de veto P≥ref, H2H vs incumbente, win2p/4p/annih, spearman BT como
> métrica de SAÚDE. Report validado de ponta a ponta nos 4029 jogos acumulados; `report.json` sem
> os campos removidos; docstring do `league_run.py` corrigido (cap real = 3 lineage por mesa 4p).

- [x] decidir se retoma a liga contínua (`league_run.py`) — RETOMADA (pedido do usuário 2026-06-10
  noite); rounds 102–112 rodaram limpos (fails=0); pausada para o bugfix do floor PGS (abaixo) e
  religada em seguida com o planner corrigido
  - [ ] verificar: qualquer liga futura só volta a valer como gate se spearman BT vs LB ≥ +0.6
    sustentado por ≥10 rounds com as âncoras atuais

> **BUGFIX 2026-06-10 (noite, DB id=171): floor do PGS não reproduzia o Producer real.**
> `_producer_entries()` criava `ProducerLiteRuntime()` ZERADO a cada chamada — perdia a memória
> rolante (`memory.movement`, ledger de lançamentos próprios reconciliado contra a próxima obs).
> Probe (`check_pgs_floor_fidelity.py`, agora com classes de severidade): 36/80 steps divergentes
> (count=3, angle_tiny=33). Fix: `PGSRuntime._floor_runtimes` persistente POR OWNER (reset no
> step 0, 1 chamada por owner/turno; `PlanetMovement.update` é gap-robusto p/ oponentes pulados).
> Pós-fix: **0/200 (seed 1000) e 0/120 (seed 4242)** — paridade EXATA. Regressão:
> `tests/test_pgs_floor_fidelity.py`; 7/7 testes pgs+isolation passam.

- [ ] decidir o que fazer com os jogos pgs_* PRÉ-FIX da liga (cont/ até r0112): o floor bugado
  era um agente ligeiramente diferente — mistura de versões no pool BT append-only (liga é
  veto-only, então o impacto é baixo, mas a medição não é pura)
  - [ ] verificar: ou (a) arquivar cont/ pré-fix e re-acumular, ou (b) aceitar a mistura e anotar
    no report que ratings pgs_* misturam versões até r0112

> **BUGFIX 2026-06-10 (madrugada, DB ids=173/174): 5 bugs de instrumentação da liga corrigidos**
> (auditoria do usuário; fixes via subagentes; 20 testes passam, report fim-a-fim OK nos 4645 jogos):
> (1) **isolamento de tarballs** — `_tarball_agent` punha o cache no `sys.path` global e os módulos
> bare-name (`_brep_weights`, `_producer_agent`, `_upstream`, `orbit_lite`) eram compartilhados via
> `sys.modules`; com brep × brep_league3 no mesmo processo, o import LAZY de `_brep_weights` no 1º
> act() resolvia pro cache ERRADO (último `sys.path[0]` vencia) → fix `_TarballIsolation` (overlay
> privado por tarball, swap em volta do load E de cada act; `tests/test_league_tarball_isolation.py`);
> (2) **faults medidos** — crash / timeout(>1s, actTimeout Kaggle) / invalid_moves por agente entram
> no game dict (`faults`, omitido se limpo) e o report imprime seção FAULTS + ⛔ (antes
> `except Exception` passava o turno sem registro no JSON); (3) **empate** — `tie:true` /
> `winner=None` (antes argmax dava vitória falsa ao assento 0; 26 casos nos artefatos antigos —
> o report novo RE-DERIVA o winner de `final_ships`, ignora o campo gravado); (4) **H2H2p puro** —
> coluna e regra de promoção usam SÓ jogos 2p decisivos (hold vs holdwave: 78-64 puro; o agregado
> inflava ~141-123 com decomposição 4p); (5) **filename** — `n[:6]` colidia (`pgs_ho`/`pgs_wa`) e
> SOBRESCREVIA: **round 92 perdeu 8 jogos reais** (holdwave×brep, único déficit em 125 rounds) —
> fix `match_filename()` com sufixo sha1[:8]. Extras: (6) banner VETO-ONLY no topo do report;
> (7) `LB_ANCHORS` refrescado via CLI (s100=1138.6, ainda subindo; resubmit 53542884=1157.7
> segue anotado como ruído ~±60, não âncora).

- [ ] decidir o destino dos jogos brep×brep_league3 PRÉ-FIX (contaminação de pesos: eram
  mirror-match dos pesos de um tarball só, não brep-real vs league3-real) — mesma decisão (a)/(b)
  do item pgs_* acima; opcional: re-rodar o matchup perdido do round 92 (holdwave×brep, 8 jogos)
  - [ ] verificar: report/decisão anota que pares brep×brep_league3 pré-fix não medem dois bots
    distintos (ou cont/ é re-acumulado)

- [~] **Liga v1 implementada e RODANDO**: `scripts/league_agents.py` (pool de 8: producer, oep, brep
  [tarball da worktree B], pgs_hold, pgs_holdwave, pgs_allscripts, ext_lb1050, ext_hellburner; instância
  fresca por jogo), `league_match.py` (2p ambos assentos + 4p com rotação; vencedor = argmax navios;
  registra aniquilação/step), `league_report.py` (matriz de payoff + rating Bradley-Terry + calibração).
  40 matchups × 4 seeds (28 pares 2p + 12 composições 4p), paralelismo 6 (`artifacts/league/v1/`).
- [x] **CALIBROU** ✅ (DB id=143): Spearman **+1.000** nas 4 âncoras (ordem idêntica ao LB:
  oep > producer > brep >> allscripts) e gate duro PASS com folga (~250 pts de gap no allscripts —
  a liga teria barrado a submissão do id=129 ANTES do slot). 272 jogos (224 2p + 48 4p).
- [x] **LIGA ADOTADA COMO GATE DE SUBMISSÃO** (decisão do usuário): candidato precisa **BT ≥ producer**.
  Leituras atuais: pgs_hold 1082 (≈ producer 1080, dentro do ruído — único candidato aprovável);
  pgs_holdwave 1035 (REPROVARIA — sangra vs oep 4-12 nos seeds da liga). **Predição registrada antes
  do score**: ref=53537753 (holdwave) fica abaixo do Producer no LB (~1050–1170), acima de 1022.
  - [~] comparar predição com o score real do ref=53537753: **preliminar T+80min = 1264 > 1228**
    (13W/7L, 4p 6W/5L, matchmaking 1100–1426) — a predição da liga (~1050–1170) está ERRANDO até
    aqui; suspeita: 4p da liga com pool Producer-pesada não representa o 4p do campo (DB id=144).
    Leitura de estabilização ~T+4h agendada (watcher).
- [~] **LIGA CONTÍNUA RODANDO** (pedido do usuário): `scripts/league_run.py` — rodadas perpétuas,
  seeds sempre novos (state.json), matchmaking 70% uniforme + 30% topo, mesas 4p com mistura de
  estilos, standings por rodada em `artifacts/league/v1/standings.log`; report v2 com bootstrap
  CI90, **P(bot ≥ producer)** (estatística do gate), LB_est e split 2p/4p. 60 rounds × 4 matchups
  em background + adensamento de 12 seeds nos 15 matchups do topo (p2x_/p4x_).
  - [x] verificado ✓ (DB id=146, 707 jogos): CI90 ~±30; **família hold separa do Producer**
    (P≥prod: oep .96, pgs_hold .94, holdwave .73); Spearman +0.8 nas DUAS métricas (única inversão
    producer×brep, gap real 20pts = empate); μ-kaggle implementado (id=145: regra real do Kaggle
    por engenharia reversa de 429 updates — E logística D=500, K exponencial decrescente, 4p ~2.3×
    menor) e μ-kaggle põe pgs_hold/holdwave no topo, consistente com LB ao vivo (1264 > 1228)
  - [ ] fechar leitura de estabilização do ref=53537753 (watcher T+4h): se > 1228 estabilizado,
    holdwave é a melhor submissão do time
  - [x] **Rodada wave guiada pela liga (/goal) ✅ COM SINAL** (DB id=148): `pgs_wave_s100` (hold +
  wave(60) desde step 100) = **1º GERAL da liga** — BT 1076, μ-kgl 839 (maior), P≥ref 0.97;
  s50 morto definitivo (0.48); 4pfloor passa (0.80) mas não supera os holds. Gradiente coerente
  s50 < s150 < s100 (ponto doce pós-expansão). Knob `floor_in_4p` adicionado ao planner.
  `wave_s100` entrou na pool do campeonato contínuo para apertar o CI (88 jogos próprios).
  - [x] **PAR FINAL SUBMETIDO** ✅ 2026-06-10 (DB id=150; regra do usuário: só as 2 últimas contam):
    `wave_s100` ref=53542864 (campeão da liga) + RESUBMIT `w60s150` ref=53542884 (config do recorde
    1244). Ambos validados; recomeçam de 600 e re-escalam. pgs_hold e holdwave originais viram inativos.
  - [ ] ler scores estabilizados do par final (watcher T+2h13 armado); atualizar LB_ANCHORS;
    flipar `GATE_REFERENCE` para o melhor PGS estabilizado
    - [ ] verificar: se s100 estabilizar ≥ s150, a liga ganha o 2º acerto preditivo e vira decisora
      plena de slots
- [~] **pgs_hold PURO SUBMETIDO** ✅ 2026-06-10 (ref=53541125, DB id=147; autorizado pelo usuário
    após confirmar que o "hold-only" de ontem rodou all-scripts). Empacotado com novo flag
    `--pgs-config` (patcha SUBMISSION_CONFIG no tarball); validação oficial OK. Predição da liga
    pré-registrada: acima do Producer (~1173), próximo do hold+wave (~1264).
    - [ ] ler scores estabilizados dos DOIS PGS (watcher T+2h) e fechar a tríade de calibração
      da família (allscripts 1022 / hold / hold+wave)

# 🎯 ATACAR AGORA — objetivo top 5

Estado operacional curto:

- **Producer é a melhor submissão operacional atual** (~1200 LB). Congelar como default até existir candidato provado.
- **OEP é útil como adversário/professor** (~1100 LB), mas não como linha de tuning: knobs/overlays OEP já foram atacados e não devem voltar sem hipótese nova.
- **PPO atual ainda é fraco** (`-1.0` vs Producer nos registros antigos). O próximo ataque é imitação + currículo forte, não PPO do zero contra heurísticas fracas.
- **Histórico detalhado fechado vive no DB `experiments.duckdb`** (`make experiments-report`). Este arquivo deve ficar só com o que ainda vamos atacar.
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
  - [ ] **Registro:** `python -m python.lab.experiments add` (DB) com baseline, candidato, comandos, margem antes/depois e decisão; commitar `experiments.duckdb`.
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
  - [ ] **Registro:** toda morte de linha vira `experiments add --status rejected` (DB) com motivo técnico, para não ressuscitar tuning morto.
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
