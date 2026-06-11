# Auditoria de Submissoes - 2026-06-11

Escopo: verificar se as submissoes anteriores de maior score tiveram bugs,
erros silenciosos ou validacao local fraca que explicam queda de performance,
e deixar os pacotes atuais auditaveis antes de nova submissao.

## Resumo Executivo

Os bugs relevantes encontrados foram principalmente de **validacao e
empacotamento**, nao de uma melhoria estrategica nova confirmada:

- PGS/OEP podiam degradar para Producer sem visibilidade suficiente.
- PGS/OEP tambem podiam degradar para Producer pelo resto do jogo apos um
  unico timeout de wrapper sob carga.
- PGS ja teve incidente de entrypoint/config: `planner.py` expunha defaults
  all-scripts rejeitados em vez do config operacional pinado.
- A liga local tinha riscos de cache stale, mistura de estado em 4p, faults
  ausentes, JSON stale e painel assimetrico entre candidatos.
- O validador oficial local assumia layout `bots/producer` e nao auditava
  corretamente tarballs flat como Producer e BReP.

Estado atual:

- PGS/OEP atuais: validados com `fallbacks=0`, `timeouts=0`,
  `timeout_thread_blocks=0`, `fallback_errors=0`.
- Producer atual: valida `DONE` em 2p/4p; sem `SUBMISSION_STATS` por ser
  baseline puro, aceito apenas com flag explicita.
- BReP 1156 atual: valida `DONE` em 2p/4p, `fallbacks=0`,
  `illegal_moves=0`, `fallback_errors=0`.
- `pgs_bigwave` foi testado como melhoria estrategica e rejeitado localmente.

## Submissoes e Diagnostico

| Submissao | Score publico | Diagnostico | Correcao / estado atual |
| --- | ---: | --- | --- |
| `53537753` PGS hold+wave w60s150 | 1228.8 | Melhor recorde. Config operacional correto: `scripts="hold", wave_min_ships=60, wave_start_step=150`. | Pacote atual `artifacts/submission_pgs.tar.gz` validado em 2p/4p all seats: `DONE`, `fallbacks=0`, `timeouts=0`, `timeout_thread_blocks=0`, `fallback_errors=0`. |
| `53542884` PGS hold+wave resubmit | 1152.1 | Mesmo config declarado do recorde, score bem menor. Evidencia insuficiente para atribuir apenas a bug; ha ruido/variancia de LB e wrappers antigos podiam cair para Producer pelo resto do jogo apos um timeout. | Wrapper atual usa Producer shadow warm, budget 0.9 e bloqueia apenas enquanto o thread atrasado ainda esta vivo; depois retoma PGS. Re-submissao so deve ser interpretada apos estabilizacao do LB. |
| `53542864` PGS wave_s100 | 1146.1 | Local antigo inflou a variante; liga/validacao anteriores tinham riscos de cache/fault/status, fallback invisivel e painel assimetrico. | Tarball atual validado sem fallback/timeout. Ainda abaixo de Producer/OEP e do recorde `pgs_holdwave`; nao promover sem regua longa final. |
| `53541125` PGS hold | 1057.6 | Localmente forte contra varias referencias, mas LB baixo. Pode refletir gap campo-local; wrappers antigos tambem eram menos auditaveis. | Tarball hold atual validado sem fallback/timeout. Ainda nao ha prova de que re-submeter superaria `pgs_holdwave`. |
| `53519882` PGS antigo | 1021.5 | Incidente confirmado: submissao "hold-only" acabou rodando defaults all-scripts por entrypoint/config errado. | `bots/pgs/planner.py` nao expoe mais `agent`/`_RUNTIME`; entrypoint operacional unico fica em `bots/pgs/agent.py`. |
| `53433131` OEP robust wrapper | 1182.7 | Wrapper robusto evitou ERROR anterior, mas template OEP ainda podia ocultar fallback antes do hardening e tambem podia cair para Producer pelo resto do jogo apos timeout unico sob carga. | `artifacts/submission_oep.tar.gz` atual validado: `DONE`, `fallbacks=0`, `timeouts=0`, `timeout_thread_blocks=0`, `fallback_errors=0`. |
| `53432895` OEP sem wrapper | ERROR | Falha de submissao por timeout/robustez insuficiente. | Mantido apenas como historico; OEP atual usa wrapper instrumentado. |
| `53366194` Producer | 1173.1 | Baseline forte; validador local nao cobria corretamente layout flat. | `artifacts/submission_producer.tar.gz` atual validado com flag de baseline: todos `DONE`, p95 27ms, max 195ms. |
| `53513962` BReP v3 | 1156.1 | Versao corrigida apos erros anteriores de callable/peso/init. O validador local nao auditava layout flat antes do fix. | Tarball atual do worktree B validado: todos `DONE`, `fallbacks=0`, `illegal_moves=0`, `fallback_errors=0`, p95 52ms, max 232ms. |
| `53513798` / `53513452` BReP | ERROR | Erros confirmados historicamente: callable selecionado errado / init-peso / fallback guardado em versoes anteriores. | Versao v3 (`53513962`) e tarball atual passam na validacao oficial local. |

## Evidencia de Validacao Atual

Comandos de referencia:

```bash
rtk .venv/bin/python scripts/validate_pgs_tarball.py \
  --tarball artifacts/submission_pgs.tar.gz --players 2 4 --seats all \
  --label pgs_holdwave_current

rtk .venv/bin/python scripts/validate_pgs_tarball.py \
  --tarball artifacts/submission_oep.tar.gz --skip-pgs-planner-check \
  --players 2 4 --seats all --label oep_current

rtk .venv/bin/python scripts/validate_pgs_tarball.py \
  --tarball artifacts/submission_producer.tar.gz --skip-pgs-planner-check \
  --allow-missing-submission-stats --seats all --label producer_current

rtk .venv/bin/python scripts/validate_pgs_tarball.py \
  --tarball /home/marcux777/projects/Kaggle/orbit-wars-lab-B/artifacts/submission_brep.tar.gz \
  --skip-pgs-planner-check --seats all --label brep_1156
```

Resultados observados nesta auditoria apos reempacotar os wrappers:

- PGS holdwave atual: `VALIDATION OK`, `fallbacks=0`, `timeouts=0`,
  `timeout_thread_blocks=0`, `fallback_errors=0`, p95 85.7ms, max 225.1ms.
- PGS hold atual: `VALIDATION OK`, `fallbacks=0`, `timeouts=0`,
  `timeout_thread_blocks=0`, `fallback_errors=0`, p95 72.8ms, max 692.3ms.
- PGS wave_s100 atual: `VALIDATION OK`, `fallbacks=0`, `timeouts=0`,
  `timeout_thread_blocks=0`, `fallback_errors=0`, p95 77.8ms, max 210.0ms.
- OEP atual: `VALIDATION OK`, `fallbacks=0`, `timeouts=0`,
  `timeout_thread_blocks=0`, `fallback_errors=0`, p95 56.7ms, max 204.1ms.
- Producer atual: `VALIDATION OK`, todos `DONE`, p95 27ms, max 195ms.
- BReP 1156: `VALIDATION OK`, `fallbacks=0`, `illegal_moves=0`,
  `fallback_errors=0`, p95 52ms, max 232ms.

Testes e commits relevantes:

- `307e14d fix(submission): harden validation gates`
- `763e75f chore(experiments): record bigwave rejection`
- `c8ceae9 fix(validation): support flat submission tarballs`
- `rtk .venv/bin/python -m pytest -q`: `307 passed` antes da auditoria
  Producer/BReP.
- `rtk .venv/bin/python -m pytest -q tests/test_validate_pgs_tarball.py`:
  `2 passed`.

## Liga Local

A regua forte foi corrigida e esta rodando em
`artifacts/league/submit_ruler/background_strict_v5`:

- candidatos agora enfrentam tambem os outros candidatos do mesmo comando;
- mapas 2p sao estaveis por adversario, independentemente do painel de
  candidatos;
- `overall_score` usa o split de campo medido: 46% 2p / 54% 4p;
- empates 2p contam como nao-vitoria no score bruto;
- ranking prioriza `PASS_LOCAL` > `INCONCLUSIVE` > `REJECT_LOCAL`;
- progresso incremental fica em `task_results.json` e os matches longos gravam
  JSON parcial por `--match-chunk-size`;
- a partir do commit `ac6ad0c`, os checkpoints de novos runs intercalam seat
  orders/rotacoes, entao o parcial fica monitoravel mais cedo.
- a partir do experimento `191`, `--skip-run` valida modo, agentes, seed slice,
  numero exato de jogos, seat orders/rotacoes, `faults` e `agent_status`; JSON
  antigo/parcial/stale falha em vez de entrar no ranking.

O smoke de 36 tarefas com `seeds=4`, `steps=20` fechou e provou que o report
final e gerado, mas e curto demais para decisao competitiva. A regua v5 longa
esta em background no PID registrado em
`artifacts/league/submit_ruler/background_strict_v5.pid`.

Observacao: a v5 foi iniciada antes do commit `ac6ad0c`, entao seus parciais
2p so ficam honestamente comparaveis depois que a ordem reversa de assentos for
gravada. O resultado final continua balanceado.

Para PPO, benchmark/selecao agora tratam crash, timeout, invalid, fallback,
policy-illegal, fallback-error e instrumentacao ausente como falhas tecnicas
zero-tolerancia. Um checkpoint com score alto mas fallback/timeout nao pode ser
selecionado como melhor submissao.

## Decisao Atual

Nao ha uma nova variante comprovadamente melhor que o recorde `pgs_holdwave`
`53537753` apenas por esta auditoria.

Decisao operacional:

- Manter `pgs_holdwave` como incumbent de submissao ate a regua longa finalizar.
- Usar os tarballs atuais regenerados se for re-submeter PGS/OEP.
- Nao promover `pgs_bigwave`.
- Tratar quedas de score de resubmit como inconclusivas sem LB estabilizado,
  porque os bugs silenciosos foram corrigidos localmente, mas Kaggle nao expoe
  `SUBMISSION_STATS`.
