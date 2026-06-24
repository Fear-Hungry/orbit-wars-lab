# Orbit Wars Lab: pós-mortem técnico

Este repositório começou como laboratório competitivo para o Kaggle **Orbit Wars**
e agora fica como portfólio técnico: um estudo de engenharia para agentes
adversariais com simulador local, validação contra ambiente oficial, heurísticas
fortes e trilhas de RL que só seriam promovidas quando passassem por gates
reprodutíveis.

## O problema

Orbit Wars é um jogo orbital de 2 ou 4 jogadores. A cada turno, o agente recebe
planetas, frotas, cometas e posições em rotação, então devolve lançamentos no
formato `[from_planet_id, direction_angle, num_ships]`. O objetivo prático não é
só capturar planetas: é sobreviver a seeds desconhecidas, assentos diferentes,
oponentes oportunistas e ao limite de tempo do Kaggle.

A primeira decisão estrutural foi separar três responsabilidades:

- `crates/`: motor Rust para simular e testar rápido localmente.
- `python/`: treino, avaliação, liga, PPO/BC e ferramentas de análise.
- `orbit_lite/` + `bots/`: agente Python puro, portátil para submissão Kaggle.

Essa fronteira foi deliberada: Rust acelerava o laboratório, mas a submissão real
precisava ser um `agent(obs)` Python validado em `kaggle-environments`.

## Estratégia

O baseline público mais importante era o **Producer**. Ele não era o teto: era o
piso que qualquer candidato precisava bater antes de merecer atenção. A estratégia
do projeto virou uma sequência de linhas testáveis:

- **Producer corrigido e empacotável** como âncora forte.
- **OEP**: planner experimental sobre o Producer, com busca curta e seleção por
  fitness de plano. Foi útil como infraestrutura, mas não promoveu.
- **PGS holdwave**: política heurística de segurar/onda, com configuração
  operacional `scripts="hold"`, `wave_min_ships=60`, `wave_start_step=150`.
- **BC/PPO**: imitação de agentes fortes e PPO com exportação para submissão,
  mantidos sob gate estrito porque acurácia offline não bastava.
- **Top-5 proxy**: pool de agentes públicos fortes para vetar candidatos que só
  pareciam bons contra Producer.

O padrão que funcionou melhor foi evitar "grandes teorias" sem régua: cada ideia
tinha que declarar o oponente, seeds, assentos, falhas técnicas proibidas e métrica
de margem antes de ser considerada.

## Validação local

A validação acabou sendo a principal peça de engenharia do projeto.

1. **Paridade com o oficial.** `docs/PARITY.md` registra os testes contra
   `kaggle-environments`: snapshots oficiais, combate, movimento, colisões,
   rotação planetária e score terminal.
2. **Gates de submissão.** Tarballs PGS/OEP/Producer precisavam rodar em 2p e 4p
   com `fallbacks=0`, `timeouts=0`, `illegal_moves=0` e sem counters proibidos em
   `SUBMISSION_STATS`.
3. **Liga local seat-rotacionada.** A régua passou a separar 2p/4p, assentos,
   lineups e status técnico. Falha de timeout, crash ou ação inválida não podia
   virar vitória por acidente.
4. **Veto por campo forte.** A pool top-5 proxy não provava leaderboard, mas
   matava candidatos frágeis que exploravam só o Producer.

Uma lição importante: a liga local é ótima como **veto**, não como oráculo de
promoção. O score Kaggle ainda tinha variância e precisava estabilizar antes de
qualquer conclusão.

## Ablações e resultados

| Linha | Resultado | Lição |
| --- | --- | --- |
| Producer corrigido | Baseline forte, score público registrado `1173.1` | A âncora precisa ser auditável antes de julgar melhorias. |
| OEP 1-ply | Chegou a rodar sem crash, mas ficou abaixo do Producer local | Busca curta sobre um modelo ruim de resposta adversarial piora rápido. |
| PGS holdwave | Melhor submissão registrada: `53537753`, score público `1228.8` | Timing de onda e política simples podem bater engenharia mais complexa. |
| PGS resubmit/variantes | Scores `1156.7`, `1147.5`, `1057.6` em variantes ou reenvios | Leaderboard imediato não é prova; wrapper/cache/fallback e variância importam. |
| BC/PPO | Saiu do colapso em alguns checks, mas não virou candidato promovido | Métrica offline e política exportada precisam concordar no ambiente real. |
| Abertura 4p | Regras de suprimir PvP ou segurar captura foram rejeitadas | Um sintoma de replay nem sempre é alavanca causal. |
| Auditoria de wrappers | Encontrou fallback/timeout silencioso e problemas de empacotamento | Robustez de submissão é parte da estratégia, não etapa burocrática. |

Os registros detalhados ficam em `docs/EXPERIMENTS_REPORT.md`,
`docs/EXPERIMENTS_REVALIDATION.md`, `docs/SUBMISSION_AUDIT_2026-06-11.md` e
`docs/OPENING_4P_ANALYSIS.md`.

## Submissões

O melhor resultado anotado no repositório foi o PGS holdwave:

- Submissão `53537753`.
- Score público `1228.8`.
- Configuração: `scripts="hold"`, `wave_min_ships=60`, `wave_start_step=150`.
- Pacote operacional: `artifacts/submission_pgs.tar.gz` nos registros de auditoria.

Também ficaram documentados incidentes importantes: OEP sem wrapper deu `ERROR`,
versões antigas podiam degradar para Producer sem visibilidade suficiente, e um
entrypoint incorreto já expôs defaults rejeitados em vez da configuração validada.
Esses casos foram mais valiosos que alguns ganhos locais, porque mudaram os gates.

## O que aprendi

Competição adversarial pune principalmente validação fraca. Em um ambiente desse
tipo, uma melhoria local pode ser só seed favorável, assento favorável, cache
stale, fallback invisível ou benchmark contra o oponente errado. A engenharia que
mais importou foi tornar essas falhas audíveis.

Também ficou claro que heurísticas fortes não são "baseline descartável". Em jogos
de estratégia pequenos, uma heurística com boa noção de tempo, segurança e
orçamento pode superar RL mal validado por muito tempo. RL só faz sentido quando a
exportação, o decoder e o gate competitivo medem exatamente o que será submetido.

Por fim, 4p muda o problema. Em 2p, a pergunta costuma ser como converter paridade
em vantagem. Em 4p, liderar cedo pode transformar o agente em alvo. A melhor régua
precisava capturar essa diferença, em vez de juntar tudo em uma média confortável.

## Mapa de leitura

- Arquitetura: [`ARCHITECTURE.md`](ARCHITECTURE.md) e
  [`DECISIONS.md`](DECISIONS.md).
- Método competitivo: [`BLUEPRINT.md`](BLUEPRINT.md) e
  [`PLAYBOOK.md`](PLAYBOOK.md).
- Validação e paridade: [`PARITY.md`](PARITY.md) e
  [`SUBMISSION.md`](SUBMISSION.md).
- Experimentos e ablações: [`EXPERIMENTS_REPORT.md`](EXPERIMENTS_REPORT.md),
  [`EXPERIMENTS_REVALIDATION.md`](EXPERIMENTS_REVALIDATION.md) e
  [`OPENING_4P_ANALYSIS.md`](OPENING_4P_ANALYSIS.md).
- Auditoria de submissões: [`SUBMISSION_AUDIT_2026-06-11.md`](SUBMISSION_AUDIT_2026-06-11.md).
