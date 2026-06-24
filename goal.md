# Objetivo Atual - Auto-Research Loop para Orbit Wars

Data: 2026-06-18

## Objetivo verificavel

Implementar no repositorio um loop inspirado no video "Recursive Self
Improvement", mas adaptado ao Orbit Wars como **Auto-Research competitivo**:

```text
agent prompt -> proposta de experimento -> patch/candidato -> treino curto
-> avaliacao fixa -> registro -> keep/discard -> proxima proposta
```

O nome "RSI" e secundario. O que queremos copiar do video e o mecanismo pratico:
um agente autonomo roda muitos experimentos pequenos, mede uma metrica objetiva,
guarda historico e itera. Para esta competicao, o loop precisa ser governado
porque o fitness local ja demonstrou risco de nao predizer o leaderboard.

## O que o video faz, traduzido para o repo

Padrao do video:

- existe um arquivo de instrucoes do agente;
- o agente propoe uma **classe de solucao**;
- ele edita/programa a solucao;
- treina por um orcamento curto;
- avalia em validacao fixa/oculta;
- registra metricas, historico e artefatos;
- continua tentando ate ser parado;
- pode platofar, hiperespecializar, trapacear a metrica ou gastar caro se o
  harness for ruim.

Adaptacao para Orbit Wars:

- "classe de solucao" = variante de planner, exploiter, alvo de value net,
  configuracao PPO curta, policy head/decoder, ou ajuste pequeno de estrategia;
- "treino curto" = budget fixo por tentativa, com seeds explicitas e timeout;
- "validacao oculta" = gate/evaluator off-limits para o gerador, com holdout de
  seeds/oponentes quando possivel;
- "error map" = relatorio por modo 2p/4p, oponente, assento, death/annihilation,
  margem, faults, timeout, invalid moves e p95;
- "git history" = patch/commit apenas para candidato aceito; rejeitados ficam no
  DB com diff/resumo e motivo;
- "dashboard" = relatorios JSON/Markdown + `experiments.duckdb`.

## Hipotese

Auto-Research deve aumentar o throughput de pesquisa se o agente otimizar dentro
de uma caixa bem definida:

- ele pode gerar candidatos e patches pequenos;
- ele pode rodar treino/avaliacao;
- ele pode ler metricas e propor a proxima tentativa;
- ele **nao** pode alterar a regua para fazer o score subir;
- ele **nao** pode decidir submissao Kaggle sozinho.

O objetivo nao e uma "explosao RSI". E engenharia repetitiva: mais experimentos
bons por dia, com menos trabalho manual.

## Invariantes especificos desta competicao

- O gargalo e fitness preditivo, nao orquestracao.
- Liga local/gate plano serve como veto, nao como prova de promocao.
- Score Kaggle imediato nao decide melhora/piora; aguardar estabilizacao.
- Submissao automatica fica fora do MVP.
- Fallback silencioso invalida a iteracao.
- Gates, seeds, thresholds, criterios de promocao e pool de validacao sao
  off-limits para o gerador.
- Toda mudanca em agente/candidato precisa ser registrada em
  `experiments.duckdb`.

## Superficies de busca

### Permitidas no MVP

- variantes de PGS/planner dentro de arquivos candidatos;
- exploiters pequenos para enriquecer avaliacao;
- novos alvos de value net;
- configs curtas de PPO/BC para smoke;
- ajustes de decoder/policy que preservem API;
- scripts de orquestracao e parsing;
- prompts/instrucoes do agente.

### Proibidas no MVP

- alterar testes, fixtures, `xfail`/`skip` ou gates para mascarar falha;
- alterar seeds, oponentes, thresholds ou criterios de promocao sem pedido
  explicito;
- introduzir dependencia nova;
- mudar empacotamento de submissao sem validar tarball;
- usar Kaggle LB como fitness denso;
- fallback silencioso;
- submit automatico.

## Arquitetura alvo

Criar um runner de Auto-Research com componentes separados:

```text
Goal/Prompt
  -> Planner de experimento
  -> Gerador de patch/candidato
  -> Executor com budget
  -> Avaliador fixo
  -> Parser de metricas
  -> Politica keep/discard
  -> Logger DuckDB
  -> Fila de candidatos promovidos
```

Contratos minimos:

- cada iteracao tem `run_id`, `parent`, `hypothesis`, `candidate`, `patch`,
  `commands`, `seeds`, `metrics`, `faults`, `decision`;
- cada comando e reproduzivel;
- cada artefato tem caminho e hash quando aplicavel;
- cada decisao e uma de `promoted`, `rejected`, `inconclusive`,
  `needs_more_seeds`, `technical_fail`;
- `technical_fail` nunca vira `rejected` competitivo.

## Fitness inicial

Usar a melhor regua disponivel, mas com humildade:

- Nash gate quando houver amostra valida;
- split 2p/4p;
- comparacao contra parent;
- faults/timeouts/invalid/bad_status obrigatoriamente zerados;
- p95 dentro do `actTimeout`;
- veto por top5-proxy/liga apenas como sanity check;
- resultado `inconclusive` quando o gate nao discriminar o topo.

Se o loop detectar que a regua nao discrimina bons candidatos, a proxima tarefa
passa a ser melhorar a regua, nao continuar gerando mutacoes.

## Modo de operacao

### Modo 0 - Dry-run

Nao edita codigo. Seleciona parent, monta hipotese, monta comando, valida que o
registro no DB funcionaria.

### Modo 1 - Smoke

Aplica uma mudanca pequena permitida, roda budget curto, avalia wiring, registra
resultado. Nao promove competitivamente.

### Modo 2 - Pesquisa local

Roda varias iteracoes com budget definido. Pode manter patch se passar criterios
locais. Nao submete.

### Modo 3 - Candidato governado

Empacota e valida candidato promovido. Submissao Kaggle continua manual ou
governada por regra explicita de 1 candidato/dia.

## Criterios de sucesso do objetivo

Marcar este objetivo como concluido somente quando:

- existe um comando unico para rodar `--dry-run --iterations 1`;
- existe um comando unico para rodar `--smoke --iterations 1`;
- o runner registra uma iteracao no `experiments.duckdb`;
- o runner gera relatorio JSON/Markdown parseavel;
- falhas tecnicas produzem `technical_fail`;
- candidato sem amostra valida produz `inconclusive` ou `needs_more_seeds`;
- nenhum caminho do runner altera validacao off-limits;
- ha teste cobrindo parser de metricas e politica keep/discard;
- uma iteracao smoke foi executada e documentada.

## Criterios de pausa

Pausar o loop se ocorrer:

- metric gaming ou alteracao da regua;
- gate plano no topo com promocao indevida;
- fault, timeout, invalid move, bad status ou fallback;
- artefato avaliado diferente do registrado;
- DB sem registro completo;
- custo/tempo por iteracao acima do budget definido.

## Validacao minima

Antes de declarar pronto:

```bash
rtk make verify-binding
rtk .venv/bin/python -m pytest -q
rtk .venv/bin/python -m python.lab.experiments report
rtk .venv/bin/python -m <modulo_auto_research> --dry-run --iterations 1
rtk .venv/bin/python -m <modulo_auto_research> --smoke --iterations 1
```

## Primeiro plano de implementacao

1. Inventariar o que ja existe:
   `python/train/competitive_cycle.py`, `python/train/evaluate_population.py`,
   `scripts/nash_gate.py`, `scripts/league_submit_ruler.py`,
   `python/lab/experiments.py`, docs de treino e relatorios existentes.
2. Criar o contrato JSON da iteracao.
3. Implementar politica keep/discard pura e testada.
4. Implementar parser de relatorios existentes.
5. Implementar runner dry-run.
6. Implementar smoke com uma superficie permitida e budget pequeno.
7. Integrar logging obrigatorio no DuckDB.
8. Documentar como deixar rodando e como interromper sem perder rastreabilidade.
9. Depois do MVP, permitir gerador LLM produzir patches pequenos sob allowlist.

## Resultado esperado desta rodada

Um Auto-Research Loop local, auditavel e especifico para Orbit Wars. Ele deve
copiar a parte util do video: agente programando, treinando, avaliando,
registrando e iterando. Ele nao deve copiar a parte perigosa: autonomia aberta,
metrica editavel, custo sem teto e decisao competitiva baseada em score local
fragil.
