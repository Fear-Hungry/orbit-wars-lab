# Playbook de experimentos

Fonte de verdade **operacional**: os comandos para reproduzir treino, benchmark e
gates. O método está em [`BLUEPRINT.md`](BLUEPRINT.md); a arquitetura em
[`ARCHITECTURE.md`](ARCHITECTURE.md). Comece pelo índice em [`README.md`](README.md).

## Primeiro check

```bash
python -m python.lab.cli doctor
python -m python.lab.cli heuristics
```

## Testar uma mudança de heurística ou decoder

Edite a estratégia em `python/agents/heuristics.py` ou o decoder em `python/orbit_wars_gym/action_decoder.py`.
Se adicionar uma heurística nova, registre-a em `python/agents/registry.py`.

Depois rode:

```bash
python -m python.lab.cli quick
```

Esse comando:

1. exporta `artifacts/submission.py`;
2. roda um benchmark curto 2p/4p;
3. salva o relatório em `artifacts/submission_benchmark.json`.

Para uma comparação um pouco mais estável:

```bash
python -m python.lab.cli bench-submission --seeds 8 --episode-steps 500
```

## Re-adjudicar ideia estrutural

Benchmarks de 16 seeds (`32` partidas 2p por oponente) servem para smoke e regressão grosseira. Eles não têm potência estatística para matar ideias estruturais com queda aparente perto de `0.05` de win rate.

Antes de rejeitar mudanças de abertura, hammer, reserva, lookahead ou fase, rode uma amostra maior contra o Producer. Ele é o oponente decisor; `submission_v_old.py`, `greedy` e `rush` são sanity checks técnicos.

```bash
rtk .venv/bin/python -m scripts.benchmark_submission \
  --submission bots/producer/agent.py \
  --opponents producer \
  --seeds 96 --episode-steps 500 --jobs 4 --skip-4p \
  --out artifacts/producer_mirror_96seed.json

rtk .venv/bin/python -m scripts.benchmark_submission \
  --submission artifacts/submission_candidate.py \
  --opponents producer \
  --seeds 96 --episode-steps 500 --jobs 4 --skip-4p \
  --out artifacts/candidate_vs_producer_96seed.json

rtk .venv/bin/python -m scripts.compare_benchmark_significance \
  --baseline artifacts/producer_mirror_96seed.json \
  --candidate artifacts/candidate_vs_producer_96seed.json \
  --min-games 128 --min-effect 0.05

rtk .venv/bin/python -m scripts.oep_promotion_gate \
  --baseline artifacts/gates/producer_fix_gates/g2_champion_vs_corrected_producer_96seed.json \
  --candidate artifacts/candidate_vs_producer_96seed.json \
  --out artifacts/gates/oep/promotion_gate.json
```

Interprete `underpowered` como "amostra insuficiente", não como regressão real. Só promova uma mudança quando a margem normalizada média contra o Producer for `>= 0.0` e nenhum veredito marcar regressão significativa. `margin_significant_improvement`, `paired_significant_improvement` ou `significant_improvement` são bônus; `inconclusive` com margem negativa é descarte, não commit.
Para o OEP, `scripts.oep_promotion_gate` torna essa regra executável: exige `192` jogos (`96` seeds × `2` lados), margem média `>= 0.0` contra Producer, crash/timeout/invalid/fallback/policy-illegal/fallback-error/instrumentation-missing `0.0`, e delta pareado de margem não-negativo contra o baseline G2.

Rode `submission_v_old.py`, `greedy`, `rush` e `4p` em baixa amostra como sanity técnico, não como decisor de melhoria 2p:

```bash
rtk .venv/bin/python -m scripts.benchmark_submission \
  --submission artifacts/submission_candidate.py \
  --opponents artifacts/submission_v_old.py greedy rush \
  --seeds 8 --episode-steps 500 --jobs 4 \
  --out artifacts/candidate_sanity_8seed.json
```

Para usar o Producer público como oponente externo local:

```bash
rtk .venv/bin/python -m scripts.benchmark_submission \
  --submission artifacts/submission.py \
  --opponents producer \
  --seeds 16 --episode-steps 500 --jobs 4 --skip-4p \
  --out artifacts/champion_vs_producer_16seed.json
```

Para empacotar o Producer fiel como submissão Kaggle:

```bash
rtk .venv/bin/python -m scripts.package_producer_submission \
  --out artifacts/submission_producer.tar.gz
```

Regra operacional:

- `16` seeds: smoke, legalidade e regressão grosseira;
- `64` seeds: triagem de ideias candidatas;
- `96` seeds contra Producer: decisão iterativa 2p por margem normalizada;
- `256+` seeds: confirmação final quando uma mudança estrutural já passou no decisor;
- rejeição por benchmark exige `significant_regression` ou falha técnica objetiva;
- mudanças avaliadas nos mesmos seeds devem usar o bloco pareado do comparador (`paired_*`);
- `submission_v_old.py`, `greedy` e `rush`: sanity de crash/legalidade, não promoção;
- quando houver margem normalizada por jogo, priorize o veredito de margem (`margin_*`) sobre win rate binário.

Para medir se o benchmark está barato o bastante antes de subir a amostra:

```bash
rtk .venv/bin/python -m scripts.measure_benchmark_throughput \
  --seeds 4 8 16 --jobs 1 4 8 --skip-4p \
  --out artifacts/throughput/summary.json
```

## Avaliar população de candidatos

Para iteração rápida:

```bash
python -m python.lab.cli eval
```

Isso usa:

- manifesto: `configs/final_candidate_pool.yaml`;
- config rápida: `configs/eval_quick.yaml`;
- saída: `artifacts/evaluation_report.json`.

Para a avaliação final pesada:

```bash
python -m python.lab.cli eval --config configs/eval_final.yaml
```

## Rodar uma iteração de liga

Depois de gerar `artifacts/evaluation_report.json`:

```bash
python -m python.lab.cli league
```

Estados persistidos:

- `artifacts/hall_of_fame.json`;
- `artifacts/map_elites.json`.

## Validar pela liga local

Antes de usar a liga para julgar um bot novo, rode o diagnóstico executável:

```bash
rtk .venv/bin/python scripts/league_doctor.py
```

Ele valida os invariantes que tornam a liga comparável ao ambiente de competição:

- `actTimeout=1s` com banco de overage `12s`;
- crash vira `ERROR`, timeout terminal vira `TIMEOUT`, e esses assentos não podem vencer;
- movimentos malformados ou overbudget são contabilizados em `faults`;
- jogos limpos ainda gravam `faults: {}` para separar jogo auditado de JSON antigo;
- empates não viram vitória falsa do assento 0;
- tarballs são isolados por conteúdo e reexportar o mesmo nome invalida cache velho;
- smoke real 2p/4p roda bots internos com status `DONE` e zero faults.

Se o diagnóstico avisar `existing_artifacts_are_fully_audited`, isso significa que o
corpus antigo em `artifacts/league/v1` contém jogos pré-instrumentação. Use esse
corpus como histórico anotado, não como prova limpa para promover bot. Para exigir
corpus 100% auditado em CI, rode:

```bash
rtk .venv/bin/python scripts/league_doctor.py --strict-existing
```

Para uma sonda isolada, sem escrever no corpus permanente:

```bash
rtk .venv/bin/python scripts/league_match.py \
  --agents candidate,producer \
  --seeds 16 --seed-base 50000 --steps 500 \
  --out /tmp/candidate_vs_producer_league.json

rtk .venv/bin/python scripts/league_report.py '/tmp/candidate_vs_producer_league.json' 100
```

Para validação competitiva de um candidato registrado em `scripts/league_agents.py`
ou em `artifacts/league/tarballs/<nome>.tar.gz`, use H2H contra o pool:

```bash
rtk .venv/bin/python scripts/league_challenger.py --candidate candidate --seeds 20 --workers 4
```

Para escolher submissão entre candidatos, use a régua pareada forte. Ela não usa
BT/ranking aleatório: roda cada candidato contra as mesmas âncoras 2p, lineups
4p fixas, H2H direto contra o incumbente e contra os outros candidatos do mesmo
comando, e só recomenda `PASS_LOCAL` se todos os gates técnicos e competitivos
passarem. O `overall_score` usa o split de campo medido (46% 2p / 54% 4p), conta
empates 2p como não-vitórias no score bruto, e o ranking é ordenado primeiro por
veredito: `PASS_LOCAL` > `INCONCLUSIVE` > `REJECT_LOCAL`. Os mapas 2p sao
derivados do nome do adversario, entao `vs producer` usa o mesmo slice mesmo que
voce rode um candidato sozinho ou dentro de um painel maior.
Em 4p, cada seed da regua decisora e jogada nas quatro rotacoes de assento; isso
evita confundir sorte de mapa com posicao inicial e tambem impede JSON parcial
de registrar seeds que o backend nao executou.

```bash
rtk .venv/bin/python scripts/league_submit_ruler.py \
  --candidates pgs_hold pgs_holdwave pgs_wave_s100 \
  --incumbent pgs_holdwave \
  --seeds 24 --steps 500 --jobs 2 \
  --match-chunk-size 8 \
  --out artifacts/league/submit_ruler/report.json
```

Em runs longos, mantenha `--match-chunk-size` ligado. Os JSONs parciais sao
gravados por chunk e, nos runs iniciados apos o commit `ac6ad0c`, os chunks sao
intercalados por seat order/rotacao para que o progresso parcial ja seja
monitoravel sem esperar o match inteiro. Nos runs apos o commit de checkpoint
balanceado, o arquivo parcial so e escrito depois de completar o par de seat
orders 2p ou o bloco de rotacoes 4p disponivel, evitando leitura enviesada no
meio do chunk.
O `--skip-run` e estrito: ele so aceita JSONs que batem exatamente com a tarefa
esperada (modo, agentes, seed slice, numero de jogos, seat orders/rotacoes,
`faults` e `agent_status`). JSON antigo, parcial ou de outro comando deve falhar.
Fallback/timeout interno reportado por `SUBMISSION_STATS` de tarball tambem e
falha audivel na liga; nao use ranking local que esteja medindo o Producer
fallback como se fosse o bot.

Interpretação:

- `PASS_LOCAL`: candidato passou faults/status, cobriu H2H mínimo, empatou/bateu
  Producer e incumbente, limpou o floor rejeitado e não morreu demais em 4p;
- `INCONCLUSIVE`: a amostra não tem decisivos suficientes; aumente `--seeds`;
- `REJECT_LOCAL`: há falha técnica ou gate competitivo local ruim.

Regra operacional: a liga é **veto-only**. Ela é boa para detectar crash, timeout,
ação inválida, perda H2H clara, fragilidade 4p e estilos exploradores. Ela não
promove sozinha entre configs próximas; promoção final exige score Kaggle/LB
estabilizado e interpretação das âncoras atuais.

## Validação

```bash
python -m python.lab.cli test
python -m python.lab.cli test --group parity
pytest -q
```

Use `--dry-run` em qualquer comando do CLI para ver o que ele chamaria sem executar.
