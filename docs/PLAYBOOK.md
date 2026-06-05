# Playbook de experimentos

Use este arquivo como entrada operacional. A arquitetura detalhada continua em `docs/BLUEPRINT.md`; aqui ficam os comandos curtos para testar ideias.

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
```

Interprete `underpowered` como "amostra insuficiente", não como regressão real. Só promova uma mudança quando a margem normalizada média contra o Producer for `>= 0.0` e nenhum veredito marcar regressão significativa. `margin_significant_improvement`, `paired_significant_improvement` ou `significant_improvement` são bônus; `inconclusive` com margem negativa é descarte, não commit.

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

## Validação

```bash
python -m python.lab.cli test
python -m python.lab.cli test --group parity
pytest -q
```

Use `--dry-run` em qualquer comando do CLI para ver o que ele chamaria sem executar.
