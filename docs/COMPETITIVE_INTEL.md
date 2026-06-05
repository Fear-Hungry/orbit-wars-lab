# Inteligencia competitiva Orbit Wars

Registro curto de achados externos usados para orientar experimentos locais. Discussao publica
do Kaggle e evidencia de notebooks sao sinais praticos, nao verdade absoluta; toda ideia abaixo
precisa passar pela regua local contra Producer antes de virar agente.

## Fontes consultadas em 2026-06-05

- Competicao: `orbit-wars`.
- Kaggle CLI: `2.2.0`.
- Topico `704095`, "Community Benchmark — 109-Agent Mega Tournament": round-robin comunitario
  com 109 agentes publicos e 9894 partidas. O topo do torneio e dominado por agentes derivados de
  heuristicas fortes, busca limitada, simulacao de timeline e variantes de redistribuicao.
- Topico `704113`, "Introducing The Producer agent": Producer usa duas regras centrais:
  projetar ganho de producao em horizonte `H` antes de enviar naves; mover naves ociosas para
  planetas amigos mais perto do inimigo. O autor reportou ~1200 de forca e 100-200 ms/turno.
- Notebooks puxados para `/tmp/orbit_wars_f3` apenas para leitura local:
  - `shummingfang/orbit-wars-exp27-search-max-actions-to-pick-2p-8`
  - `yashm917/orbit-wars-sim-value-search-agent`
  - `emanuellcs/orbit-wars-advanced-timeline-simulation-agent`
  - `byfone/orbit-dominance-based-fleet-redistribution`

## Sinais extraidos

- Producer e piso, nao teto. O benchmark comunitario lista varios agentes acima do Producer
  publico com sinais de busca, simulacao de timeline, politica de seguranca de lancamento e
  redistribuicao.
- `exp27` combina busca limitada por acoes (`SEARCH_MAX_ACTIONS_TO_PICK_2P=8`), forward sim curto
  (`FWD_SIM_HORIZON=7`), filtros de captura segura e ataques coordenados tipo hammer/multiprong.
- `sim-value-search-agent` usa defesa ponderada por producao, gera top-K candidatos por fonte,
  simula cada candidato por 20 ticks e pontua estado terminal com funcao de valor.
- `advanced-timeline-simulation-agent` projeta timeline longa (~115 turnos), resolve arrivals/
  combate por planeta e monta missoes de rescue, recapture, reinforce, snipe e crash exploit.
- `dominance-based-fleet-redistribution` calcula dominancia local por soma assinada ponderada por
  distancia e redistribui surplus para regioes com deficit/baixa dominancia.

## Ideias acionaveis

1. **OEP action shortlist tipo top-K por fonte.**
   Hipotese: antes de criar genomas, ranquear os candidatos Producer/OEP por valor diferencial e
   limitar a `K` por fonte reduz custo sem perder os casos bons. Teste: varrer `K in {3,5,8}` contra
   Producer, 16 seeds smoke e depois gate OEP 96 seeds; aceitar apenas se margem nao cair e p95 ms
   cair materialmente.

2. **Fitness OEP com risco de timeline curta.**
   Hipotese: adicionar penalidade de defesa/recaptura baseada em arrivals nos proximos 7-20 turnos
   evita launches que parecem bons por producao mas abrem colapso local. Teste: OEP vs Producer
   com mesma shortlist, comparar `mean_score_margin`, `paired_margin_delta` e `max_decision_ms`;
   `greedy`/`rush` continuam sanity de crash, nao regua de qualidade.

3. **Redistribuicao por dominancia como candidato de plano, nao substituto do Producer.**
   Hipotese: o calculo `signed ships / distance` identifica fronts que o Producer nao prioriza
   quando a producao projetada empata. Teste: gerar entries de regroup por dominancia e inserir no
   torneio de fitness como mais um candidato; promover so se bater Producer por margem no gate OEP.

4. **Missoes coordenadas de multiprong/hammer como genes compostos.**
   Hipotese: quando o Producer concentra defesa em um alvo de alta producao, um gene composto com
   alvo principal + prong no planeta reforcador explora a resposta gulosa. Teste: implementar primeiro
   detector offline em replays contra Producer; so depois emitir moves. Aceite exige margem positiva
   e sem aumento de timeout.

## Lacunas

- A CLI `forums topics show` nesta instalacao nao aceita `--json`; os topicos foram lidos em texto.
- Notebooks publicos podem ter mudado depois da extracao em `/tmp`; nao foram versionados no repo.
- Nenhuma ideia acima foi validada localmente ainda. F3 fecha inteligencia/hipoteses, nao promocao.
