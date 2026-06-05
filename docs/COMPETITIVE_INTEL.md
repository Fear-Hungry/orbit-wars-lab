# Inteligência competitiva Orbit Wars

Registro curto de achados externos usados para orientar experimentos locais.
Discussão pública do Kaggle e evidência de notebooks são sinais práticos, não
verdade absoluta; toda ideia abaixo precisa passar pela régua local contra o
Producer antes de virar agente.

> **Fonte canônica de "Producer é piso, não teto".** Quando outros documentos
> citarem essa premissa, apontam para cá.

## Fontes consultadas em 2026-06-05

- Competição: `orbit-wars`. Kaggle CLI: `2.2.0`.
- Tópico `704095`, "Community Benchmark — 109-Agent Mega Tournament": round-robin comunitário com 109 agentes públicos e 9894 partidas. O topo é dominado por agentes derivados de heurísticas fortes, busca limitada, simulação de timeline e variantes de redistribuição.
- Tópico `704113`, "Introducing The Producer agent": Producer usa duas regras centrais — projetar ganho de produção em horizonte `H` antes de enviar naves; mover naves ociosas para planetas amigos mais perto do inimigo. O autor reportou ~1200 de força e 100–200 ms/turno.
- Notebooks lidos localmente (não versionados): `shummingfang/orbit-wars-exp27-search-max-actions-to-pick-2p-8`, `yashm917/orbit-wars-sim-value-search-agent`, `emanuellcs/orbit-wars-advanced-timeline-simulation-agent`, `byfone/orbit-dominance-based-fleet-redistribution`.

## Sinais extraídos

- **Producer é piso, não teto.** O benchmark comunitário lista vários agentes acima do Producer público com sinais de busca, simulação de timeline, política de segurança de lançamento e redistribuição.
- `exp27` combina busca limitada por ações (`SEARCH_MAX_ACTIONS_TO_PICK_2P=8`), forward sim curto (`FWD_SIM_HORIZON=7`), filtros de captura segura e ataques coordenados tipo hammer/multiprong.
- `sim-value-search-agent` usa defesa ponderada por produção, gera top-K candidatos por fonte, simula cada candidato por 20 ticks e pontua estado terminal com função de valor.
- `advanced-timeline-simulation-agent` projeta timeline longa (~115 turnos), resolve arrivals/combate por planeta e monta missões de rescue, recapture, reinforce, snipe e crash exploit.
- `dominance-based-fleet-redistribution` calcula dominância local por soma assinada ponderada por distância e redistribui surplus para regiões com déficit/baixa dominância.

## Ideias acionáveis

1. **OEP action shortlist tipo top-K por fonte.** Antes de criar genomas, ranquear os candidatos Producer/OEP por valor diferencial e limitar a `K` por fonte reduz custo sem perder os casos bons. Teste: varrer `K in {3,5,8}` contra Producer, 16 seeds smoke e depois gate OEP 96 seeds; aceitar só se a margem não cair e p95 ms cair materialmente.
2. **Fitness OEP com risco de timeline curta.** Adicionar penalidade de defesa/recaptura baseada em arrivals nos próximos 7–20 turnos evita launches que parecem bons por produção mas abrem colapso local. Teste: OEP vs Producer com mesma shortlist, comparar `mean_score_margin`, `paired_margin_delta` e `max_decision_ms`.
3. **Redistribuição por dominância como candidato de plano**, não substituto do Producer. O cálculo `signed ships / distance` identifica fronts que o Producer não prioriza quando a produção projetada empata. Promover só se bater Producer por margem no gate OEP.
4. **Missões coordenadas de multiprong/hammer como genes compostos.** Quando o Producer concentra defesa em um alvo de alta produção, um gene composto com alvo principal + prong no planeta reforçador explora a resposta gulosa. Implementar primeiro detector offline em replays; só depois emitir moves.

## Lacunas

- A CLI `forums topics show` nesta instalação não aceita `--json`; os tópicos foram lidos em texto.
- Notebooks públicos podem ter mudado após a extração; não foram versionados no repo.
- Nenhuma ideia acima foi validada localmente ainda — esta página fecha inteligência/hipóteses, não promoção.
