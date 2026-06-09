# Plano de treinamento

Fonte de verdade do **estado atual** do treino. O *qual algoritmo* e o *porquê*
ficam em D4 de [`DECISIONS.md`](DECISIONS.md); as fases abaixo são o *como*.

## Decisão atual

PPO/self-play fica **deferido** enquanto o caminho OEP (planner com busca sobre o
Producer) ainda for o candidato ativo. A ativação do PPO só volta para a fila
quando uma destas condições for verdadeira:

- OEP passa o gate de promoção contra Producer e vira baseline a ser batido por aprendizado;
- OEP esgota o ganho mensurável contra Producer, com experimentos registrados no store `experiments.duckdb` (`make experiments-report`);
- surge oponente externo forte que exija diversidade de política em vez de busca sobre o planner.

Quando ativado, PPO deve treinar contra Producer/heurístico e só gerar candidato
promovível se o checkpoint exportado tiver margem contra Producer ≥ baseline OEP,
com crash/timeout/fallback igual a zero. Runs smoke de PPO podem existir para
infraestrutura, mas não são caminho de submissão.

## Fases (quando o aprendizado for ativado)

### Fase 0 — baseline funcional
- ambiente 2p, sem cometas
- PPO contra greedy/defensive/rush
- reward shaping leve
- medir captura de neutros e sobrevivência inicial

### Fase 1 — órbitas
- ativar rotação
- decoder prevê posição futura
- penalizar perda para sol e borda

### Fase 2 — self-play simples
- política atual contra snapshots anteriores
- Elo local; hall-of-fame pequeno

### Fase 3 — liga completa
- população PPO
- PBT em hiperparâmetros
- heurísticas especializadas
- MAP-Elites

### Fase 4 — cometas
- ativar cometas
- reward auxiliar temporário para custo-benefício
- remover dependência excessiva do shaping no final

### Fase 5 — 4p
- treinar política separada (ver D9 em [`DECISIONS.md`](DECISIONS.md))
- aumentar importância de vulnerabilidade e terceiro jogador

### Fase 6 — seleção final
- seeds retidas; round-robin massivo
- pior decil de score margin
- análise de replays ruins
- exportação de 2 submissões candidatas
