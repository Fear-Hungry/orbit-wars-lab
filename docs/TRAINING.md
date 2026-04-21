# Plano de treinamento

## Fase 0 — baseline funcional

- ambiente 2p
- sem cometas
- PPO contra greedy/defensive/rush
- reward shaping leve
- medir captura de neutros e sobrevivência inicial

## Fase 1 — órbitas

- ativar rotação
- decoder prevê posição futura
- penalizar perda para sol e borda

## Fase 2 — self-play simples

- política atual contra snapshots anteriores
- Elo local
- hall-of-fame pequeno

## Fase 3 — liga completa

- população PPO
- PBT em hiperparâmetros
- heurísticas especializadas
- MAP-Elites

## Fase 4 — cometas

- ativar cometas
- criar reward auxiliar temporário para custo-benefício
- remover dependência excessiva do shaping no final

## Fase 5 — 4p

- treinar política separada
- aumentar importância de vulnerabilidade e terceiro jogador

## Fase 6 — seleção final

- seeds retidas
- round-robin massivo
- pior decil de score margin
- análise de replays ruins
- exportação de 2 submissões candidatas
