# Plano de treinamento

## Decisão atual

PPO/self-play fica deferido enquanto o caminho OEP ainda for o candidato ativo contra o Producer.
A ativação do PPO só volta para a fila quando uma destas condições for verdadeira:

- OEP passa o gate de promoção contra Producer e vira baseline a ser batido por aprendizado;
- OEP esgota o ganho mensurável contra Producer, com experimentos registrados em `EXPERIMENTS.md`;
- surgir oponente externo forte que exija diversidade de politica em vez de busca sobre o planner.

Quando ativado, PPO deve treinar contra Producer/heuristico e só gerar candidato promovivel se o
checkpoint exportado tiver margem contra Producer >= baseline OEP, com crash/timeout/fallback igual
a zero. Runs smoke de PPO podem existir para infraestrutura, mas nao sao caminho de submissao.

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
