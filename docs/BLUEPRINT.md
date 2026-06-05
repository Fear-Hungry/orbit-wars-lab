# Blueprint competitivo

O método de seleção que o projeto persegue. A arquitetura que o suporta está em
[`ARCHITECTURE.md`](ARCHITECTURE.md); o estado atual (PPO deferido, planner OEP
ativo) em [`TRAINING.md`](TRAINING.md).

## Objetivo terminal

Gerar um conjunto pequeno de agentes finais robustos contra estratégias variadas
em 2p e 4p, com baixa taxa de erro, baixa explorabilidade e bom desempenho em
seeds desconhecidas.

> A meta da competição é **terminar no top 5**. Bater o Producer público é a
> *entrada*, não a meta — ele é piso, não teto (ver
> [`COMPETITIVE_INTEL.md`](COMPETITIVE_INTEL.md)).

## Núcleo

```text
Rust core -> simulação em massa
PyO3 -> interface Python
Gymnasium/PettingZoo -> treino RL
PPO -> aprendizado de política
PBT/GA -> seleção e mutação de hiperparâmetros/decoders
Hall-of-fame -> memória estratégica
MAP-Elites -> diversidade comportamental
Submission exporter -> agente Kaggle leve
```

## Ciclo de geração

```text
1. Criar população: heurísticas + políticas PPO + mutantes.
2. Parear por Elo e diversidade.
3. Rodar partidas em Rust batch.
4. Atualizar PPO para agentes treináveis.
5. Avaliar em seeds separadas.
6. Atualizar Elo e fitness.
7. Preservar elites e estilos raros.
8. Clonar/mutar piores.
9. Congelar snapshots no hall-of-fame.
10. Repetir.
```

## Espaço de ação

Ação neural compacta:

```text
[source_rank, target_rank, fraction_idx, offset_idx]
```

Decoder:

```text
source_rank  -> planeta próprio ordenado por força
target_rank  -> alvo ordenado por valor estratégico
fraction_idx -> fração discreta de naves
offset_idx   -> pequeno desvio angular
```

Vantagem: reduz ações inválidas e deixa a rede aprender intenção, não trigonometria
básica.

## Métrica de fitness

```text
fitness =
  2.00 * win_rate_vs_league
+ 0.80 * normalized_score_margin
+ 0.50 * win_rate_vs_hall_of_fame
+ 0.25 * robustness_across_seeds
+ 0.20 * novelty_bonus
- 1.00 * crash_rate
- 0.20 * invalid_action_rate
```

## Critérios de promoção

Um candidato só entra no hall-of-fame se:

- vence pelo menos uma fração mínima da liga;
- não aumenta crash/timeout;
- melhora desempenho em seeds de avaliação;
- ou ocupa uma célula comportamental vazia no MAP-Elites.

A régua executável (gate OEP contra o Producer) está em [`PLAYBOOK.md`](PLAYBOOK.md).

## Antipadrões

- Treinar contra um único bot.
- Usar GA puro em rede grande sem PPO.
- Permitir ação contínua bruta desde o início.
- Medir só média e ignorar pior decil.
- Otimizar para mapa simplificado e esquecer paridade.
- Submeter agente sem fallback instrumentado.
