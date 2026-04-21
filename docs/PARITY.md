# Plano de paridade com Kaggle

## Alvo oficial

A compatibilidade deve mirar o `kaggle-environments` oficial para Orbit Wars.

Elementos conhecidos:

- `episodeSteps = 500`
- `actTimeout = 1`
- `agents = [2, 4]`
- `shipSpeed = 6.0`
- `cometSpeed = 4.0`
- observação: `planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`, `next_fleet_id`, `comets`, `comet_planet_ids`
- ação: lista de `[from_planet_id, direction_angle, num_ships]`

## Testes obrigatórios

1. Inicialização por snapshot oficial.
2. Sanitização de ação inválida.
3. Lançamento de frota fora do raio do planeta.
4. Produção depois do lançamento.
5. Fórmula de velocidade por tamanho de frota.
6. Colisão contínua com borda, sol e planetas.
7. Rotação planetária.
8. Sweep collision de planeta móvel contra frota.
9. Combate maior força vs segunda maior.
10. Captura/reforço de planeta.
11. Expiração e movimento de cometas.
12. Score terminal.

## Estratégia

Não tente reproduzir o RNG Python em Rust para treinamento. Para paridade, carregue snapshots oficiais. Para treino, use gerador Rust determinístico com distribuição fiel.

## Tolerância

- `owner`, `ships`, `id`, `done`: igualdade exata.
- `x`, `y`, `angle`: tolerância inicial `1e-9`, relaxável apenas se houver diferença justificada de float.
