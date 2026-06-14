# Plano de paridade com Kaggle

Este documento é a **fonte de verdade** das constantes e regras do ambiente
oficial. Outros docs (ex.: o `README`) referenciam estes valores em vez de
repeti-los.

## Alvo oficial

A compatibilidade deve mirar o `kaggle-environments` oficial para Orbit Wars,
definido por dois arquivos da fonte oficial:

- `orbit_wars.json` (v1.0.9) — schema de observação/ação e parâmetros do episódio.
- `orbit_wars.py` — constantes físicas, geração de planetas/cometas, lançamento, produção, movimento, colisão contínua, rotação planetária, sweep collision e combate.

URLs de referência:

- https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/orbit_wars/orbit_wars.json
- https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/orbit_wars/orbit_wars.py

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

Cobertos em `tests/test_official_spec.py`, `tests/test_official_snapshots.py` e
`tests/test_parity_tolerances.py`. A paridade sob **combate ativo** (launch → colisão →
combate → captura → reforço, que os testes passivos NÃO exercem) é coberta por
`tests/test_parity_actions.py` e `tests/test_movement_fidelity.py`.

> **Combate ativo — bug corrigido 2026-06-09:** `compute_planet_paths` (`step.rs`) calculava a
> posição orbital por rotação matricial; o oficial usa reconstrução **polar**
> `r·cos(atan2(dy,dx)+ω·step)`. Idênticas na matemática, mas float-diferentes (~1e-10) — dentro da
> tolerância da paridade passiva, mas o bastante para virar a colisão *knife-edge* de uma frota
> recém-lançada contra seu planeta de origem em órbita. Fix: replicar o caminho polar exato.

## Frescor do binding (CRÍTICO — ler antes de confiar em QUALQUER resultado de motor)

O `RustBatchBackend` é a "verdade" do motor de treino/régua. Se o `.so` instalado no venv está
**stale** vs `crates/`, todo teste/treino/eval roda o motor ERRADO **em silêncio** (e o "expected"
dos testes de fidelidade vira a verdade errada). Duas armadilhas, ambas já custaram horas:

1. **`maturin develop` nem sempre recompila** (fingerprint de mtime no WSL) — confirme o build.
2. **`uv run` (com auto-sync) REVERTE o `.so` fresco** reinstalando `orbit-wars-lab` de um wheel em
   cache (`Uninstalled 1/Installed 1` na saída). Por isso `make` usa `uv run --no-sync` em tudo que
   RODA código, e `make build` faz `sync-binding` (force-copy de `target/release/liborbit_wars_rs.so`
   para o venv). **Nunca** rode testes/treino com `uv run` puro depois de um build.

**Sempre:** `make build` (compila + sincroniza o binding) → rode com `make test`/`make <target>` ou
`uv run --no-sync …` ou `.venv/bin/python …`. Verifique com `make verify-binding`. Nunca trate o
binding compilado como verdade sem confirmar frescor.

## Estratégia

Não tente reproduzir o RNG Python em Rust para treinamento. Para paridade, carregue
snapshots oficiais. Para treino, use gerador Rust determinístico com distribuição
fiel (ver D2 em [`DECISIONS.md`](DECISIONS.md)).

## Tolerância

- `owner`, `ships`, `id`, `done`: igualdade exata.
- `x`, `y`, `angle`: tolerância inicial `1e-9`, relaxável apenas se houver diferença justificada de float.
