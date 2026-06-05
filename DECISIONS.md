# Decisões já tomadas

## D1 — Rust para simulação, Python para aprendizado

A engine de treino fica em Rust porque fitness evolutivo exige milhões de turnos por geração. Python fica responsável por PyTorch, PPO, liga, logging e exportação.

## D2 — Dois modos de simulação

- **Parity mode**: carrega snapshots oficiais gerados pelo `kaggle-environments` e compara step a step.
- **Training mode**: gera mapas próprios com distribuição fiel e domain randomization.

A paridade protege contra treino em universo falso. O training mode maximiza throughput.

## D3 — Ação abstrata + decoder tático

A rede não deve emitir diretamente listas arbitrárias `[from_planet_id, angle, ships]`. Ela escolhe macrodecisões: origem, alvo, fração de naves e offset. O decoder calcula ângulo, previsão orbital, segurança contra sol e legalidade.

## D4 — PPO como aprendiz base

PPO é o algoritmo base porque é estável, fácil de auditar e compatível com action masking. CleanRL é a referência de implementação, mas este repositório mantém hooks próprios para ambiente batched.

## D5 — Evolução não deve mutar pesos grandes às cegas no início

O GA/PBT entra em:

- seleção de checkpoints;
- hiperparâmetros;
- pesos de reward shaping;
- parâmetros do decoder;
- população de heurísticas;
- estilos estratégicos.

Mutação direta de rede grande fica para experimentos posteriores.

## D6 — Liga com hall-of-fame é obrigatória

Self-play sem memória histórica gera esquecimento estratégico. O hall-of-fame força robustez contra versões anteriores, heurísticas especializadas e anti-meta.

## D7 — MAP-Elites preserva diversidade

Elo sozinho seleciona monocultura. O archive por comportamento preserva rushers, expansivos, defensivos, comet-hunters e estilos híbridos.

## D8 — Recompensa oficial só no fim; shaping apenas para treino

O agente final é avaliado pela regra oficial. Durante treino, shaping potencial acelera aprendizado, mas não deve substituir vitória e score final.

## D9 — Políticas separadas para 2p e 4p

Orbit Wars 2p e 4p têm dinâmicas diferentes. Em 2p, dano direto é mais limpo. Em 4p, overextension vira vulnerabilidade para terceiros. Treine e avalie separadamente.

## D10 — Submissão final sem Rust

Rust é infraestrutura local. O `submission.py` final deve ser Python leve, com fallback heurístico, para reduzir risco de timeout, incompatibilidade e crash.

## D11 — Fronteira Rust/Python é um invariante, não uma preferência (consolida D1 + D10)

"Só Rust para ser rápido" **não é uma opção**: a plataforma fixa o formato. O Kaggle Orbit Wars executa um `agent(obs)` **Python** (via `kaggle-environments`); não há como submeter binário/extensão Rust. E velocidade não é o gargalo — o agente roda em 4–290 ms/step contra um `actTimeout` de **1 s** (`tests/test_official_spec.py`); o limite é qualidade do modelo de estado/valor, não throughput de compute.

Regra de fronteira a preservar:

> **Rust só simula; Python decide e é o que se submete.** Nenhum bot em `bots/` nem submissão em `artifacts/` pode importar `orbit_wars_core` / `orbit_wars_py`. O crate Rust é importado apenas pelo simulador local (`python/orbit_wars_gym/backend.py`) e pela CLI (`python/lab/cli.py`).

Verificado em 2026-06-05: nenhum agente/submissão importa o crate. Dois modos de envio válidos:

- **arquivo único Python puro** (`artifacts/submission.py`, só `math` — dependência zero; o `export_submission.py` serializa a rede PPO em forward pass Python puro, sem torch em runtime);
- **tarball** com `orbit_lite` + torch (ex.: Producer, ~53 KB).

Não compilar o planner para `.so` e submeter: casar Python/arquitetura do Kaggle é frágil e um `.so` que não carrega = submissão morta, sem ganho (não é CPU-bound). O único crescimento legítimo de Rust é **acelerar o treino local** (batch step vetorizado no `orbit_wars_core`), e só após perfilar e confirmar que o rollout em CPU é o gargalo.

**Guarda recomendada:** um teste de arquitetura (`test_no_native_in_submission`) que falha se `artifacts/submission.py` ou o tarball importarem o crate Rust, para impedir regressão silenciosa que quebraria a submissão.
