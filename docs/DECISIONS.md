# Decisões de arquitetura (ADR)

Registro das escolhas estruturais e do *porquê* de cada uma. O *como* operacional
fica em [`PLAYBOOK.md`](PLAYBOOK.md); o *estado atual* de treino em
[`TRAINING.md`](TRAINING.md).

## D1 — Rust para simulação, Python para aprendizado

A engine de treino fica em Rust porque fitness evolutivo exige milhões de turnos
por geração. Python fica responsável por PyTorch, PPO, liga, logging e exportação.

## D2 — Dois modos de simulação

- **Parity mode**: carrega snapshots oficiais gerados pelo `kaggle-environments` e compara step a step.
- **Training mode**: gera mapas próprios com distribuição fiel e domain randomization.

A paridade protege contra treino em universo falso; o training mode maximiza
throughput. Detalhes em [`PARITY.md`](PARITY.md).

## D3 — Ação abstrata + decoder tático

A rede não emite diretamente listas arbitrárias `[from_planet_id, angle, ships]`.
Ela escolhe macrodecisões: origem, alvo, fração de naves e offset. O decoder
calcula ângulo, previsão orbital, segurança contra sol e legalidade. O espaço de
ação concreto está em [`BLUEPRINT.md`](BLUEPRINT.md).

## D4 — PPO é o aprendiz base — quando o aprendizado for ativado

PPO é o algoritmo base escolhido por ser estável, fácil de auditar e compatível
com action masking (referência de implementação: CleanRL, com hooks próprios para
ambiente batched).

> **Estado atual:** o aprendizado por PPO está **deferido**. O caminho ativo é o
> planner OEP com busca sobre o Producer. Esta decisão registra *qual* algoritmo
> será usado quando o aprendizado entrar; as condições de ativação e o estado
> vigente são donos de [`TRAINING.md`](TRAINING.md).

## D5 — Evolução não muta pesos grandes às cegas no início

O GA/PBT entra em: seleção de checkpoints; hiperparâmetros; pesos de reward
shaping; parâmetros do decoder; população de heurísticas; estilos estratégicos.
Mutação direta de rede grande fica para experimentos posteriores.

## D6 — Liga com hall-of-fame é obrigatória

Self-play sem memória histórica gera esquecimento estratégico. O hall-of-fame
força robustez contra versões anteriores, heurísticas especializadas e anti-meta.

## D7 — MAP-Elites preserva diversidade

Elo sozinho seleciona monocultura. O archive por comportamento preserva rushers,
expansivos, defensivos, comet-hunters e estilos híbridos.

## D8 — Recompensa oficial só no fim; shaping apenas para treino

O agente final é avaliado pela regra oficial. Durante treino, shaping potencial
acelera aprendizado, mas não substitui vitória e score final.

## D9 — Políticas separadas para 2p e 4p

Orbit Wars 2p e 4p têm dinâmicas diferentes. Em 2p, dano direto é mais limpo. Em
4p, overextension vira vulnerabilidade para terceiros. Treine e avalie separadamente.

## D10 — Submissão final sem Rust

Rust é infraestrutura local. O `submission.py` final deve ser Python leve, com
fallback **instrumentado** (nunca silencioso), para reduzir risco de timeout,
incompatibilidade e crash. A regra operacional e o checklist ficam em
[`SUBMISSION.md`](SUBMISSION.md).

## D11 — A fronteira Rust/Python é um invariante, não uma preferência (consolida D1 + D10)

"Só Rust para ser rápido" **não é uma opção**: a plataforma fixa o formato. O
Kaggle Orbit Wars executa um `agent(obs)` **Python** (via `kaggle-environments`);
não há como submeter binário/extensão Rust. E velocidade não é o gargalo — o
agente roda em 4–290 ms/step contra um `actTimeout` de **1 s**
(`tests/test_official_spec.py`); o limite é qualidade do modelo de estado/valor,
não throughput de compute.

Regra de fronteira a preservar:

> **Rust só simula; Python decide e é o que se submete.** Nenhum bot em `bots/`
> nem submissão em `artifacts/` pode importar `orbit_wars_core` / `orbit_wars_py`.
> O crate Rust é importado apenas pelo simulador local
> (`python/orbit_wars_gym/backend.py`) e pela CLI (`python/lab/cli.py`).

Verificado em 2026-06-05: nenhum agente/submissão importa o crate. Dois modos de
envio válidos:

- **arquivo único Python puro** (`artifacts/submission.py`, só `math` — dependência zero; o `export_submission.py` serializa a rede PPO em forward pass Python puro, sem torch em runtime);
- **tarball** com `orbit_lite` + torch (ex.: Producer, ~53 KB).

Não compilar o planner para `.so` e submeter: casar Python/arquitetura do Kaggle é
frágil e um `.so` que não carrega = submissão morta, sem ganho (não é CPU-bound). O
único crescimento legítimo de Rust é **acelerar o treino local** (batch step
vetorizado no `orbit_wars_core`), e só após perfilar e confirmar que o rollout em
CPU é o gargalo.

**Guarda recomendada:** um teste de arquitetura (`test_no_native_in_submission`)
que falha se `artifacts/submission.py` ou o tarball importarem o crate Rust, para
impedir regressão silenciosa que quebraria a submissão.
