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
