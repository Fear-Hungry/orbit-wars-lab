# Plano de treinamento

Fonte de verdade do **estado atual** do treino. O *qual algoritmo* e o *porquê*
ficam em D4 de [`DECISIONS.md`](DECISIONS.md); as fases abaixo são o *como*.

## Decisão atual

PPO/BReP voltou para a trilha ativa, mas não como treino aberto. O caminho
aceito é uma campanha com validação rígida: checkpoints PPO são exportados para
submissão Kaggle e avaliados pela régua DRL em `scripts/drl_promotion_gate.py`.
Essa exportação agora valida paridade entre checkpoint PyTorch, arquivo `.py`
renderizado e tarball Kaggle com `main.py`; qualquer fallback/illegal increment
em `SUBMISSION_STATS` reprova a paridade. Com paridade ativa, o gate registra
o tarball validado na liga local.
Um checkpoint só é promovível se passar em 2p e 4p contra o pool congelado:
`pgs_holdwave`, `producer`, `oep`, `pgs_bigwave`, `greedy`, `rush` e `brep`,
com `bad_status=0`, `fault_games=0`, sem crash, timeout, invalid, fallback ou
instrumentação ausente.

Comando de auditoria/promoção:

```bash
rtk .venv/bin/python scripts/audit_ppo_checkpoints.py \
  "artifacts/ppo/**/*.pt" "artifacts/bc/*.pt" \
  --out-dir artifacts/ppo/audit_current

rtk .venv/bin/python scripts/drl_promotion_gate.py \
  --checkpoint "artifacts/ppo/**/*.pt" \
  --candidate brep \
  --profile quick \
  --out-dir artifacts/drl_promotion_gate
```

Campanha PPO com gate forte por chunk:

```bash
rtk .venv/bin/python scripts/ppo_campaign.py \
  --init artifacts/ppo/bc_seed0.pt \
  --out-dir artifacts/ppo/campaign_drl \
  --opponents producer,producer,oep,pgs_holdwave,pgs_bigwave,brep,greedy,rush \
  --eval-opponents producer,oep \
  --strict-drl-gate \
  --drl-profile quick
```

Campanha 4p separada:

```bash
rtk .venv/bin/python scripts/ppo_campaign.py \
  --init artifacts/ppo/bc_seed0.pt \
  --out-dir artifacts/ppo/campaign_phase5_4p \
  --training-track phase5_4p \
  --opponents producer+oep+pgs_holdwave,producer+brep+pgs_bigwave,oep+greedy+rush,pgs_holdwave+brep+pgs_bigwave \
  --strict-drl-gate \
  --pfsp \
  --drl-profile quick
```

Em `phase5_4p`, o campaign constrói a config por `build_phase5_4p_config`,
ativando um shaping potencial de margem normalizada (`0.15 -> 0.04`) alinhado
à régua local, shaping estratégico 4p e penalidade imediata de eliminação
(`elimination_penalty=0.35`). Essa penalidade existe porque a liga mostrou
PPOs phase5 morrendo em 100% das aparições 4p antes do fim global da partida;
sem esse sinal, o PPO só recebe a punição terminal tarde demais. O formato
`a+b+c` cria uma lineup 4p heterogênea e isola bots com estado por assento; isso
evita treinar contra três cópias compartilhando o mesmo runtime interno. O
benchmark leve do histórico da campanha também inclui 4p nesse track
(`eval_include_4p=true`); o gate DRL estrito continua sendo a autoridade para
promoção. Com `--pfsp`, cada relatório de gate repondera o pool do próximo chunk
para repetir adversários/lineups com winrate pareado mais perto da fronteira de
aprendizado (`35%` a `65%`), sem alterar os critérios de promoção.

O `best.pt` dessa campanha só é escrito quando algum chunk recebe
`PASS_LOCAL` no gate DRL. Se nenhum chunk passar, o resultado correto é não
promover PPO.

Warm-start por imitação forte:

```bash
rtk .venv/bin/python scripts/collect_imitation_dataset.py \
  --datasets league_strong_mix \
  --seeds 0-31 \
  --num-players 4 \
  --episode-steps 64 \
  --enable-comets \
  --launch-oversample 6 \
  --out-dir artifacts/imitation/league_strong_mix \
  --self-check

rtk .venv/bin/python -m python.train.train_bc \
  --dataset artifacts/imitation/league_strong_mix/league_strong_mix.npz \
  --arch entity \
  --epochs 60 \
  --batch-size 256 \
  --checkpoint-out artifacts/bc/bc_league_strong_mix.pt
```

`league_strong_mix` rotaciona `producer`, `pgs_holdwave`, `brep`,
`pgs_bigwave`, `oep`, `rush` e `greedy` por seed, para que 2p/4p não usem
sempre apenas os primeiros assentos do pool. `--launch-oversample` só repete
decisões não vazias no split de treino; validação e teste ficam na distribuição
natural. `--calibrate-launch` existe para diagnóstico de bias do launch head,
mas em datasets curtos pode escolher passividade extrema por CE; não usar como
evidência de promoção sem gate de liga.

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
