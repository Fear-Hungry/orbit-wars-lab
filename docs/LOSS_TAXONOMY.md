# Loss taxonomy — onde estão os pontos (G1.2)

> **Dona deste tópico.** Mineração das derrotas REAIS do leaderboard (não liga
> local) das submissões Producer/OEP. Decide ONDE perdemos; daqui pra frente
> todo experiment cita qual classe está atacando.

Gerado em 2026-06-14. Força da evidência: **parcial** — a *fonte* é forte (382
replays reais do campo Kaggle), o *classificador* é regra-based com limiares
ajustados à mão (auditável linha a linha no CSV, não aprendido).

## Dados / como reproduzir

Replays reais via `EpisodeService` (descoberto empiricamente):

- Lista de episódios + resultado: `POST .../api/i/competitions.EpisodeService/ListEpisodes`
  body `{"submissionId": <id>}` (auth básica kaggle.json). `reward` ±1 = vitória/derrota
  do ponto de vista do agente; `len(agents)` = 2p/4p; `index` = nosso slot.
- Replay completo (sem auth): `GET https://www.kaggleusercontent.com/episodes/<id>.json`
  — as rotas `GetEpisodeReplay` clássicas dão 400/404 nesta competição; só a CDN funciona.

Pipeline (`scripts/replay_mining/`):

```bash
.venv/bin/python scripts/replay_mining/mine.py collect    # ListEpisodes + baixa derrotas
.venv/bin/python scripts/replay_mining/mine.py taxonomy   # parseia + classifica + CSV/exemplares
```

Saídas em `artifacts/replay_mining/`: `episode_index.json` (660 episódios),
`loss_taxonomy.csv` (382 derrotas, 1 linha/feature por replay),
`exemplars/<classe>/` (≤10 replays por classe), `loss_taxonomy_summary.md`.

Submissões mineradas (as que jogaram o campo real): Producer `53366194`,
OEP `53433131`, OEP `53582886`. **Score-proxy** por step = naves controladas
(guarnições + frotas em trânsito); estado é full-observability, então leio a obs
mais completa de cada step (visão congelada de eliminado não contamina a série).

## Achado-cabeçalho

| formato | derrotas | jogos | taxa de derrota | % dos jogos |
|---------|----------|-------|-----------------|-------------|
| 2p | 136 | 290 | **46.9%** | 44% |
| 4p | 246 | 370 | **66.5%** | 56% |

**Perdemos 2 em cada 3 jogos 4p, e 4p é a maioria do campo.** 64% das nossas
derrotas (246/382) são em 4p. Confirma e quantifica `field_is_majority_4p`: 4p é
a alavanca primária.

## Taxonomia (382 derrotas reais)

| # | classe | derrotas | % | 2p | 4p |
|---|--------|----------|---|----|----|
| 1 | Abertura ruim (atrás cedo, nunca recupera) | 129 | 33.8% | 5 | 124 |
| 2 | Jogo parelho perdido na atrição do fim | 126 | 33.0% | 125 | 1 |
| 3 | 4p kingmaker / engolido após liderar | 66 | 17.3% | 0 | 66 |
| 4 | Overextension (picou e colapsou) | 55 | 14.4% | 2 | 53 |
| 5 | Redistribuição tardia (reserva ociosa) | 2 | 0.5% | 1 | 1 |
| 6 | Comet | 2 | 0.5% | 2 | 0 |
| 7 | Não-classificado | 2 | 0.5% | 1 | 1 |
| — | Sem defesa / nunca recaptura | 0 | 0% | 0 | 0 |
| — | Timeout / crash | 0 | 0% | 0 | 0 |

(As duas últimas existem como flags mas nunca são primárias: casos de erosão de
planetas caem em abertura-ruim/atrição, que têm prioridade maior.)

(Critérios de classe: ver `scripts/replay_mining/parse.py::classify_loss`. Limiares
relativos à fração justa `share×n_players` para comparar 2p e 4p. Cada derrota
recebe UMA classe primária por prioridade + todas as flags disparadas no CSV.)

## Top-3 padrões (o que cada um ataca)

### 1. Abertura ruim em 4p — 124 derrotas (perdemos a corrida de expansão)
Por volta do step 60 já estamos na metade de baixo (mid_rank 3–4) e somos
eliminados cedo. Ex.: `78983963` (share 25%→4% até step 65, elim step 63).
**Ataca:** disputa de expansão/abertura em 4p — captura neutra agressiva e segura
nos primeiros ~60 steps, não deixar dois vizinhos crescerem sem contestação.

### 2. Atrição parelha em 2p — 125 derrotas (perdemos o fim de jogo apertado)
Começamos parelhos (~50%), trocamos planetas a vida toda (recapturas ≈ perdas) e
desmoronamos no fim. Ex.: `78997920` (50%→55%→40%→13%→1%, elim step 134).
**Ataca:** decisividade no end-game 2p — converter paridade em vantagem (timing de
onda grande, hoard→golpe) em vez de empatar a troca até perder o último embate.

### 3. 4p kingmaker / engolido — 66 derrotas (lideramos e fomos derrubados)
Estávamos em 1º/2º no step 60 (mid_rank ≤ 2, muitas vezes acima da fração justa)
e caímos para último/eliminados. Ex.: `78998339` (líder 32% no step 60 → 0% no
step 120). **Ataca:** sobrevivência 4p quando viramos alvo — não expor a liderança,
ler coalizão/foco adversário, evitar virar o "rei" que os outros derrubam.

> Overextension (4) é primo do #3: lideramos forte (rel ≥ 1.5) e colapsamos. Junto
> com kingmaker, **119 das 246 derrotas 4p são "tínhamos posição e a perdemos"** —
> o outro tanto (124) é "nunca tivemos posição". 4p exige tanto abertura quanto
> defesa de vantagem.

## Achados negativos (descartados como driver)

- **Comet: 2/382 (0.5%). Timeout/crash: 0/382.** Não são causa material de
  derrota — parar de gastar esforço em robustez a comet/timeout como prioridade.
  (Wrappers anti-timeout já cumpriram o papel; ver `project_kaggle_submission`.)
- "Sem defesa/recaptura" puro quase não aparece como primário: nas derrotas de
  atrição NÓS recapturamos (perdas ≈ recapturas) — o problema não é recapturar, é
  **fechar o jogo**, não defender.

## Consistência entre submissões

Mesmo padrão em Producer e nas duas OEP (abertura + atrição dominam, kingmaker +
overextension em seguida) → é característica da **família de agentes**, não de uma
config específica. Producer `53366194`: bad_opening 69 / atrição 59 / kingmaker 40
/ overext 33. OEP `53433131`: 52 / 45 / 21 / 20. OEP `53582886` (mais 2p): atrição
22 / abertura 8 / kingmaker 5 / overext 2.

## Diretriz para Fase 2/3

Todo experimento daqui pra frente **cita a classe que ataca**:
1. `bad_opening_4p` — abertura/expansão 4p (maior balde).
2. `even_attrition_2p` — decisividade do end-game 2p.
3. `kingmaker_4p` + `overextension` — defender vantagem em 4p.
