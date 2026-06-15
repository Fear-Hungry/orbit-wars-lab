# Selector baseline — `selector_baseline_2026_06_12`

Registro rastreável do baseline congelado antes de qualquer mudança na régua
(etapa 1 do plano do seletor). O artefato em si vive em
`artifacts/league/baselines/selector_baseline_2026_06_12/` (gitignored, por isso
este espelho do manifest).

**Por quê**: depois de mudar o scoring, "melhorou a predição" só é testável
re-pontuando OS MESMOS jogos sob a régua nova — sem baseline, toda mudança de
score é indistinguível de mudança de régua.

## Manifest (espelho)

- **Fonte**: `orbit-wars-lab/artifacts/league/submit_ruler/background_strict_v11_hard_timeout/task_results.json`
  (sha1 `ver manifest.json`), 36 tasks, todos `returncode=0`, jogos copiados para `games/`.
- **Git HEAD no freeze**: `5c775b3` (worktree dirty — frente pgs_v2 em andamento; o
  baseline congela os DADOS, não o código).
- **Candidatos (âncoras)**: `pgs_hold`, `pgs_holdwave`, `pgs_wave_s100`.
- **Settings**: seeds=48, steps=500, seed_base=5433500, chunk_size=8, perfil strong-like.
- **Incumbente**: `pgs_holdwave`; gate reference: `producer`.
- **LB_ANCHORS no dia** (espelho de `scripts/league_agents.py`): producer 1173.1,
  oep 1182.7, brep 1156.1, pgs_allscripts 1021.5, pgs_holdwave 1228.8,
  pgs_hold 1057.6, pgs_wave_s100 1146.1 (ruído de resubmit ~±60).

## Regra de uso

- O diretório do baseline é IMUTÁVEL (o freeze recusa overwrite sem `--force`).
- Qualquer mudança de scoring da régua deve ser validada re-pontuando estes jogos
  e comparando a predição (Spearman vs LB_ANCHORS do manifest) antes/depois.
- Reuso dos jogos em calibrações: assinatura `(mode, names, seeds, seed_base=5433500, steps)`.
