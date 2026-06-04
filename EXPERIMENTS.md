# Experimentos

Registro curto para comparar ideias de heurística, decoder, modelo e liga.

## Como registrar

Use uma linha por hipótese testada. Mudanças no agente só podem ser commitadas depois de registrar
o antes/depois contra a régua honesta: `submission_v_old.py` + heurísticos principais.

```text
YYYY-MM-DD | ideia | comando | antes | depois | resultado | decisão
```

Métricas úteis:

- `2p_win_rate` por oponente;
- `4p_win_rate`;
- `mean_score_margin`;
- `invalid_action_rate`;
- `timeout_rate`;
- `crash_rate`;
- pior seed ou lineup ruim.

## Baseline atual

Artefato: `artifacts/submission.py`

Régua honesta: `artifacts/honest_benchmark.json`

Resumo da régua honesta conhecida:

- 2p vs `submission_v_old`: `0.46875`, margem `-0.08788`;
- 2p vs `greedy`: `0.90625`, margem `0.83750`;
- 2p vs `rush`: `0.93750`, margem `0.91085`;
- 4p misto: `0.68750`, margem `0.37500`;
- crashes/timeouts/ações inválidas: `0.0`.

Regra Kaggle: não julgar submissão pelo score imediato. Aguardar cerca de 1 hora para o score
estabilizar antes de concluir se uma mudança melhorou ou piorou.

## Resultados recentes

```text
2026-06-03 | régua honesta da heurística atual | uv run --extra dev python -m scripts.benchmark_submission --submission artifacts/submission.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --out artifacts/honest_benchmark.json | baseline anterior não registrado nessa régua | vs old=0.46875, greedy=0.90625, rush=0.93750, 4p=0.68750 | atual não supera a versão antiga no self-play histórico | usar como bloqueio antes de novos commits de agente
2026-06-03 | PPO cand1 exportado | uv run --extra dev python -m scripts.benchmark_ppo_submission --checkpoint artifacts/ppo/phase0_targeted_seed21_lr1e4_16384.pt --submission-out artifacts/ppo/cand1_phase3_submission.py --out artifacts/ppo/cand1_phase3_4seed_benchmark.json --seeds 4 --opponents artifacts/submission_v_old.py greedy rush | heurística: old=0.46875, greedy=0.90625, rush=0.93750 | PPO: old=0.125, greedy=0.125, rush=0.500, 4p=0.000 | PPO exportado pior e mais lento | não submeter
2026-06-03 | habilitar hammer plans também em 2p | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_2p_hammer.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/hammer_2p_honest_16seed.json | old=0.46875, greedy=0.90625, rush=0.93750, 4p=0.68750 | old=0.56250, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | melhora coordenação sem regressão na régua honesta | aceitar e exportar submissão
2026-06-03 | filtrar neutros inseguros quando existe neutral seguro na abertura 0-10 | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_safe_opening_filter.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/safe_opening_filter_16seed.json | old=0.56250, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | melhora abertura sem regressão na régua honesta | aceitar e exportar submissão
2026-06-03 | bonus conservador para neutros seguros na abertura adaptativa 15-80 | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_adaptive_opening_score.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/adaptive_opening_score_16seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | implementa scoring da fase 15-80 sem regressão local | aceitar como mudança estrutural
2026-06-03 | reserva 5 apenas na abertura/adaptativa 2p sem ameaça | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_opening_reserve5.py --opponents artifacts/submission_v_old.py greedy rush --seeds 4 --episode-steps 500 --jobs 8 --out artifacts/opening_reserve5_4seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | 4 seeds: old=1.00000, greedy=1.00000, rush=0.75000, 4p=0.75000; crash/timeout/invalid=0 | regressão rápida contra rush; reserva menor ainda abre failure mode | rejeitar
2026-06-03 | bonus de hammer contra inimigo overextended | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_overextended_hammer.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/overextended_hammer_16seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | transforma enemy_overextended em alvo de coordenação sem regressão local | aceitar como mudança estrutural
2026-06-03 | abertura 0-10 estrita: neutro só se for seguro | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_strict_safe_opening.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/strict_safe_opening_16seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.59375, greedy=0.87500, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | fiel ao diagnóstico, mas perde produção em seeds sem neutro seguro inicial | rejeitar
2026-06-03 | reserva zero somente em TOTAL_WAR 2p sem ameaça chegando | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_total_war_reserve0.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/total_war_reserve0_16seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | implementa parte segura da reserva adaptativa de TOTAL_WAR sem regressão local | aceitar como mudança estrutural
2026-06-04 | penalidade depth-2 suave para neutro recapturável | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_neutral_recapture_penalty.py --opponents artifacts/submission_v_old.py greedy rush --seeds 4 --episode-steps 500 --jobs 8 --out artifacts/neutral_recapture_penalty_4seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | 4 seeds: old=0.75000, greedy=1.00000, rush=0.75000, 4p=0.75000; crash/timeout/invalid=0 | penalidade atrapalha corrida de abertura contra old/rush | rejeitar
2026-06-04 | throttle reduzido em fases urgentes/oportunistas | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_adaptive_throttle_2p.py --opponents artifacts/submission_v_old.py greedy rush --seeds 4 --episode-steps 500 --jobs 8 --out artifacts/adaptive_throttle_2p_4seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | 4 seeds: old=0.62500, greedy=1.00000, rush=1.00000, 4p=0.75000; crash/timeout/invalid=0 | melhora âncoras simples mas não recupera self-play histórico no smoke | rejeitar
2026-06-04 | foco orbital 10-13 quando orbital seguro existe | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_orbital_focus.py --opponents artifacts/submission_v_old.py greedy rush --seeds 4 --episode-steps 500 --jobs 8 --out artifacts/orbital_focus_4seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | 4 seeds: old=1.00000, greedy=1.00000, rush=0.87500, 4p=0.75000; crash/timeout/invalid=0 | foco orbital cria regressão rápida contra rush | rejeitar
2026-06-04 | reserva 30 no late-game 2p sem expansão nem TOTAL_WAR | rtk .venv/bin/python -m scripts.benchmark_submission --submission artifacts/submission_candidate_late_reserve30.py --opponents artifacts/submission_v_old.py greedy rush --seeds 16 --episode-steps 500 --jobs 8 --out artifacts/late_reserve30_16seed.json | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000 | old=0.62500, greedy=0.93750, rush=0.93750, 4p=0.75000; crash/timeout/invalid=0 | implementa parte segura da reserva adaptativa de late game sem regressão local | aceitar como mudança estrutural
```

## Próximas hipóteses

```text
2026-05-31 | reduzir perdas contra rush | python -m python.lab.cli quick | n/a | win_rate vs rush | pendente | testar
2026-05-31 | melhorar decisão 4p quando anti_meta+defensive aparecem juntos | python -m python.lab.cli bench-submission --seeds 8 --episode-steps 500 | n/a | 4p_win_rate | pendente | testar
```
