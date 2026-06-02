# Submissão Kaggle

## Regra

A submissão final não deve depender do motor Rust. O motor Rust serve para treino local.

## Estrutura final

```python
def agent(obs):
    try:
        features = encode(obs)
        action = policy_forward(features)
        moves = decode(action, obs)
        return moves
    except Exception:
        return fallback_greedy(obs)
```

## Checklist

- roda contra si mesmo em 2p e 4p;
- não importa pacotes ausentes;
- não escreve arquivos;
- não usa rede externa;
- não excede tempo;
- sempre retorna lista;
- fallback não quebra;
- ação nunca contém nave negativa, NaN ou origem inválida.

## Ciclo atual

- Data: 2026-06-02
- Commit: `c4b9467`
- Kaggle ref: `53290356`
- Arquivo: `artifacts/submission.py`
- Mensagem: `c4b9467 forward simulation comet evacuation surplus safeguards`

Validação local antes da submissão:

- `make gate-check-final`: passou
- Final 20 seeds:
  - `win_rate_2p_mean`: `0.975`
  - `mean_score_margin`: `0.9324010976130529`
  - `worst_decile_score_margin`: `0.32401097613052904`
  - 4p: `0.90`

Decisão do próximo ciclo:

- Se o public score subir forte, aprofundar forward simulation, usar este heurístico como oponente de PPO e fortalecer redistribuição 4p.
- Se o public score não mover, priorizar análise de replays/estratégias públicas e considerar self-play/PPO contra este heurístico.
