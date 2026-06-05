# Submissão Kaggle

## Regra

A submissão final não deve depender do motor Rust. O motor Rust serve para treino local.

## Estrutura final

```python
def agent(obs):
    features = encode(obs)
    action = policy_forward(features)
    moves = decode(action, obs)
    return validate_moves(moves, obs)
```

Falhas devem ser atribuiveis no teste local. Nao degrade silenciosamente para outra politica
(`fallback_greedy`, Producer, hold por timeout ou qualquer substituto) quando o caminho principal
quebrar. Se a submissao precisar de guarda defensiva por causa do ambiente Kaggle, ela deve ser
instrumentada e o gate local deve falhar com `fallback_rate > 0`.

## Checklist

- roda contra si mesmo em 2p e 4p;
- não importa pacotes ausentes;
- não escreve arquivos;
- não usa rede externa;
- não excede tempo;
- sempre retorna lista;
- não usa fallback silencioso;
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

- Caminho ativo: OEP/Producer. Um candidato só promove se bater a margem contra o Producer local
  pela régua decisora.
- PPO/self-play fica deferido ate o OEP bater o Producer ou esgotar o caminho OEP. Quando ativado,
  treinar contra Producer/heuristico e exigir checkpoint com win/margem contra Producer >= baseline
  OEP, sem crash/timeout/fallback.
