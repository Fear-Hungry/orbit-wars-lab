# Submissão Kaggle

A **regra e a justificativa** da submissão sem Rust são donas de D10/D11 em
[`DECISIONS.md`](DECISIONS.md). Esta página é o *operacional*: estrutura, checklist
e estado do ciclo.

## Estrutura final

```python
def agent(obs):
    features = encode(obs)
    action = policy_forward(features)
    moves = decode(action, obs)
    return validate_moves(moves, obs)
```

Falhas devem ser atribuíveis no teste local. **Não degrade silenciosamente** para
outra política (`fallback_greedy`, Producer, hold por timeout ou qualquer
substituto) quando o caminho principal quebrar. Se a submissão precisar de guarda
defensiva por causa do ambiente Kaggle, ela deve ser instrumentada e o gate local
deve falhar com `fallback_rate > 0`.

## Checklist

- roda contra si mesmo em 2p e 4p;
- não importa pacotes ausentes;
- não escreve arquivos;
- não usa rede externa;
- não excede tempo;
- sempre retorna lista;
- não usa fallback silencioso;
- ação nunca contém nave negativa, NaN ou origem inválida;
- não importa `orbit_wars_core` / `orbit_wars_py` (D11).

## Estado do ciclo (snapshot)

> Instantâneo manual; o registro vivo de experimentos fica em [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

- Data: 2026-06-02 · Commit `c4b9467` · Kaggle ref `53290356`
- Arquivo: `artifacts/submission.py`
- Validação local (`make gate-check-final`: passou), final 20 seeds:
  - `win_rate_2p_mean`: `0.975`
  - `mean_score_margin`: `0.932`
  - `worst_decile_score_margin`: `0.324`
  - 4p: `0.90`

**Caminho ativo:** OEP/Producer. Um candidato só promove se bater a margem contra
o Producer local pela régua decisora (ver [`PLAYBOOK.md`](PLAYBOOK.md)). PPO/self-play
está deferido (ver [`TRAINING.md`](TRAINING.md)).
