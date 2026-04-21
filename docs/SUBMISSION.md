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
