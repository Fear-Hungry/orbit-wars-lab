"""Nash averaging para avaliação de população (Balduzzi et al. 2018, "Re-evaluating
Evaluation", Zotero HJJAI6Q6).

Por que: a régua atual rankeia por win-rate média-vs-pool, que é ENVIESADA por
agentes redundantes/fracos (Spearman -0.6 local-vs-LB, ver memória
local_league_is_submission_gate). Nash averaging agrega a matriz payoff
agente-vs-agente pelo equilíbrio de Nash de máxima entropia do jogo zero-soma
simétrico definido pela matriz ANTISSIMÉTRICA de vantagem — e é invariante a
redundância: adicionar cópias de um agente fraco NÃO muda o rating dos outros.

Uso no loop: rankear um candidato vs o pool de referências pela sua vantagem
esperada contra o equilíbrio de Nash do pool (não contra a média do pool).

Matemática (jogo zero-soma simétrico, payoff antissimétrico A = -Aᵀ):
  - valor do jogo = 0;
  - p é Nash  <=>  (A p)_j <= 0  para todo j  (nenhum agente bate a mistura);
  - maxent Nash = o p de MÁXIMA ENTROPIA nesse politopo (único, estável a
    redundância — cópias dividem massa em vez de inflar vizinhos).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def winrate_to_advantage(winrate: np.ndarray) -> np.ndarray:
    """Matriz antissimétrica de vantagem a partir de win-rates pareados.

    winrate[i, j] = P(i vence j) em [0, 1]. Retorna A[i, j] = winrate[i,j] -
    winrate[j,i] em [-1, 1] (antissimétrica por construção, robusta a diagonais
    ou a winrate[i,j]+winrate[j,i] != 1)."""
    W = np.asarray(winrate, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"winrate deve ser quadrada, got {W.shape}")
    A = W - W.T
    np.fill_diagonal(A, 0.0)
    return A


def maxent_nash(advantage: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """Equilíbrio de Nash de máxima entropia do jogo zero-soma simétrico com
    payoff antissimétrico `advantage`. Retorna a distribuição p* (soma 1)."""
    A = np.asarray(advantage, dtype=float)
    n = A.shape[0]
    if n == 1:
        return np.ones(1)
    A = 0.5 * (A - A.T)  # forçar antissimetria

    def neg_entropy(p):
        q = np.clip(p, 1e-15, 1.0)
        return float(np.sum(q * np.log(q)))

    def neg_entropy_grad(p):
        q = np.clip(p, 1e-15, 1.0)
        return np.log(q) + 1.0

    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0,
         "jac": lambda p: np.ones_like(p)},
        # Nash: (A p)_j <= 0  =>  -(A p)_j >= 0
        {"type": "ineq", "fun": lambda p: -(A @ p),
         "jac": lambda p: -A},
    ]
    best = None
    # múltiplos starts: uniforme + perturbações determinísticas (sem RNG).
    starts = [np.full(n, 1.0 / n)]
    for i in range(n):
        s = np.full(n, (1.0 - 0.5) / (n - 1))
        s[i] = 0.5
        starts.append(s)
    for x0 in starts:
        res = minimize(
            neg_entropy, x0, jac=neg_entropy_grad, method="SLSQP",
            bounds=[(0.0, 1.0)] * n, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success:
            continue
        p = np.clip(res.x, 0.0, None)
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        # validar Nash (folga numérica) e escolher o de MAIOR entropia
        if np.max(A @ p) <= 1e-6:
            ent = -neg_entropy(p)
            if best is None or ent > best[1] + tol:
                best = (p, ent)
    if best is None:
        return np.full(n, 1.0 / n)
    return best[0]


def nash_rating(winrate: np.ndarray) -> np.ndarray:
    """Rating Nash-averaged de cada agente: vantagem esperada vs a mistura de
    Nash de máxima entropia da população. <=0, ~0 no suporte (não-dominado),
    <0 se dominado/redundante. Invariante a redundância."""
    A = winrate_to_advantage(winrate)
    p = maxent_nash(A)
    return A @ p


def nash_advantage_vs_pool(
    cand_vs_pool: np.ndarray, pool_vs_cand: np.ndarray, pool_winrate: np.ndarray
) -> float:
    """Vantagem Nash-averaged de UM candidato contra um pool de referências.

    Computa o equilíbrio de Nash p* SOBRE O POOL (refs vs refs) e mede a
    vantagem do candidato vs essa mistura — invariante a refs redundantes/fracas
    (o ponto: bater 5 cópias do mesmo bot fraco não infla a nota; bater a
    FRONTEIRA não-redundante, sim). >0 = candidato bate o equilíbrio do pool.

    cand_vs_pool[j] = P(candidato vence ref j); pool_vs_cand[j] = P(ref j vence
    candidato); pool_winrate[i, j] = P(ref i vence ref j)."""
    cand_vs_pool = np.asarray(cand_vs_pool, dtype=float)
    pool_vs_cand = np.asarray(pool_vs_cand, dtype=float)
    p = maxent_nash(winrate_to_advantage(pool_winrate))
    adv = cand_vs_pool - pool_vs_cand  # vantagem antissimétrica vs cada ref
    return float(adv @ p)
