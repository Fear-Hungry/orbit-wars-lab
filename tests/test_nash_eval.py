from __future__ import annotations

import numpy as np
from scripts.nash_eval import (
    maxent_nash,
    nash_advantage_vs_pool,
    nash_rating,
    winrate_to_advantage,
)


def test_rps_nash_is_uniform():
    # Rock-paper-scissors: antissimétrico, Nash de maxent = uniforme.
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    p = maxent_nash(A)
    assert np.allclose(p, 1 / 3, atol=1e-2), p


def test_dominant_agent_takes_all_mass():
    # Agente 0 vence 1 e 2; 1 e 2 empatam. Nash concentra no 0.
    A = np.array([[0, 1, 1], [-1, 0, 0], [-1, 0, 0]], dtype=float)
    p = maxent_nash(A)
    assert p[0] > 0.95, p


def test_redundancy_invariance_splits_mass():
    # RPS com a estratégia "S" DUPLICADA (S e S'). A propriedade-chave do Nash
    # averaging: a cópia redundante DIVIDE a massa de S (1/3 -> 1/6 + 1/6) sem
    # mexer em R e P (continuam 1/3). Win-rate média inflaria P espuriamente.
    # idx: 0=R, 1=P, 2=S, 3=S'
    A = np.array([
        [0, -1, 1, 1],
        [1, 0, -1, -1],
        [-1, 1, 0, 0],
        [-1, 1, 0, 0],
    ], dtype=float)
    p = maxent_nash(A)
    assert abs(p[0] - 1 / 3) < 0.03, p
    assert abs(p[1] - 1 / 3) < 0.03, p
    assert abs(p[2] - 1 / 6) < 0.03, p
    assert abs(p[3] - 1 / 6) < 0.03, p
    # ratings ~0 para todos (todos no suporte): nenhum é inflado pela cópia.
    r = A @ p
    assert np.allclose(r, 0.0, atol=1e-2), r


def test_winrate_to_advantage_antisymmetric():
    W = np.array([[0.5, 0.7, 0.9], [0.3, 0.5, 0.6], [0.1, 0.4, 0.5]])
    A = winrate_to_advantage(W)
    assert np.allclose(A, -A.T)
    assert np.allclose(np.diag(A), 0.0)
    assert A[0, 1] == 0.7 - 0.3


def test_nash_rating_ranks_dominant_first():
    # Agente forte (bate todos) deve ter rating >= os outros.
    W = np.array([[0.5, 0.8, 0.8], [0.2, 0.5, 0.5], [0.2, 0.5, 0.5]])
    r = nash_rating(W)
    assert r[0] >= r[1] - 1e-9 and r[0] >= r[2] - 1e-9, r


def test_nash_advantage_vs_pool_ignores_redundant_weak_refs():
    # O fix anti-overfit: um candidato que SÓ bate refs fracas redundantes mas
    # PERDE para a fronteira forte não deve pontuar positivo. Pool: 1 ref forte
    # (idx 0) + 2 cópias de uma ref fraca (idx 1,2). Nash do pool concentra na
    # forte. Candidato bate as fracas (0.9) mas perde p/ a forte (0.2).
    pool_W = np.array([
        [0.5, 0.9, 0.9],   # ref forte domina as fracas
        [0.1, 0.5, 0.5],
        [0.1, 0.5, 0.5],
    ])
    cand_vs_pool = np.array([0.2, 0.9, 0.9])   # perde p/ forte, bate fracas
    pool_vs_cand = np.array([0.8, 0.1, 0.1])
    adv = nash_advantage_vs_pool(cand_vs_pool, pool_vs_cand, pool_W)
    # Nash do pool poe ~toda massa na ref forte -> vantagem ~ (0.2-0.8) < 0.
    assert adv < 0.0, adv
    # Sanidade: média ingênua vs pool seria positiva (enganaria a régua).
    naive = float(np.mean(cand_vs_pool - pool_vs_cand))
    assert naive > 0.0, naive
