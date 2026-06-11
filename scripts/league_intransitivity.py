"""H5 (parte 1) — mede a INTRANSITIVIDADE da matriz de payoff da liga.

Rating escalar (Elo/BT) assume transitividade: se A>B e B>C então A>C. Quando o
jogo tem ciclos pedra-papel-tesoura, um número só não captura o meta — foi o que
enganou o s100 (liga #1, mas perdia H2H p/ hold). Balduzzi et al. 2018
("Re-evaluating Evaluation", arXiv:1806.02643) decompõe a matriz antissimétrica
de payoff em parte TRANSITIVA (gradiente de um rating) + parte CÍCLICA (rotação
sem fonte/sumidouro). A fração cíclica quantifica o quanto o ranking escalar
PERDE — e justifica trocar o rating único por uma mistura de Nash (H5).

Decomposição (combinatorial Hodge / Balduzzi):
  A[i,j] = p(i bate j) - p(j bate i)           (antissimétrica, em [-1,1])
  r[i]   = média_j A[i,j]                       (rating de mínimos quadrados)
  A_t[i,j] = r[i] - r[j]                        (componente transitiva = grad r)
  A_c    = A - A_t                              (componente cíclica = rotacional)
  intransitividade = ||A_c||_F / ||A||_F        (0 = puro rating; 1 = puro ciclo)

Uso: PYTHONPATH=. .venv/bin/python scripts/league_intransitivity.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GLOBS = ("artifacts/league/v1/p*.json,artifacts/league/v1/cont/*.json,"
                 "artifacts/league/v1/waveround/*.json,artifacts/league/v1/bl3round/*.json")


def load_pairwise(pattern):
    """wins[a][b] = # vezes que a venceu b (2p direto; 4p winner-beats-each-loser)."""
    from scripts.league_report import decisive_winner

    wins = defaultdict(lambda: defaultdict(float))
    for pat in pattern.split(","):
        for f in glob.glob(pat.strip()):
            d = json.load(open(f))
            for g in d["games"]:
                # shared rule (ties, wipes, Kaggle agent_status eligibility)
                w = decisive_winner(g)
                if w is None:
                    continue
                for s in g["seats"]:
                    if s != w:
                        wins[w][s] += 1.0
    return wins


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOBS
    # restringe ao pelotão competitivo (excluir comida ext_* infla a transitiva)
    focus = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "pgs_holdwave", "pgs_hold", "pgs_wave_s100", "pgs_wave_s50",
        "pgs_bigwave", "brep", "oep", "producer",
    ]
    wins = load_pairwise(pattern)
    names = [n for n in focus if n in wins or any(n in wins[o] for o in wins)]
    N = len(names)
    idx = {n: i for i, n in enumerate(names)}

    # matriz antissimétrica A e contagem mínima de jogos
    A = [[0.0] * N for _ in range(N)]
    npair = [[0] * N for _ in range(N)]
    for a in names:
        for b in names:
            if a == b:
                continue
            w, loss = wins[a].get(b, 0.0), wins[b].get(a, 0.0)
            n = w + loss
            npair[idx[a]][idx[b]] = int(n)
            if n > 0:
                A[idx[a]][idx[b]] = (w - loss) / n  # = 2*winrate - 1

    # rating de mínimos quadrados r[i] = média das linhas; transitiva = r_i - r_j
    r = [sum(A[i][j] for j in range(N) if j != i) / (N - 1) for i in range(N)]
    norm_A = norm_Ac = 0.0
    cyc = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            at = r[i] - r[j]
            ac = A[i][j] - at
            cyc[i][j] = ac
            norm_A += A[i][j] ** 2
            norm_Ac += ac ** 2
    intransitivity = (norm_Ac / norm_A) ** 0.5 if norm_A else 0.0

    # 3-ciclos diretos A>B>C>A (win>0.5 em cada aresta) — leitura intuitiva
    def beats(a, b):
        return A[idx[a]][idx[b]] > 0.0 and npair[idx[a]][idx[b]] >= 10
    cycles = []
    for a, b, c in combinations(names, 3):
        for x, y, z in ((a, b, c), (a, c, b)):
            if beats(x, y) and beats(y, z) and beats(z, x):
                cycles.append((x, y, z))

    print(f"agentes ({N}): {', '.join(names)}")
    print("\nrating LS (transitivo):")
    for n in sorted(names, key=lambda n: -r[idx[n]]):
        print(f"  {n:16s} r={r[idx[n]]:+.3f}")
    print(f"\n||A_ciclico|| / ||A|| = INTRANSITIVIDADE = {intransitivity:.3f}  "
          f"(0=rating puro, 1=ciclo puro)")
    print(f"3-ciclos diretos (A>B>C>A, n>=10 cada aresta): {len(cycles)}")
    for x, y, z in cycles[:12]:
        wxy = (A[idx[x]][idx[y]] + 1) / 2
        wyz = (A[idx[y]][idx[z]] + 1) / 2
        wzx = (A[idx[z]][idx[x]] + 1) / 2
        print(f"  {x} >{wxy:.2f}> {y} >{wyz:.2f}> {z} >{wzx:.2f}> {x}")
    verdict = "INTRANSITIVA (rating escalar perde sinal — H5 justificada)" if (
        intransitivity >= 0.15 or cycles) else "majoritariamente transitiva"
    print(f"\nveredito: matriz {verdict}")


if __name__ == "__main__":
    main()
