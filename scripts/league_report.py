"""Aggregate league match JSONs -> payoff matrix + Bradley-Terry rating
+ bootstrap CIs + veto verdict.

Pairwise outcomes: 2p games directly; 4p games decompose winner-beats-each-loser
(same winner-take-all signal the LB emits). Ties in final ships = draw (no
pairwise outcome for the tied players).

Calibration health check: Spearman between BT rating and LB_ANCHORS + hard gate
"pgs_allscripts below producer/oep/brep". Gate stat: P(bot >= producer) from
bootstrap resampling of games.

REMOVED 2026-06-10 (falsified — Spearman vs LB collapsed to ~0 by round 101):
the µ-kaggle online-rating replica and the least-squares BT->LB projection
(LB_est). Both implied predictive power over the LB that the league does not
have with this pool. The Spearman BT line stays as a HEALTH metric only.

PROMOTION RULE (since the 2026-06-10 field falsification — pgs_hold and
pgs_wave_s100 passed P(>=producer)=1.00 and landed 115-135 LB points below
producer): the league is a VETO instrument only. A candidate may be CONSIDERED
for an LB probe only if ALL hold — none is sufficient on its own:
  1. P(cand >= GATE_REFERENCE) >= 0.6           (veto floor)
  2. CI90(cand) does NOT overlap CI90(INCUMBENT) from below
  3. PURE 2p head-to-head vs INCUMBENT >= 0.5 with N >= 60 decisive 2p games
     (NOT the aggregated pairwise matrix — that one decomposes 4p games into
     pairwise credits and inflates the count; measured 2026-06-10: pgs_hold vs
     pgs_holdwave 141-123 aggregated vs 73-61 pure 2p)
Final promotion = stabilized LB score only.
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_agents import GATE_REFERENCE, INCUMBENT, LB_ANCHORS  # noqa: E402

DEFAULT_GLOBS = ("artifacts/league/v1/p*.json,artifacts/league/v1/cont/*.json,"
                 "artifacts/league/v1/waveround/*.json,artifacts/league/v1/bl3round/*.json")


def load_games(pattern):
    """pattern: one glob or several comma-separated globs."""
    import os

    games = []
    for pat in pattern.split(","):
        for f in sorted(glob.glob(pat.strip()), key=lambda p: (os.path.getmtime(p), p)):
            d = json.load(open(f))
            for g in d["games"]:
                g["mode"] = d["mode"]
                games.append(g)
    return games


def decisive_winner(g):
    """Nome do vencedor único, ou None se empate/aniquilação total.

    Derivado dos final_ships, NUNCA do campo "winner": artefatos antigos
    carregam winner falso de argmax mesmo em empate; os novos (2026-06-10)
    marcam "tie": true e winner=None / winner_seat=-1 — tudo tolerado aqui.

    2026-06-11 (semântica Kaggle): assento com "agent_status" ERROR/TIMEOUT/
    INVALID recebe reward=None no Kaggle e NÃO pode vencer, mesmo com max
    ships — sai do argmax. Chave ausente (jogos antigos) = todos elegíveis.
    """
    if g.get("tie"):
        return None
    statuses = g.get("agent_status") or ["DONE"] * len(g["seats"])
    eligible = [(s, sh) for s, st, sh in
                zip(g["seats"], statuses, g["final_ships"], strict=False)
                if st not in ("ERROR", "TIMEOUT", "INVALID")]
    if not eligible:
        return None
    top = max(sh for _, sh in eligible)
    winners = [s for s, sh in eligible if sh == top]
    return winners[0] if (len(winners) == 1 and top > 0) else None


def pairwise_outcomes(games):
    """[(winner, loser), ...] with draws skipped."""
    out = []
    for g in games:
        w = decisive_winner(g)
        if w is None:
            continue  # draw/tie (or total wipe) — no signal
        out.append((g, w, [s for s in g["seats"] if s != w]))
    return out


def h2h_2p(games, incumbent):
    """H2H PURO vs o incumbente: só jogos 2p decisivos. {name: (wins, losses)}.

    2026-06-10: a coluna antiga vinha da matriz pairwise AGREGADA, que decompõe
    4p em créditos par-a-par e inflava o N (pgs_hold 141-123 agregado vs 73-61
    puro). A condição 3 da regra de promoção exige o número 2p puro.
    """
    h2h = defaultdict(lambda: [0, 0])
    for g in games:
        seats = g["seats"]
        if len(seats) != 2 or incumbent not in seats:
            continue
        w = decisive_winner(g)
        if w is None:
            continue  # empate/aniquilação — não conta como jogo decisivo
        other = seats[1] if seats[0] == incumbent else seats[0]
        if other == incumbent:
            continue  # espelho incumbente vs incumbente — sem sinal
        h2h[other][0 if w == other else 1] += 1
    return {n: tuple(v) for n, v in h2h.items()}


def aggregate_faults(games):
    """Soma o campo "faults" por agente -> {name: {crashes,timeouts,invalid_moves}}.

    Chave AUSENTE = jogo pré-instrumentação (2026-06-10), NÃO auditado — não é
    "sem fault" (use fault_audit p/ a cobertura). Presente-e-vazia = auditado
    limpo. Agente com crash > 0 passa turnos e corrompe H2H/BT silenciosamente
    — por isso o ⛔ no veredito.
    """
    agg = defaultdict(lambda: {"crashes": 0, "timeouts": 0, "invalid_moves": 0})
    for g in games:
        for name, f in (g.get("faults") or {}).items():
            for k in ("crashes", "timeouts", "invalid_moves"):
                agg[name][k] += int(f.get(k, 0))
    return {n: f for n, f in agg.items() if any(f.values())}


# Calibração mínima p/ o BT voltar a valer como ORDEM (regra 2026-06-10, após a
# falsificação id=159): spearman vs LB >= +0.6 — e sustentado por >=10 rounds,
# condição longitudinal que este report (stateless) NÃO verifica sozinho.
PREDICTIVE_SPEARMAN_MIN = 0.6
MIN_ANCHORS = 5
# Ruído de resubmissão Kaggle: config idêntica do holdwave variou ~±60
# (refs 53537753 vs 53542884) — gaps de LB menores que isso não separam nada.
LB_NOISE = 60.0


def lb_inversions(bt, anchors, noise=LB_NOISE):
    """Pares ancorados onde a ordem BT INVERTE a ordem do LB real.

    Só conta inversão com gap de LB acima do ruído de resubmissão (~±60):
    é a evidência par-a-par da falsificação do BT como preditor (id=159) —
    um spearman ~0 global não diz QUEM está trocado. Retorna
    [(acima_no_bt, acima_no_lb, d_bt, d_lb), ...] por gap de LB decrescente.
    """
    names = [n for n in anchors if n in bt]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d_bt = bt[a] - bt[b]
            d_lb = anchors[a] - anchors[b]
            if abs(d_lb) <= noise or d_bt == 0:
                continue
            if (d_bt > 0) != (d_lb > 0):
                hi_bt, hi_lb = (a, b) if d_bt > 0 else (b, a)
                out.append((hi_bt, hi_lb, abs(d_bt), abs(d_lb)))
    return sorted(out, key=lambda t: -t[3])


def fault_audit(games):
    """Cobertura da auditoria de faults: {"audited": n, "unaudited": n}.

    Jogos SEM a chave "faults" foram gravados antes da instrumentação —
    crash/timeout/invalid são INVISÍVEIS neles; ausência != limpo. O
    league_match atual sempre grava a chave ({} quando limpo)."""
    audited = sum(1 for g in games if "faults" in g)
    return {"audited": audited, "unaudited": len(games) - audited}


def bt_from_outcomes(outcomes, names, iters=300):
    wins = defaultdict(lambda: defaultdict(float))
    for _, w, losers in outcomes:
        for loser in losers:
            wins[w][loser] += 1.0
    s = {n: 1.0 for n in names}
    for _ in range(iters):
        new = {}
        for i in names:
            num = sum(wins[i][j] for j in names if j != i)
            den = 0.0
            for j in names:
                if j == i:
                    continue
                nij = wins[i][j] + wins[j][i]
                if nij:
                    den += nij / (s[i] + s[j])
            new[i] = num / den if den > 0 else s[i]
        z = sum(new.values()) / len(new)
        s = {k: max(v / z, 1e-9) for k, v in new.items()}
    return {k: 400.0 * math.log10(v) + 1000.0 for k, v in s.items()}, wins


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry, strict=False)) / (n * (n * n - 1))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOBS
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    games = load_games(pattern)
    outcomes = pairwise_outcomes(games)
    names = sorted({s for g in games for s in g["seats"]})
    bt, wins = bt_from_outcomes(outcomes, names)

    # bootstrap over games -> rating CI + P(bot >= producer)
    rng = random.Random(7)
    boots = []
    for _ in range(n_boot):
        sample = [outcomes[rng.randrange(len(outcomes))] for _ in range(len(outcomes))]
        b, _ = bt_from_outcomes(sample, names, iters=150)
        boots.append(b)
    ci = {}
    p_ge_prod = {}
    gate_ref_available = GATE_REFERENCE in bt
    for n in names:
        vals = sorted(b[n] for b in boots)
        ci[n] = (vals[int(0.05 * n_boot)], vals[int(0.95 * n_boot)])
        if gate_ref_available:
            p_ge_prod[n] = sum(1 for b in boots if b[n] >= b[GATE_REFERENCE]) / n_boot
        else:
            p_ge_prod[n] = None

    # per-mode breakdown
    stats = {n: {"2p": [0, 0], "4p": [0, 0], "annih": [0, 0]} for n in names}
    for g in games:
        w = decisive_winner(g)
        for seat, nm in enumerate(g["seats"]):
            st = stats[nm]
            st[g["mode"]][1] += 1
            if nm == w:
                st[g["mode"]][0] += 1
            st["annih"][1] += 1
            if g["died_at"][seat] is not None:
                st["annih"][0] += 1

    ranking = sorted(names, key=lambda n: -bt[n])
    anchored = [n for n in ranking if n in LB_ANCHORS]
    rho = spearman([bt[n] for n in anchored], [LB_ANCHORS[n] for n in anchored])
    inversions = lb_inversions(bt, LB_ANCHORS)
    bt_predictive = (len(anchored) >= MIN_ANCHORS
                     and not math.isnan(rho) and rho >= PREDICTIVE_SPEARMAN_MIN)
    hard_gate = ("pgs_allscripts" in bt and
                 all(bt["pgs_allscripts"] < bt[n] for n in ("producer", "oep", "brep") if n in bt))

    # H2H PURO 2p vs o incumbente (regra de promoção, cond. 3). A matriz
    # agregada (wins) decompõe 4p em créditos pairwise — serve para o BT, mas
    # NÃO para H2H (2026-06-10: 141-123 agregado vs 73-61 puro no mesmo par).
    h2h = h2h_2p(games, INCUMBENT) if INCUMBENT in bt else {}
    faults = aggregate_faults(games)
    audit = fault_audit(games)

    print("=" * 78)
    print("LIGA = VETO-ONLY: BT NÃO promove (Spearman vs LB ~0; promoção exige a "
          "regra de 3 condições + sonda LB)")
    print("=" * 78)
    print(f"games: { {m: sum(1 for g in games if g['mode']==m) for m in ('2p','4p')} } "
          f"(decisive pairwise: {len(outcomes)}; bootstrap n={n_boot})")
    print(f"\n{'bot':16s} {'BT':>5s} {'CI90':>12s} {'LB':>5s} {'P>=ref':>7s} "
          f"{'H2H2p':>9s} {'win2p':>6s} {'win4p':>6s} {'annih':>6s}")
    for n in ranking:
        lo, hi = ci[n]
        lb = f"{LB_ANCHORS[n]:5.0f}" if n in LB_ANCHORS else "    -"
        s = stats[n]
        w2 = f"{s['2p'][0]/s['2p'][1]:.2f}" if s["2p"][1] else "  -"
        w4 = f"{s['4p'][0]/s['4p'][1]:.2f}" if s["4p"][1] else "  -"
        an = f"{s['annih'][0]/s['annih'][1]:.2f}"
        if n == INCUMBENT:
            hh = "     INC."
        elif n in h2h and sum(h2h[n]):
            w, loss = h2h[n]
            hh = f"{w:3.0f}-{loss:3.0f}".rjust(9)
        else:
            hh = "        -"
        p_ref = f"{p_ge_prod[n]:7.2f}" if p_ge_prod[n] is not None else "    n/a"
        print(f"{n:16s} {bt[n]:5.0f} [{lo:4.0f},{hi:4.0f}] {lb} "
              f"{p_ref} {hh} {w2:>6s} {w4:>6s} {an:>6s}")
    if faults:
        print("\nFAULTS (somados nos jogos AUDITADOS):")
        for n in sorted(faults):
            f = faults[n]
            print(f"  {n:16s} crashes={f['crashes']} timeouts={f['timeouts']} "
                  f"invalid_moves={f['invalid_moves']}")
    if audit["unaudited"]:
        print(f"\n⚠ auditoria de faults: {audit['audited']}/{len(games)} jogos auditados — "
              f"{audit['unaudited']} pré-instrumentação SEM a chave 'faults' "
              f"(crash/timeout/invalid INVISÍVEIS nesses jogos; ausência != limpo)")

    print(f"\ncalibration: spearman BT = {rho:+.3f} | "
          f"hard gate (allscripts < cluster): {'PASS' if hard_gate else 'FAIL'}")
    if not bt_predictive:
        print(f"⚠ BT NÃO-PREDITIVO do LB (exige spearman >= {PREDICTIVE_SPEARMAN_MIN:+.1f} "
              f"com >= {MIN_ANCHORS} âncoras, sustentado >= 10 rounds): a ordem da tabela "
              f"é INTERNA da pool — não usar como ranking de submissão")
        for a, b, dbt, dlb in inversions:
            print(f"  inversão vs LB: BT põe {a} +{dbt:.0f} acima de {b}, mas no campo "
                  f"{b} está +{dlb:.0f} acima (> ruído ±{LB_NOISE:.0f})")
    if not gate_ref_available:
        print(f"⚠ referência de veto ausente no corpus atual: {GATE_REFERENCE!r}. "
              "P>=ref fica NaN até essa âncora entrar nos jogos.")
    print(f"VETO floor (ref={GATE_REFERENCE}): reprovado se P(bot >= ref) < 0.6")
    print(f"PROMOÇÃO (regra 2026-06-10, liga NÃO promove sozinha): probe de LB só se "
          f"CI90 não sobrepõe o incumbente ({INCUMBENT}) por baixo E H2H PURO 2p "
          f"(coluna H2H2p) >= 0.5 com N >= 60 decisivos; "
          f"promoção final = score de LB estabilizado")
    for n in sorted(faults):
        if faults[n]["crashes"] > 0:
            print(f"⛔ {n}: {faults[n]['crashes']} crashes — resultados não confiáveis "
                  f"(passa turnos: corrompe H2H/BT)")

    out = {
        "bt": bt, "ci90": ci, "p_ge_producer": p_ge_prod,
        "incumbent": INCUMBENT,
        # 2026-06-10: chave renomeada — h2h_incumbent_2p é o número da regra de
        # promoção (puro 2p); o agregado segue derivável da matriz "wins".
        "h2h_incumbent_2p": {n: list(v) for n, v in h2h.items()},
        "faults": faults,
        "fault_audit": audit,
        # ranking = ordem BT INTERNA da pool; bt_predictive diz se ela pode ser
        # lida como ordem de LB (falsificação id=159: hoje NÃO pode)
        "ranking": ranking, "spearman_lb": rho, "hard_gate": hard_gate,
        "bt_predictive": bt_predictive,
        "lb_inversions": [[a, b, round(dbt), round(dlb)] for a, b, dbt, dlb in inversions],
        "per_mode": stats,
        "wins": {r: dict(wins[r]) for r in names},
    }
    out_path = Path("artifacts/league/v1/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
