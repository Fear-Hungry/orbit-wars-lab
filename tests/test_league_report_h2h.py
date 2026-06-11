"""Testes do league_report: H2H puro 2p, empates não-decisivos e faults.

Contexto (2026-06-10): a coluna H2H vs incumbente vinha da matriz pairwise
AGREGADA — que decompõe jogos 4p em créditos par-a-par e inflava o N (medido:
141-123 agregado vs 73-61 puro no mesmo par). A condição 3 da regra de
promoção exige o H2H 2p PURO; aqui validamos isso com jogos sintéticos, mais
a tolerância a empates (winner=None / winner_seat=-1 / "tie": true) e a
agregação do campo opcional "faults" (crash > 0 => resultados não confiáveis).
"""
from __future__ import annotations

import json
import sys

from scripts.league_agents import GATE_REFERENCE, INCUMBENT
from scripts.league_report import (
    aggregate_faults,
    decisive_winner,
    fault_audit,
    h2h_2p,
    lb_inversions,
    load_games,
    main,
    pairwise_outcomes,
)


def _g(seats, ships, mode="2p", **extra):
    """Jogo sintético no formato emitido por league_match.py."""
    top = max(ships)
    winners = [s for s, sh in zip(seats, ships, strict=False) if sh == top]
    g = {
        "seed": 1000,
        "seats": list(seats),
        "final_ships": [float(x) for x in ships],
        "winner_seat": ships.index(top) if top > 0 else -1,
        "winner": winners[0] if (top > 0 and len(winners) == 1) else None,
        "died_at": [None] * len(seats),
        "mode": mode,
    }
    g.update(extra)
    return g


# ---------------------------------------------------------------- H2H puro 2p

def test_h2h_2p_counts_only_2p_games():
    """Mistura 2p e 4p do mesmo par: só os 2p entram no H2H puro."""
    games = [
        _g(["cand", "inc"], [10, 0]),                                # 2p: cand
        _g(["inc", "cand"], [10, 0]),                                # 2p: inc
        _g(["cand", "inc"], [5, 1]),                                 # 2p: cand
        _g(["cand", "inc", "a", "b"], [9, 1, 0, 0], mode="4p"),      # 4p: fora
        _g(["inc", "cand", "a", "b"], [9, 1, 0, 0], mode="4p"),      # 4p: fora
        _g(["cand", "x"], [10, 0]),                                  # sem inc: fora
    ]
    assert h2h_2p(games, "inc")["cand"] == (2, 1)
    # o agregado (base do BT) decompõe os 4p e dá 3-2 — exatamente o viés que
    # a coluna H2H2p remove
    agg = [0, 0]
    for _, w, losers in pairwise_outcomes(games):
        if w == "cand" and "inc" in losers:
            agg[0] += 1
        elif w == "inc" and "cand" in losers:
            agg[1] += 1
    assert tuple(agg) == (3, 2)
    assert tuple(agg) != h2h_2p(games, "inc")["cand"]


def test_tie_games_are_not_decisive():
    """Empates (winner=None / winner_seat=-1 / tie=true) não contam decisivos."""
    tie_new = _g(["cand", "inc"], [5, 5], winner=None, winner_seat=-1, tie=True)
    # artefato antigo: empate em ships com winner falso de argmax
    tie_old_fake = _g(["cand", "inc"], [7, 7], winner="cand", winner_seat=0)
    # tie flag domina mesmo com ships distintos (ex.: corte por timeout)
    tie_flag = _g(["cand", "inc"], [5, 4], tie=True)
    wipe = _g(["cand", "inc"], [0, 0])                       # aniquilação total
    win = _g(["cand", "inc"], [3, 1])
    games = [tie_new, tie_old_fake, tie_flag, wipe, win]

    assert decisive_winner(tie_new) is None
    assert decisive_winner(tie_old_fake) is None
    assert decisive_winner(tie_flag) is None
    assert decisive_winner(wipe) is None
    assert decisive_winner(win) == "cand"
    assert h2h_2p(games, "inc")["cand"] == (1, 0)
    assert len(pairwise_outcomes(games)) == 1


def test_load_games_tolerates_none_winner(tmp_path):
    """JSONs com winner=None/winner_seat=-1 carregam e agregam sem crash."""
    payload = {"mode": "2p", "games": [
        _g(["a", "b"], [4, 4], winner=None, winner_seat=-1, tie=True),
        _g(["a", "b"], [4, 1]),
    ]}
    for g in payload["games"]:
        g.pop("mode")  # load_games injeta o mode do arquivo
    (tmp_path / "p0.json").write_text(json.dumps(payload))
    games = load_games(str(tmp_path / "*.json"))
    assert len(games) == 2 and all(g["mode"] == "2p" for g in games)
    assert len(pairwise_outcomes(games)) == 1


# ------------------------------------------------- BT como preditor de LB

def test_lb_inversions_flags_only_gaps_above_noise():
    """Par invertido com gap de LB > ruído entra; gap <= ruído fica fora."""
    bt = {"s100": 1086, "holdwave": 1057, "producer": 1028, "hold": 1063}
    anchors = {"s100": 1138.6, "holdwave": 1228.8, "producer": 1173.1,
               "hold": 1057.6}
    inv = lb_inversions(bt, anchors, noise=60.0)
    pairs = {(a, b) for a, b, _, _ in inv}
    # BT põe s100 acima do holdwave; LB diz holdwave +90 (> 60) => inversão
    assert ("s100", "holdwave") in pairs
    # BT põe hold acima do producer; LB diz producer +115 => inversão
    assert ("hold", "producer") in pairs
    # producer vs holdwave: BT e LB concordam (holdwave acima) => fora
    assert ("producer", "holdwave") not in pairs and ("holdwave", "producer") not in pairs
    # gap de LB <= ruído nunca conta, mesmo com BT invertido
    assert lb_inversions({"a": 10, "b": 20}, {"a": 1050, "b": 1000}, noise=60.0) == []
    # ordenado pelo gap de LB decrescente (producer+115 antes de holdwave+90)
    assert inv[0][3] >= inv[-1][3]


# --------------------------------------------------------------------- faults

def test_faults_aggregation_surfaces_crashes():
    games = [
        _g(["a", "b"], [1, 0],
           faults={"a": {"crashes": 2, "timeouts": 0, "invalid_moves": 1}}),
        _g(["a", "b"], [0, 1], faults={"a": {"crashes": 1}, "b": {"timeouts": 3}}),
        _g(["a", "b"], [1, 0]),                       # JSON antigo: sem a chave
        _g(["a", "z"], [1, 0], faults={"z": {"crashes": 0}}),  # tudo zero: fora
    ]
    f = aggregate_faults(games)
    assert f["a"] == {"crashes": 3, "timeouts": 0, "invalid_moves": 1}
    assert f["b"] == {"crashes": 0, "timeouts": 3, "invalid_moves": 0}
    assert "z" not in f and "c" not in f


def test_errored_seat_cannot_be_decisive_winner():
    """Semântica Kaggle: ERROR/TIMEOUT => reward None => fora do argmax."""
    # agente errado com MAX ships não vence; o sobrevivente leva
    g = _g(["dead", "alive"], [9, 4], agent_status=["TIMEOUT", "DONE"])
    assert decisive_winner(g) == "alive"
    # ambos errados: sem vencedor
    g2 = _g(["dead", "dead2"], [9, 4], agent_status=["ERROR", "TIMEOUT"])
    assert decisive_winner(g2) is None
    # único elegível mas aniquilado (0 ships): não é decisivo
    g3 = _g(["dead", "wiped"], [9, 0], agent_status=["ERROR", "DONE"])
    assert decisive_winner(g3) is None
    # jogo antigo sem a chave: todos elegíveis (comportamento inalterado)
    g4 = _g(["a", "b"], [9, 4])
    assert decisive_winner(g4) == "a"


def test_fault_audit_separates_unaudited_from_clean():
    """Chave ausente = pré-instrumentação (NÃO auditado); {} = auditado limpo."""
    games = [
        _g(["a", "b"], [1, 0]),                                  # antigo: sem chave
        _g(["a", "b"], [0, 1]),                                  # antigo: sem chave
        _g(["a", "b"], [1, 0], faults={}),                       # novo: limpo
        _g(["a", "b"], [0, 1], faults={"a": {"crashes": 1}}),    # novo: com fault
    ]
    assert fault_audit(games) == {"audited": 2, "unaudited": 2}
    assert fault_audit(games[2:]) == {"audited": 2, "unaudited": 0}


# ----------------------------------------------- saída do main (veto + faults)

def test_main_prints_veto_header_and_crash_flag(tmp_path, monkeypatch, capsys):
    """End-to-end no glob sintético: header VETO-ONLY, coluna H2H2p e ⛔."""
    seats2 = [GATE_REFERENCE, INCUMBENT]
    games = (
        [_g(seats2, [3, 1]) for _ in range(2)]
        + [_g(seats2, [1, 3]) for _ in range(3)]
        + [_g(["crashy", INCUMBENT], [1, 3],
              faults={"crashy": {"crashes": 4, "timeouts": 1, "invalid_moves": 0}})]
        + [_g([GATE_REFERENCE, INCUMBENT, "crashy", "x"], [4, 3, 1, 0], mode="4p")]
    )
    for g in games:
        g.pop("mode", None)
    by_mode = {"2p": games[:6], "4p": games[6:]}
    for mode, gs in by_mode.items():
        (tmp_path / f"p_{mode}.json").write_text(json.dumps({"mode": mode, "games": gs}))
    (tmp_path / "artifacts/league/v1").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["league_report.py", str(tmp_path / "*.json"), "20"])

    main()
    out = capsys.readouterr().out
    assert "LIGA = VETO-ONLY" in out
    assert "H2H2p" in out and "H2Hinc" not in out
    assert "⛔ crashy: 4 crashes — resultados não confiáveis" in out
    assert "FAULTS" in out and "crashes=4 timeouts=1" in out
    # 6 dos 7 jogos sintéticos não têm a chave faults → pré-instrumentação
    assert "⚠ auditoria de faults: 1/7 jogos auditados — 6 pré-instrumentação" in out

    # com < 5 âncoras o BT nunca é preditivo — aviso explícito na saída
    assert "⚠ BT NÃO-PREDITIVO do LB" in out

    rep = json.loads((tmp_path / "artifacts/league/v1/report.json").read_text())
    # H2H da regra = puro 2p (2-3 nos 2p; o 4p onde producer ganha NÃO conta)
    assert rep["h2h_incumbent_2p"][GATE_REFERENCE] == [2, 3]
    assert rep["h2h_incumbent_2p"]["crashy"] == [0, 1]
    assert rep["faults"]["crashy"]["crashes"] == 4
    assert rep["fault_audit"] == {"audited": 1, "unaudited": 6}
    assert rep["bt_predictive"] is False
    assert isinstance(rep["lb_inversions"], list)


def test_main_tolerates_missing_gate_reference_in_fresh_corpus(tmp_path, monkeypatch, capsys):
    """Round inicial pode ainda não ter o Producer; report não deve quebrar."""
    games = [
        _g(["a", "b"], [3, 1]),
        _g(["b", "a"], [2, 0]),
    ]
    for g in games:
        g.pop("mode", None)
    path = tmp_path / "fresh.json"
    path.write_text(json.dumps({"mode": "2p", "games": games}))
    (tmp_path / "artifacts/league/v1").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["league_report.py", str(path), "5"])

    main()

    out = capsys.readouterr().out
    assert "referência de veto ausente" in out
    rep = json.loads((tmp_path / "artifacts/league/v1/report.json").read_text())
    assert set(rep["p_ge_producer"]) == {"a", "b"}
    assert all(value is None for value in rep["p_ge_producer"].values())
