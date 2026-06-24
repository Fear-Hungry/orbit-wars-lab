"""Functional gates do plano PGS v2/v3 (testes obrigatórios por etapa).

- Etapa D: hammer multi-source captura alvo que NENHUMA fonte captura sozinha
  (este teste é a prova de que o eixo estratégico multi-source abriu).
- Etapa E: timeline pune morte intermediária e só premia captura SUSTENTADA.
- Etapa G: wave v2 descarta pending envelhecida em vez de soltar spray.
- Hoard: HOLD_SOURCE veta fonte cujo plano é só spray de ataques pequenos.
"""
from __future__ import annotations

import math

import torch
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import LaunchEntries, ensure_planet_movement


def _board(planets, *, step=0, me=0, H=40, player_count=2):
    obs = {
        "planets": planets,
        "fleets": [],
        "step": step,
        "player": me,
        "angular_velocity": 0.03,
    }
    ot = single_obs_to_tensor(obs, player_id=me)
    movement = ensure_planet_movement(
        obs_tensors=ot,
        expected_cfg=MovementConfig(
            movement_horizon=H,
            drift_epsilon=1e-3,
            track_fleets=True,
            player_count=player_count,
            max_tracked_fleets=128,
        ),
        cached_movement=None,
    )
    return ot, movement


def _entries(source: list[int], target: list[int], ships: list[float]) -> LaunchEntries:
    return LaunchEntries(
        source_slots=torch.tensor(source, dtype=torch.long),
        target_slots=torch.tensor(target, dtype=torch.long),
        ships=torch.tensor(ships, dtype=torch.float32),
        angle=torch.zeros(len(source), dtype=torch.float32),
        eta=torch.ones(len(source), dtype=torch.float32),
        valid=torch.ones(len(source), dtype=torch.bool),
    )


# planeta: [id, owner, x, y, radius, ships, production]
# tabuleiro oficial: 100x100, centro (50,50), órbitas rígidas (ω comum)
_HAMMER_PLANETS = [
    [0, 0, 70.0, 58.0, 3.0, 36.0, 1.0],   # minha fonte A
    [1, 0, 70.0, 42.0, 3.0, 36.0, 1.0],   # minha fonte B (simétrica)
    [2, 1, 80.0, 50.0, 3.0, 60.0, 1.0],   # alvo inimigo: 60 defensores
    [3, 1, 15.0, 50.0, 3.0, 100.0, 1.0],  # home inimigo forte (incapturável)
]


def _hammer_setup(H: int = 40):
    from bots.pgs.planner import EffectivePGSPolicy, PGSConfig, PGSRuntime

    ot, movement = _board(_HAMMER_PLANETS, H=H)
    status = movement.garrison_status(max_horizon=H)
    owner0 = status.owner[:, 0]
    avail = torch.zeros_like(movement.planet_ships)
    avail[0] = 35.0
    avail[1] = 35.0
    runtime = PGSRuntime(PGSConfig(mission_mode=True, enabled_missions="hammer,recapture"))
    policy = EffectivePGSPolicy(
        mode="producer-like",
        hammer_enabled=True,
        hammer_min_total_ships=50.0,
        hammer_hold_window=10,
        recapture_enabled=True,
    )
    return runtime, ot, movement, owner0, avail, policy


def test_hammer_two_sources_capture_what_neither_can_alone() -> None:
    runtime, ot, movement, owner0, avail, policy = _hammer_setup()

    # prova negativa: single-source (35 vs 60+) NÃO captura
    single = runtime._mission_recapture(
        movement=movement,
        script_movement=movement,
        obs_tensors=ot,
        owner0=owner0,
        avail=avail,
        sources=[0, 1],
        me=0,
        H=40,
        policy=policy,
    )
    assert single == [], "35 navios não podem capturar 60 defensores sozinhos"

    hammers = runtime._mission_hammer(
        movement=movement,
        script_movement=movement,
        obs_tensors=ot,
        owner0=owner0,
        avail=avail,
        sources=[0, 1],
        me=0,
        H=40,
        policy=policy,
    )
    target2 = [m for m in hammers if 2 in m.exclusive_targets]
    assert target2, "hammer multi-source deve gerar missão válida contra o alvo de 60"
    mission = target2[0]
    used = {int(s.item()) for s in mission.entries.source_slots[mission.entries.valid]}
    assert used == {0, 1}, "a captura exige as DUAS fontes"
    total = float(mission.entries.ships[mission.entries.valid].sum().item())
    assert total >= 61.0


def test_assemble_removes_producer_actions_of_hammer_sources() -> None:
    runtime, ot, movement, owner0, avail, policy = _hammer_setup()
    hammers = runtime._mission_hammer(
        movement=movement,
        script_movement=movement,
        obs_tensors=ot,
        owner0=owner0,
        avail=avail,
        sources=[0, 1],
        me=0,
        H=40,
        policy=policy,
    )
    mission = [m for m in hammers if 2 in m.exclusive_targets][0]
    # base Producer: as duas fontes do grupo lançam spray para o slot 3
    base = _entries([0, 1], [3, 3], [12.0, 9.0])
    assembled = runtime._assemble_missions(base, [mission])
    spray = assembled.valid & (assembled.target_slots == 3)
    assert not bool(spray.any()), "assemble deve remover as ações Producer das fontes usadas"


def _timeline_board(H: int = 40):
    planets = [
        [0, 0, 70.0, 50.0, 3.0, 5.0, 1.0],    # meu único planeta (fraco)
        [1, 1, 80.0, 50.0, 3.0, 200.0, 2.0],  # inimigo forte ao lado
    ]
    return _board(planets, H=H)


def test_timeline_death_penalty_dominates_margins() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(value_mode="timeline"))
    ot, movement = _timeline_board()
    aim = intercept_angle(
        movement,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([100.0], dtype=movement.dtype),
    )
    assert bool(aim["viable"][0])
    attack = LaunchEntries(
        source_slots=torch.tensor([1]),
        target_slots=torch.tensor([0]),
        ships=torch.tensor([100.0], dtype=movement.dtype),
        angle=aim["angle"].to(movement.dtype),
        eta=aim["eta"].to(movement.dtype),
        valid=torch.tensor([True]),
    )
    empty_mine = LaunchEntries(
        source_slots=torch.zeros(0, dtype=torch.long),
        target_slots=torch.zeros(0, dtype=torch.long),
        ships=torch.zeros(0, dtype=movement.dtype),
        angle=torch.zeros(0, dtype=movement.dtype),
        eta=torch.zeros(0, dtype=movement.dtype),
        valid=torch.zeros(0, dtype=torch.bool),
    )
    died = runtime._plan_value_timeline(movement, ot, empty_mine, [(1, attack)], 0)
    survived = runtime._plan_value_timeline(movement, ot, empty_mine, [], 0)
    assert died < -50_000.0, "perder todos os planetas no horizonte deve ser dominado"
    assert survived > -10_000.0
    assert survived > died


def _capture_board(H: int = 40):
    planets = [
        [0, 0, 70.0, 50.0, 3.0, 60.0, 1.0],   # minha fonte forte
        [1, 1, 80.0, 50.0, 3.0, 10.0, 3.0],   # alvo inimigo fraco, prod alta
        [2, 1, 20.0, 50.0, 3.0, 100.0, 1.0],  # home inimigo distante e forte
    ]
    return _board(planets, H=H)


def test_timeline_rewards_only_sustained_capture() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    ot, movement = _capture_board()
    aim = intercept_angle(
        movement,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([40.0], dtype=movement.dtype),
    )
    assert bool(aim["viable"][0])
    eta = float(aim["eta"][0].item())
    assert math.isfinite(eta) and eta + 10 <= 40, "captura precisa caber na janela de hold"
    capture = LaunchEntries(
        source_slots=torch.tensor([0]),
        target_slots=torch.tensor([1]),
        ships=torch.tensor([40.0], dtype=movement.dtype),
        angle=aim["angle"].to(movement.dtype),
        eta=aim["eta"].to(movement.dtype),
        valid=torch.tensor([True]),
    )
    with_bonus = PGSRuntime(PGSConfig(value_mode="timeline", timeline_capture_hold_weight=8.0))
    no_bonus = PGSRuntime(PGSConfig(value_mode="timeline", timeline_capture_hold_weight=0.0))
    v_with = with_bonus._plan_value_timeline(movement, ot, capture, [], 0)
    v_without = no_bonus._plan_value_timeline(movement, ot, capture, [], 0)
    # bônus = weight * prod(alvo) = 8 * 3 — presente SÓ quando a captura segura
    assert abs((v_with - v_without) - 8.0 * 3.0) < 1e-3


def test_wave_v2_age_discards_instead_of_spraying() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    owner0 = torch.tensor([0, 1, 1], dtype=torch.long)
    small_attack = _entries([0], [1], [20.0])  # 20 < wave_min_ships=60

    v1 = PGSRuntime(PGSConfig(wave_min_ships=60.0, wave_max_delay=8, wave_release_on_age=True))
    v1._wave_pending = {1: 90}
    released, pending_v1 = v1._wave_merge_filter(small_attack, owner0, 0, 100)
    assert bool(released.valid.any()), "v1: grupo envelhecido (age 10 >= 8) é liberado"
    assert pending_v1 == {}

    v2 = PGSRuntime(
        PGSConfig(
            wave_min_ships=60.0,
            wave_max_delay=8,
            wave_release_on_age=False,
            wave_discard_after=8,
        )
    )
    v2._wave_pending = {1: 90}
    withheld, pending_v2 = v2._wave_merge_filter(small_attack, owner0, 0, 100)
    assert not bool(withheld.valid.any()), "v2: NUNCA solta ataque pequeno por idade"
    assert pending_v2 == {1: 100}, "v2: pending descartada reinicia a janela de idade"


def test_wave_filter_does_not_mutate_runtime_state() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(wave_min_ships=60.0))
    runtime._wave_pending = {7: 33}
    owner0 = torch.tensor([0, 1], dtype=torch.long)
    runtime._wave_merge_filter(_entries([0], [1], [10.0]), owner0, 0, 40)
    assert runtime._wave_pending == {7: 33}, "commit do pending é do caller, não do filtro"


def test_hold_source_vetoes_only_pure_small_attack_spray() -> None:
    from bots.pgs.planner import PGSConfig, PGSRuntime

    runtime = PGSRuntime(PGSConfig(mission_mode=True, enabled_missions="hold_source"))
    owner0 = torch.tensor([0, 0, 0, -1, 1, 1], dtype=torch.long)
    base = _entries(
        [0, 0, 1, 2],
        [4, 5, 4, 3],
        [10.0, 12.0, 30.0, 10.0],
    )  # fonte 0: só spray pequeno; fonte 1: ataque grande; fonte 2: expansão (neutro)

    missions = runtime._mission_hold_source(base_entries=base, owner0=owner0, me=0, step_now=100)
    assert [m.name for m in missions] == ["hold_source:0"]
    assert missions[0].replace_sources == frozenset({0})

    # antes do hoard_min_step: nada
    early = runtime._mission_hold_source(base_entries=base, owner0=owner0, me=0, step_now=50)
    assert early == []
