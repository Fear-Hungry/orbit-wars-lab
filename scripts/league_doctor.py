"""Executable health check for the local Orbit Wars league.

The league is useful only if it fails loudly on the same classes of problems
that break or distort Kaggle submissions: stale tarballs, shared module state,
crashes, timeouts, invalid actions, fake tie winners and unaudited historical
results. This script runs those canaries plus a short real-bot smoke match and
prints a machine-readable report.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import league_agents, league_match, league_report  # noqa: E402


def _check(name: str, passed: bool, details: dict[str, Any] | None = None,
           *, severity: str = "fail") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "details": details or {},
    }


def _planet_field(planet: Any, index: int, key: str, default: float = 0.0) -> float:
    if isinstance(planet, dict):
        return float(planet.get(key, default))
    return float(planet[index])


def _fake_make(sleep_s: float = 0.02) -> Callable[[str], Callable[[dict[str, Any]], list]]:
    def make(name: str):
        if name == "crasher":
            def crasher(_obs):
                raise RuntimeError("league doctor crash canary")
            return crasher

        if name == "badmove":
            def badmove(_obs):
                return [[1.0], [float("nan"), 0.0, 1.0], [1.0, 0.0, 1.0, 9.9]]
            return badmove

        if name == "overbudget":
            def overbudget(obs):
                player = int(obs.get("player", 0))
                owned = [
                    p for p in obs.get("planets", [])
                    if int(_planet_field(p, 1, "owner", -1)) == player
                ]
                if not owned:
                    return []
                src = owned[0]
                ships = int(_planet_field(src, 5, "ships", 0.0))
                if ships < 2:
                    return []
                planet_id = int(_planet_field(src, 0, "id", 0.0))
                return [[planet_id, 0.0, ships], [planet_id, 0.1, 1]]
            return overbudget

        if name == "sleeper":
            def sleeper(_obs):
                time.sleep(sleep_s)
                return []
            return sleeper

        return lambda _obs: []

    return make


def _fault_canaries() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    old_make = league_match.make
    old_timeout = league_match.ACT_TIMEOUT_S
    old_bank = league_match.OVERAGE_BANK_S
    old_totals = league_match._totals
    try:
        checks.append(_check(
            "league_constants_match_kaggle_contract",
            league_match.ACT_TIMEOUT_S == 1.0 and league_match.OVERAGE_BANK_S == 12.0,
            {"act_timeout_s": league_match.ACT_TIMEOUT_S, "overage_bank_s": league_match.OVERAGE_BANK_S},
        ))

        league_match.make = _fake_make()

        crash = league_match.play_batch(["crasher", "passer"], [9010], 4, {}, {})[0]
        checks.append(_check(
            "crash_becomes_error_and_cannot_win",
            crash["agent_status"] == ["ERROR", "DONE"]
            and crash["winner"] == "passer"
            and crash["faults"].get("crasher", {}).get("crashes") == 1,
            {"game": crash},
        ))

        invalid = league_match.play_batch(["badmove", "passer"], [9011], 3, {}, {})[0]
        checks.append(_check(
            "malformed_moves_are_counted",
            invalid["agent_status"] == ["DONE", "DONE"]
            and invalid["faults"].get("badmove", {}).get("invalid_moves") == 9,
            {"faults": invalid["faults"]},
        ))

        overbudget = league_match.play_batch(["overbudget", "passer"], [9012], 3, {}, {})[0]
        checks.append(_check(
            "aggregate_source_overbudget_is_counted",
            overbudget["agent_status"] == ["DONE", "DONE"]
            and overbudget["faults"].get("overbudget", {}).get("invalid_moves", 0) >= 1,
            {"faults": overbudget["faults"]},
        ))

        league_match.ACT_TIMEOUT_S = 0.001
        league_match.OVERAGE_BANK_S = 0.001
        timeout = league_match.play_batch(["sleeper", "passer"], [9013], 3, {}, {})[0]
        checks.append(_check(
            "timeout_exhausts_bank_and_cannot_win",
            timeout["agent_status"] == ["TIMEOUT", "DONE"]
            and timeout["winner"] == "passer"
            and timeout["faults"].get("sleeper", {}).get("timeouts") == 1,
            {"game": timeout},
        ))

        league_match.ACT_TIMEOUT_S = old_timeout
        league_match.OVERAGE_BANK_S = old_bank
        clean = league_match.play_batch(["passer", "passer"], [9014], 2, {}, {})[0]
        checks.append(_check(
            "clean_games_still_carry_audited_faults_key",
            "faults" in clean and clean["faults"] == {},
            {"game": clean},
        ))

        league_match._totals = lambda _state, num_players: [10.0] * num_players
        tie = league_match.play_batch(["passer", "passer"], [9015], 2, {}, {})[0]
        checks.append(_check(
            "ties_are_not_fake_seat0_wins",
            tie["tie"] is True and tie["winner"] is None and tie["winner_seat"] == -1,
            {"game": tie},
        ))
    finally:
        league_match.make = old_make
        league_match.ACT_TIMEOUT_S = old_timeout
        league_match.OVERAGE_BANK_S = old_bank
        league_match._totals = old_totals
    return checks


def _make_tarball(root: Path, name: str, sentinel: str) -> Path:
    src = root / f"src_{name}"
    src.mkdir()
    (src / "_weights.py").write_text(f"VALUE = {sentinel!r}\n")
    (src / "main.py").write_text(
        "def agent(obs):\n"
        "    import _weights\n"
        "    return _weights.VALUE\n"
    )
    tar_path = root / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(src.iterdir()):
            tar.add(path, arcname=path.name)
    return tar_path


def _tarball_canaries() -> list[dict[str, Any]]:
    old_root = league_agents.ROOT
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        try:
            league_agents.ROOT = root
            tar_a = _make_tarball(root, "a", "A")
            tar_b = _make_tarball(root, "b", "B")
            a = league_agents._tarball_agent(tar_a, "a")
            b = league_agents._tarball_agent(tar_b, "b")
            isolated = [a({}), b({}), a({}), b({})]
            checks.append(_check(
                "tarball_modules_are_isolated",
                isolated == ["A", "B", "A", "B"],
                {"observed": isolated},
            ))

            state_src = root / "src_stateful"
            state_src.mkdir()
            (state_src / "_state.py").write_text("COUNTER = 0\n")
            (state_src / "main.py").write_text(
                "def agent(obs):\n"
                "    import _state\n"
                "    _state.COUNTER += 1\n"
                "    return _state.COUNTER\n"
            )
            state_tar = root / "stateful.tar.gz"
            with tarfile.open(state_tar, "w:gz") as tar:
                for path in sorted(state_src.iterdir()):
                    tar.add(path, arcname=path.name)
            state_a = league_agents._tarball_agent(state_tar, "stateful")
            state_b = league_agents._tarball_agent(state_tar, "stateful")
            state_observed = [state_a({}), state_b({}), state_a({}), state_b({})]
            checks.append(_check(
                "tarball_instances_have_isolated_runtime_state",
                state_observed == [1, 1, 2, 2],
                {"observed": state_observed},
            ))

            tar_c = _make_tarball(root, "champ", "V1")
            old_agent = league_agents._tarball_agent(tar_c, "champ")
            tar_c.write_bytes(_make_tarball(root, "champ_v2", "V2").read_bytes())
            new_agent = league_agents._tarball_agent(tar_c, "champ")
            checks.append(_check(
                "tarball_reexport_invalidates_cache",
                old_agent({}) == "V1" and new_agent({}) == "V2",
                {"old": old_agent({}), "new": new_agent({})},
            ))
        finally:
            league_agents.ROOT = old_root
    return checks


def _p95_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return float(vals[max(0, int(len(vals) * 0.95) - 1)])


def _real_smoke(seeds: int, seed_base: int, steps: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cases = [
        ("real_2p_producer_vs_pgs_hold", ["producer", "pgs_hold"]),
        ("real_4p_builtin_mix", ["producer", "oep", "pgs_holdwave", "pgs_bigwave"]),
    ]
    for name, agents in cases:
        decision_ms: dict[str, list[float]] = {}
        crashes: dict[str, int] = {}
        games = league_match.play_batch(
            agents,
            list(range(seed_base, seed_base + seeds)),
            steps,
            decision_ms,
            crashes,
        )
        fault_games = [g for g in games if g.get("faults")]
        bad_status = [g for g in games if any(st != "DONE" for st in g.get("agent_status", []))]
        checks.append(_check(
            name,
            len(games) == seeds and not fault_games and not bad_status and not crashes,
            {
                "agents": agents,
                "games": len(games),
                "fault_games": len(fault_games),
                "bad_status_games": len(bad_status),
                "crashes": crashes,
                "decision_ms_p95": {k: _p95_ms(v) for k, v in decision_ms.items()},
            },
        ))
    return checks


def _existing_artifact_audit(pattern: str, strict: bool) -> list[dict[str, Any]]:
    games = league_report.load_games(pattern)
    if not games:
        return [_check("existing_artifacts_present", True, {"games": 0}, severity="warn")]
    audit = league_report.fault_audit(games)
    faults = league_report.aggregate_faults(games)
    severity = "fail" if strict else "warn"
    checks = [
        _check(
            "existing_artifacts_are_fully_audited",
            audit["unaudited"] == 0,
            {"games": len(games), **audit},
            severity=severity,
        )
    ]
    if faults:
        checks.append(_check(
            "existing_artifacts_have_no_recorded_faults",
            False,
            {"faults": faults},
            severity=severity,
        ))
    return checks


def run_diagnostics(*, smoke_seeds: int = 2, seed_base: int = 9100, smoke_steps: int = 40,
                    skip_real_smoke: bool = False, existing_glob: str = league_report.DEFAULT_GLOBS,
                    strict_existing: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(_fault_canaries())
    checks.extend(_tarball_canaries())
    if not skip_real_smoke:
        checks.extend(_real_smoke(smoke_seeds, seed_base, smoke_steps))
    checks.extend(_existing_artifact_audit(existing_glob, strict_existing))
    failed = [c for c in checks if not c["passed"] and c["severity"] == "fail"]
    warnings = [c for c in checks if not c["passed"] and c["severity"] == "warn"]
    return {
        "passed": not failed,
        "checks": checks,
        "failed": [c["name"] for c in failed],
        "warnings": [c["name"] for c in warnings],
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"league_doctor: {'PASS' if report['passed'] else 'FAIL'}")
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else check["severity"].upper()
        print(f"[{marker}] {check['name']}")
        if not check["passed"] or check["name"].startswith("real_"):
            print(json.dumps(check["details"], indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-seeds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=9100)
    parser.add_argument("--smoke-steps", type=int, default=40)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--existing-glob", default=league_report.DEFAULT_GLOBS)
    parser.add_argument("--strict-existing", action="store_true",
                        help="treat unaudited/faulted existing league artifacts as failures")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_diagnostics(
        smoke_seeds=args.smoke_seeds,
        seed_base=args.seed_base,
        smoke_steps=args.smoke_steps,
        skip_real_smoke=args.skip_real_smoke,
        existing_glob=args.existing_glob,
        strict_existing=args.strict_existing,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
