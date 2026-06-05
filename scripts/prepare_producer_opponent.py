from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

WRAPPER = '''from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_upstream():
    module_path = Path(__file__).with_name("_upstream.py")
    spec = importlib.util.spec_from_file_location("_orbit_wars_producer_upstream", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Producer upstream module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_UPSTREAM = _load_upstream()


def _planet_row(planet: Any) -> Any:
    if not isinstance(planet, dict):
        return planet
    return [
        planet["id"],
        planet["owner"],
        planet["x"],
        planet["y"],
        planet["radius"],
        planet["ships"],
        planet["production"],
    ]


def _fleet_row(fleet: Any) -> Any:
    if not isinstance(fleet, dict):
        return fleet
    return [
        fleet["id"],
        fleet["owner"],
        fleet["x"],
        fleet["y"],
        fleet["angle"],
        fleet["from_planet_id"],
        fleet["ships"],
    ]


def _to_list_observation(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return obs
    converted = dict(obs)
    converted["planets"] = [_planet_row(planet) for planet in obs.get("planets", [])]
    converted["initial_planets"] = [
        _planet_row(planet) for planet in obs.get("initial_planets", [])
    ]
    converted["fleets"] = [_fleet_row(fleet) for fleet in obs.get("fleets", [])]
    return converted


def agent(obs: Any):
    return _UPSTREAM.agent(_to_list_observation(obs))
'''


def _safe_extract(archive: Path, target: Path) -> None:
    target_root = target.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target / member.name).resolve()
            if target_root not in (member_path, *member_path.parents):
                raise RuntimeError(f"refusing unsafe archive member: {member.name}")
        tar.extractall(target, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Slawek Biel's Producer as a local benchmark opponent."
    )
    parser.add_argument("--kernel", default="slawekbiel/the-producer-agent")
    parser.add_argument("--bot-dir", type=Path, default=Path("bots/producer"))
    parser.add_argument("--orbit-lite-dir", type=Path, default=Path("orbit_lite"))
    parser.add_argument(
        "--tmp-dir", type=Path, default=Path("/tmp/orbit_wars_public/the-producer-agent-output")
    )
    args = parser.parse_args()

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", str(args.kernel), "-p", str(args.tmp_dir)],
        check=True,
    )

    archive = args.tmp_dir / "submission.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"kernel output did not include {archive}")

    extracted_dir = args.tmp_dir / "extracted"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, extracted_dir)

    main_py = extracted_dir / "main.py"
    orbit_lite = extracted_dir / "orbit_lite"
    if not main_py.exists() or not orbit_lite.is_dir():
        raise FileNotFoundError(f"kernel archive missing main.py or orbit_lite in {extracted_dir}")

    args.bot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_py, args.bot_dir / "_upstream.py")
    (args.bot_dir / "agent.py").write_text(WRAPPER, encoding="utf-8")
    if args.orbit_lite_dir.exists():
        shutil.rmtree(args.orbit_lite_dir)
    shutil.copytree(orbit_lite, args.orbit_lite_dir)
    print({
        "producer_agent": str(args.bot_dir / "agent.py"),
        "producer_upstream": str(args.bot_dir / "_upstream.py"),
        "orbit_lite": str(args.orbit_lite_dir),
    })


if __name__ == "__main__":
    main()
