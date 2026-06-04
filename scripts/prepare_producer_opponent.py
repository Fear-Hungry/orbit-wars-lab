from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

WRAPPER = """from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent / "producer"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location("_producer_main", _ROOT / "main.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load Producer agent from {_ROOT}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _planet_row(planet):
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


def _fleet_row(fleet):
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


def _to_list_observation(obs):
    if not isinstance(obs, dict):
        return obs
    converted = dict(obs)
    converted["planets"] = [_planet_row(planet) for planet in obs.get("planets", [])]
    converted["initial_planets"] = [_planet_row(planet) for planet in obs.get("initial_planets", [])]
    converted["fleets"] = [_fleet_row(fleet) for fleet in obs.get("fleets", [])]
    return converted


def agent(obs):
    return _MODULE.agent(_to_list_observation(obs))
"""


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
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/opponents"))
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

    producer_dir = args.out_dir / "producer"
    if producer_dir.exists():
        shutil.rmtree(producer_dir)
    producer_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, producer_dir)

    wrapper_path = args.out_dir / "producer_agent.py"
    wrapper_path.write_text(WRAPPER, encoding="utf-8")
    print({"producer_dir": str(producer_dir), "wrapper": str(wrapper_path)})


if __name__ == "__main__":
    main()
