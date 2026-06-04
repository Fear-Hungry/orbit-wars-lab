from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def _add_file(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    tar.add(source, arcname=arcname, recursive=False)


def _add_tree(tar: tarfile.TarFile, source: Path, arc_prefix: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            tar.add(path, arcname=str(Path(arc_prefix) / path.relative_to(source)), recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the prepared Producer agent as a Kaggle submission tar.gz."
    )
    parser.add_argument("--producer-dir", type=Path, default=Path("artifacts/opponents/producer"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_producer.tar.gz"))
    args = parser.parse_args()

    main_py = args.producer_dir / "main.py"
    orbit_lite = args.producer_dir / "orbit_lite"
    if not main_py.exists() or not orbit_lite.is_dir():
        raise FileNotFoundError(
            f"missing Producer files in {args.producer_dir}; run scripts.prepare_producer_opponent first"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        _add_file(tar, main_py, "main.py")
        _add_tree(tar, orbit_lite, "orbit_lite")
    print({"wrote": str(args.out), "producer_dir": str(args.producer_dir)})


if __name__ == "__main__":
    main()
