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
    parser.add_argument("--producer-agent", type=Path, default=Path("bots/producer/agent.py"))
    parser.add_argument("--producer-upstream", type=Path, default=Path("bots/producer/_upstream.py"))
    parser.add_argument("--orbit-lite-dir", type=Path, default=Path("orbit_lite"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_producer.tar.gz"))
    args = parser.parse_args()

    if (
        not args.producer_agent.exists()
        or not args.producer_upstream.exists()
        or not args.orbit_lite_dir.is_dir()
    ):
        raise FileNotFoundError(
            "missing Producer packaging inputs: "
            f"{args.producer_agent}, {args.producer_upstream}, and {args.orbit_lite_dir}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        _add_file(tar, args.producer_agent, "main.py")
        _add_file(tar, args.producer_upstream, "_upstream.py")
        _add_tree(tar, args.orbit_lite_dir, "orbit_lite")
    print({
        "wrote": str(args.out),
        "producer_agent": str(args.producer_agent),
        "producer_upstream": str(args.producer_upstream),
        "orbit_lite": str(args.orbit_lite_dir),
    })


if __name__ == "__main__":
    main()
