from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

MAX_MEMORY_BYTES = 16 * 1024 * 1024 * 1024


def _read_first_existing(paths: list[str]) -> str | None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def _parse_limit(value: str | None, label: str) -> int:
    if value is None:
        raise RuntimeError(f"{label} cgroup file was not found")
    if value == "max":
        raise RuntimeError(f"{label} is unlimited")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not an integer: {value!r}") from exc


def memory_limit_bytes() -> int:
    return _parse_limit(
        _read_first_existing(
            [
                "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            ]
        ),
        "memory limit",
    )


def validate_swap_limit(memory_limit: int) -> str:
    cgroup_v2_swap = _read_first_existing(["/sys/fs/cgroup/memory.swap.max"])
    if cgroup_v2_swap is not None:
        if cgroup_v2_swap == "max":
            raise RuntimeError("swap limit is unlimited")
        swap_limit = int(cgroup_v2_swap)
        if swap_limit > 0:
            raise RuntimeError(f"swap limit allows {swap_limit} extra bytes")
        return "disabled"

    cgroup_v1_memsw = _read_first_existing(
        ["/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"]
    )
    if cgroup_v1_memsw is not None:
        memswap_limit = _parse_limit(cgroup_v1_memsw, "mem+swap limit")
        if memswap_limit > memory_limit:
            raise RuntimeError(
                f"mem+swap limit {memswap_limit} exceeds memory limit {memory_limit}"
            )
        return "not_above_memory_limit"

    raise RuntimeError("swap limit cgroup file was not found")


def nvidia_smi_summary() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip() or None
    return result.stdout.strip()


def build_report(require_cuda: bool, max_memory_bytes: int) -> dict[str, Any]:
    limit = memory_limit_bytes()
    if limit > max_memory_bytes:
        raise RuntimeError(
            f"memory limit {limit} bytes exceeds max {max_memory_bytes} bytes"
        )

    import orbit_wars_rs  # noqa: F401
    import torch

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "memory_limit_bytes": limit,
        "max_memory_bytes": max_memory_bytes,
        "swap": validate_swap_limit(limit),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": nvidia_smi_summary(),
        "orbit_wars_rs": "import_ok",
    }

    if require_cuda:
        if torch.version.cuda is None:
            raise RuntimeError("PyTorch was installed without CUDA support")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("no CUDA devices are visible inside the container")
        report["cuda_device_0"] = torch.cuda.get_device_name(0)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-memory-bytes", type=int, default=MAX_MEMORY_BYTES)
    args = parser.parse_args()

    report = build_report(args.require_cuda, args.max_memory_bytes)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
