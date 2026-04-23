#!/usr/bin/env bash
set -euo pipefail

max_bytes="${MAX_CONTAINER_MEMORY_BYTES:-17179869184}"

read_cgroup_memory_limit() {
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        cat /sys/fs/cgroup/memory.max
        return
    fi
    if [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
        cat /sys/fs/cgroup/memory/memory.limit_in_bytes
        return
    fi
    echo "unknown"
}

memory_limit="$(read_cgroup_memory_limit)"
if [[ "${memory_limit}" == "max" || "${memory_limit}" == "unknown" ]]; then
    echo "ERROR: Docker memory limit is not enforced; refusing to start." >&2
    echo "Set mem_limit/memswap_limit to 16g or less before running this image." >&2
    exit 64
fi

if ! [[ "${memory_limit}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not parse Docker memory limit: ${memory_limit}" >&2
    exit 64
fi

if (( memory_limit > max_bytes )); then
    echo "ERROR: Docker memory limit ${memory_limit} bytes exceeds max ${max_bytes} bytes." >&2
    exit 64
fi

if [[ "${ORBIT_ENFORCE_NO_SWAP:-1}" == "1" ]]; then
    if [[ -r /sys/fs/cgroup/memory.swap.max ]]; then
        swap_limit="$(cat /sys/fs/cgroup/memory.swap.max)"
        if [[ "${swap_limit}" == "max" ]]; then
            echo "ERROR: Docker swap limit is not enforced; refusing to start." >&2
            exit 64
        fi
        if [[ "${swap_limit}" =~ ^[0-9]+$ ]] && (( swap_limit > 0 )); then
            echo "ERROR: Docker swap limit allows ${swap_limit} extra bytes; refusing to start." >&2
            exit 64
        fi
    elif [[ -r /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes ]]; then
        memswap_limit="$(cat /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes)"
        if [[ "${memswap_limit}" =~ ^[0-9]+$ ]] && (( memswap_limit > memory_limit )); then
            echo "ERROR: Docker mem+swap limit exceeds memory limit; refusing to start." >&2
            exit 64
        fi
    else
        echo "ERROR: Cannot verify Docker swap limit; refusing to start." >&2
        exit 64
    fi
fi

if [[ "${ORBIT_REQUIRE_CUDA:-0}" == "1" ]]; then
    python - <<'PY'
import sys

import torch

if torch.version.cuda is None:
    raise SystemExit("ERROR: PyTorch was not installed with CUDA support.")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available inside the container.")
if torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: No CUDA devices are visible inside the container.")

device_name = torch.cuda.get_device_name(0)
print(f"CUDA ready: torch={torch.__version__} cuda={torch.version.cuda} device={device_name}", file=sys.stderr)
PY
fi

exec "$@"
