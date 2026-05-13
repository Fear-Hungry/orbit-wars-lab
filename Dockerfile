# syntax=docker/dockerfile:1.7

ARG ML_CPU_BASE_IMAGE=local/ml-python311-cpu:bookworm

FROM rust:1.86-slim-bookworm AS rust-toolchain

FROM node:22-bookworm-slim AS node-runtime

FROM ${ML_CPU_BASE_IMAGE}

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
ARG TORCH_COMPUTE_PLATFORM=cpu
ARG TORCH_VERSION=
ARG MAX_CONTAINER_MEMORY_BYTES=17179869184

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/home/orbit/.local/bin:/opt/venv/bin:/usr/local/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PYO3_PYTHON=/opt/venv/bin/python \
    PYTHONPATH=/workspace:/workspace/python \
    MAX_CONTAINER_MEMORY_BYTES=${MAX_CONTAINER_MEMORY_BYTES} \
    ORBIT_ENFORCE_NO_SWAP=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    RAYON_NUM_THREADS=4 \
    CARGO_BUILD_JOBS=2 \
    CARGO_INCREMENTAL=0 \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    CODEX_HOME=/home/orbit/.codex

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    libssl-dev \
    libstdc++6 \
    pkg-config \
    procps \
    ripgrep \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY docker/entrypoint.sh /usr/local/bin/orbit-entrypoint

RUN chmod -R a+rX /usr/local/cargo /usr/local/rustup \
    && chmod 0755 /usr/local/bin/orbit-entrypoint

RUN groupadd --gid "${GID}" orbit \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash orbit \
    && mkdir -p /workspace /home/orbit/.codex /home/orbit/.local/bin /home/orbit/.local/lib/node_modules "${VIRTUAL_ENV}"

RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

COPY requirements.txt pyproject.toml Cargo.toml Cargo.lock ./
COPY crates/orbit_wars_core/Cargo.toml crates/orbit_wars_core/Cargo.toml
COPY crates/orbit_wars_py/Cargo.toml crates/orbit_wars_py/Cargo.toml
COPY crates/orbit_wars_core/src crates/orbit_wars_core/src
COPY crates/orbit_wars_py/src crates/orbit_wars_py/src

RUN --mount=type=cache,target=/root/.cache/pip \
    case "${TORCH_COMPUTE_PLATFORM}" in \
        cpu|cu118|cu126|cu128) ;; \
        *) echo "Unsupported TORCH_COMPUTE_PLATFORM=${TORCH_COMPUTE_PLATFORM}" >&2; exit 2 ;; \
    esac \
    && \
    grep -vE '^torch([<>=].*)?$' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && if [ -n "${TORCH_VERSION}" ]; then torch_spec="torch==${TORCH_VERSION}"; else torch_spec="torch>=2.2"; fi \
    && if [ "${TORCH_COMPUTE_PLATFORM}" = "cpu" ]; then \
        pip install "${torch_spec}" --index-url "https://download.pytorch.org/whl/cpu"; \
    else \
        pip install "${torch_spec}" --index-url "https://download.pytorch.org/whl/${TORCH_COMPUTE_PLATFORM}" \
        && pip install nvidia-ml-py; \
    fi

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    maturin develop --release -m crates/orbit_wars_py/Cargo.toml \
    && python -c "import orbit_wars_rs, torch" \
    && if [ "${TORCH_COMPUTE_PLATFORM}" = "cpu" ]; then \
        python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"; \
    else \
        python -c "import torch; assert torch.version.cuda is not None, torch.__version__"; \
    fi

RUN chown -R orbit:orbit /workspace /home/orbit

USER orbit

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/orbit-entrypoint"]
CMD ["bash"]
