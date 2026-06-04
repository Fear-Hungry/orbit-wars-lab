# syntax=docker/dockerfile:1.7

FROM rust:1.86-slim-bookworm AS rust-toolchain

FROM node:22-bookworm-slim AS node-runtime

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
ARG TORCH_COMPUTE_PLATFORM=cpu

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/home/orbit/.cache/pip \
    VIRTUAL_ENV=/opt/venv \
    PATH=/home/orbit/.local/bin:/opt/venv/bin:/usr/local/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_HOME=/home/orbit/.cargo \
    CARGO_TARGET_DIR=/home/orbit/.cargo-target \
    RUSTUP_HOME=/usr/local/rustup \
    TORCH_HOME=/home/orbit/.cache/torch \
    PYO3_PYTHON=/opt/venv/bin/python \
    PYTHONPATH=/workspace:/workspace/python \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    RAYON_NUM_THREADS=4 \
    CARGO_BUILD_JOBS=2 \
    CARGO_INCREMENTAL=0 \
    TOKENIZERS_PARALLELISM=false \
    CODEX_HOME=/home/orbit/.codex

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    git \
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

RUN groupadd --gid "${GID}" orbit \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash orbit \
    && mkdir -p /workspace /home/orbit/.cache/pip /home/orbit/.cargo /home/orbit/.cargo-target /home/orbit/.codex /home/orbit/.local/bin /home/orbit/.local/lib/node_modules "${VIRTUAL_ENV}" \
    && chown -R orbit:orbit /workspace /home/orbit "${VIRTUAL_ENV}"

RUN chmod -R a+rX /usr/local/cargo /usr/local/rustup

USER orbit

RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

COPY --chown=orbit:orbit requirements.txt pyproject.toml Cargo.toml Cargo.lock ./
COPY --chown=orbit:orbit crates/orbit_wars_core/Cargo.toml crates/orbit_wars_core/Cargo.toml
COPY --chown=orbit:orbit crates/orbit_wars_py/Cargo.toml crates/orbit_wars_py/Cargo.toml
COPY --chown=orbit:orbit crates/orbit_wars_core/src crates/orbit_wars_core/src
COPY --chown=orbit:orbit crates/orbit_wars_py/src crates/orbit_wars_py/src

RUN --mount=type=cache,id=pip-shared,target=/home/orbit/.cache/pip,uid=1000,gid=1000 \
    --mount=type=cache,id=torch-shared,target=/home/orbit/.cache/torch,uid=1000,gid=1000 \
    grep -vE '^torch([<>=].*)?$' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && if [ "${TORCH_COMPUTE_PLATFORM}" = "cpu" ]; then \
        pip install "torch>=2.2" --index-url "https://download.pytorch.org/whl/cpu"; \
    else \
        pip install "torch>=2.2" --index-url "https://download.pytorch.org/whl/${TORCH_COMPUTE_PLATFORM}"; \
    fi

RUN --mount=type=cache,id=cargo-registry,target=/home/orbit/.cargo/registry,uid=1000,gid=1000 \
    --mount=type=cache,id=cargo-git,target=/home/orbit/.cargo/git,uid=1000,gid=1000 \
    cargo fetch --locked

COPY --chown=orbit:orbit . .

RUN --mount=type=cache,id=cargo-registry,target=/home/orbit/.cargo/registry,uid=1000,gid=1000 \
    --mount=type=cache,id=cargo-git,target=/home/orbit/.cargo/git,uid=1000,gid=1000 \
    --mount=type=cache,id=cargo-target,target=/home/orbit/.cargo-target,uid=1000,gid=1000 \
    maturin develop --release -m crates/orbit_wars_py/Cargo.toml \
    && python -c "import orbit_wars_rs, torch"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
