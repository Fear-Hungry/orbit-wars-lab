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
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/home/orbit/.local/bin:/opt/venv/bin:/usr/local/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
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

RUN chmod -R a+rX /usr/local/cargo /usr/local/rustup

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
    grep -vE '^torch([<>=].*)?$' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && if [ "${TORCH_COMPUTE_PLATFORM}" = "cpu" ]; then \
        pip install "torch>=2.2" --index-url "https://download.pytorch.org/whl/cpu"; \
    else \
        pip install "torch>=2.2" --index-url "https://download.pytorch.org/whl/${TORCH_COMPUTE_PLATFORM}"; \
    fi

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    maturin develop --release -m crates/orbit_wars_py/Cargo.toml \
    && python -c "import orbit_wars_rs, torch"

RUN chown -R orbit:orbit /workspace /home/orbit

USER orbit

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
