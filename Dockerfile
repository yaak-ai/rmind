# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential libexpat1 ffmpeg git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# rust: the private rbyte[yaak] deps (ymmv, m2df; #261) are Rust crates built from git
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:$PATH" \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    MATURIN_NO_INSTALL_RUST=1

# private git deps are fetched over the build's forwarded ssh agent:
#   docker build --ssh default ...
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# just
RUN curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# ytt (used by generate-config recipe)
RUN curl -sSfL https://github.com/carvel-dev/ytt/releases/latest/download/ytt-linux-amd64 \
        -o /usr/local/bin/ytt && chmod +x /usr/local/bin/ytt

WORKDIR /app

# dependency layer — cached as long as uv.lock doesn't change
COPY pyproject.toml uv.lock ./
RUN --mount=type=ssh uv sync --all-extras --all-groups --locked --no-install-project

# source + config templates
COPY . .
RUN --mount=type=ssh uv sync --all-extras --all-groups --locked

# image is built from this exact commit (enforced clean+pushed by `just docker-build`)
ARG COMMIT_SHA
ENV WANDB_GIT_COMMIT=$COMMIT_SHA \
    WANDB_GIT_REMOTE_URL=https://github.com/yaak-ai/rmind
LABEL org.opencontainers.image.revision=$COMMIT_SHA
