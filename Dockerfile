FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# just
RUN curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# ytt (used by generate-config recipe)
RUN curl -sSfL https://github.com/carvel-dev/ytt/releases/latest/download/ytt-linux-amd64 \
        -o /usr/local/bin/ytt && chmod +x /usr/local/bin/ytt

WORKDIR /app

# dependency layer — cached as long as uv.lock doesn't change
COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --all-groups --locked --no-install-project

# source + config templates
COPY . .
RUN uv sync --all-extras --all-groups --locked

# image is built from this exact commit (enforced clean+pushed by `just docker-build`)
ARG COMMIT_SHA
ENV WANDB_GIT_COMMIT=$COMMIT_SHA \
    WANDB_GIT_REMOTE_URL=https://github.com/yaak-ai/rmind
LABEL org.opencontainers.image.revision=$COMMIT_SHA
