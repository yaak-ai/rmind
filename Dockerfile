FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
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
