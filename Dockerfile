FROM python:3.13-slim

# Copy uv binary 
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /bin/

WORKDIR /app

# Install build tools for native compilation (torch, tree-sitter)
RUN apt-get update && \ 
    apt-get install -y make gcc build-essential vim && \
    rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Install only external dependencies to cache this layer
# and prevent re-installation when source code changes.
RUN uv sync --no-install-project

# Copy source code
COPY . .

# Activate project virtual environment
ENV PATH="/app/.venv/bin:$PATH"
