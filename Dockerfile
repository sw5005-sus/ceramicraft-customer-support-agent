FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY serve.py ./
RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "python", "serve.py"]
