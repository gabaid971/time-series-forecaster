FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /bin/uv

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install locked dependencies first for better caching (no dev group)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# Copy application code
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
