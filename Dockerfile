FROM python:3.11-slim

# System utilities (curl for debugging/cron if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install Python dependencies first (cached Docker layer)
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application source code & register entrypoints
COPY . .
RUN uv pip install --system --no-cache --no-deps .

# Ensure local fallback directories exist
RUN mkdir -p data logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# Default command: binds 0.0.0.0 and honors Railway's injected $PORT automatically
CMD ["python", "scripts/start_web.py", "--host", "0.0.0.0"]
