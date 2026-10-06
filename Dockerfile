FROM python:3.11-slim

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
COPY nadirclaw/ nadirclaw/
RUN pip install --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8856/health')" || exit 1

EXPOSE 8856

# Listening on all interfaces requires NADIRCLAW_AUTH_TOKEN; serve refuses to start without it.
CMD ["nadirclaw", "serve", "--host", "0.0.0.0"]
