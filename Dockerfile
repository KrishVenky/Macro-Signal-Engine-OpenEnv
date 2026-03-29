# Macro Signal Engine — Dockerfile
# Builds a minimal FastAPI server for HF Spaces (port 7860, non-root user)
# Runtime: python:3.11-slim | Dependencies: fastapi, uvicorn, pydantic, websockets

FROM python:3.11-slim

# HF Spaces security requirement: non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (cache layer)
COPY --chown=user pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.29" \
        "pydantic>=2.0" \
        "websockets>=12.0"

# Copy source and data
COPY --chown=user src/ ./src/
COPY --chown=user data/ ./data/

# Install the package itself (for module resolution)
RUN pip install --no-cache-dir -e .

# HF Spaces default port — MUST be 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=50

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD uvicorn macro_signal.server.app:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info
