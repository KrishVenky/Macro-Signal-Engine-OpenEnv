FROM public.ecr.aws/docker/library/python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src/ ./src/
COPY --chown=user data/ ./data/
COPY --chown=user server/ ./server/

ENV PYTHONPATH="/app/src/envs:${PYTHONPATH}"
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=50
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD uvicorn macro_signal.server.app:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info
