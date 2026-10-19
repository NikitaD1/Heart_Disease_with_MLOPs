# ---------------------------------------------------------------------------
# Heart Disease Risk API - production image
#   docker build -t heart-disease-api:1.0.0 .
#   docker run -p 8000:8000 heart-disease-api:1.0.0
# Requires models/heart_model.joblib (run `make train` or `python -m heart.train`).
# BASE_IMAGE can be overridden (e.g. an internal registry mirror).
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.11-slim

# ---------- stage 1: build an isolated virtualenv with serving deps only ----------
FROM ${BASE_IMAGE} AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-serve.txt

# ---------- stage 2: slim runtime image ----------
FROM ${BASE_IMAGE} AS runtime
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service predicting heart disease risk" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/heart_model.joblib \
    MODEL_META_PATH=/app/models/model_metadata.json \
    PORT=8000

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/heart/ src/heart/
COPY app/ app/
COPY models/heart_model.joblib models/model_metadata.json models/

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
