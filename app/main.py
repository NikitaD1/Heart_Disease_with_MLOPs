"""FastAPI model-serving service for the heart-disease classifier.

Endpoints
  GET  /           service info
  GET  /health     liveness probe (process is up)
  GET  /ready      readiness probe (model is loaded) -> 503 until then
  GET  /model      model metadata (version, metrics, training run)
  POST /predict    one patient -> prediction + confidence
  POST /predict/batch  up to 1000 patients
  GET  /metrics    Prometheus metrics

Every request is logged as one JSON line (request id, path, status, latency)
and every prediction is logged with its inputs and output.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.schemas import BatchRequest, BatchResponse, PatientFeatures, Prediction
from heart.config import MODEL_FILE, MODEL_META_FILE
from heart.predict import HeartModel

SERVICE_NAME = "heart-disease-api"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODEL_FILE))
META_PATH = Path(os.environ.get("MODEL_META_PATH", MODEL_META_FILE))
LOG_FILE = os.environ.get("LOG_FILE", "")  # e.g. logs/api.log; empty -> stdout only


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, default=str)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger(SERVICE_NAME)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)
    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    logger.propagate = False
    return logger


log = _setup_logging()


def _log(level: int, msg: str, **fields) -> None:
    log.log(level, msg, extra={"extra_fields": fields})


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
PRED_PROBA = Histogram("model_prediction_probability", "Predicted probability of disease",
                       buckets=[i / 10 for i in range(1, 11)])
ERRORS = Counter("model_prediction_errors_total", "Failed prediction requests")
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded and ready")
MODEL_INFO = Gauge("model_info", "Model version information", ["version", "algorithm"])

STATE: dict = {"model": None, "started": time.time()}
# In-memory ring buffers that feed the built-in /dashboard (Prometheus is the
# durable store; these only cover the most recent traffic of this replica).
RECENT_REQUESTS: deque = deque(maxlen=2000)
RECENT_PREDICTIONS: deque = deque(maxlen=2000)
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


def _load_model() -> None:
    try:
        model = HeartModel(MODEL_PATH, META_PATH)
        STATE["model"] = model
        MODEL_LOADED.set(1)
        MODEL_INFO.labels(model.version, model.metadata.get("algorithm", "unknown")).set(1)
        _log(logging.INFO, "model loaded", model_path=str(MODEL_PATH),
             model_version=model.version, algorithm=model.metadata.get("algorithm"))
    except Exception as exc:  # noqa: BLE001 - service stays up, /ready reports 503
        MODEL_LOADED.set(0)
        _log(logging.ERROR, "model load failed", error=str(exc), model_path=str(MODEL_PATH))


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan,
              description="Predicts the presence of heart disease from 13 clinical "
                          "attributes (UCI Cleveland dataset).")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:  # pragma: no cover - re-raised after logging
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        if endpoint not in ("/metrics", "/stats", "/dashboard"):
            RECENT_REQUESTS.append({"ts": time.time(), "method": request.method,
                                    "path": endpoint, "status": status,
                                    "latency_ms": round(elapsed * 1000, 2),
                                    "request_id": request_id})
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            _log(logging.INFO, "request", request_id=request_id, method=request.method,
                 path=endpoint, status=status, latency_ms=round(elapsed * 1000, 2),
                 client=request.client.host if request.client else None)
    response.headers["x-request-id"] = request_id
    return response


def _require_model() -> HeartModel:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _predict(model: HeartModel, patients: list[PatientFeatures]) -> list[Prediction]:
    records = [p.model_dump() for p in patients]
    try:
        results = model.predict_records(records)
    except Exception as exc:
        ERRORS.inc()
        _log(logging.ERROR, "prediction failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    out = []
    for rec, res in zip(records, results):
        PREDICTIONS.labels(res["label"]).inc()
        PRED_PROBA.observe(res["probability_disease"])
        RECENT_PREDICTIONS.append({"ts": time.time(), **res})
        _log(logging.INFO, "prediction", inputs=rec, **res, model_version=model.version)
        out.append(Prediction(**res, model_version=model.version))
    return out


@app.get("/")
def root():
    return {"service": SERVICE_NAME, "docs": "/docs", "predict": "POST /predict",
            "metrics": "/metrics"}


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STATE["started"], 1)}


@app.get("/ready")
def ready():
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "ready", "model_version": STATE["model"].version}


@app.get("/model")
def model_info():
    return _require_model().metadata


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures):
    return _predict(_require_model(), [patient])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    return BatchResponse(predictions=_predict(_require_model(), batch.instances))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


@app.get("/stats")
def stats(window_s: int = 900, bucket_s: int = 30):
    """Aggregated recent traffic for the dashboard (this replica only)."""
    now = time.time()
    reqs = [r for r in RECENT_REQUESTS if now - r["ts"] <= window_s]
    preds = [p for p in RECENT_PREDICTIONS if now - p["ts"] <= window_s]
    lat = [r["latency_ms"] for r in reqs]
    n_buckets = max(1, window_s // bucket_s)
    series = [{"t": int(now - (n_buckets - i) * bucket_s), "requests": 0, "errors": 0}
              for i in range(n_buckets)]
    for r in reqs:
        idx = min(n_buckets - 1, int((r["ts"] - (now - window_s)) // bucket_s))
        series[idx]["requests"] += 1
        series[idx]["errors"] += int(r["status"] >= 400)
    by_status: dict = {}
    by_path: dict = {}
    for r in reqs:
        by_status[str(r["status"])] = by_status.get(str(r["status"]), 0) + 1
        by_path[r["path"]] = by_path.get(r["path"], 0) + 1
    positives = sum(p["prediction"] for p in preds)
    model = STATE["model"]
    return {
        "window_s": window_s, "bucket_s": bucket_s,
        "uptime_s": round(now - STATE["started"], 1),
        "model_loaded": model is not None,
        "model_version": model.version if model else None,
        "algorithm": model.metadata.get("algorithm") if model else None,
        "requests": len(reqs),
        "error_rate": round(sum(r["status"] >= 400 for r in reqs) / len(reqs), 4) if reqs else 0,
        "latency_ms": {"p50": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95),
                       "p99": _percentile(lat, 0.99)},
        "by_status": by_status, "by_path": by_path,
        "predictions": len(preds), "predicted_positive": positives,
        "positive_share": round(positives / len(preds), 4) if preds else None,
        "mean_probability": round(sum(p["probability_disease"] for p in preds) / len(preds), 4)
        if preds else None,
        "series": series,
        "recent": list(reversed(list(RECENT_REQUESTS)[-15:])),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Self-contained monitoring page (no external assets) that polls /stats."""
    return HTMLResponse(DASHBOARD_HTML.read_text())
