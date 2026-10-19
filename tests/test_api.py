"""API tests using FastAPI's TestClient against a temporary model."""
import pytest
from fastapi.testclient import TestClient

import app.main as api
from heart.predict import HeartModel


@pytest.fixture
def client(model_files):
    with TestClient(api.app) as c:          # runs lifespan (loads models/ if present)
        api.STATE["model"] = HeartModel(*model_files)   # force the temp test model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "test"


def test_predict_returns_prediction_and_confidence(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["label"] in ("heart disease", "no heart disease")
    assert body["model_version"] == "test"
    assert "x-request-id" in r.headers


def test_positive_prediction_confidence_matches_probability(client, high_risk_patient):
    body = client.post("/predict", json=high_risk_patient).json()
    assert body["prediction"] == 1 and body["label"] == "heart disease"
    assert body["confidence"] == body["probability_disease"] >= 0.5


def test_predict_accepts_null_ca_thal(client, patient):
    r = client.post("/predict", json={**patient, "ca": None, "thal": None})
    assert r.status_code == 200


@pytest.mark.parametrize("change", [
    {"age": 500}, {"cp": 9}, {"sex": 2}, {"chol": -1}, {"thal": 5}, {"unexpected": 1},
])
def test_predict_rejects_invalid_input(client, patient, change):
    r = client.post("/predict", json={**patient, **change})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client, patient):
    patient.pop("thalach")
    assert client.post("/predict", json=patient).status_code == 422


def test_batch_predict(client, patient):
    r = client.post("/predict/batch", json={"instances": [patient, {**patient, "age": 40}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_endpoint_exposes_counters(client, patient):
    client.post("/predict", json=patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert "api_request_latency_seconds_bucket" in text


def test_returns_503_without_model(client, patient):
    api.STATE["model"] = None
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json=patient).status_code == 503


def test_stats_aggregates_recent_traffic(client, patient):
    client.post("/predict", json=patient)
    client.post("/predict", json={**patient, "age": 999})
    s = client.get("/stats").json()
    assert s["requests"] >= 2 and s["predictions"] >= 1
    assert "422" in s["by_status"] and "/predict" in s["by_path"]
    assert s["latency_ms"]["p95"] >= s["latency_ms"]["p50"] >= 0
    assert len(s["series"]) == s["window_s"] // s["bucket_s"]


def test_dashboard_page_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200 and "live monitor" in r.text
