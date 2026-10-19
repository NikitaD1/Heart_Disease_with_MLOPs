#!/usr/bin/env bash
# End-to-end smoke test against a running API.  Usage: scripts/smoke_test.sh [BASE_URL]
# Exits non-zero (and prints why) on the first failed check, so CI fails loudly.
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> waiting for ${BASE_URL}/ready"
for i in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/ready" >/dev/null 2>&1; then break; fi
  [ "$i" -eq 30 ] && { echo "FAIL: service not ready after 60s"; exit 1; }
  sleep 2
done

check() {  # name expected_status method path [json_file|json_string]
  local name="$1" expected="$2" method="$3" path="$4" body="${5:-}"
  local args=(-s -o /tmp/smoke_body -w "%{http_code}" -X "$method" "${BASE_URL}${path}")
  if [ -n "$body" ]; then
    if [ -f "$body" ]; then args+=(-H "content-type: application/json" --data @"$body");
    else args+=(-H "content-type: application/json" --data "$body"); fi
  fi
  local code; code=$(curl "${args[@]}")
  if [ "$code" != "$expected" ]; then
    echo "FAIL: $name -> HTTP $code (expected $expected)"; cat /tmp/smoke_body; echo; exit 1
  fi
  echo "PASS: $name -> HTTP $code $(head -c 160 /tmp/smoke_body | tr "\n" " ")"
}

check "health"                200 GET  /health
check "ready"                 200 GET  /ready
check "predict (low risk)"    200 POST /predict "$HERE/samples/patient.json"
check "predict (high risk)"   200 POST /predict "$HERE/samples/patient_high_risk.json"
check "batch predict"         200 POST /predict/batch "$HERE/samples/batch.json"
check "invalid input -> 422"  422 POST /predict '{"age": 500}'
check "metrics"               200 GET  /metrics
grep -q "model_predictions_total" /tmp/smoke_body || { echo "FAIL: metrics missing"; exit 1; }
echo "All smoke tests passed."
