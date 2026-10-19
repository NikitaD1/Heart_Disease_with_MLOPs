#!/usr/bin/env bash
# Runs the same stages as .github/workflows/ci-cd.yml on your machine.
# Stops at the first failing stage (non-zero exit) and prints which stage failed.
# & "C:\Program Files\Git\bin\bash.exe" ./scripts/run_ci_locally.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# Always test/train the code in *this* checkout, even if another copy is pip-installed.
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR=${LOG_DIR:-ci-logs}; mkdir -p "$LOG_DIR"
DOCKER_ARGS=${DOCKER_ARGS:-}
stage() { echo; echo "================ STAGE: $1 ================"; }
trap 'echo; echo "!!!!!!!! PIPELINE FAILED in stage: ${CURRENT:-unknown} (see $LOG_DIR) !!!!!!!!"' ERR

CURRENT=lint;  stage lint;  flake8 | tee "$LOG_DIR/lint.log"
CURRENT=test;  stage test;  python -m pytest -q --junitxml="$LOG_DIR/junit.xml" \
                              --cov=src/heart --cov=app --cov-report=term 2>&1 | tee "$LOG_DIR/test.log"
CURRENT=train; stage train; python -m heart.train 2>&1 | tee "$LOG_DIR/train.log"
CURRENT=gate;  stage "quality gate"
python -c "import json,sys; a=json.load(open('reports/metrics.json'))['cv']['cv_roc_auc_mean']; print(f'CV ROC-AUC {a:.3f}'); sys.exit(a < 0.85)"
if command -v docker >/dev/null 2>&1; then
  CURRENT=docker; stage "docker build"
  docker build $DOCKER_ARGS -t heart-disease-api:ci . 2>&1 | tee "$LOG_DIR/docker-build.log"
  docker rm -f heart-api-ci >/dev/null 2>&1 || true
  docker run -d --name heart-api-ci -p 8001:8000 heart-disease-api:ci
  CURRENT=smoke; stage "container smoke test"
  scripts/smoke_test.sh http://localhost:8001 | tee "$LOG_DIR/smoke.log"
  docker logs heart-api-ci > "$LOG_DIR/container.log" 2>&1; docker rm -f heart-api-ci >/dev/null
fi
echo; echo "================ PIPELINE PASSED ================"
