# Common tasks. Run `make help` for the list.
PY      ?= python
IMAGE   ?= heart-disease-api
TAG     ?= 1.0.0
NS      ?= heart-mlops

.PHONY: help setup data eda train test lint notebooks mlflow-ui serve docker-build docker-run \
        smoke compose-up compose-down k8s-deploy k8s-status k8s-delete helm-install ci clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

setup:           ## create venv deps (run inside an activated virtualenv)
	pip install -r requirements.txt && pip install -e .

data:            ## download + clean the UCI dataset
	$(PY) scripts/download_data.py

eda:             ## regenerate EDA figures in reports/figures
	$(PY) -m heart.eda

train:           ## tune/train models, log to MLflow, package the champion
	$(PY) -m heart.train && $(PY) scripts/export_mlflow_runs.py

test:            ## unit tests with coverage
	$(PY) -m pytest -v --cov=src/heart --cov=app --cov-report=term-missing

lint:            ## flake8
	flake8

notebooks:       ## execute all notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

mlflow-ui:       ## MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:           ## run the API locally on :8000
	uvicorn app.main:app --reload --port 8000

docker-build:    ## build the serving image
	docker build -t $(IMAGE):$(TAG) .

docker-run:      ## run the serving image on :8000
	docker run --rm -d --name heart-api -p 8000:8000 $(IMAGE):$(TAG)

smoke:           ## smoke-test a running API (BASE_URL=http://localhost:8000)
	scripts/smoke_test.sh $${BASE_URL:-http://localhost:8000}

compose-up:      ## API + Prometheus + Grafana
	docker compose up -d --build

compose-down:
	docker compose down

k8s-deploy:      ## apply manifests to the current kube context
	kubectl apply -k k8s/ && kubectl -n $(NS) rollout status deploy/heart-api --timeout=180s

k8s-status:
	kubectl -n $(NS) get deploy,pods,svc,ingress -o wide

k8s-delete:
	kubectl delete -k k8s/

helm-install:    ## alternative to k8s-deploy
	helm upgrade --install heart-api helm/heart-api -n $(NS) --create-namespace

ci:              ## run the CI stages locally (lint, test, train, docker, smoke)
	scripts/run_ci_locally.sh

clean:
	rm -rf .pytest_cache .coverage htmlcov test-results logs
