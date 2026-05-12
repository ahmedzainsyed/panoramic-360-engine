# ============================================================
# 360° Construction Site Semantic Understanding Engine
# Makefile - Complete build, dev, test, and deploy commands
# ============================================================

.PHONY: help dev prod prod-gpu build test clean lint format shell logs \
        migrate seed train deploy monitoring setup docs

# ─── Colors ──────────────────────────────────────────────────────────────────
BLUE  := \033[0;34m
GREEN := \033[0;32m
YELLOW:= \033[1;33m
RED   := \033[0;31m
NC    := \033[0m  # No Color

# ─── Config ──────────────────────────────────────────────────────────────────
PROJECT   := panoramic-360-engine
VERSION   := $(shell git describe --tags --always --dirty 2>/dev/null || echo "v1.0.0")
REGISTRY  := ghcr.io/your-org
ENV_FILE  := .env
DC        := docker compose
DC_DEV    := $(DC) -f docker/docker-compose.dev.yml
DC_PROD   := $(DC) -f docker/docker-compose.prod.yml
DC_GPU    := $(DC) -f docker/docker-compose.gpu.yml

# ─── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     360° Construction Site Intelligence Platform         ║$(NC)"
	@echo "$(BLUE)║     Version: $(VERSION)                                  ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(YELLOW)<target>$(NC)\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─── Environment Setup ────────────────────────────────────────────────────────
setup: ## Initial environment setup (run first)
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@cp -n .env.example .env || true
	@pip install pre-commit
	@pre-commit install
	@$(MAKE) _check-docker
	@echo "$(GREEN)✓ Setup complete$(NC)"

_check-docker:
	@docker info > /dev/null 2>&1 || (echo "$(RED)Docker is not running$(NC)" && exit 1)
	@docker compose version > /dev/null 2>&1 || (echo "$(RED)Docker Compose not found$(NC)" && exit 1)

# ─── Development ─────────────────────────────────────────────────────────────
dev: ## Start full development stack
	@echo "$(BLUE)Starting development environment...$(NC)"
	@$(DC_DEV) --env-file $(ENV_FILE) up --build -d
	@$(MAKE) _wait-healthy
	@$(MAKE) _print-urls-dev

dev-backend: ## Start only backend services
	@$(DC_DEV) --env-file $(ENV_FILE) up -d postgres redis minio api celery_worker

dev-frontend: ## Start only frontend dev server
	@cd frontend && npm run dev

dev-ml: ## Start ML inference services
	@$(DC_DEV) --env-file $(ENV_FILE) up -d api celery_worker triton

dev-rebuild: ## Rebuild and restart development stack
	@$(DC_DEV) --env-file $(ENV_FILE) up --build --force-recreate -d

# ─── Production ──────────────────────────────────────────────────────────────
prod: ## Start production stack (CPU)
	@echo "$(BLUE)Starting production environment...$(NC)"
	@$(DC_PROD) --env-file $(ENV_FILE) up -d
	@$(MAKE) _wait-healthy
	@$(MAKE) _print-urls-prod

prod-gpu: ## Start production stack with GPU support
	@echo "$(BLUE)Starting GPU-accelerated production environment...$(NC)"
	@$(DC_GPU) --env-file $(ENV_FILE) up -d
	@$(MAKE) _wait-healthy
	@$(MAKE) _print-urls-prod

prod-scale: ## Scale inference workers
	@$(DC_PROD) --env-file $(ENV_FILE) up -d --scale celery_worker=4

# ─── Build ────────────────────────────────────────────────────────────────────
build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker build -f docker/backend/Dockerfile -t $(REGISTRY)/$(PROJECT)-api:$(VERSION) .
	@docker build -f docker/frontend/Dockerfile -t $(REGISTRY)/$(PROJECT)-frontend:$(VERSION) .
	@echo "$(GREEN)✓ Images built$(NC)"

build-push: build ## Build and push images to registry
	@docker push $(REGISTRY)/$(PROJECT)-api:$(VERSION)
	@docker push $(REGISTRY)/$(PROJECT)-frontend:$(VERSION)
	@echo "$(GREEN)✓ Images pushed to $(REGISTRY)$(NC)"

# ─── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@$(DC_DEV) exec api alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

migrate-create: ## Create new migration (name=<migration_name>)
	@$(DC_DEV) exec api alembic revision --autogenerate -m "$(name)"

migrate-down: ## Rollback last migration
	@$(DC_DEV) exec api alembic downgrade -1

seed: ## Seed database with sample data
	@$(DC_DEV) exec api python scripts/seed_database.py

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run full test suite
	@echo "$(BLUE)Running test suite...$(NC)"
	@$(DC_DEV) exec api pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	@$(DC_DEV) exec api pytest tests/unit/ -v

test-integration: ## Run integration tests
	@$(DC_DEV) exec api pytest tests/integration/ -v --tb=long

test-ml: ## Run ML module tests
	@$(DC_DEV) exec api pytest tests/unit/ml/ -v -x

test-api: ## Run API tests
	@$(DC_DEV) exec api pytest tests/integration/test_api.py -v

test-geometry: ## Run spherical geometry tests
	@$(DC_DEV) exec api pytest tests/unit/ml/test_spherical_geometry.py -v

coverage: ## Run tests with coverage report
	@$(DC_DEV) exec api pytest tests/ --cov=app --cov=ml \
	    --cov-report=html:coverage_html --cov-report=term-missing

benchmark: ## Run performance benchmarks
	@$(DC_DEV) exec api python scripts/benchmark_inference.py

# ─── Code Quality ────────────────────────────────────────────────────────────
lint: ## Run linters (ruff, mypy, eslint)
	@echo "$(BLUE)Running linters...$(NC)"
	@$(DC_DEV) exec api ruff check .
	@$(DC_DEV) exec api mypy app/ ml/
	@cd frontend && npm run lint

format: ## Auto-format code (ruff, black, prettier)
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(DC_DEV) exec api ruff format .
	@$(DC_DEV) exec api black .
	@cd frontend && npm run format

type-check: ## Run mypy type checking
	@$(DC_DEV) exec api mypy app/ ml/ --ignore-missing-imports

security-scan: ## Run security scan (bandit, safety)
	@$(DC_DEV) exec api bandit -r app/ ml/ -f json -o reports/security.json
	@$(DC_DEV) exec api safety check

# ─── ML Operations ───────────────────────────────────────────────────────────
train-segmentation: ## Train segmentation model
	@echo "$(BLUE)Starting segmentation training...$(NC)"
	@$(DC_DEV) exec api python scripts/training/train_segmentation.py \
	    --config configs/training/segformer_config.yaml

train-ppe: ## Train PPE detection model
	@$(DC_DEV) exec api python scripts/training/train_ppe.py \
	    --config configs/training/ppe_config.yaml

export-onnx: ## Export models to ONNX
	@$(DC_DEV) exec api python scripts/export_models.py --format onnx

export-tensorrt: ## Export models to TensorRT
	@$(DC_DEV) exec api python scripts/export_models.py --format tensorrt

download-models: ## Download pretrained model weights
	@bash scripts/setup/download_models.sh

# ─── Data Management ─────────────────────────────────────────────────────────
data-pull: ## Pull dataset from DVC remote
	@dvc pull

data-push: ## Push dataset to DVC remote
	@dvc push

data-process: ## Run data preprocessing pipeline
	@$(DC_DEV) exec api python scripts/data/process_panoramas.py

create-sample-data: ## Create sample dataset for testing
	@$(DC_DEV) exec api python scripts/data/create_samples.py

# ─── Monitoring ──────────────────────────────────────────────────────────────
monitoring: ## Start monitoring stack (Prometheus + Grafana)
	@$(DC) -f docker/docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✓ Grafana: http://localhost:3001$(NC)"
	@echo "$(GREEN)✓ Prometheus: http://localhost:9090$(NC)"

monitoring-down: ## Stop monitoring stack
	@$(DC) -f docker/docker-compose.monitoring.yml down

# ─── Logs & Shell ────────────────────────────────────────────────────────────
logs: ## Tail all service logs
	@$(DC_DEV) logs -f

logs-api: ## Tail API logs
	@$(DC_DEV) logs -f api

logs-worker: ## Tail Celery worker logs
	@$(DC_DEV) logs -f celery_worker

shell: ## Open shell in API container
	@$(DC_DEV) exec api /bin/bash

shell-db: ## Open PostgreSQL shell
	@$(DC_DEV) exec postgres psql -U panoramic -d panoramic360

# ─── Deployment ──────────────────────────────────────────────────────────────
deploy-k8s: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	@bash scripts/deploy/deploy_k8s.sh

helm-install: ## Install Helm chart
	@helm install 360-engine ./kubernetes/helm/360-engine \
	    --namespace panoramic --create-namespace \
	    -f kubernetes/helm/360-engine/values-prod.yaml

helm-upgrade: ## Upgrade Helm release
	@helm upgrade 360-engine ./kubernetes/helm/360-engine \
	    --namespace panoramic \
	    -f kubernetes/helm/360-engine/values-prod.yaml

helm-uninstall: ## Uninstall Helm release
	@helm uninstall 360-engine --namespace panoramic

# ─── Cleanup ─────────────────────────────────────────────────────────────────
stop: ## Stop all services
	@$(DC_DEV) down

clean: ## Stop and remove all containers, volumes
	@$(DC_DEV) down -v --remove-orphans
	@$(DC_PROD) down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f

clean-all: clean ## Deep clean including images and caches
	@docker rmi $(docker images -q $(REGISTRY)/$(PROJECT)*) 2>/dev/null || true
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete

# ─── Documentation ────────────────────────────────────────────────────────────
docs: ## Build API documentation
	@$(DC_DEV) exec api python -m mkdocs build -d docs/site
	@echo "$(GREEN)✓ Docs built at docs/site/$(NC)"

docs-serve: ## Serve documentation locally
	@$(DC_DEV) exec api python -m mkdocs serve --dev-addr=0.0.0.0:8001

# ─── Utilities ───────────────────────────────────────────────────────────────
_wait-healthy:
	@echo "$(YELLOW)Waiting for services to be healthy...$(NC)"
	@sleep 5
	@echo "$(GREEN)✓ Services started$(NC)"

_print-urls-dev:
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║         Development URLs                    ║$(NC)"
	@echo "$(GREEN)║  Dashboard:  http://localhost:3000          ║$(NC)"
	@echo "$(GREEN)║  API Docs:   http://localhost:8000/docs     ║$(NC)"
	@echo "$(GREEN)║  MLflow:     http://localhost:5000          ║$(NC)"
	@echo "$(GREEN)║  MinIO:      http://localhost:9001          ║$(NC)"
	@echo "$(GREEN)║  Airflow:    http://localhost:8080          ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════╝$(NC)"

_print-urls-prod:
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║         Production URLs                     ║$(NC)"
	@echo "$(GREEN)║  Dashboard:  https://your-domain.com        ║$(NC)"
	@echo "$(GREEN)║  API:        https://api.your-domain.com    ║$(NC)"
	@echo "$(GREEN)║  Grafana:    http://localhost:3001          ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════╝$(NC)"

version: ## Show current version
	@echo "$(PROJECT) $(VERSION)"
