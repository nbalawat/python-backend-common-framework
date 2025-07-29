.PHONY: help install install-dev test test-cov lint format type-check clean build publish docs

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python3
UV := uv

# Directories
MODULES := core cloud k8s testing events pipelines workflows llm agents data

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install all modules in development mode
	$(UV) sync

install-dev: ## Install all modules with development dependencies
	$(UV) sync --all-extras

test: ## Run tests for all modules
	@for module in $(MODULES); do \
		echo "Testing $$module..."; \
		cd $$module && $(UV) run pytest || exit 1; \
		cd ..; \
	done

test-cov: ## Run tests with coverage
	@for module in $(MODULES); do \
		echo "Testing $$module with coverage..."; \
		cd $$module && $(UV) run pytest --cov=src --cov-report=html --cov-report=term || exit 1; \
		cd ..; \
	done

test-module: ## Run tests for specific module (usage: make test-module MODULE=core)
	@if [ -z "$(MODULE)" ]; then \
		echo "Please specify MODULE=<module_name>"; \
		exit 1; \
	fi
	cd $(MODULE) && $(UV) run pytest

lint: ## Run linting checks
	$(UV) run ruff check .
	$(UV) run black --check .

format: ## Format code
	$(UV) run black .
	$(UV) run ruff check --fix .

type-check: ## Run type checking
	@for module in $(MODULES); do \
		echo "Type checking $$module..."; \
		cd $$module && $(UV) run mypy src || exit 1; \
		cd ..; \
	done

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +

build: clean ## Build all modules
	@for module in $(MODULES); do \
		echo "Building $$module..."; \
		cd $$module && $(UV) build || exit 1; \
		cd ..; \
	done

publish: ## Publish all modules to PyPI
	@for module in $(MODULES); do \
		echo "Publishing $$module..."; \
		cd $$module && $(UV) publish || exit 1; \
		cd ..; \
	done

publish-test: ## Publish all modules to TestPyPI
	@for module in $(MODULES); do \
		echo "Publishing $$module to TestPyPI..."; \
		cd $$module && $(UV) publish --registry testpypi || exit 1; \
		cd ..; \
	done

docs: ## Build documentation
	cd docs && $(UV) run mkdocs build

docs-serve: ## Serve documentation locally
	cd docs && $(UV) run mkdocs serve

deps-update: ## Update dependencies
	$(UV) sync --upgrade

deps-check: ## Check for outdated dependencies
	$(UV) pip list --outdated

security-check: ## Run security checks
	$(UV) run bandit -r */src
	$(UV) run safety check

pre-commit: lint type-check test ## Run all pre-commit checks

setup-dev: ## Setup development environment
	$(UV) venv
	$(UV) sync --all-extras
	$(UV) run pre-commit install

# Module-specific targets
core-test:
	cd core && $(UV) run pytest

cloud-test:
	cd cloud && $(UV) run pytest

k8s-test:
	cd k8s && $(UV) run pytest

# Docker targets
docker-build: ## Build Docker images
	docker build -t python-commons:latest .

docker-test: ## Run tests in Docker
	docker run --rm python-commons:latest make test

# CI/CD helpers
ci-test: ## Run CI tests
	$(UV) run pytest --cov=src --cov-report=xml --cov-report=term

ci-lint: ## Run CI linting
	$(UV) run ruff check . --format=github
	$(UV) run black --check .

ci-security: ## Run CI security checks
	$(UV) run bandit -r */src -f json -o bandit-report.json
	$(UV) run safety check --json