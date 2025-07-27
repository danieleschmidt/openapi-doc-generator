.PHONY: help install dev test lint format security build clean docs docker-build docker-run

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in production mode
	pip install -e .

dev: ## Install package in development mode with all dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run all tests with coverage
	pytest -v --cov=src --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage for faster feedback
	pytest -v -x

test-unit: ## Run only unit tests
	pytest -v -m "unit"

test-integration: ## Run only integration tests
	pytest -v -m "integration"

test-e2e: ## Run end-to-end tests
	pytest -v -m "e2e"

test-performance: ## Run performance tests
	pytest -v -m "performance"

test-security: ## Run security tests
	pytest -v -m "security"

test-contract: ## Run contract tests
	pytest -v -m "contract"

test-mutation: ## Run mutation testing
	mutmut run

test-parallel: ## Run tests in parallel
	pytest -v -n auto

lint: ## Run all linting checks
	ruff check .
	ruff format --check .
	mypy src
	bandit -r src

format: ## Format all code
	ruff format .
	ruff check --fix .

security: ## Run security scans
	bandit -r src -f json -o security_results.json
	safety check
	pip-audit

build: ## Build the package
	python -m build

clean: ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Generate documentation
	mkdir -p docs/generated
	openapi-doc-generator --app examples/app.py --format markdown --output docs/generated/API.md
	openapi-doc-generator --app examples/app.py --format openapi --output docs/generated/openapi.json
	openapi-doc-generator --app examples/app.py --format html --output docs/generated/playground.html

docker-build: ## Build Docker image
	docker build -t openapi-doc-generator:latest .

docker-run: ## Run application in Docker container
	docker run --rm -v $(PWD):/workspace openapi-doc-generator:latest /workspace/examples/app.py --format openapi

ci: ## Run full CI pipeline locally
	make lint
	make test
	make security
	make build

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

performance: ## Run performance benchmarks
	pytest tests/test_performance_benchmarks.py -v --benchmark-only

complexity: ## Check code complexity
	radon cc src -n C -s

install-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

update-deps: ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -e .[dev]
	pre-commit autoupdate

release-patch: ## Bump patch version and create release
	bump2version patch
	git push origin --tags

release-minor: ## Bump minor version and create release
	bump2version minor
	git push origin --tags

release-major: ## Bump major version and create release
	bump2version major
	git push origin --tags