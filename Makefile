.PHONY: help install dev test lint format security build clean docs docker-build docker-run \
        autonomous-discover autonomous-execute autonomous-cycle value-report \
        sbom compliance pre-commit-setup health-check

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
	@echo "Setting up Terragon autonomous SDLC..."
	@mkdir -p .terragon
	@echo "✅ Development environment ready. Run 'make autonomous-discover' to start."

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
	make autonomous-discover
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

# 🤖 TERRAGON AUTONOMOUS SDLC OPERATIONS 🤖

autonomous-discover: ## Discover new work items using value-based analysis
	@echo "🔍 Running autonomous value discovery..."
	python3 .terragon/backlog-discovery.py
	@echo "📊 Discovery complete. Check AUTONOMOUS_BACKLOG.md for results."

autonomous-execute: ## Execute the highest-value work item
	@echo "⚡ Running autonomous execution cycle..."
	python3 .terragon/autonomous-executor.py
	@echo "✅ Execution cycle complete."

autonomous-cycle: autonomous-discover autonomous-execute ## Full autonomous cycle: discover + execute
	@echo "🔄 Full autonomous SDLC cycle completed."
	@echo "📈 Value delivered. Check docs/automation/ for generated improvements."

value-report: ## Generate comprehensive value metrics report
	@echo "📊 Generating value delivery report..."
	@python3 -c "import json; metrics=json.load(open('.terragon/value-metrics.json')); print(f'Completed Tasks: {metrics.get(\"valueDelivered\", {}).get(\"completedTasks\", 0)}'); print(f'Total Score: {metrics.get(\"valueDelivered\", {}).get(\"totalScore\", 0):.1f}'); print(f'Items in Backlog: {metrics.get(\"backlogMetrics\", {}).get(\"totalItems\", 0)}')"

sbom: ## Generate Software Bill of Materials
	python scripts/generate_sbom.py

compliance: ## Run compliance checks
	python scripts/compliance_check.py

health-check: ## Run comprehensive repository health check
	@echo "🏥 Running repository health check..."
	@echo "1. Test coverage check..."
	@coverage report --show-missing | tail -1 || echo "No coverage data"
	@echo "2. Security posture..."
	@test -f security_results.json && echo "✅ Security scan results available" || echo "⚠️  Run 'make security' first"
	@echo "3. Autonomous backlog..."
	@test -f AUTONOMOUS_BACKLOG.md && echo "✅ Autonomous backlog active" || echo "⚠️  Run 'make autonomous-discover' first"
	@echo "4. Pre-commit hooks..."
	@test -f .git/hooks/pre-commit && echo "✅ Pre-commit hooks installed" || echo "⚠️  Run 'make install-hooks' first"