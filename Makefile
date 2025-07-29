# QuantumEdge Development Makefile
# Professional development workflow automation

.PHONY: help install install-dev clean test test-unit test-integration lint format type-check benchmark security docs dev-setup all-checks deploy-local

# Default target
help: ## Show this help message
	@echo "QuantumEdge Development Commands:"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Python environment
install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v -m unit

test-int:
	pytest tests/integration/ -v -m integration

test-bench:
	pytest benchmarks/ -v --benchmark-only

# Code quality
lint:
	flake8 src/ tests/
	pylint src/
	bandit -r src/ -f json -o bandit-report.json

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

# Cleaning
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf bandit-report.json

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development setup
dev-setup: install-dev
	@echo "Setting up development environment..."
	python -m ipykernel install --user --name quantumedge --display-name "QuantumEdge"
	@echo "Development environment ready!"

# Run the application
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Database migrations (placeholder for future use)
db-migrate:
	@echo "Running database migrations..."
	# alembic upgrade head

# Generate documentation
docs:
	cd docs && make html

# Performance profiling
profile:
	python -m cProfile -o profile.stats src/main.py
	snakeviz profile.stats