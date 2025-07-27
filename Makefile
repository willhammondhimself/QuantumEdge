.PHONY: help install install-dev test lint format type-check clean docker-build docker-up docker-down

# Default target
help:
	@echo "QuantumEdge Development Commands"
	@echo "================================"
	@echo "install       - Install production dependencies"
	@echo "install-dev   - Install development dependencies"
	@echo "test          - Run all tests"
	@echo "test-unit     - Run unit tests only"
	@echo "test-int      - Run integration tests only"
	@echo "lint          - Run linting checks"
	@echo "format        - Format code with black and isort"
	@echo "type-check    - Run mypy type checking"
	@echo "clean         - Clean build artifacts"
	@echo "docker-build  - Build Docker images"
	@echo "docker-up     - Start Docker services"
	@echo "docker-down   - Stop Docker services"
	@echo "dev-setup     - Complete development setup"

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