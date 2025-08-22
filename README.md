# QuantumEdge - Quantum-Inspired Portfolio Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-comprehensive-green.svg)](#testing)

A portfolio optimization framework implementing quantum-inspired algorithms alongside classical methods. Features a production-ready FastAPI backend with comprehensive testing and modular architecture.

## Overview

QuantumEdge explores quantum computing applications in financial portfolio optimization by implementing VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm) through classical simulation. The framework includes classical mean-variance optimization for comparison and evaluation.

## Features

### Core Algorithms
- **Variational Quantum Eigensolver (VQE)** - Eigenportfolio discovery using parameterized quantum circuits
- **Quantum Approximate Optimization Algorithm (QAOA)** - Combinatorial portfolio selection with cardinality constraints
- **Classical Mean-Variance Optimization** - Multiple objectives including Sharpe ratio, CVaR, and Sortino ratio
- **Constraint Handling** - Long-only, box constraints, and cardinality restrictions

### API & Infrastructure
- **FastAPI Backend** - Async endpoints with WebSocket support for real-time portfolio updates
- **Production Ready** - Docker containerization, monitoring, and comprehensive error handling
- **Automated Testing** - Comprehensive test suite with unit, integration, and performance tests
- **Documentation** - Type hints, docstrings, and comprehensive test coverage

## Installation

```bash
git clone https://github.com/willhammond/QuantumEdge.git
cd QuantumEdge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest tests/

# Start API server
uvicorn src.api.main:app --reload
```

## Usage

```python
from src.optimization import MeanVarianceOptimizer
from src.quantum_algorithms import VQE, QAOA

# Classical optimization
optimizer = MeanVarianceOptimizer()
result = optimizer.optimize(
    returns,
    objective=ObjectiveType.MAXIMIZE_SHARPE,
    constraints={"long_only": True}
)

# Quantum-inspired approaches
vqe = VQE(num_qubits=4, depth=2)
eigenportfolio = vqe.find_eigenportfolio(covariance_matrix)

qaoa = QAOA(num_assets=10, cardinality=5)
selection = qaoa.optimize(expected_returns, risk_matrix)
```

## Technology Stack

- **Optimization**: NumPy, SciPy, CVXPY, JAX (optional)
- **Backend**: FastAPI, Pydantic, AsyncIO
- **Testing**: Pytest, GitHub Actions
- **Infrastructure**: Docker, Docker Compose
- **Development**: Black, Flake8, MyPy, Pre-commit hooks

## API Endpoints

- `POST /optimize/classical` - Classical mean-variance optimization
- `POST /optimize/vqe` - VQE-based eigenportfolio optimization  
- `POST /optimize/qaoa` - QAOA combinatorial selection
- `WS /portfolio/stream` - Real-time portfolio updates
- `GET /health` - Health check and system status

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run development server
make dev-setup
make run

# Code quality checks
make lint
make test
make benchmark
```

## Testing

- **Unit Tests** - Algorithm implementations and core functionality
- **Integration Tests** - API endpoints and database interactions
- **Performance Tests** - Benchmarking and regression detection
- **Test Coverage** - Comprehensive test scenarios covering core functionality and edge cases

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code standards and style requirements
- Testing expectations and coverage
- Pull request process and review criteria
- Development workflow and best practices

## Security

Security is important to us. Please see [SECURITY.md](SECURITY.md) for:
- Vulnerability reporting procedures
- Security best practices
- Supported versions and updates
- Data handling guidelines

## License

MIT License - see [LICENSE](LICENSE) for details.
