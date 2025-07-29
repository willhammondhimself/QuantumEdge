# QuantumEdge - Quantum-Inspired Portfolio Optimization Framework 🚀

[![Build Status](https://github.com/willhammond/QuantumEdge/workflows/CI/badge.svg)](https://github.com/willhammond/QuantumEdge/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-focused implementation of quantum-inspired algorithms for portfolio optimization, demonstrating modern software engineering practices and quantitative finance concepts.

## 🎯 Project Overview

QuantumEdge explores the intersection of quantum computing concepts and financial portfolio optimization by implementing classical simulations of quantum algorithms. This project serves as a foundation for researching advanced optimization techniques beyond traditional mean-variance approaches.

### What This Project Demonstrates

- **Quantum Algorithm Implementation**: Working implementations of VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm) using classical simulation
- **Modern API Architecture**: Production-ready FastAPI backend with WebSocket support for real-time portfolio updates
- **Software Engineering Best Practices**: Comprehensive CI/CD pipeline, testing framework, and professional development workflow
- **Quantitative Finance Foundation**: Classical mean-variance optimization with multiple objective functions (Sharpe, CVaR, Sortino)

## 🛠️ Technical Implementation

### Core Algorithms

**1. Variational Quantum Eigensolver (VQE)**
- Classical simulation of quantum circuits for eigenportfolio discovery
- Parameterized quantum circuits with customizable depth
- Integration with JAX for automatic differentiation (when available)
- Scipy optimization for parameter tuning

**2. Quantum Approximate Optimization Algorithm (QAOA)**
- Implementation for combinatorial portfolio selection problems
- Customizable mixer and problem Hamiltonians
- Support for cardinality constraints

**3. Classical Optimization Baseline**
- CVXPY-based mean-variance optimization
- Multiple objective functions: Sharpe ratio, CVaR, Sortino ratio
- Comprehensive constraint handling (long-only, box constraints, cardinality)

### Technology Stack

- **Backend**: FastAPI with async support and WebSocket endpoints
- **Optimization**: CVXPY, NumPy, SciPy, JAX (optional)
- **Testing**: Pytest with comprehensive unit and integration tests
- **CI/CD**: GitHub Actions with automated testing and code quality checks
- **Documentation**: Well-documented code with type hints and docstrings

## 🚀 Installation & Usage

```bash
# Clone repository
git clone https://github.com/willhammond/QuantumEdge.git
cd QuantumEdge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start API server
uvicorn src.api.main:app --reload
```

### Example Usage

```python
from src.optimization import MeanVarianceOptimizer
from src.quantum_algorithms import VQE

# Classical optimization
optimizer = MeanVarianceOptimizer()
result = optimizer.optimize(
    returns,
    objective=ObjectiveType.MAXIMIZE_SHARPE,
    constraints={"long_only": True}
)

# Quantum-inspired approach
vqe = VQE(num_qubits=4)
eigenportfolio = vqe.find_eigenportfolio(covariance_matrix)
```

## 📊 Current Capabilities

### Implemented Features
- ✅ VQE implementation with parameterized quantum circuits
- ✅ QAOA for portfolio selection problems
- ✅ Classical mean-variance optimization with multiple objectives
- ✅ FastAPI backend with RESTful endpoints
- ✅ Comprehensive test suite with 80%+ coverage
- ✅ Professional CI/CD pipeline
- ✅ Type hints and documentation

### Research Opportunities
- 🔬 Performance comparison between quantum-inspired and classical methods
- 🔬 Crisis period robustness analysis (implementation pending)
- 🔬 Real-world backtesting framework (in development)
- 🔬 GPU acceleration for larger portfolios

## 🎓 Learning & Development

This project demonstrates proficiency in:
- **Quantum Computing Concepts**: Understanding of VQE, QAOA, and quantum circuit design
- **Financial Engineering**: Portfolio optimization theory and risk management
- **Software Engineering**: Clean architecture, testing, CI/CD, and API design
- **Scientific Computing**: NumPy, SciPy, and optimization algorithms

## 🤝 Contributing

This is an active research project. Contributions are welcome in:
- Algorithm improvements and optimizations
- Additional test coverage and edge cases
- Documentation and examples
- Performance benchmarking tools

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📈 Future Roadmap

- [ ] Implement backtesting framework with historical data
- [ ] Add performance benchmarking suite
- [ ] Integrate real market data feeds
- [ ] Explore hybrid classical-quantum algorithms
- [ ] Add GPU acceleration support
- [ ] Implement additional risk measures

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note**: This is a research project exploring quantum-inspired approaches to portfolio optimization. Performance claims require further validation with real market data and comprehensive backtesting.