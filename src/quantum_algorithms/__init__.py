"""
Quantum-inspired algorithms for portfolio optimization.

This module contains implementations of:
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Tensor network based quantum circuit simulation
"""

from typing import List

from .quantum_gates import QuantumGates, create_controlled_gate
from .quantum_circuit import QuantumCircuit, Operation
from .vqe import QuantumVQE, ParameterizedCircuit, VQEResult, JAX_AVAILABLE
from .qaoa import PortfolioQAOA, QAOACircuit, QAOAResult

__all__: List[str] = [
    "QuantumGates",
    "create_controlled_gate", 
    "QuantumCircuit",
    "Operation",
    "QuantumVQE",
    "ParameterizedCircuit", 
    "VQEResult",
    "JAX_AVAILABLE",
    "PortfolioQAOA",
    "QAOACircuit",
    "QAOAResult"
]