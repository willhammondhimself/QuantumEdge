"""
Quantum Approximate Optimization Algorithm (QAOA) for portfolio selection.

This module implements QAOA for solving combinatorial portfolio optimization
problems on classical hardware using quantum-inspired techniques.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import logging

from .quantum_circuit import QuantumCircuit
from .quantum_gates import QuantumGates

logger = logging.getLogger(__name__)


@dataclass
class QAOAResult:
    """Result from QAOA optimization."""
    optimal_portfolio: np.ndarray
    optimal_value: float
    optimal_params: np.ndarray
    optimization_history: List[float]
    probability_distribution: np.ndarray
    num_iterations: int
    success: bool


class QAOACircuit:
    """QAOA circuit implementation for portfolio optimization."""
    
    def __init__(self, num_assets: int, num_layers: int = 1):
        """
        Initialize QAOA circuit.
        
        Args:
            num_assets: Number of assets in portfolio
            num_layers: Number of QAOA layers (p parameter)
        """
        self.num_assets = num_assets
        self.num_qubits = num_assets  # One qubit per asset
        self.num_layers = num_layers
        self.gates = QuantumGates()
        
        # Number of parameters: 2 * num_layers (gamma and beta for each layer)
        self.num_params = 2 * num_layers
    
    def create_initial_state(self) -> QuantumCircuit:
        """
        Create initial superposition state |+⟩^⊗n.
        
        Returns:
            Quantum circuit in superposition state
        """
        circuit = QuantumCircuit(self.num_qubits)
        
        # Apply Hadamard to all qubits to create equal superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        return circuit
    
    def apply_cost_hamiltonian(
        self, 
        circuit: QuantumCircuit, 
        gamma: float,
        cost_matrix: np.ndarray
    ) -> None:
        """
        Apply cost Hamiltonian evolution exp(-i*gamma*H_C).
        
        Args:
            circuit: Quantum circuit to modify
            gamma: Cost Hamiltonian parameter
            cost_matrix: Cost matrix for optimization problem
        """
        # Apply single-qubit Z rotations for diagonal terms
        for i in range(self.num_qubits):
            if i < cost_matrix.shape[0]:
                circuit.rz(i, 2 * gamma * cost_matrix[i, i])
        
        # Apply two-qubit interactions for off-diagonal terms
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if i < cost_matrix.shape[0] and j < cost_matrix.shape[1]:
                    interaction_strength = cost_matrix[i, j]
                    if abs(interaction_strength) > 1e-10:
                        # ZZ interaction: exp(-i*gamma*J_ij*Z_i*Z_j)
                        circuit.cnot(i, j)
                        circuit.rz(j, 2 * gamma * interaction_strength)
                        circuit.cnot(i, j)
    
    def apply_mixer_hamiltonian(self, circuit: QuantumCircuit, beta: float) -> None:
        """
        Apply mixer Hamiltonian evolution exp(-i*beta*H_M).
        
        Args:
            circuit: Quantum circuit to modify
            beta: Mixer Hamiltonian parameter
        """
        # Apply X rotations to all qubits (standard mixer)
        for qubit in range(self.num_qubits):
            circuit.rx(qubit, 2 * beta)
    
    def create_qaoa_circuit(
        self, 
        params: np.ndarray, 
        cost_matrix: np.ndarray
    ) -> QuantumCircuit:
        """
        Create full QAOA circuit.
        
        Args:
            params: QAOA parameters [gamma_1, beta_1, gamma_2, beta_2, ...]
            cost_matrix: Cost matrix for optimization
            
        Returns:
            Complete QAOA circuit
        """
        if len(params) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameters, got {len(params)}")
        
        # Start with initial superposition
        circuit = self.create_initial_state()
        
        # Apply QAOA layers
        for layer in range(self.num_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            # Apply cost Hamiltonian
            self.apply_cost_hamiltonian(circuit, gamma, cost_matrix)
            
            # Apply mixer Hamiltonian
            self.apply_mixer_hamiltonian(circuit, beta)
        
        return circuit
    
    def create_constrained_mixer(self, circuit: QuantumCircuit, beta: float) -> None:
        """
        Apply constrained mixer that preserves constraint subspace.
        
        This mixer preserves the number of selected assets (for cardinality constraints).
        
        Args:
            circuit: Quantum circuit to modify
            beta: Mixer parameter
        """
        # Ring mixer: preserves Hamming weight
        for i in range(self.num_qubits):
            j = (i + 1) % self.num_qubits
            
            # Apply ring mixer operations
            circuit.cnot(i, j)
            circuit.ry(j, beta)
            circuit.cnot(i, j)


class PortfolioQAOA:
    """QAOA for portfolio optimization problems."""
    
    def __init__(
        self,
        num_assets: int,
        num_layers: int = 1,
        optimizer: str = 'COBYLA',
        max_iterations: int = 1000,
        constraint_type: str = 'none'
    ):
        """
        Initialize portfolio QAOA solver.
        
        Args:
            num_assets: Number of assets
            num_layers: Number of QAOA layers
            optimizer: Classical optimizer
            max_iterations: Maximum optimization iterations
            constraint_type: Type of constraints ('none', 'cardinality', 'budget')
        """
        self.num_assets = num_assets
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.constraint_type = constraint_type
        
        # Initialize QAOA circuit
        self.qaoa_circuit = QAOACircuit(num_assets, num_layers)
        
        # Optimization tracking
        self.optimization_history: List[float] = []
        self.iteration_count = 0
    
    def _create_cost_matrix(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """
        Create cost matrix for portfolio optimization.
        
        This formulates the mean-variance portfolio problem as a QUBO
        (Quadratic Unconstrained Binary Optimization) problem.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            risk_aversion: Risk aversion parameter
            
        Returns:
            Cost matrix for QAOA
        """
        n = self.num_assets
        cost_matrix = np.zeros((n, n))
        
        # Diagonal terms: -expected_returns + risk_aversion * diag(covariance)
        for i in range(n):
            cost_matrix[i, i] = -expected_returns[i] + risk_aversion * covariance_matrix[i, i]
        
        # Off-diagonal terms: risk_aversion * covariance
        for i in range(n):
            for j in range(i + 1, n):
                cost_matrix[i, j] = risk_aversion * covariance_matrix[i, j]
                cost_matrix[j, i] = cost_matrix[i, j]  # Symmetric
        
        return cost_matrix
    
    def _add_cardinality_constraint(
        self,
        cost_matrix: np.ndarray,
        target_assets: int,
        penalty_strength: float = 10.0
    ) -> np.ndarray:
        """
        Add cardinality constraint penalty to cost matrix.
        
        Args:
            cost_matrix: Original cost matrix
            target_assets: Target number of assets to select
            penalty_strength: Penalty coefficient
            
        Returns:
            Modified cost matrix with penalty terms
        """
        n = self.num_assets
        modified_matrix = cost_matrix.copy()
        
        # Add penalty for deviation from target cardinality
        # Penalty = penalty_strength * (sum(x_i) - target_assets)^2
        
        # Linear terms: penalty_strength * (2*target_assets - 1)
        for i in range(n):
            modified_matrix[i, i] += penalty_strength * (1 - 2 * target_assets)
        
        # Quadratic terms: penalty_strength * 2
        for i in range(n):
            for j in range(i + 1, n):
                modified_matrix[i, j] += 2 * penalty_strength
                modified_matrix[j, i] = modified_matrix[i, j]
        
        return modified_matrix
    
    def _cost_function(self, params: np.ndarray, cost_matrix: np.ndarray) -> float:
        """
        QAOA cost function to minimize.
        
        Args:
            params: QAOA parameters
            cost_matrix: Problem cost matrix
            
        Returns:
            Expected cost value
        """
        # Create QAOA circuit
        circuit = self.qaoa_circuit.create_qaoa_circuit(params, cost_matrix)
        
        # Create cost Hamiltonian
        hamiltonian = self._cost_matrix_to_hamiltonian(cost_matrix)
        
        # Calculate expectation value
        expectation = circuit.expectation_value(hamiltonian)
        
        # Track optimization progress
        self.optimization_history.append(expectation)
        self.iteration_count += 1
        
        if self.iteration_count % 50 == 0:
            logger.info(f"QAOA iteration {self.iteration_count}: cost = {expectation:.6f}")
        
        return expectation
    
    def _cost_matrix_to_hamiltonian(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Convert cost matrix to quantum Hamiltonian.
        
        Args:
            cost_matrix: QUBO cost matrix
            
        Returns:
            Hamiltonian matrix
        """
        n = self.num_assets
        hamiltonian = np.zeros((2**n, 2**n), dtype=complex)
        
        # Convert QUBO to Ising Hamiltonian
        # x_i ∈ {0,1} -> (1 - Z_i)/2
        
        gates = self.gates = QuantumGates()
        
        for i in range(n):
            for j in range(i, n):
                coeff = cost_matrix[i, j]
                
                if i == j:
                    # Diagonal term: coeff * (1 - Z_i)/2
                    pauli_ops = []
                    for k in range(n):
                        if k == i:
                            pauli_ops.append(gates.Z)
                        else:
                            pauli_ops.append(gates.I)
                    
                    z_op = gates.tensor_product(*pauli_ops)
                    hamiltonian += coeff * (np.eye(2**n) - z_op) / 2
                else:
                    # Off-diagonal term: coeff * (1 - Z_i)(1 - Z_j)/4
                    pauli_ops_ii = []
                    pauli_ops_ij = []
                    pauli_ops_ji = []
                    pauli_ops_jj = []
                    
                    for k in range(n):
                        if k == i:
                            pauli_ops_ii.append(gates.I)
                            pauli_ops_ij.append(gates.I)
                            pauli_ops_ji.append(gates.Z)
                            pauli_ops_jj.append(gates.Z)
                        elif k == j:
                            pauli_ops_ii.append(gates.I)
                            pauli_ops_ij.append(gates.Z)
                            pauli_ops_ji.append(gates.I)
                            pauli_ops_jj.append(gates.Z)
                        else:
                            pauli_ops_ii.append(gates.I)
                            pauli_ops_ij.append(gates.I)
                            pauli_ops_ji.append(gates.I)
                            pauli_ops_jj.append(gates.I)
                    
                    ii_op = gates.tensor_product(*pauli_ops_ii)
                    ij_op = gates.tensor_product(*pauli_ops_ij)
                    ji_op = gates.tensor_product(*pauli_ops_ji)
                    jj_op = gates.tensor_product(*pauli_ops_jj)
                    
                    hamiltonian += coeff * (ii_op - ij_op - ji_op + jj_op) / 4
        
        return hamiltonian
    
    def solve_portfolio_selection(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        cardinality_constraint: Optional[int] = None,
        initial_params: Optional[np.ndarray] = None
    ) -> QAOAResult:
        """
        Solve portfolio selection using QAOA.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            risk_aversion: Risk aversion parameter
            cardinality_constraint: Maximum number of assets to select
            initial_params: Initial QAOA parameters
            
        Returns:
            QAOA optimization result
        """
        logger.info(f"Starting QAOA portfolio optimization for {self.num_assets} assets")
        
        # Create cost matrix
        cost_matrix = self._create_cost_matrix(
            expected_returns, covariance_matrix, risk_aversion
        )
        
        # Add cardinality constraint if specified
        if cardinality_constraint is not None:
            cost_matrix = self._add_cardinality_constraint(
                cost_matrix, cardinality_constraint
            )
        
        # Initialize parameters
        if initial_params is None:
            # Good initialization: start with small random values
            initial_params = np.random.uniform(0, np.pi/4, size=self.qaoa_circuit.num_params)
        
        # Reset optimization tracking
        self.optimization_history = []
        self.iteration_count = 0
        
        # Optimize using classical optimizer
        if self.optimizer == 'COBYLA':
            result = minimize(
                fun=self._cost_function,
                x0=initial_params,
                args=(cost_matrix,),
                method='COBYLA',
                options={'maxiter': self.max_iterations, 'rhobeg': 0.1}
            )
        else:
            result = minimize(
                fun=self._cost_function,
                x0=initial_params,
                args=(cost_matrix,),
                method=self.optimizer,
                options={'maxiter': self.max_iterations}
            )
        
        # Extract solution from optimal QAOA state
        optimal_circuit = self.qaoa_circuit.create_qaoa_circuit(result.x, cost_matrix)
        probabilities = optimal_circuit.get_probabilities()
        
        # Find most probable portfolio
        optimal_portfolio = self._extract_portfolio(probabilities)
        
        logger.info(f"QAOA optimization completed. Optimal value: {result.fun:.6f}")
        
        return QAOAResult(
            optimal_portfolio=optimal_portfolio,
            optimal_value=result.fun,
            optimal_params=result.x,
            optimization_history=self.optimization_history,
            probability_distribution=probabilities,
            num_iterations=self.iteration_count,
            success=result.success
        )
    
    def _extract_portfolio(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Extract portfolio allocation from probability distribution.
        
        Args:
            probabilities: Quantum measurement probabilities
            
        Returns:
            Binary portfolio allocation
        """
        # Find most probable state
        max_prob_idx = np.argmax(probabilities)
        
        # Convert to binary string
        binary_string = format(max_prob_idx, f'0{self.num_assets}b')
        
        # Convert to portfolio allocation
        portfolio = np.array([int(bit) for bit in binary_string])
        
        return portfolio
    
    def sample_portfolios(
        self,
        params: np.ndarray,
        cost_matrix: np.ndarray,
        num_samples: int = 1000
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Sample multiple portfolios from QAOA distribution.
        
        Args:
            params: QAOA parameters
            cost_matrix: Problem cost matrix
            num_samples: Number of samples
            
        Returns:
            List of (portfolio, probability) tuples
        """
        # Create QAOA circuit
        circuit = self.qaoa_circuit.create_qaoa_circuit(params, cost_matrix)
        probabilities = circuit.get_probabilities()
        
        # Sample from distribution
        samples = []
        for _ in range(num_samples):
            # Sample state index
            state_idx = np.random.choice(len(probabilities), p=probabilities)
            
            # Convert to portfolio
            binary_string = format(state_idx, f'0{self.num_assets}b')
            portfolio = np.array([int(bit) for bit in binary_string])
            
            samples.append((portfolio, probabilities[state_idx]))
        
        return samples
    
    def evaluate_portfolio_cost(
        self,
        portfolio: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> float:
        """
        Evaluate cost of a given portfolio.
        
        Args:
            portfolio: Binary portfolio allocation
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            
        Returns:
            Portfolio cost (negative utility)
        """
        # Expected return
        expected_return = np.dot(portfolio, expected_returns)
        
        # Portfolio risk
        portfolio_risk = np.dot(portfolio, np.dot(covariance_matrix, portfolio))
        
        # Cost = -return + risk_aversion * risk
        cost = -expected_return + risk_aversion * portfolio_risk
        
        return cost