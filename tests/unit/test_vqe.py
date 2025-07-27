"""
Unit tests for VQE implementation.
"""

import numpy as np
import pytest
from src.quantum_algorithms.vqe import (
    QuantumVQE, ParameterizedCircuit, VQEResult, JAXVQEOptimizer, JAX_AVAILABLE
)


class TestParameterizedCircuit:
    """Test suite for ParameterizedCircuit class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_qubits = 3
        self.depth = 2
        self.circuit_template = ParameterizedCircuit(self.num_qubits, self.depth)
        
    def test_initialization(self):
        """Test circuit template initialization."""
        assert self.circuit_template.num_qubits == 3
        assert self.circuit_template.depth == 2
        
        # Check parameter count calculation
        expected_params = self.depth * (3 * self.num_qubits + self.num_qubits - 1)
        assert self.circuit_template.num_params == expected_params
    
    def test_ansatz_creation(self):
        """Test parameterized ansatz creation."""
        params = np.random.uniform(0, 2*np.pi, size=self.circuit_template.num_params)
        
        circuit = self.circuit_template.create_ansatz(params)
        
        # Check circuit is created with correct number of qubits
        assert circuit.num_qubits == self.num_qubits
        
        # Check that operations were added
        assert len(circuit.operations) > 0
        
        # State should be modified from initial |000⟩
        state = circuit.get_state_vector()
        initial_state = np.zeros(2**self.num_qubits)
        initial_state[0] = 1
        assert not np.allclose(state, initial_state)
    
    def test_invalid_parameter_count(self):
        """Test error handling for wrong parameter count."""
        wrong_params = np.random.uniform(0, 2*np.pi, size=5)  # Wrong size
        
        with pytest.raises(ValueError, match="Expected .* parameters"):
            self.circuit_template.create_ansatz(wrong_params)
    
    def test_efficient_ansatz(self):
        """Test efficient ansatz creation."""
        # For efficient ansatz, we need fewer parameters
        params = np.random.uniform(0, 2*np.pi, size=self.depth * self.num_qubits)
        
        circuit = self.circuit_template.create_efficient_ansatz(params)
        
        # Check circuit is created
        assert circuit.num_qubits == self.num_qubits
        assert len(circuit.operations) > 0


class TestQuantumVQE:
    """Test suite for QuantumVQE class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_assets = 4
        self.vqe = QuantumVQE(
            num_assets=self.num_assets,
            depth=1,  # Small depth for testing
            max_iterations=50  # Few iterations for testing
        )
    
    def test_initialization(self):
        """Test VQE initialization."""
        assert self.vqe.num_assets == 4
        assert self.vqe.num_qubits == 2  # ceil(log2(4))
        assert self.vqe.depth == 1
        assert self.vqe.optimizer == 'COBYLA'
    
    def test_covariance_to_hamiltonian(self):
        """Test covariance matrix to Hamiltonian conversion."""
        # Create simple covariance matrix
        cov_matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        
        hamiltonian = self.vqe._covariance_to_hamiltonian(cov_matrix)
        
        # Check dimensions
        expected_dim = 2 ** self.vqe.num_qubits
        assert hamiltonian.shape == (expected_dim, expected_dim)
        
        # Should be Hermitian
        assert np.allclose(hamiltonian, hamiltonian.conj().T)
    
    def test_cost_function(self):
        """Test VQE cost function."""
        # Create test Hamiltonian
        hamiltonian = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        # Random parameters
        params = np.random.uniform(0, 2*np.pi, size=self.vqe.circuit_template.num_params)
        
        # Calculate cost
        cost = self.vqe._cost_function(params, hamiltonian)
        
        # Should be a real number
        assert isinstance(cost, (int, float, np.floating))
        assert not np.isnan(cost)
    
    def test_extract_portfolio_weights(self):
        """Test portfolio weight extraction."""
        # Test probabilities
        probabilities = np.array([0.4, 0.3, 0.2, 0.1])
        num_assets = 3
        
        weights = self.vqe._extract_portfolio_weights(probabilities, num_assets)
        
        # Check properties
        assert len(weights) == num_assets
        assert np.allclose(np.sum(weights), 1.0)  # Should sum to 1
        assert np.all(weights >= 0)  # Should be non-negative
    
    def test_extract_portfolio_weights_zero_sum(self):
        """Test portfolio weight extraction with zero probabilities."""
        probabilities = np.array([0.0, 0.0, 0.0, 0.0])
        num_assets = 3
        
        weights = self.vqe._extract_portfolio_weights(probabilities, num_assets)
        
        # Should fallback to equal weights
        expected_weights = np.ones(num_assets) / num_assets
        assert np.allclose(weights, expected_weights)
    
    def test_solve_eigenportfolio_small(self):
        """Test VQE eigenportfolio solving with small problem."""
        # Create simple 2x2 covariance matrix
        cov_matrix = np.array([
            [1.0, 0.3],
            [0.3, 1.0]
        ])
        
        # Solve with few iterations for testing
        self.vqe.max_iterations = 20
        result = self.vqe.solve_eigenportfolio(cov_matrix)
        
        # Check result structure
        assert isinstance(result, VQEResult)
        assert isinstance(result.eigenvalue, (int, float, np.floating))
        assert len(result.eigenvector) == 2
        assert np.allclose(np.sum(result.eigenvector), 1.0)
        assert len(result.optimization_history) > 0
        assert result.num_iterations > 0
    
    @pytest.mark.slow
    def test_solve_eigenportfolio_convergence(self):
        """Test VQE convergence with more iterations."""
        # Create positive definite covariance matrix
        np.random.seed(42)
        A = np.random.randn(3, 3)
        cov_matrix = A @ A.T
        
        vqe = QuantumVQE(
            num_assets=3,
            depth=2,
            max_iterations=100,
            optimizer='COBYLA'
        )
        
        result = vqe.solve_eigenportfolio(cov_matrix)
        
        # Check convergence
        assert result.success or len(result.optimization_history) > 50
        
        # Check eigenvalue is reasonable
        eigenvalues = np.linalg.eigvals(cov_matrix)
        min_eigenvalue = np.min(eigenvalues)
        assert result.eigenvalue >= min_eigenvalue - 1.0  # Some tolerance
    
    def test_multiple_eigenportfolios(self):
        """Test computation of multiple eigenportfolios."""
        # Simple covariance matrix
        cov_matrix = np.array([
            [2.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.5]
        ])
        
        # Small numbers for testing
        vqe = QuantumVQE(
            num_assets=3,
            depth=1,
            max_iterations=20
        )
        
        results = vqe.compute_multiple_eigenportfolios(
            cov_matrix,
            num_eigenportfolios=2,
            num_random_starts=2
        )
        
        # Check results
        assert len(results) == 2
        assert all(isinstance(r, VQEResult) for r in results)
        
        # Should be sorted by eigenvalue
        assert results[0].eigenvalue <= results[1].eigenvalue
    
    def test_gradient_cost_function(self):
        """Test gradient computation using parameter-shift rule."""
        # Simple Hamiltonian
        hamiltonian = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        # Small parameter vector for testing
        params = np.array([0.5, 1.0, 1.5])[:self.vqe.circuit_template.num_params]
        if len(params) < self.vqe.circuit_template.num_params:
            params = np.random.uniform(0, 2*np.pi, size=self.vqe.circuit_template.num_params)
        
        # Calculate gradients
        gradients = self.vqe._gradient_cost_function(params, hamiltonian)
        
        # Check gradient properties
        assert len(gradients) == len(params)
        assert np.all(np.isfinite(gradients))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXVQEOptimizer:
    """Test suite for JAX VQE optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = JAXVQEOptimizer(num_qubits=3, depth=2)
    
    def test_initialization(self):
        """Test JAX optimizer initialization."""
        assert self.optimizer.num_qubits == 3
        assert self.optimizer.depth == 2
    
    @pytest.mark.slow
    def test_jax_optimization(self):
        """Test JAX-based optimization."""
        # Simple covariance matrix
        cov_matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        
        result = self.optimizer.optimize_with_jax(
            cov_matrix,
            learning_rate=0.1,
            num_steps=50  # Small number for testing
        )
        
        # Check result structure
        assert isinstance(result, VQEResult)
        assert len(result.optimization_history) == 50
        assert result.num_iterations == 50
        assert result.success


def test_jax_optimizer_without_jax():
    """Test JAX optimizer error when JAX not available."""
    if not JAX_AVAILABLE:
        with pytest.raises(ImportError, match="JAX is required"):
            JAXVQEOptimizer(num_qubits=3, depth=2)


class TestVQEResult:
    """Test suite for VQEResult dataclass."""
    
    def test_vqe_result_creation(self):
        """Test VQEResult creation."""
        result = VQEResult(
            eigenvalue=1.5,
            eigenvector=np.array([0.6, 0.4]),
            optimal_params=np.array([1.0, 2.0, 3.0]),
            optimization_history=[2.0, 1.8, 1.5],
            num_iterations=100,
            success=True
        )
        
        assert result.eigenvalue == 1.5
        assert np.array_equal(result.eigenvector, [0.6, 0.4])
        assert result.num_iterations == 100
        assert result.success is True


if __name__ == '__main__':
    pytest.main([__file__])