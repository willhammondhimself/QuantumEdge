"""
Unit tests for QAOA implementation.
"""

import numpy as np
import pytest
from src.quantum_algorithms.qaoa import PortfolioQAOA, QAOACircuit, QAOAResult


class TestQAOACircuit:
    """Test suite for QAOACircuit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_assets = 3
        self.num_layers = 2
        self.qaoa_circuit = QAOACircuit(self.num_assets, self.num_layers)

    def test_initialization(self):
        """Test QAOA circuit initialization."""
        assert self.qaoa_circuit.num_assets == 3
        assert self.qaoa_circuit.num_qubits == 3
        assert self.qaoa_circuit.num_layers == 2
        assert self.qaoa_circuit.num_params == 4  # 2 * num_layers

    def test_initial_state_creation(self):
        """Test initial superposition state creation."""
        circuit = self.qaoa_circuit.create_initial_state()

        # Should have 3 qubits
        assert circuit.num_qubits == 3

        # State should be equal superposition |+++⟩
        state = circuit.get_state_vector()
        expected_state = np.ones(8) / np.sqrt(8)  # |+++⟩
        assert np.allclose(state, expected_state)

    def test_cost_hamiltonian_application(self):
        """Test cost Hamiltonian application."""
        circuit = self.qaoa_circuit.create_initial_state()

        # Simple cost matrix
        cost_matrix = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]])

        initial_state = circuit.get_state_vector()

        # Apply cost Hamiltonian
        self.qaoa_circuit.apply_cost_hamiltonian(
            circuit, gamma=0.5, cost_matrix=cost_matrix
        )

        # State should be modified
        final_state = circuit.get_state_vector()
        assert not np.allclose(initial_state, final_state)

    def test_mixer_hamiltonian_application(self):
        """Test mixer Hamiltonian application."""
        circuit = self.qaoa_circuit.create_initial_state()
        initial_state = circuit.get_state_vector()

        # Apply mixer Hamiltonian
        self.qaoa_circuit.apply_mixer_hamiltonian(circuit, beta=0.3)

        # State should be modified
        final_state = circuit.get_state_vector()
        assert not np.allclose(initial_state, final_state)

    def test_full_qaoa_circuit_creation(self):
        """Test full QAOA circuit creation."""
        params = np.array([0.5, 0.3, 0.7, 0.2])  # [gamma1, beta1, gamma2, beta2]
        cost_matrix = np.eye(3)  # Simple identity matrix

        circuit = self.qaoa_circuit.create_qaoa_circuit(params, cost_matrix)

        # Should have operations applied
        assert len(circuit.operations) > 0

        # State should be non-trivial
        state = circuit.get_state_vector()
        uniform_state = np.ones(8) / np.sqrt(8)
        assert not np.allclose(state, uniform_state)

    def test_invalid_parameter_count(self):
        """Test error handling for wrong parameter count."""
        wrong_params = np.array([0.5, 0.3, 0.7])  # Wrong size
        cost_matrix = np.eye(3)

        with pytest.raises(ValueError, match="Expected .* parameters"):
            self.qaoa_circuit.create_qaoa_circuit(wrong_params, cost_matrix)

    def test_constrained_mixer(self):
        """Test constrained mixer application."""
        circuit = self.qaoa_circuit.create_initial_state()
        initial_state = circuit.get_state_vector()

        # Apply constrained mixer
        self.qaoa_circuit.create_constrained_mixer(circuit, beta=0.3)

        # State should be modified
        final_state = circuit.get_state_vector()
        assert not np.allclose(initial_state, final_state)


class TestPortfolioQAOA:
    """Test suite for PortfolioQAOA class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_assets = 3
        self.qaoa = PortfolioQAOA(
            num_assets=self.num_assets,
            num_layers=1,  # Simple for testing
            max_iterations=50,  # Few iterations for testing
        )

    def test_initialization(self):
        """Test QAOA initialization."""
        assert self.qaoa.num_assets == 3
        assert self.qaoa.num_layers == 1
        assert self.qaoa.optimizer == "COBYLA"
        assert self.qaoa.constraint_type == "none"

    def test_cost_matrix_creation(self):
        """Test cost matrix creation for portfolio optimization."""
        expected_returns = np.array([0.1, 0.15, 0.12])
        covariance_matrix = np.array(
            [[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.06]]
        )
        risk_aversion = 2.0

        cost_matrix = self.qaoa._create_cost_matrix(
            expected_returns, covariance_matrix, risk_aversion
        )

        # Check dimensions
        assert cost_matrix.shape == (3, 3)

        # Should be symmetric
        assert np.allclose(cost_matrix, cost_matrix.T)

        # Diagonal elements should include return and risk terms
        for i in range(3):
            expected_diag = (
                -expected_returns[i] + risk_aversion * covariance_matrix[i, i]
            )
            assert np.isclose(cost_matrix[i, i], expected_diag)

    def test_cardinality_constraint_addition(self):
        """Test cardinality constraint penalty addition."""
        cost_matrix = np.eye(3)
        target_assets = 2
        penalty_strength = 5.0

        modified_matrix = self.qaoa._add_cardinality_constraint(
            cost_matrix, target_assets, penalty_strength
        )

        # Should be different from original
        assert not np.allclose(cost_matrix, modified_matrix)

        # Should still be symmetric
        assert np.allclose(modified_matrix, modified_matrix.T)

    def test_cost_function(self):
        """Test QAOA cost function."""
        params = np.array([0.5, 0.3])  # [gamma, beta] for 1 layer
        cost_matrix = np.eye(3)

        cost = self.qaoa._cost_function(params, cost_matrix)

        # Should be a real number
        assert isinstance(cost, (int, float, np.floating))
        assert not np.isnan(cost)

    def test_extract_portfolio(self):
        """Test portfolio extraction from probabilities."""
        # Create probability distribution favoring state |101⟩ (index 5)
        probabilities = np.zeros(8)
        probabilities[5] = 0.8  # |101⟩
        probabilities[0] = 0.2  # |000⟩

        portfolio = self.qaoa._extract_portfolio(probabilities)

        # Should extract |101⟩ = [1, 0, 1]
        expected_portfolio = np.array([1, 0, 1])
        assert np.array_equal(portfolio, expected_portfolio)

    def test_portfolio_cost_evaluation(self):
        """Test portfolio cost evaluation."""
        portfolio = np.array([1, 0, 1])  # Select assets 0 and 2
        expected_returns = np.array([0.1, 0.15, 0.12])
        covariance_matrix = np.array(
            [[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.06]]
        )
        risk_aversion = 1.0

        cost = self.qaoa.evaluate_portfolio_cost(
            portfolio, expected_returns, covariance_matrix, risk_aversion
        )

        # Should be a finite real number
        assert isinstance(cost, (int, float, np.floating))
        assert np.isfinite(cost)

        # Manual calculation check
        expected_return = (
            portfolio[0] * expected_returns[0] + portfolio[2] * expected_returns[2]
        )
        risk = (
            portfolio[0] ** 2 * covariance_matrix[0, 0]
            + portfolio[2] ** 2 * covariance_matrix[2, 2]
            + 2 * portfolio[0] * portfolio[2] * covariance_matrix[0, 2]
        )
        expected_cost = -expected_return + risk_aversion * risk

        assert np.isclose(cost, expected_cost)

    def test_cost_matrix_to_hamiltonian(self):
        """Test conversion from cost matrix to Hamiltonian."""
        cost_matrix = np.array([[1.0, 0.5], [0.5, 2.0]])

        # Create 2-asset QAOA for this test
        qaoa_2 = PortfolioQAOA(num_assets=2, num_layers=1)
        hamiltonian = qaoa_2._cost_matrix_to_hamiltonian(cost_matrix)

        # Should be 4x4 matrix for 2 qubits
        assert hamiltonian.shape == (4, 4)

        # Should be Hermitian
        assert np.allclose(hamiltonian, hamiltonian.conj().T)

    def test_solve_portfolio_selection_small(self):
        """Test portfolio selection with small problem."""
        expected_returns = np.array([0.1, 0.15, 0.08])
        covariance_matrix = np.array(
            [[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.03]]
        )
        risk_aversion = 1.0

        # Small number of iterations for testing
        self.qaoa.max_iterations = 20

        result = self.qaoa.solve_portfolio_selection(
            expected_returns, covariance_matrix, risk_aversion
        )

        # Check result structure
        assert isinstance(result, QAOAResult)
        assert len(result.optimal_portfolio) == 3
        assert all(x in [0, 1] for x in result.optimal_portfolio)  # Binary
        assert isinstance(result.optimal_value, (int, float, np.floating))
        assert len(result.optimization_history) > 0
        assert result.num_iterations > 0
        assert len(result.probability_distribution) == 8  # 2^3

    @pytest.mark.slow
    def test_solve_with_cardinality_constraint(self):
        """Test portfolio selection with cardinality constraint."""
        expected_returns = np.array([0.1, 0.15, 0.08, 0.12])
        covariance_matrix = np.eye(4) * 0.04  # Simple diagonal covariance

        qaoa = PortfolioQAOA(num_assets=4, num_layers=1, max_iterations=30)

        result = qaoa.solve_portfolio_selection(
            expected_returns,
            covariance_matrix,
            risk_aversion=1.0,
            cardinality_constraint=2,
        )

        # Check result
        assert isinstance(result, QAOAResult)
        assert len(result.optimal_portfolio) == 4

        # Portfolio should ideally have 2 assets (though constraint is soft)
        num_selected = np.sum(result.optimal_portfolio)
        assert num_selected >= 1  # At least one asset

    def test_sample_portfolios(self):
        """Test portfolio sampling from QAOA distribution."""
        params = np.array([0.5, 0.3])
        cost_matrix = np.eye(3)
        num_samples = 50

        samples = self.qaoa.sample_portfolios(params, cost_matrix, num_samples)

        # Check samples structure
        assert len(samples) == num_samples

        for portfolio, prob in samples:
            assert len(portfolio) == 3
            assert all(x in [0, 1] for x in portfolio)  # Binary
            assert 0 <= prob <= 1  # Valid probability

    def test_different_optimizers(self):
        """Test different classical optimizers."""
        qaoa_bfgs = PortfolioQAOA(
            num_assets=2, num_layers=1, optimizer="L-BFGS-B", max_iterations=20
        )

        expected_returns = np.array([0.1, 0.15])
        covariance_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])

        result = qaoa_bfgs.solve_portfolio_selection(
            expected_returns, covariance_matrix
        )

        # Should still produce valid result
        assert isinstance(result, QAOAResult)
        assert len(result.optimal_portfolio) == 2


class TestQAOAResult:
    """Test suite for QAOAResult dataclass."""

    def test_qaoa_result_creation(self):
        """Test QAOAResult creation."""
        result = QAOAResult(
            optimal_portfolio=np.array([1, 0, 1]),
            optimal_value=1.5,
            optimal_params=np.array([0.5, 0.3]),
            optimization_history=[2.0, 1.8, 1.5],
            probability_distribution=np.array([0.1, 0.2, 0.3, 0.4]),
            num_iterations=100,
            success=True,
        )

        assert np.array_equal(result.optimal_portfolio, [1, 0, 1])
        assert result.optimal_value == 1.5
        assert result.num_iterations == 100
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__])
