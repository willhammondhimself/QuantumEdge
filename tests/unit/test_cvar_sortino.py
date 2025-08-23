"""Test CVaR and Sortino objective implementations."""

import pytest
import numpy as np
from src.optimization.mean_variance import (
    ObjectiveType,
    MeanVarianceOptimizer,
    PortfolioConstraints,
)
from src.optimization.classical_solvers import (
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    ParticleSwarmOptimizer,
    OptimizerParameters,
    compare_classical_methods,
)


class TestCVaRSortinoImplementation:
    """Test cases for CVaR and Sortino ratio optimization."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        np.random.seed(42)
        n_assets = 5
        n_periods = 252

        # Generate expected returns (annualized)
        expected_returns = np.array([0.08, 0.10, 0.12, 0.07, 0.09])

        # Generate covariance matrix
        correlations = np.array(
            [
                [1.0, 0.3, 0.2, 0.1, 0.2],
                [0.3, 1.0, 0.4, 0.2, 0.3],
                [0.2, 0.4, 1.0, 0.3, 0.2],
                [0.1, 0.2, 0.3, 1.0, 0.4],
                [0.2, 0.3, 0.2, 0.4, 1.0],
            ]
        )
        volatilities = np.array([0.15, 0.20, 0.25, 0.12, 0.18])
        covariance_matrix = np.outer(volatilities, volatilities) * correlations

        # Generate historical returns data
        returns_data = np.random.multivariate_normal(
            expected_returns / 252,  # Daily returns
            covariance_matrix / 252,  # Daily covariance
            size=n_periods,
        )

        return {
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
            "returns_data": returns_data,
            "n_assets": n_assets,
        }

    def test_mean_variance_cvar_optimization(self, sample_data):
        """Test CVaR optimization with MeanVarianceOptimizer."""
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test with CVaR objective
        result = optimizer.optimize_portfolio(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            objective=ObjectiveType.MINIMIZE_CVAR,
            constraints=constraints,
            returns_data=sample_data["returns_data"],
            cvar_confidence=0.95,
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)  # Long-only constraint

    def test_mean_variance_sortino_optimization(self, sample_data):
        """Test Sortino optimization with MeanVarianceOptimizer."""
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test with Sortino objective
        result = optimizer.optimize_portfolio(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            objective=ObjectiveType.MAXIMIZE_SORTINO,
            constraints=constraints,
            returns_data=sample_data["returns_data"],
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)  # Long-only constraint

    def test_genetic_algorithm_cvar(self, sample_data):
        """Test CVaR optimization with Genetic Algorithm."""
        params = OptimizerParameters(max_iterations=100, population_size=30, seed=42)
        optimizer = GeneticAlgorithmOptimizer(params)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test with CVaR objective
        result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MINIMIZE_CVAR,
            returns_data=sample_data["returns_data"],
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)

    def test_simulated_annealing_sortino(self, sample_data):
        """Test Sortino optimization with Simulated Annealing."""
        params = OptimizerParameters(
            max_iterations=200, temperature=100.0, cooling_rate=0.95, seed=42
        )
        optimizer = SimulatedAnnealingOptimizer(params)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test with Sortino objective
        result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MAXIMIZE_SORTINO,
            returns_data=sample_data["returns_data"],
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)

    def test_particle_swarm_cvar(self, sample_data):
        """Test CVaR optimization with Particle Swarm."""
        params = OptimizerParameters(max_iterations=100, population_size=20, seed=42)
        optimizer = ParticleSwarmOptimizer(params)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test with CVaR objective
        result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MINIMIZE_CVAR,
            returns_data=sample_data["returns_data"],
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)

    def test_synthetic_returns_generation(self, sample_data):
        """Test that synthetic returns are generated when not provided."""
        params = OptimizerParameters(max_iterations=50, population_size=20, seed=42)
        optimizer = GeneticAlgorithmOptimizer(params)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Test CVaR without providing returns_data
        result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MINIMIZE_CVAR,
            # Note: returns_data not provided
        )

        assert result.success
        assert result.weights is not None
        assert len(result.weights) == sample_data["n_assets"]
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_compare_methods_with_cvar(self, sample_data):
        """Test compare_classical_methods with CVaR objective."""
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        results = compare_classical_methods(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MINIMIZE_CVAR,
            returns_data=sample_data["returns_data"],
        )

        assert len(results) == 3  # GA, SA, PSO
        for method_name, result in results.items():
            assert result.success
            assert result.weights is not None
            assert len(result.weights) == sample_data["n_assets"]
            assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_compare_methods_with_sortino(self, sample_data):
        """Test compare_classical_methods with Sortino objective."""
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        results = compare_classical_methods(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MAXIMIZE_SORTINO,
            returns_data=sample_data["returns_data"],
        )

        assert len(results) == 3  # GA, SA, PSO
        for method_name, result in results.items():
            assert result.success
            assert result.weights is not None
            assert len(result.weights) == sample_data["n_assets"]
            assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_objective_values_make_sense(self, sample_data):
        """Test that CVaR and Sortino objectives produce sensible values."""
        params = OptimizerParameters(max_iterations=100, population_size=30, seed=42)
        optimizer = GeneticAlgorithmOptimizer(params)
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        # Get results for different objectives
        sharpe_result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MAXIMIZE_SHARPE,
        )

        cvar_result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MINIMIZE_CVAR,
            returns_data=sample_data["returns_data"],
        )

        sortino_result = optimizer.optimize(
            expected_returns=sample_data["expected_returns"],
            covariance_matrix=sample_data["covariance_matrix"],
            constraints=constraints,
            objective=ObjectiveType.MAXIMIZE_SORTINO,
            returns_data=sample_data["returns_data"],
        )

        # CVaR portfolios should be more conservative (lower volatility)
        sharpe_vol = np.sqrt(sharpe_result.expected_variance)
        cvar_vol = np.sqrt(cvar_result.expected_variance)

        # Sortino focuses on downside risk, so might have different characteristics
        sortino_vol = np.sqrt(sortino_result.expected_variance)

        # All should produce valid portfolios
        assert sharpe_result.success
        assert cvar_result.success
        assert sortino_result.success

        # CVaR objective value should be positive (it's a risk measure)
        assert cvar_result.objective_value > 0

        # Sortino objective should be negative (we minimize negative Sortino)
        assert sortino_result.objective_value < 0
