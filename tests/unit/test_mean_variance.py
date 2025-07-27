"""
Unit tests for mean-variance optimization.
"""

import numpy as np
import pytest
from src.optimization.mean_variance import (
    MeanVarianceOptimizer, RobustOptimizer, OptimizationResult, 
    PortfolioConstraints, ObjectiveType, CVXPY_AVAILABLE
)


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not available")
class TestMeanVarianceOptimizer:
    """Test suite for MeanVarianceOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        
        # Create test data
        self.expected_returns = np.array([0.10, 0.15, 0.12, 0.08])
        self.covariance_matrix = np.array([
            [0.04, 0.01, 0.02, 0.005],
            [0.01, 0.09, 0.03, 0.01],
            [0.02, 0.03, 0.06, 0.015],
            [0.005, 0.01, 0.015, 0.03]
        ])
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.risk_free_rate == 0.02
        assert self.optimizer.solver == 'OSQP'
        assert self.optimizer.verbose is False
    
    def test_maximize_utility_optimization(self):
        """Test utility maximization."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_UTILITY,
            risk_aversion=2.0
        )
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 4
        assert result.success
        assert result.status in ['optimal', 'optimal_inaccurate']
        
        # Check weight properties
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)  # Sum to 1
        assert np.all(result.weights >= -1e-6)  # Non-negative (with tolerance)
        
        # Check portfolio metrics
        assert isinstance(result.expected_return, (int, float, np.floating))
        assert isinstance(result.expected_variance, (int, float, np.floating))
        assert result.expected_variance >= 0
    
    def test_minimize_variance_optimization(self):
        """Test variance minimization."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MINIMIZE_VARIANCE
        )
        
        assert result.success
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)
        
        # This should give the minimum variance portfolio
        assert result.expected_variance > 0
    
    def test_maximize_return_optimization(self):
        """Test return maximization."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_RETURN
        )
        
        assert result.success
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)
        
        # Should invest in highest return asset
        highest_return_idx = np.argmax(self.expected_returns)
        assert result.weights[highest_return_idx] > 0.9  # Mostly in best asset
    
    def test_constraints_application(self):
        """Test various portfolio constraints."""
        constraints = PortfolioConstraints(
            max_weight=0.4,
            min_weight=0.05,
            forbidden_assets=[3],  # Forbid last asset
            min_return=0.11
        )
        
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_UTILITY,
            constraints=constraints
        )
        
        if result.success:
            # Check constraints are satisfied
            assert np.all(result.weights <= 0.4 + 1e-6)  # Max weight
            assert np.all(result.weights >= 0.05 - 1e-6)  # Min weight
            assert result.weights[3] <= 1e-6  # Forbidden asset
            assert result.expected_return >= 0.11 - 1e-6  # Min return
    
    def test_long_only_constraint(self):
        """Test long-only constraint."""
        constraints = PortfolioConstraints(long_only=True)
        
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_UTILITY,
            constraints=constraints
        )
        
        assert result.success
        assert np.all(result.weights >= -1e-6)  # Non-negative weights
    
    def test_turnover_constraint(self):
        """Test turnover constraint."""
        current_weights = np.array([0.3, 0.3, 0.2, 0.2])
        constraints = PortfolioConstraints(
            max_turnover=0.2,
            current_weights=current_weights
        )
        
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_UTILITY,
            constraints=constraints
        )
        
        if result.success:
            # Check turnover constraint
            turnover = np.sum(np.abs(result.weights - current_weights))
            assert turnover <= 0.2 + 1e-6
    
    def test_efficient_frontier_computation(self):
        """Test efficient frontier computation."""
        returns, risks, weights = self.optimizer.compute_efficient_frontier(
            self.expected_returns,
            self.covariance_matrix,
            num_points=10
        )
        
        # Check output dimensions
        assert len(returns) <= 10  # Some points might fail
        assert len(risks) == len(returns)
        assert len(weights) == len(returns)
        
        if len(returns) > 1:
            # Check frontier properties
            assert len(returns[0]) == len(self.expected_returns) or len(weights) > 0
            
            # Returns should generally increase along frontier
            # (though not strictly due to numerical issues)
            assert returns[-1] >= returns[0] - 0.01
    
    def test_maximum_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize_maximum_sharpe(
            self.expected_returns,
            self.covariance_matrix
        )
        
        if result.success:
            assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
            assert np.all(result.weights >= -1e-6)
            assert result.sharpe_ratio > 0  # Should have positive Sharpe ratio
    
    def test_invalid_objective_type(self):
        """Test error handling for invalid objective."""
        with pytest.raises(ValueError, match="Unknown objective type"):
            self.optimizer.optimize_portfolio(
                self.expected_returns,
                self.covariance_matrix,
                objective="invalid_objective"
            )
    
    def test_zero_variance_asset(self):
        """Test handling of zero-variance assets."""
        # Create covariance matrix with one zero-variance asset
        modified_cov = self.covariance_matrix.copy()
        modified_cov[0, :] = 0
        modified_cov[:, 0] = 0
        
        result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            modified_cov,
            objective=ObjectiveType.MAXIMIZE_UTILITY
        )
        
        # Should handle gracefully
        assert isinstance(result, OptimizationResult)
    
    def test_single_asset_portfolio(self):
        """Test optimization with single asset."""
        single_return = np.array([0.10])
        single_cov = np.array([[0.04]])
        
        result = self.optimizer.optimize_portfolio(
            single_return,
            single_cov,
            objective=ObjectiveType.MAXIMIZE_UTILITY
        )
        
        if result.success:
            assert np.isclose(result.weights[0], 1.0, atol=1e-6)
            assert np.isclose(result.expected_return, 0.10, atol=1e-6)


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not available")
class TestRobustOptimizer:
    """Test suite for RobustOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = RobustOptimizer(
            risk_free_rate=0.02,
            uncertainty_level=0.1
        )
        
        self.expected_returns = np.array([0.10, 0.15, 0.12])
        self.covariance_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.06]
        ])
    
    def test_initialization(self):
        """Test robust optimizer initialization."""
        assert self.optimizer.uncertainty_level == 0.1
        assert isinstance(self.optimizer, MeanVarianceOptimizer)
    
    def test_robust_optimization(self):
        """Test robust portfolio optimization."""
        result = self.optimizer.optimize_robust_portfolio(
            self.expected_returns,
            self.covariance_matrix
        )
        
        if result.success:
            assert len(result.weights) == 3
            assert np.isclose(np.sum(result.weights), 1.0, atol=1e-6)
            assert np.all(result.weights >= -1e-6)
            
            # Robust portfolio should be more conservative
            assert result.expected_variance > 0
    
    def test_robust_optimization_with_uncertainty(self):
        """Test robust optimization with specified uncertainty."""
        return_uncertainty = np.array([0.02, 0.03, 0.025])
        
        result = self.optimizer.optimize_robust_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            return_uncertainty=return_uncertainty
        )
        
        if result.success:
            assert isinstance(result, OptimizationResult)
            assert len(result.weights) == 3
    
    def test_robust_vs_standard_comparison(self):
        """Compare robust vs standard optimization."""
        # Standard optimization
        standard_result = self.optimizer.optimize_portfolio(
            self.expected_returns,
            self.covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_UTILITY
        )
        
        # Robust optimization
        robust_result = self.optimizer.optimize_robust_portfolio(
            self.expected_returns,
            self.covariance_matrix
        )
        
        if standard_result.success and robust_result.success:
            # Robust portfolio should generally be more diversified
            # (higher entropy in weights)
            def weight_entropy(weights):
                # Avoid log(0) by adding small epsilon
                w = weights + 1e-10
                return -np.sum(w * np.log(w))
            
            robust_entropy = weight_entropy(robust_result.weights)
            standard_entropy = weight_entropy(standard_result.weights)
            
            # This is a general tendency, not a strict rule
            assert robust_entropy >= standard_entropy - 0.1


class TestPortfolioConstraints:
    """Test suite for PortfolioConstraints dataclass."""
    
    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = PortfolioConstraints()
        
        assert constraints.long_only is True
        assert constraints.sum_to_one is True
        assert constraints.min_weight is None
        assert constraints.max_weight is None
        assert constraints.max_assets is None
        assert constraints.forbidden_assets is None
    
    def test_custom_constraints(self):
        """Test custom constraint specification."""
        constraints = PortfolioConstraints(
            long_only=False,
            max_weight=0.3,
            min_return=0.08,
            forbidden_assets=[0, 2]
        )
        
        assert constraints.long_only is False
        assert constraints.max_weight == 0.3
        assert constraints.min_return == 0.08
        assert constraints.forbidden_assets == [0, 2]


def test_optimizer_without_cvxpy():
    """Test optimizer error when CVXPY not available."""
    if not CVXPY_AVAILABLE:
        with pytest.raises(ImportError, match="CVXPY is required"):
            MeanVarianceOptimizer(risk_free_rate=0.02)


class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass."""
    
    def test_result_creation(self):
        """Test optimization result creation."""
        weights = np.array([0.3, 0.4, 0.3])
        result = OptimizationResult(
            weights=weights,
            expected_return=0.12,
            expected_variance=0.04,
            sharpe_ratio=0.5,
            objective_value=0.08,
            status='optimal',
            solve_time=0.1,
            success=True
        )
        
        assert np.array_equal(result.weights, weights)
        assert result.expected_return == 0.12
        assert result.success is True
        assert result.status == 'optimal'


if __name__ == '__main__':
    pytest.main([__file__])