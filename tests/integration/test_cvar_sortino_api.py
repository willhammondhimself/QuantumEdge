"""Integration tests for CVaR and Sortino objectives through API."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.models import ObjectiveFunction

client = TestClient(app)


class TestCVaRSortinoAPI:
    """Test CVaR and Sortino objectives through API endpoints."""

    @pytest.fixture
    def sample_portfolio_data(self):
        """Generate sample portfolio data for testing."""
        np.random.seed(42)
        n_assets = 5
        n_periods = 252

        # Generate expected returns (annualized)
        expected_returns = [0.08, 0.10, 0.12, 0.07, 0.09]

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
        covariance_matrix = (
            np.outer(volatilities, volatilities) * correlations
        ).tolist()

        # Generate historical returns data for CVaR/Sortino
        returns_data = np.random.multivariate_normal(
            np.array(expected_returns) / 252,  # Daily returns
            np.array(covariance_matrix) / 252,  # Daily covariance
            size=n_periods,
        ).tolist()

        return {
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
            "returns_data": returns_data,
        }

    def test_mean_variance_cvar_objective(self, sample_portfolio_data):
        """Test CVaR optimization through mean-variance endpoint."""
        request_data = {
            "expected_returns": sample_portfolio_data["expected_returns"],
            "covariance_matrix": sample_portfolio_data["covariance_matrix"],
            "objective": ObjectiveFunction.MINIMIZE_CVAR.value,
            "returns_data": sample_portfolio_data["returns_data"],
            "cvar_confidence": 0.05,  # 5% CVaR (95% confidence)
            "constraints": {"long_only": True, "sum_to_one": True},
        }

        response = client.post("/api/v1/optimize/mean-variance", json=request_data)

        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.json()}")
        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["portfolio"] is not None
        assert len(result["portfolio"]["weights"]) == 5
        assert abs(sum(result["portfolio"]["weights"]) - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result["portfolio"]["weights"])

    def test_mean_variance_sortino_objective(self, sample_portfolio_data):
        """Test Sortino optimization through mean-variance endpoint."""
        request_data = {
            "expected_returns": sample_portfolio_data["expected_returns"],
            "covariance_matrix": sample_portfolio_data["covariance_matrix"],
            "objective": ObjectiveFunction.MAXIMIZE_SORTINO.value,
            "returns_data": sample_portfolio_data["returns_data"],
            "constraints": {"long_only": True, "sum_to_one": True},
        }

        response = client.post("/api/v1/optimize/mean-variance", json=request_data)

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["portfolio"] is not None
        assert len(result["portfolio"]["weights"]) == 5
        assert abs(sum(result["portfolio"]["weights"]) - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result["portfolio"]["weights"])

    def test_classical_optimizer_cvar(self, sample_portfolio_data):
        """Test CVaR optimization through classical optimizer endpoint."""
        request_data = {
            "expected_returns": sample_portfolio_data["expected_returns"],
            "covariance_matrix": sample_portfolio_data["covariance_matrix"],
            "objective": ObjectiveFunction.MINIMIZE_CVAR.value,
            "returns_data": sample_portfolio_data["returns_data"],
            "constraints": {"long_only": True, "sum_to_one": True},
        }

        # Test with Genetic Algorithm
        response = client.post(
            "/api/v1/optimize/classical?method=genetic_algorithm", json=request_data
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["portfolio"] is not None
        assert "genetic_algorithm" in result["message"]

        # Test with Simulated Annealing
        response = client.post(
            "/api/v1/optimize/classical?method=simulated_annealing", json=request_data
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["portfolio"] is not None
        assert "simulated_annealing" in result["message"]

    def test_classical_optimizer_sortino(self, sample_portfolio_data):
        """Test Sortino optimization through classical optimizer endpoint."""
        request_data = {
            "expected_returns": sample_portfolio_data["expected_returns"],
            "covariance_matrix": sample_portfolio_data["covariance_matrix"],
            "objective": ObjectiveFunction.MAXIMIZE_SORTINO.value,
            "returns_data": sample_portfolio_data["returns_data"],
            "constraints": {"long_only": True, "sum_to_one": True},
        }

        # Test with Particle Swarm
        response = client.post(
            "/api/v1/optimize/classical?method=particle_swarm", json=request_data
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["portfolio"] is not None
        assert "particle_swarm" in result["message"]

    def test_compare_objectives(self, sample_portfolio_data):
        """Compare different objectives to ensure they produce different portfolios."""
        objectives = [
            ObjectiveFunction.MAXIMIZE_SHARPE.value,
            ObjectiveFunction.MINIMIZE_CVAR.value,
            ObjectiveFunction.MAXIMIZE_SORTINO.value,
        ]

        portfolios = {}

        for objective in objectives:
            request_data = {
                "expected_returns": sample_portfolio_data["expected_returns"],
                "covariance_matrix": sample_portfolio_data["covariance_matrix"],
                "objective": objective,
                "returns_data": sample_portfolio_data["returns_data"],
                "constraints": {"long_only": True, "sum_to_one": True},
            }

            response = client.post("/api/v1/optimize/mean-variance", json=request_data)
            assert response.status_code == 200

            result = response.json()
            assert result["success"] is True

            portfolios[objective] = result["portfolio"]["weights"]

        # Verify that different objectives produce different portfolios
        sharpe_weights = np.array(portfolios[ObjectiveFunction.MAXIMIZE_SHARPE.value])
        cvar_weights = np.array(portfolios[ObjectiveFunction.MINIMIZE_CVAR.value])
        sortino_weights = np.array(portfolios[ObjectiveFunction.MAXIMIZE_SORTINO.value])

        # CVaR should be more conservative (lower variance) than Sharpe
        sharpe_variance = (
            sharpe_weights
            @ np.array(sample_portfolio_data["covariance_matrix"])
            @ sharpe_weights
        )
        cvar_variance = (
            cvar_weights
            @ np.array(sample_portfolio_data["covariance_matrix"])
            @ cvar_weights
        )

        # Sortino focuses on downside risk, so might have different characteristics
        # At minimum, the portfolios should be different
        assert not np.allclose(sharpe_weights, cvar_weights, atol=0.01)
        assert not np.allclose(sharpe_weights, sortino_weights, atol=0.01)
        assert not np.allclose(cvar_weights, sortino_weights, atol=0.01)

    def test_invalid_objective_data(self):
        """Test error handling when returns_data is missing for CVaR/Sortino."""
        request_data = {
            "expected_returns": [0.08, 0.10, 0.12],
            "covariance_matrix": [
                [0.04, 0.02, 0.01],
                [0.02, 0.05, 0.02],
                [0.01, 0.02, 0.06],
            ],
            "objective": ObjectiveFunction.MINIMIZE_CVAR.value,
            # Note: returns_data is missing
            "constraints": {"long_only": True, "sum_to_one": True},
        }

        response = client.post("/api/v1/optimize/mean-variance", json=request_data)

        # Should fail because returns_data is required for CVaR
        assert response.status_code == 500
        assert "returns_data required" in response.json()["detail"]
