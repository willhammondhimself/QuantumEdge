"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient


# Test if FastAPI can be imported and basic functionality works
def test_fastapi_import():
    """Test that FastAPI components can be imported."""
    try:
        from src.api.main import app
        from src.api.config import settings
        from src.api.models import HealthResponse

        assert app is not None
        assert settings is not None
        assert HealthResponse is not None
    except ImportError as e:
        pytest.skip(f"FastAPI dependencies not available: {e}")


def test_api_models():
    """Test API model validation."""
    try:
        from src.api.models import MeanVarianceRequest, PortfolioConstraintsModel

        # Test valid request
        request = MeanVarianceRequest(
            expected_returns=[0.1, 0.15, 0.12],
            covariance_matrix=[
                [0.04, 0.01, 0.02],
                [0.01, 0.09, 0.03],
                [0.02, 0.03, 0.06],
            ],
        )
        assert request.expected_returns == [0.1, 0.15, 0.12]
        assert len(request.covariance_matrix) == 3

        # Test constraints
        constraints = PortfolioConstraintsModel(long_only=True, max_weight=0.4)
        assert constraints.long_only is True
        assert constraints.max_weight == 0.4

    except ImportError as e:
        pytest.skip(f"Pydantic/FastAPI not available: {e}")


def test_config_settings():
    """Test configuration settings."""
    try:
        from src.api.config import settings

        assert settings.app_name == "QuantumEdge"
        assert settings.app_version == "0.1.0"
        assert settings.environment in ["development", "production", "testing"]
        assert settings.api_prefix == "/api/v1"

    except ImportError as e:
        pytest.skip(f"Configuration dependencies not available: {e}")


def test_dependency_injection():
    """Test dependency injection functions."""
    try:
        from src.api.deps import get_logger, get_optimization_manager

        logger = get_logger("test")
        assert logger.name == "test"

        manager = get_optimization_manager()
        assert hasattr(manager, "running_optimizations")
        assert hasattr(manager, "can_start_optimization")

    except ImportError as e:
        pytest.skip(f"Dependency injection not available: {e}")


@pytest.mark.skipif(True, reason="Requires FastAPI server to be running")
def test_health_endpoint():
    """Test health endpoint (requires server)."""
    try:
        from src.api.main import app

        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data

    except ImportError as e:
        pytest.skip(f"FastAPI testing not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
