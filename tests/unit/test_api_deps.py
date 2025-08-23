"""Tests for API dependencies."""

import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
import redis

from src.api.deps import (
    get_redis_client,
    get_logger,
    get_optimization_manager,
    OptimizationManager,
)
from src.api.config import settings


class TestRedisDependency:
    """Test Redis client dependency."""

    @patch("redis.from_url")
    def test_get_redis_client_success(self, mock_from_url):
        """Test successful Redis client creation."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        # Test the dependency
        client = get_redis_client()
        assert client == mock_redis
        mock_from_url.assert_called_once_with(settings.redis_url)
        mock_redis.ping.assert_called_once()

    @patch("redis.from_url")
    def test_get_redis_client_connection_error(self, mock_from_url):
        """Test Redis connection error handling."""
        mock_from_url.side_effect = ConnectionError("Redis connection failed")

        # Should return None when Redis is unavailable
        client = get_redis_client()
        assert client is None

    @patch("redis.from_url")
    def test_get_redis_client_ping_failure(self, mock_from_url):
        """Test Redis ping failure."""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.ping.side_effect = Exception("Ping failed")
        mock_from_url.return_value = mock_redis

        # Should return None when ping fails
        client = get_redis_client()
        assert client is None

    @patch("src.api.deps.REDIS_AVAILABLE", False)
    def test_get_redis_client_not_available(self):
        """Test when Redis module is not available."""
        client = get_redis_client()
        assert client is None


class TestLoggerDependency:
    """Test logger dependency."""

    def test_get_logger_default(self):
        """Test getting logger with default name."""
        logger = get_logger("quantumedge")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "quantumedge"

    def test_get_logger_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """Test that same logger instance is returned for same name."""
        logger1 = get_logger("test.logger")
        logger2 = get_logger("test.logger")
        assert logger1 is logger2

    def test_get_logger_different_instances(self):
        """Test that different instances are returned for different names."""
        logger1 = get_logger("test.logger1")
        logger2 = get_logger("test.logger2")
        assert logger1 is not logger2


class TestOptimizationManager:
    """Test OptimizationManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create OptimizationManager instance."""
        return OptimizationManager()

    def test_start_optimization(self, manager):
        """Test starting an optimization."""
        success = manager.start_optimization("opt_123")

        assert success is True
        assert "opt_123" in manager.running_optimizations
        assert manager.running_optimizations["opt_123"]["status"] == "running"

    def test_can_start_optimization(self, manager):
        """Test checking if optimization can start."""
        # Fill up to max concurrent
        for i in range(manager.max_concurrent):
            assert manager.can_start_optimization() is True
            manager.start_optimization(f"opt_{i}")

        # Should not be able to start more
        assert manager.can_start_optimization() is False

    def test_finish_optimization(self, manager):
        """Test finishing optimization."""
        manager.start_optimization("opt_123")
        assert "opt_123" in manager.running_optimizations

        # Finish optimization
        manager.finish_optimization("opt_123")

        assert "opt_123" not in manager.running_optimizations

    def test_finish_optimization_nonexistent(self, manager):
        """Test finishing non-existent optimization."""
        # Should not raise error
        manager.finish_optimization("nonexistent_id")

    def test_get_optimization_status(self, manager):
        """Test getting optimization status."""
        # Non-existent optimization
        status = manager.get_optimization_status("invalid_id")
        assert status is None

        # Existing optimization
        manager.start_optimization("opt_123")

        status = manager.get_optimization_status("opt_123")
        assert status["status"] == "running"
        assert status["start_time"] is None  # Not set by manager

    def test_max_concurrent_limit(self, manager):
        """Test max concurrent optimizations limit."""
        # Start optimizations up to the limit
        for i in range(manager.max_concurrent):
            success = manager.start_optimization(f"opt_{i}")
            assert success is True

        # Try to start one more
        success = manager.start_optimization("opt_extra")
        assert success is False

        # Finish one and try again
        manager.finish_optimization("opt_0")
        success = manager.start_optimization("opt_extra")
        assert success is True

    def test_concurrent_optimizations(self, manager):
        """Test managing multiple concurrent optimizations."""
        # Start multiple optimizations (up to limit)
        opt_ids = []
        for i in range(min(3, manager.max_concurrent)):
            success = manager.start_optimization(f"opt_{i}")
            assert success is True
            opt_ids.append(f"opt_{i}")

        assert len(manager.running_optimizations) == len(opt_ids)

        # Finish some
        manager.finish_optimization(opt_ids[0])

        # Check remaining
        assert opt_ids[0] not in manager.running_optimizations
        assert opt_ids[1] in manager.running_optimizations

    def test_optimization_state(self, manager):
        """Test optimization state management."""
        # Start an optimization
        success = manager.start_optimization("opt_state_test")
        assert success is True

        # Check state
        state = manager.get_optimization_status("opt_state_test")
        assert state is not None
        assert state["status"] == "running"

        # Clean up
        manager.finish_optimization("opt_state_test")
        state = manager.get_optimization_status("opt_state_test")
        assert state is None


class TestDependencyInjection:
    """Test dependency injection in API endpoints."""

    def test_get_optimization_manager(self):
        """Test getting optimization manager dependency."""
        manager1 = get_optimization_manager()
        manager2 = get_optimization_manager()

        # Should return same instance (singleton)
        assert manager1 is manager2
        assert isinstance(manager1, OptimizationManager)

    def test_dependency_in_endpoint(self):
        """Test using dependencies in an endpoint."""
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        # Since get_logger requires a parameter, we need to create a wrapper
        def get_logger_dep():
            return get_logger("test")

        @app.get("/test")
        def test_endpoint(
            redis_client=Depends(get_redis_client),
            logger=Depends(get_logger_dep),
            opt_manager=Depends(get_optimization_manager),
        ):
            return {
                "has_redis": redis_client is not None,
                "has_logger": logger is not None,
                "has_manager": opt_manager is not None,
            }

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        data = response.json()
        assert data["has_logger"] is True
        assert data["has_manager"] is True
        # Redis might be None if not available
        assert isinstance(data["has_redis"], bool)
