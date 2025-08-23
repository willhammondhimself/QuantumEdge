"""Unit tests for WebSocket optimization service."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import time

from src.streaming.optimization_service import (
    WebSocketOptimizationService,
    OptimizationRequest,
    OptimizationResult,
    OptimizationTrigger,
    OptimizationCache,
)
from src.streaming.websocket import (
    WebSocketConnectionManager,
    WebSocketMessage,
    MessageType,
)
from src.streaming.data_pipeline import DataPipeline
from src.optimization.mean_variance import ObjectiveType, PortfolioConstraints


class TestOptimizationCache:
    """Test optimization cache functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        cache = OptimizationCache(ttl_seconds=300)

        symbols = ["AAPL", "GOOGL", "MSFT"]
        objective = ObjectiveType.MAXIMIZE_SHARPE
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)
        method = "mean_variance"

        # Generate key twice with same parameters
        key1 = cache._generate_key(symbols, objective, constraints, method)
        key2 = cache._generate_key(symbols, objective, constraints, method)

        assert key1 == key2

        # Different order of symbols should produce same key
        symbols_reordered = ["GOOGL", "AAPL", "MSFT"]
        key3 = cache._generate_key(symbols_reordered, objective, constraints, method)

        assert key1 == key3

    def test_cache_set_and_get(self):
        """Test setting and getting cached results."""
        cache = OptimizationCache(ttl_seconds=300)

        # Create test result
        result = OptimizationResult(
            request_id="test_123",
            client_id="client_1",
            weights={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8,
            optimization_time=1.5,
            timestamp=time.time(),
            trigger=OptimizationTrigger.MANUAL,
            method="mean_variance",
            success=True,
        )

        symbols = ["AAPL", "GOOGL", "MSFT"]
        objective = ObjectiveType.MAXIMIZE_SHARPE
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)
        method = "mean_variance"

        # Cache the result
        cache.set(symbols, objective, constraints, method, result)

        # Retrieve the result
        cached_result = cache.get(symbols, objective, constraints, method)

        assert cached_result is not None
        assert cached_result.request_id == result.request_id
        assert cached_result.weights == result.weights

        # Access count should be 1
        stats = cache.get_stats()
        assert stats["total_accesses"] == 1

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = OptimizationCache(ttl_seconds=0.1)  # 100ms TTL

        result = OptimizationResult(
            request_id="test_exp",
            client_id="client_1",
            weights={"AAPL": 1.0},
            expected_return=0.1,
            expected_volatility=0.2,
            sharpe_ratio=0.5,
            optimization_time=1.0,
            timestamp=time.time(),
            trigger=OptimizationTrigger.MANUAL,
            method="mean_variance",
            success=True,
        )

        symbols = ["AAPL"]
        objective = ObjectiveType.MAXIMIZE_SHARPE
        constraints = PortfolioConstraints()
        method = "mean_variance"

        # Cache the result
        cache.set(symbols, objective, constraints, method, result)

        # Should get result immediately
        assert cache.get(symbols, objective, constraints, method) is not None

        # Wait for expiration
        time.sleep(0.2)

        # Should return None after expiration
        assert cache.get(symbols, objective, constraints, method) is None

    def test_clear_expired(self):
        """Test clearing expired cache entries."""
        cache = OptimizationCache(ttl_seconds=0.1)

        # Add multiple entries
        for i in range(3):
            result = OptimizationResult(
                request_id=f"test_{i}",
                client_id="client_1",
                weights={"AAPL": 1.0},
                expected_return=0.1,
                expected_volatility=0.2,
                sharpe_ratio=0.5,
                optimization_time=1.0,
                timestamp=time.time(),
                trigger=OptimizationTrigger.MANUAL,
                method="mean_variance",
                success=True,
            )

            cache.set(
                [f"SYMBOL{i}"],
                ObjectiveType.MAXIMIZE_SHARPE,
                PortfolioConstraints(),
                "mean_variance",
                result,
            )

        assert len(cache.cache) == 3

        # Wait for expiration
        time.sleep(0.2)

        # Clear expired entries
        cache.clear_expired()

        assert len(cache.cache) == 0


@pytest.mark.asyncio
class TestWebSocketOptimizationService:
    """Test WebSocket optimization service."""

    async def test_service_initialization(self):
        """Test service initialization."""
        connection_manager = Mock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager,
            data_pipeline=data_pipeline,
            cache_ttl=300,
            max_queue_size=50,
        )

        assert service.connection_manager == connection_manager
        assert service.data_pipeline == data_pipeline
        assert service.cache.ttl == 300
        assert service.max_queue_size == 50
        assert not service._running

    async def test_start_stop_service(self):
        """Test starting and stopping the service."""
        connection_manager = Mock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        # Start service
        await service.start()
        assert service._running
        assert len(service._tasks) > 0

        # Stop service
        await service.stop()
        assert not service._running
        assert len(service._tasks) == 0

    async def test_request_optimization(self):
        """Test requesting optimization."""
        connection_manager = AsyncMock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        client_id = "test_client"
        symbols = ["AAPL", "GOOGL"]
        objective = ObjectiveType.MAXIMIZE_SHARPE
        constraints = PortfolioConstraints(long_only=True)

        request_id = await service.request_optimization(
            client_id=client_id,
            symbols=symbols,
            objective=objective,
            constraints=constraints,
        )

        assert request_id.startswith(f"opt_{client_id}_")
        assert len(service.optimization_queue) == 1
        assert service.stats["total_requests"] == 1

        # Check acknowledgment was sent
        connection_manager.send_to_client.assert_called_once()
        call_args = connection_manager.send_to_client.call_args
        assert call_args[0][0] == client_id
        assert call_args[0][1].message_type == MessageType.PORTFOLIO_UPDATE

    async def test_cache_hit(self):
        """Test optimization cache hit."""
        connection_manager = AsyncMock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        # Pre-populate cache
        result = OptimizationResult(
            request_id="cached_123",
            client_id="test_client",
            weights={"AAPL": 0.6, "GOOGL": 0.4},
            expected_return=0.15,
            expected_volatility=0.20,
            sharpe_ratio=0.75,
            optimization_time=2.0,
            timestamp=time.time(),
            trigger=OptimizationTrigger.PRICE_CHANGE,
            method="mean_variance",
            success=True,
        )

        symbols = ["AAPL", "GOOGL"]
        objective = ObjectiveType.MAXIMIZE_SHARPE
        constraints = PortfolioConstraints(long_only=True)
        method = "mean_variance"

        service.cache.set(symbols, objective, constraints, method, result)

        # Request optimization with automatic trigger (should use cache)
        await service.request_optimization(
            client_id="test_client",
            symbols=symbols,
            objective=objective,
            constraints=constraints,
            method=method,
            trigger=OptimizationTrigger.PRICE_CHANGE,
        )

        # Should not add to queue (cache hit)
        assert len(service.optimization_queue) == 0
        assert service.stats["cache_hits"] == 1

        # Result should be sent immediately
        connection_manager.send_to_client.assert_called()

    async def test_queue_overflow(self):
        """Test optimization queue overflow handling."""
        connection_manager = AsyncMock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager,
            data_pipeline=data_pipeline,
            max_queue_size=2,
        )

        # Fill the queue
        for i in range(2):
            await service.request_optimization(
                client_id=f"client_{i}",
                symbols=["AAPL"],
                objective=ObjectiveType.MAXIMIZE_SHARPE,
                constraints=PortfolioConstraints(),
            )

        assert len(service.optimization_queue) == 2

        # Try to add one more
        await service.request_optimization(
            client_id="overflow_client",
            symbols=["AAPL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
        )

        # Queue should still be at max size
        assert len(service.optimization_queue) == 2

        # Error should be sent to client
        calls = connection_manager.send_to_client.call_args_list
        last_call = calls[-1]
        assert last_call[0][0] == "overflow_client"
        assert last_call[0][1].message_type == MessageType.ERROR

    @patch("src.streaming.optimization_service.MeanVarianceOptimizer")
    async def test_perform_optimization_success(self, mock_optimizer_class):
        """Test successful optimization execution."""
        connection_manager = AsyncMock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        # Mock cached data
        data_pipeline.get_cached_data.return_value = {
            "market_data": {"AAPL": {"price": 150.0}, "GOOGL": {"price": 2800.0}},
            "portfolios": {},
            "risk_metrics": {},
        }

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        # Mock optimization result
        mock_optimizer = mock_optimizer_class.return_value
        mock_result = Mock()
        mock_result.success = True
        mock_result.weights = np.array([0.6, 0.4])
        mock_result.expected_return = 0.12
        mock_result.expected_variance = 0.04
        mock_result.sharpe_ratio = 0.8
        mock_result.solve_time = 0.5
        mock_result.status = "optimal"

        mock_optimizer.optimize_portfolio.return_value = mock_result

        # Create request
        request = OptimizationRequest(
            request_id="test_opt_123",
            client_id="test_client",
            symbols=["AAPL", "GOOGL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(long_only=True),
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        # Mock _get_market_data
        service._get_market_data = AsyncMock(
            return_value={
                "expected_returns": np.array([0.10, 0.12]),
                "covariance_matrix": np.array([[0.04, 0.01], [0.01, 0.05]]),
                "returns_data": None,
                "timestamp": time.time(),
            }
        )

        # Perform optimization
        result = await service._perform_optimization(request)

        assert result.success
        assert result.request_id == request.request_id
        assert result.client_id == request.client_id
        assert result.expected_return == 0.12
        assert result.expected_volatility == 0.2  # sqrt(0.04)
        assert result.sharpe_ratio == 0.8
        assert result.method == "mean_variance"
        assert len(result.weights) == 2

    async def test_perform_optimization_failure(self):
        """Test optimization failure handling."""
        connection_manager = AsyncMock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        # Mock market data retrieval failure
        service._get_market_data = AsyncMock(return_value=None)

        request = OptimizationRequest(
            request_id="test_fail_123",
            client_id="test_client",
            symbols=["INVALID"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        result = await service._perform_optimization(request)

        assert not result.success
        assert result.error is not None
        assert "Unable to retrieve market data" in result.error
        assert result.weights == {}

    async def test_get_stats(self):
        """Test getting service statistics."""
        connection_manager = Mock(spec=WebSocketConnectionManager)
        data_pipeline = Mock(spec=DataPipeline)

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        # Set some stats
        service.stats["total_requests"] = 10
        service.stats["successful_optimizations"] = 8
        service.stats["failed_optimizations"] = 2
        service.stats["cache_hits"] = 3

        stats = service.get_stats()

        assert stats["total_requests"] == 10
        assert stats["successful_optimizations"] == 8
        assert stats["failed_optimizations"] == 2
        assert stats["cache_hits"] == 3
        assert "cache_stats" in stats
        assert "optimizers" in stats
