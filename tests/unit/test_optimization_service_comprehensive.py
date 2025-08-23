"""Comprehensive tests for optimization service module."""

import pytest
import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque
import hashlib

from src.streaming.optimization_service import (
    OptimizationTrigger,
    OptimizationRequest,
    OptimizationResult,
    OptimizationCache,
    WebSocketOptimizationService,
)
from src.streaming.websocket import (
    WebSocketConnectionManager,
    WebSocketMessage,
    MessageType,
)
from src.streaming.data_pipeline import DataPipeline
from src.optimization.mean_variance import (
    MeanVarianceOptimizer,
    ObjectiveType,
    PortfolioConstraints,
    OptimizationResult as MVOptimizationResult,
)


class TestOptimizationTrigger:
    """Test OptimizationTrigger enum."""

    def test_trigger_types(self):
        """Test all trigger types are defined."""
        assert OptimizationTrigger.MANUAL == "manual"
        assert OptimizationTrigger.PRICE_CHANGE == "price_change"
        assert OptimizationTrigger.TIME_BASED == "time_based"
        assert OptimizationTrigger.RISK_ALERT == "risk_alert"
        assert OptimizationTrigger.CONSTRAINT_VIOLATION == "constraint_violation"
        assert OptimizationTrigger.MARKET_EVENT == "market_event"


class TestOptimizationRequest:
    """Test OptimizationRequest dataclass."""

    def test_request_creation(self):
        """Test creating optimization request."""
        constraints = PortfolioConstraints(
            long_only=True, sum_to_one=True, min_weight=0.0, max_weight=0.4
        )

        request = OptimizationRequest(
            request_id="opt_123",
            client_id="client_456",
            symbols=["AAPL", "GOOGL", "MSFT"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=constraints,
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        assert request.request_id == "opt_123"
        assert request.client_id == "client_456"
        assert len(request.symbols) == 3
        assert request.objective == ObjectiveType.MAXIMIZE_SHARPE
        assert request.method == "mean_variance"
        assert request.trigger == OptimizationTrigger.MANUAL
        assert request.metadata == {}

    def test_request_with_metadata(self):
        """Test request with metadata."""
        request = OptimizationRequest(
            request_id="opt_123",
            client_id="client_456",
            symbols=["AAPL"],
            objective=ObjectiveType.MINIMIZE_VARIANCE,
            constraints=PortfolioConstraints(),
            method="vqe",
            trigger=OptimizationTrigger.PRICE_CHANGE,
            timestamp=time.time(),
            metadata={"threshold": 0.05, "portfolio_id": "port_789"},
        )

        assert request.metadata["threshold"] == 0.05
        assert request.metadata["portfolio_id"] == "port_789"


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_result_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            request_id="opt_123",
            client_id="client_456",
            weights={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8,
            optimization_time=2.5,
            timestamp=time.time(),
            trigger=OptimizationTrigger.MANUAL,
            method="mean_variance",
            success=True,
        )

        assert result.request_id == "opt_123"
        assert result.weights["AAPL"] == 0.4
        assert result.expected_return == 0.12
        assert result.sharpe_ratio == 0.8
        assert result.success is True
        assert result.error is None

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        timestamp = time.time()
        result = OptimizationResult(
            request_id="opt_123",
            client_id="client_456",
            weights={"AAPL": 0.5, "GOOGL": 0.5},
            expected_return=0.10,
            expected_volatility=0.12,
            sharpe_ratio=0.83,
            optimization_time=1.5,
            timestamp=timestamp,
            trigger=OptimizationTrigger.TIME_BASED,
            method="genetic_algorithm",
            success=True,
            metadata={"iterations": 100},
        )

        result_dict = result.to_dict()

        assert result_dict["request_id"] == "opt_123"
        assert result_dict["client_id"] == "client_456"
        assert result_dict["weights"]["AAPL"] == 0.5
        assert result_dict["metrics"]["expected_return"] == 0.10
        assert result_dict["metrics"]["sharpe_ratio"] == 0.83
        assert result_dict["trigger"] == OptimizationTrigger.TIME_BASED
        assert result_dict["method"] == "genetic_algorithm"
        assert result_dict["success"] is True
        assert result_dict["metadata"]["iterations"] == 100

    def test_failed_result(self):
        """Test failed optimization result."""
        result = OptimizationResult(
            request_id="opt_failed",
            client_id="client_456",
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            optimization_time=0.0,
            timestamp=time.time(),
            trigger=OptimizationTrigger.MANUAL,
            method="mean_variance",
            success=False,
            error="Optimization failed: Singular matrix",
        )

        assert result.success is False
        assert result.error == "Optimization failed: Singular matrix"
        assert result.weights == {}


class TestOptimizationCache:
    """Test OptimizationCache class."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return OptimizationCache(ttl_seconds=300)

    @pytest.fixture
    def sample_params(self):
        """Create sample optimization parameters."""
        return {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "objective": ObjectiveType.MAXIMIZE_SHARPE,
            "constraints": PortfolioConstraints(
                long_only=True, sum_to_one=True, min_weight=0.0, max_weight=0.4
            ),
            "method": "mean_variance",
        }

    @pytest.fixture
    def sample_result(self):
        """Create sample optimization result."""
        return OptimizationResult(
            request_id="opt_123",
            client_id="client_456",
            weights={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8,
            optimization_time=2.5,
            timestamp=time.time(),
            trigger=OptimizationTrigger.MANUAL,
            method="mean_variance",
            success=True,
        )

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.ttl == 300
        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

    def test_generate_key(self, cache, sample_params):
        """Test cache key generation."""
        key = cache._generate_key(**sample_params)

        # Key should be consistent for same parameters
        key2 = cache._generate_key(**sample_params)
        assert key == key2

        # Key should be different for different parameters
        sample_params["symbols"] = ["AAPL", "GOOGL"]  # Different symbols
        key3 = cache._generate_key(**sample_params)
        assert key != key3

    def test_generate_key_symbol_order(self, cache):
        """Test cache key is consistent regardless of symbol order."""
        key1 = cache._generate_key(
            symbols=["AAPL", "GOOGL", "MSFT"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
        )

        key2 = cache._generate_key(
            symbols=["MSFT", "AAPL", "GOOGL"],  # Different order
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
        )

        assert key1 == key2

    def test_set_and_get(self, cache, sample_params, sample_result):
        """Test setting and getting cached results."""
        # Set result
        cache.set(**sample_params, result=sample_result)

        # Get result
        cached_result = cache.get(**sample_params)
        assert cached_result == sample_result

        # Check access count
        key = cache._generate_key(**sample_params)
        assert cache.access_count[key] == 1

        # Get again
        cached_result2 = cache.get(**sample_params)
        assert cached_result2 == sample_result
        assert cache.access_count[key] == 2

    def test_cache_miss(self, cache, sample_params):
        """Test cache miss."""
        result = cache.get(**sample_params)
        assert result is None

    def test_cache_expiration(self, cache, sample_params, sample_result):
        """Test cache expiration."""
        # Set with very short TTL
        cache.ttl = 0.1  # 100ms
        cache.set(**sample_params, result=sample_result)

        # Should get result immediately
        assert cache.get(**sample_params) is not None

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get(**sample_params) is None

    def test_clear_expired(self, cache, sample_params, sample_result):
        """Test clearing expired entries."""
        # Add multiple entries with short TTL
        cache.ttl = 0.1

        # Add first entry
        cache.set(**sample_params, result=sample_result)

        # Add second entry with different method
        sample_params["method"] = "genetic_algorithm"
        cache.set(**sample_params, result=sample_result)

        assert len(cache.cache) == 2

        # Wait for expiration
        time.sleep(0.2)

        # Clear expired
        cache.clear_expired()

        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0

    def test_get_stats(self, cache, sample_params, sample_result):
        """Test cache statistics."""
        # Empty cache stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["total_accesses"] == 0
        assert stats["ttl_seconds"] == 300
        assert stats["most_accessed"] == []

        # Add some entries and access them
        cache.set(**sample_params, result=sample_result)
        cache.get(**sample_params)
        cache.get(**sample_params)

        # Different objective
        sample_params["objective"] = ObjectiveType.MINIMIZE_VARIANCE
        cache.set(**sample_params, result=sample_result)
        cache.get(**sample_params)

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_accesses"] == 3
        assert len(stats["most_accessed"]) == 2
        assert stats["most_accessed"][0][1] == 2  # First entry accessed twice


class TestWebSocketOptimizationService:
    """Test WebSocketOptimizationService class."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = Mock(spec=WebSocketConnectionManager)
        manager.send_to_client = AsyncMock()
        manager.connections = {"client_456": Mock()}
        return manager

    @pytest.fixture
    def mock_data_pipeline(self):
        """Create mock data pipeline."""
        pipeline = Mock(spec=DataPipeline)
        pipeline.get_market_data = Mock()
        pipeline.get_portfolios = Mock(return_value={})
        return pipeline

    @pytest.fixture
    def optimization_service(self, mock_connection_manager, mock_data_pipeline):
        """Create optimization service instance."""
        return WebSocketOptimizationService(
            connection_manager=mock_connection_manager,
            data_pipeline=mock_data_pipeline,
            cache_ttl=300,
            max_queue_size=100,
            optimization_interval=60.0,
        )

    def test_initialization(self, optimization_service):
        """Test service initialization."""
        assert optimization_service.cache.ttl == 300
        assert optimization_service.max_queue_size == 100
        assert optimization_service.optimization_interval == 60.0
        assert len(optimization_service.optimization_queue) == 0
        assert len(optimization_service.active_optimizations) == 0
        assert optimization_service._running is False
        assert optimization_service.stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, optimization_service):
        """Test starting and stopping the service."""
        # Start service
        await optimization_service.start()
        assert optimization_service._running is True
        assert len(optimization_service._tasks) == 3  # 3 background tasks

        # Start again should be no-op
        await optimization_service.start()
        assert len(optimization_service._tasks) == 3

        # Stop service
        await optimization_service.stop()
        assert optimization_service._running is False
        assert len(optimization_service._tasks) == 0

    @pytest.mark.asyncio
    async def test_request_optimization_manual(
        self, optimization_service, mock_connection_manager
    ):
        """Test manual optimization request."""
        constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

        request_id = await optimization_service.request_optimization(
            client_id="client_456",
            symbols=["AAPL", "GOOGL", "MSFT"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=constraints,
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
        )

        assert request_id.startswith("opt_client_456_")
        assert len(optimization_service.optimization_queue) == 1
        assert optimization_service.stats["total_requests"] == 1

        # Check acknowledgment sent
        mock_connection_manager.send_to_client.assert_called_once()
        call_args = mock_connection_manager.send_to_client.call_args
        assert call_args[0][0] == "client_456"
        message = call_args[0][1]
        assert message.message_type == MessageType.PORTFOLIO_UPDATE
        assert message.data["type"] == "optimization_queued"
        assert message.data["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_request_optimization_cached(
        self, optimization_service, mock_connection_manager
    ):
        """Test optimization request with cached result."""
        constraints = PortfolioConstraints()

        # Pre-populate cache
        cached_result = OptimizationResult(
            request_id="opt_cached",
            client_id="client_456",
            weights={"AAPL": 0.5, "GOOGL": 0.5},
            expected_return=0.10,
            expected_volatility=0.12,
            sharpe_ratio=0.83,
            optimization_time=1.5,
            timestamp=time.time(),
            trigger=OptimizationTrigger.TIME_BASED,
            method="mean_variance",
            success=True,
        )

        optimization_service.cache.set(
            symbols=["AAPL", "GOOGL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=constraints,
            method="mean_variance",
            result=cached_result,
        )

        # Request with automatic trigger (uses cache)
        request_id = await optimization_service.request_optimization(
            client_id="client_456",
            symbols=["AAPL", "GOOGL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=constraints,
            method="mean_variance",
            trigger=OptimizationTrigger.TIME_BASED,
        )

        # Should use cached result
        assert optimization_service.stats["cache_hits"] == 1
        assert len(optimization_service.optimization_queue) == 0  # Not queued

        # Check result sent directly
        assert mock_connection_manager.send_to_client.call_count == 1

    @pytest.mark.asyncio
    async def test_request_optimization_queue_full(
        self, optimization_service, mock_connection_manager
    ):
        """Test optimization request when queue is full."""
        # Fill the queue
        optimization_service.optimization_queue = deque(
            [Mock() for _ in range(100)], maxlen=100
        )

        # The queue is already full, cannot add more
        assert len(optimization_service.optimization_queue) == 100

        request_id = await optimization_service.request_optimization(
            client_id="client_456",
            symbols=["AAPL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
        )

        # Should not increase queue size
        assert len(optimization_service.optimization_queue) == 100

        # Check error sent
        mock_connection_manager.send_to_client.assert_called_once()
        message = mock_connection_manager.send_to_client.call_args[0][1]
        assert message.message_type == MessageType.ERROR
        assert "queue full" in message.data["error"]

    @pytest.mark.asyncio
    async def test_perform_optimization_mean_variance(
        self, optimization_service, mock_data_pipeline
    ):
        """Test performing mean-variance optimization."""
        # Mock market data
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[
                {"returns": np.array([0.01, 0.02, -0.01, 0.03, 0.01])},
                {"returns": np.array([0.02, -0.01, 0.01, 0.02, 0.00])},
                {"returns": np.array([-0.01, 0.01, 0.02, 0.01, 0.02])},
            ]
        )

        # Create request
        request = OptimizationRequest(
            request_id="opt_test",
            client_id="client_456",
            symbols=["AAPL", "GOOGL", "MSFT"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        # Mock optimizer
        mock_mv_result = MVOptimizationResult(
            weights=np.array([0.4, 0.3, 0.3]),
            expected_return=0.12,
            expected_variance=0.0225,
            sharpe_ratio=0.8,
            objective_value=0.8,
            solve_time=1.5,
            status="optimal",
            success=True,
        )

        with patch.object(
            optimization_service.mean_variance_optimizer,
            "optimize_portfolio",
            return_value=mock_mv_result,
        ):
            result = await optimization_service._perform_optimization(request)

        assert result.success is True
        assert result.request_id == "opt_test"
        assert result.weights["AAPL"] == 0.4
        assert result.expected_return == 0.12
        assert result.expected_volatility == pytest.approx(0.15)  # sqrt(0.0225)
        assert result.sharpe_ratio == 0.8
        assert result.method == "mean_variance"

    @pytest.mark.asyncio
    async def test_perform_optimization_classical(
        self, optimization_service, mock_data_pipeline
    ):
        """Test performing classical optimization (genetic algorithm)."""
        # Mock market data
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[
                {"returns": np.array([0.01, 0.02])},
                {"returns": np.array([0.02, -0.01])},
            ]
        )

        request = OptimizationRequest(
            request_id="opt_ga",
            client_id="client_456",
            symbols=["AAPL", "GOOGL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="genetic_algorithm",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        # Mock classical optimizer
        mock_result = Mock()
        mock_result.weights = np.array([0.6, 0.4])
        mock_result.expected_return = 0.11
        mock_result.expected_variance = 0.02
        mock_result.sharpe_ratio = 0.78
        mock_result.objective_value = 0.78
        mock_result.solve_time = 3.0
        mock_result.status = "optimal"
        mock_result.success = True

        with patch(
            "src.optimization.classical_solvers.ClassicalOptimizerFactory.create_optimizer"
        ) as mock_factory:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_result
            mock_factory.return_value = mock_optimizer

            result = await optimization_service._perform_optimization(request)

        assert result.success is True
        assert result.weights["AAPL"] == 0.6
        assert result.method == "genetic_algorithm"

    @pytest.mark.asyncio
    async def test_perform_optimization_vqe(
        self, optimization_service, mock_data_pipeline
    ):
        """Test performing VQE optimization."""
        # Mock market data
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[{"returns": np.array([0.01])}, {"returns": np.array([0.02])}]
        )

        request = OptimizationRequest(
            request_id="opt_vqe",
            client_id="client_456",
            symbols=["AAPL", "GOOGL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="vqe",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        # Mock VQE availability and optimizer
        with patch("src.streaming.optimization_service.VQE_AVAILABLE", True):
            with patch(
                "src.streaming.optimization_service.VQEOptimizer"
            ) as mock_vqe_class:
                mock_vqe = Mock()
                mock_result = Mock()
                mock_result.weights = np.array([0.5, 0.5])
                mock_result.expected_return = 0.10
                mock_result.expected_variance = 0.018
                mock_result.sharpe_ratio = 0.74
                mock_result.objective_value = 0.74
                mock_result.solve_time = 5.0
                mock_result.status = "optimal"
                mock_result.success = True

                mock_vqe.optimize_portfolio.return_value = mock_result
                mock_vqe_class.return_value = mock_vqe

                result = await optimization_service._perform_optimization(request)

        assert result.success is True
        assert result.method == "vqe"
        assert result.weights["AAPL"] == 0.5

    @pytest.mark.asyncio
    async def test_perform_optimization_vqe_not_available(
        self, optimization_service, mock_data_pipeline
    ):
        """Test VQE optimization when not available."""
        # Mock market data to avoid early failure
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[{"returns": np.array([0.01, 0.02])}]
        )

        request = OptimizationRequest(
            request_id="opt_vqe_fail",
            client_id="client_456",
            symbols=["AAPL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="vqe",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        with patch("src.streaming.optimization_service.VQE_AVAILABLE", False):
            result = await optimization_service._perform_optimization(request)

        assert result.success is False
        assert "VQE optimizer not available" in result.error

    @pytest.mark.asyncio
    async def test_perform_optimization_error(
        self, optimization_service, mock_data_pipeline
    ):
        """Test optimization error handling."""
        # Mock market data to return None (causing retrieval error)
        mock_data_pipeline.get_market_data = Mock(return_value=None)

        request = OptimizationRequest(
            request_id="opt_error",
            client_id="client_456",
            symbols=["AAPL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )

        result = await optimization_service._perform_optimization(request)

        assert result.success is False
        assert "Unable to retrieve market data" in result.error
        assert result.weights == {}

    @pytest.mark.asyncio
    async def test_get_market_data_with_returns(
        self, optimization_service, mock_data_pipeline
    ):
        """Test getting market data with returns data."""
        # Mock data pipeline responses
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[
                {"price": 150.0, "returns": np.array([0.01, 0.02, -0.01, 0.03, 0.01])},
                {"price": 2500.0, "returns": np.array([0.02, -0.01, 0.01, 0.02, 0.00])},
            ]
        )

        market_data = await optimization_service._get_market_data(["AAPL", "GOOGL"])

        assert market_data is not None
        assert "expected_returns" in market_data
        assert "covariance_matrix" in market_data
        assert "returns_data" in market_data
        assert market_data["expected_returns"].shape == (2,)
        assert market_data["covariance_matrix"].shape == (2, 2)

    @pytest.mark.asyncio
    async def test_get_market_data_fallback(
        self, optimization_service, mock_data_pipeline
    ):
        """Test market data fallback when no returns available."""
        # Mock data without returns
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[{"price": 150.0}, {"price": 2500.0}]
        )

        market_data = await optimization_service._get_market_data(["AAPL", "GOOGL"])

        assert market_data is not None
        assert "expected_returns" in market_data
        assert "covariance_matrix" in market_data
        assert market_data["returns_data"] is None
        # Check fallback values are reasonable
        assert np.all(market_data["expected_returns"] >= 0.05)
        assert np.all(market_data["expected_returns"] <= 0.15)

    @pytest.mark.asyncio
    async def test_process_optimization_queue(
        self, optimization_service, mock_connection_manager
    ):
        """Test processing optimization queue."""
        # Add request to queue
        request = OptimizationRequest(
            request_id="opt_queue",
            client_id="client_456",
            symbols=["AAPL"],
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            constraints=PortfolioConstraints(),
            method="mean_variance",
            trigger=OptimizationTrigger.MANUAL,
            timestamp=time.time(),
        )
        optimization_service.optimization_queue.append(request)

        # Mock successful optimization
        with patch.object(
            optimization_service, "_perform_optimization"
        ) as mock_perform:
            mock_result = OptimizationResult(
                request_id="opt_queue",
                client_id="client_456",
                weights={"AAPL": 1.0},
                expected_return=0.10,
                expected_volatility=0.15,
                sharpe_ratio=0.67,
                optimization_time=1.0,
                timestamp=time.time(),
                trigger=OptimizationTrigger.MANUAL,
                method="mean_variance",
                success=True,
            )
            mock_perform.return_value = mock_result

            # Create a task for queue processing
            optimization_service._running = True
            task = asyncio.create_task(
                optimization_service._process_optimization_queue()
            )

            # Wait for processing
            await asyncio.sleep(0.2)

            # Stop the service
            optimization_service._running = False

            # Cancel and wait for task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check request was processed
        assert len(optimization_service.optimization_queue) == 0
        assert optimization_service.stats["successful_optimizations"] == 1
        assert mock_connection_manager.send_to_client.called

    @pytest.mark.asyncio
    async def test_check_price_triggers(self, optimization_service, mock_data_pipeline):
        """Test price-based trigger checking."""
        # Mock portfolio data
        portfolio_data = {
            "portfolio_123": {
                "client_id": "client_456",
                "symbols": ["AAPL", "GOOGL"],
                "optimization_settings": {"price_threshold": 0.03},  # 3% threshold
                "objective": "maximize_sharpe",
                "constraints": {},
                "optimization_method": "mean_variance",
            }
        }
        mock_data_pipeline.get_portfolios.return_value = portfolio_data

        # Mock market data with significant price change
        mock_data_pipeline.get_market_data = Mock(
            side_effect=[
                {"price_change_percent": 0.05},  # 5% change for AAPL
                {"price_change_percent": 0.01},  # 1% change for GOOGL
            ]
        )

        # Mock request_optimization to track calls
        with patch.object(optimization_service, "request_optimization") as mock_request:
            await optimization_service._check_price_triggers()

            # Should trigger optimization due to AAPL price change
            mock_request.assert_called_once()
            call_args = mock_request.call_args[1]
            assert call_args["client_id"] == "client_456"
            assert call_args["trigger"] == OptimizationTrigger.PRICE_CHANGE
            assert call_args["metadata"]["portfolio_id"] == "portfolio_123"

    @pytest.mark.asyncio
    async def test_check_risk_triggers(self, optimization_service, mock_data_pipeline):
        """Test risk-based trigger checking."""
        # Mock portfolio data with high volatility scenario
        portfolio_data = {
            "portfolio_risk": {
                "client_id": "client_789",
                "symbols": ["AAPL", "GOOGL"],
                "weights": {"AAPL": 0.8, "GOOGL": 0.2},  # Concentrated portfolio
                "risk_settings": {"max_volatility": 0.15},  # 15% max volatility
                "constraints": {},
                "optimization_method": "mean_variance",
            }
        }
        mock_data_pipeline.get_portfolios.return_value = portfolio_data

        # Mock market data with high correlation
        with patch.object(optimization_service, "_get_market_data") as mock_get_data:
            mock_get_data.return_value = {
                "expected_returns": np.array([0.10, 0.12]),
                "covariance_matrix": np.array(
                    [[0.06, 0.05], [0.05, 0.08]]
                ),  # High volatility and correlation
                "timestamp": time.time(),
            }

            # Mock request_optimization
            with patch.object(
                optimization_service, "request_optimization"
            ) as mock_request:
                mock_request.return_value = asyncio.create_task(
                    asyncio.sleep(0)
                )  # Return a coroutine

                await optimization_service._check_risk_triggers()

                # Should trigger optimization due to high volatility
                # Portfolio variance = w^T * Cov * w
                # = [0.8, 0.2]^T * [[0.06, 0.05], [0.05, 0.08]] * [0.8, 0.2]
                # = [0.8, 0.2] * [0.058, 0.056]
                # = 0.0576
                # Portfolio volatility = sqrt(0.0576) = 0.24 > 0.15

                mock_request.assert_called_once()
                call_args = mock_request.call_args[1]
                assert call_args["client_id"] == "client_789"
                assert call_args["trigger"] == OptimizationTrigger.RISK_ALERT
                assert call_args["objective"] == ObjectiveType.MINIMIZE_VARIANCE

    def test_get_stats(self, optimization_service):
        """Test getting service statistics."""
        # Set some stats
        optimization_service.stats = {
            "total_requests": 100,
            "successful_optimizations": 95,
            "failed_optimizations": 5,
            "cache_hits": 30,
            "average_optimization_time": 2.5,
        }

        # Add some active items
        optimization_service.optimization_queue.append(Mock())
        optimization_service.active_optimizations["opt_1"] = Mock()

        stats = optimization_service.get_stats()

        assert stats["total_requests"] == 100
        assert stats["successful_optimizations"] == 95
        assert stats["queue_size"] == 1
        assert stats["active_optimizations"] == 1
        assert "cache_stats" in stats
        assert "optimizers" in stats


class TestOptimizationServiceIntegration:
    """Integration tests for optimization service."""

    @pytest.mark.asyncio
    async def test_full_optimization_flow(self):
        """Test complete optimization flow from request to result."""
        # Create real instances with mocks
        connection_manager = Mock(spec=WebSocketConnectionManager)
        connection_manager.send_to_client = AsyncMock()
        connection_manager.connections = {"client_test": Mock()}

        data_pipeline = Mock(spec=DataPipeline)
        data_pipeline.get_portfolios = Mock(return_value={})

        service = WebSocketOptimizationService(
            connection_manager=connection_manager,
            data_pipeline=data_pipeline,
            cache_ttl=300,
            max_queue_size=10,
        )

        # Mock market data
        data_pipeline.get_market_data = Mock(
            side_effect=[
                {"returns": np.array([0.01, 0.02, -0.01, 0.03])},
                {"returns": np.array([0.02, -0.01, 0.01, 0.02])},
                {"returns": np.array([-0.01, 0.01, 0.02, 0.01])},
            ]
        )

        # Start service
        await service.start()

        try:
            # Submit optimization request
            request_id = await service.request_optimization(
                client_id="client_test",
                symbols=["AAPL", "GOOGL", "MSFT"],
                objective=ObjectiveType.MAXIMIZE_SHARPE,
                constraints=PortfolioConstraints(
                    long_only=True, sum_to_one=True, min_weight=0.1, max_weight=0.5
                ),
                method="mean_variance",
            )

            # Wait for processing
            await asyncio.sleep(0.5)

            # Check results
            assert service.stats["total_requests"] == 1
            assert (
                request_id in service.last_optimization.values()
                or service.stats["successful_optimizations"] > 0
            )

            # Verify message sent
            assert connection_manager.send_to_client.called

        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_concurrent_optimization_requests(self):
        """Test handling multiple concurrent optimization requests."""
        connection_manager = Mock(spec=WebSocketConnectionManager)
        connection_manager.send_to_client = AsyncMock()
        connection_manager.connections = {
            "client_1": Mock(),
            "client_2": Mock(),
            "client_3": Mock(),
        }

        data_pipeline = Mock(spec=DataPipeline)
        data_pipeline.get_portfolios = Mock(return_value={})
        data_pipeline.get_market_data = Mock(
            return_value={"returns": np.array([0.01, 0.02])}
        )

        service = WebSocketOptimizationService(
            connection_manager=connection_manager, data_pipeline=data_pipeline
        )

        await service.start()

        try:
            # Submit multiple requests
            tasks = []
            for i in range(3):
                task = service.request_optimization(
                    client_id=f"client_{i+1}",
                    symbols=["AAPL", "GOOGL"],
                    objective=ObjectiveType.MAXIMIZE_SHARPE,
                    constraints=PortfolioConstraints(),
                    method="mean_variance",
                )
                tasks.append(task)

            request_ids = await asyncio.gather(*tasks)

            # All requests should be queued
            assert len(request_ids) == 3
            assert service.stats["total_requests"] == 3

        finally:
            await service.stop()
