"""Tests for data pipeline module."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.streaming.data_pipeline import (
    DataPipeline,
    MarketDataUpdate,
    PortfolioUpdate,
    RiskUpdate,
)
from src.streaming.websocket import WebSocketConnectionManager


class TestMarketDataUpdate:
    """Test MarketDataUpdate dataclass."""

    def test_market_data_creation(self):
        """Test creating market data update."""
        update = MarketDataUpdate(
            symbol="AAPL",
            price=150.50,
            change=2.50,
            change_percent=0.0169,
            volume=1000000,
            timestamp=datetime.now(),
        )

        assert update.symbol == "AAPL"
        assert update.price == 150.50
        assert update.change == 2.50
        assert update.change_percent == 0.0169
        assert update.volume == 1000000
        assert update.timestamp is not None

    def test_market_data_to_dict(self):
        """Test converting market data to dict."""
        timestamp = datetime.now()
        update = MarketDataUpdate(
            symbol="AAPL",
            price=150.50,
            change=2.50,
            change_percent=0.0169,
            volume=1000000,
            timestamp=timestamp,
        )

        data = update.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["price"] == 150.50
        assert data["change"] == 2.50
        assert data["change_percent"] == 0.0169
        assert data["volume"] == 1000000
        assert data["timestamp"] == timestamp.isoformat()


class TestPortfolioUpdate:
    """Test PortfolioUpdate dataclass."""

    def test_portfolio_update_creation(self):
        """Test creating portfolio update."""
        positions = {
            "AAPL": {"shares": 100, "value": 15000, "weight": 0.3},
            "GOOGL": {"shares": 50, "value": 140000, "weight": 0.7},
        }

        update = PortfolioUpdate(
            portfolio_id="test_123",
            total_value=155000,
            change=5000,
            change_percent=0.0333,
            positions=positions,
            timestamp=datetime.now(),
        )

        assert update.portfolio_id == "test_123"
        assert update.total_value == 155000
        assert update.change == 5000
        assert update.change_percent == 0.0333
        assert update.positions == positions
        assert update.timestamp is not None

    def test_portfolio_update_to_dict(self):
        """Test converting portfolio update to dict."""
        timestamp = datetime.now()
        positions = {"AAPL": {"shares": 100, "value": 15000, "weight": 0.3}}

        update = PortfolioUpdate(
            portfolio_id="test_123",
            total_value=15000,
            change=1000,
            change_percent=0.0714,
            positions=positions,
            timestamp=timestamp,
        )

        data = update.to_dict()
        assert data["portfolio_id"] == "test_123"
        assert data["total_value"] == 15000
        assert data["positions"] == positions
        assert data["timestamp"] == timestamp.isoformat()


class TestRiskUpdate:
    """Test RiskUpdate dataclass."""

    def test_risk_update_creation(self):
        """Test creating risk update."""
        update = RiskUpdate(
            portfolio_id="test_123",
            volatility=0.15,
            var_95=0.05,
            cvar_95=0.08,
            beta=1.2,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            timestamp=datetime.now(),
        )

        assert update.portfolio_id == "test_123"
        assert update.volatility == 0.15
        assert update.var_95 == 0.05
        assert update.cvar_95 == 0.08
        assert update.beta == 1.2
        assert update.sharpe_ratio == 1.5
        assert update.max_drawdown == 0.10
        assert update.timestamp is not None

    def test_risk_update_to_dict(self):
        """Test converting risk update to dict."""
        timestamp = datetime.now()
        update = RiskUpdate(
            portfolio_id="test_123",
            volatility=0.15,
            var_95=0.05,
            cvar_95=0.08,
            beta=1.2,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            timestamp=timestamp,
        )

        data = update.to_dict()
        assert data["portfolio_id"] == "test_123"
        assert data["volatility"] == 0.15
        assert data["sharpe_ratio"] == 1.5
        assert data["timestamp"] == timestamp.isoformat()


class TestDataPipeline:
    """Test DataPipeline functionality."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock WebSocket connection manager."""
        manager = Mock(spec=WebSocketConnectionManager)
        manager.broadcast = AsyncMock()
        manager.send_to_client = AsyncMock()
        manager.broadcast_to_topic = AsyncMock()
        return manager

    @pytest.fixture
    def pipeline(self, mock_connection_manager):
        """Create DataPipeline instance."""
        return DataPipeline(
            connection_manager=mock_connection_manager, update_interval=1
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, pipeline):
        """Test starting and stopping the pipeline."""
        # Start pipeline
        await pipeline.start()
        assert pipeline._running is True

        # Stop pipeline
        await pipeline.stop()
        assert pipeline._running is False

    def test_register_market_data_source(self, pipeline):
        """Test registering market data source."""

        async def mock_source():
            return {"symbol": "AAPL", "price": 150}

        pipeline.register_market_data_source("yahoo", mock_source)
        assert "yahoo" in pipeline.market_data_sources
        assert pipeline.market_data_sources["yahoo"] == mock_source

    def test_register_portfolio_calculator(self, pipeline):
        """Test registering portfolio calculator."""

        async def mock_calculator(portfolio_id):
            return {"total_value": 100000}

        pipeline.register_portfolio_calculator("test_123", mock_calculator)
        assert "test_123" in pipeline.portfolio_calculators
        assert pipeline.portfolio_calculators["test_123"] == mock_calculator

    def test_register_risk_calculator(self, pipeline):
        """Test registering risk calculator."""

        async def mock_calculator(portfolio_id):
            return {"volatility": 0.15}

        pipeline.register_risk_calculator("test_123", mock_calculator)
        assert "test_123" in pipeline.risk_calculators
        assert pipeline.risk_calculators["test_123"] == mock_calculator

    @pytest.mark.asyncio
    async def test_subscribe_market_data(self, pipeline):
        """Test subscribing to market data."""
        await pipeline.subscribe_market_data(["AAPL", "GOOGL"])
        assert "AAPL" in pipeline.market_data_subscribers
        assert "GOOGL" in pipeline.market_data_subscribers

    @pytest.mark.asyncio
    async def test_subscribe_portfolio(self, pipeline):
        """Test subscribing to portfolio updates."""
        await pipeline.subscribe_portfolio("test_123")
        assert "test_123" in pipeline.portfolio_subscribers

    @pytest.mark.asyncio
    async def test_subscribe_risk_metrics(self, pipeline):
        """Test subscribing to risk metrics."""
        await pipeline.subscribe_risk_metrics("test_123")
        assert "test_123" in pipeline.risk_subscribers

    @pytest.mark.asyncio
    async def test_send_alert(self, pipeline, mock_connection_manager):
        """Test sending alert."""
        await pipeline.send_alert(
            alert_type="portfolio_threshold",
            message="Portfolio down 5%",
            severity="warning",
        )

        # Should broadcast alert
        mock_connection_manager.broadcast_to_topic.assert_called_once()

        # Check the alert message format
        call_args = mock_connection_manager.broadcast_to_topic.call_args
        topic = call_args[0][0]
        message = call_args[0][1]

        assert topic == "alerts"
        assert message.message_type.value == "alert"
        assert message.data["type"] == "portfolio_threshold"
        assert message.data["message"] == "Portfolio down 5%"
        assert message.data["severity"] == "warning"

    def test_get_cached_data(self, pipeline):
        """Test getting cached data."""
        # Add some cached data
        pipeline.market_data_cache["AAPL"] = MarketDataUpdate(
            symbol="AAPL",
            price=150.50,
            change=2.50,
            change_percent=0.0169,
            volume=1000000,
            timestamp=datetime.now(),
        )
        pipeline.portfolio_cache["test_123"] = PortfolioUpdate(
            portfolio_id="test_123",
            total_value=100000,
            change=5000,
            change_percent=0.05,
            positions={},
            timestamp=datetime.now(),
        )

        cached = pipeline.get_cached_data()

        assert "market_data" in cached
        assert "portfolios" in cached
        assert "risk_metrics" in cached

        assert "AAPL" in cached["market_data"]
        assert cached["market_data"]["AAPL"]["price"] == 150.50

        assert "test_123" in cached["portfolios"]
        assert cached["portfolios"]["test_123"]["total_value"] == 100000

    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, pipeline):
        """Test handling multiple concurrent subscriptions."""
        # Subscribe to multiple data types concurrently
        await asyncio.gather(
            pipeline.subscribe_market_data(["AAPL", "GOOGL"]),
            pipeline.subscribe_portfolio("portfolio_1"),
            pipeline.subscribe_portfolio("portfolio_2"),
            pipeline.subscribe_risk_metrics("portfolio_1"),
        )

        assert len(pipeline.market_data_subscribers) == 2
        assert len(pipeline.portfolio_subscribers) == 2
        assert len(pipeline.risk_subscribers) == 1

    @pytest.mark.asyncio
    async def test_pipeline_lifecycle(self, pipeline):
        """Test complete pipeline lifecycle."""

        # Register sources
        async def mock_market_source():
            return MarketDataUpdate(
                symbol="AAPL",
                price=150.50,
                change=2.50,
                change_percent=0.0169,
                volume=1000000,
                timestamp=datetime.now(),
            )

        pipeline.register_market_data_source("yahoo", mock_market_source)

        # Subscribe to data
        await pipeline.subscribe_market_data(["AAPL"])

        # Start pipeline
        await pipeline.start()
        assert pipeline._running is True

        # Stop pipeline
        await pipeline.stop()
        assert pipeline._running is False

        # Verify cleanup
        assert not pipeline._running
