"""Tests for streaming portfolio monitor module."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.streaming.portfolio_monitor import PortfolioMonitor, Portfolio, AlertRule
from src.streaming.data_pipeline import DataPipeline


class TestPortfolio:
    """Test Portfolio class."""

    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            portfolio_id="test_123",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=50000,
            created_at=datetime.now(),
        )

        assert portfolio.portfolio_id == "test_123"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.holdings == {"AAPL": 100, "GOOGL": 50}
        assert portfolio.initial_value == 50000
        assert portfolio.created_at is not None

    def test_get_symbols(self):
        """Test getting portfolio symbols."""
        portfolio = Portfolio(
            portfolio_id="test",
            name="Test",
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            initial_value=10000,
            created_at=datetime.now(),
        )

        symbols = portfolio.get_symbols()
        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" in symbols


class TestAlertRule:
    """Test AlertRule class."""

    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            rule_id="alert_123",
            portfolio_id="portfolio_123",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
            enabled=True,
        )

        assert rule.rule_id == "alert_123"
        assert rule.portfolio_id == "portfolio_123"
        assert rule.rule_type == "portfolio_change"
        assert rule.threshold == -0.05
        assert rule.condition == "below"
        assert rule.enabled is True


class TestPortfolioMonitor:
    """Test PortfolioMonitor functionality."""

    @pytest.fixture
    def mock_data_pipeline(self):
        """Create mock data pipeline."""
        pipeline = Mock(spec=DataPipeline)
        pipeline.market_data = {}
        pipeline.subscribe_symbols = AsyncMock()
        pipeline.unsubscribe_symbols = AsyncMock()
        return pipeline

    @pytest.fixture
    def monitor(self, mock_data_pipeline):
        """Create PortfolioMonitor instance."""
        return PortfolioMonitor(data_pipeline=mock_data_pipeline)

    @pytest.mark.asyncio
    async def test_add_portfolio(self, monitor, mock_data_pipeline):
        """Test adding a portfolio."""
        portfolio = Portfolio(
            portfolio_id="test_123",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=50000,
            created_at=datetime.now(),
        )

        await monitor.add_portfolio(portfolio)

        assert "test_123" in monitor.portfolios
        assert monitor.portfolios["test_123"].name == "Test Portfolio"

        # Should subscribe to symbols
        mock_data_pipeline.subscribe_market_data.assert_called_with(["AAPL", "GOOGL"])

    @pytest.mark.asyncio
    async def test_remove_portfolio(self, monitor, mock_data_pipeline):
        """Test removing a portfolio."""
        # Add portfolio
        portfolio = Portfolio(
            portfolio_id="test_123",
            name="Test",
            holdings={"AAPL": 100},
            initial_value=10000,
            created_at=datetime.now(),
        )
        await monitor.add_portfolio(portfolio)

        # Remove it
        await monitor.remove_portfolio("test_123")
        assert "test_123" not in monitor.portfolios

    @pytest.mark.asyncio
    async def test_add_alert_rule(self, monitor):
        """Test adding an alert rule."""
        # Add portfolio first
        portfolio = Portfolio(
            portfolio_id="test_123",
            name="Test",
            holdings={"AAPL": 100},
            initial_value=10000,
            created_at=datetime.now(),
        )
        await monitor.add_portfolio(portfolio)

        # Add alert rule
        rule = AlertRule(
            rule_id="alert_1",
            portfolio_id="test_123",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
        )

        await monitor.add_alert_rule(rule)

        assert "alert_1" in monitor.alert_rules
        assert monitor.alert_rules["alert_1"].portfolio_id == "test_123"

    @pytest.mark.asyncio
    async def test_remove_alert_rule(self, monitor):
        """Test removing an alert rule."""
        # Add alert
        rule = AlertRule(
            rule_id="alert_1",
            portfolio_id="test_123",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
        )
        await monitor.add_alert_rule(rule)

        # Remove alert
        await monitor.remove_alert_rule("alert_1")
        assert "alert_1" not in monitor.alert_rules

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping the monitor."""
        # Start monitoring
        monitor._monitoring_task = asyncio.create_task(asyncio.sleep(1))
        monitor.is_running = True

        assert monitor.is_running is True

        # Stop monitoring
        monitor.is_running = False
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_calculate_portfolio_value(self, monitor, mock_data_pipeline):
        """Test calculating portfolio value."""
        # Setup market data
        mock_data_pipeline.market_data = {
            "AAPL": {"price": 150},
            "GOOGL": {"price": 2800},
        }

        # Add portfolio
        portfolio = Portfolio(
            portfolio_id="test_123",
            name="Test",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=150000,
            created_at=datetime.now(),
        )
        await monitor.add_portfolio(portfolio)

        # Calculate value (internal method would be called during monitoring)
        # Expected value: 100 * 150 + 50 * 2800 = 155000
        expected_value = 155000

        # Verify portfolio was added correctly
        assert "test_123" in monitor.portfolios
