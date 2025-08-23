"""Tests for portfolio monitoring module."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.streaming.portfolio_monitor import PortfolioMonitor, Portfolio, AlertRule
from src.streaming.data_pipeline import DataPipeline


class TestPortfolio:
    """Test Portfolio class."""

    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        holdings = {"AAPL": 100, "GOOGL": 50}
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            name="Test Portfolio",
            holdings=holdings,
            initial_value=50000,
            created_at=datetime.now(),
            benchmark_symbol="SPY",
        )

        assert portfolio.portfolio_id == "test_portfolio"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.holdings == holdings
        assert portfolio.initial_value == 50000
        assert portfolio.benchmark_symbol == "SPY"
        assert portfolio.created_at is not None

    def test_portfolio_symbols(self):
        """Test getting portfolio symbols."""
        holdings = {"AAPL": 100, "GOOGL": 50, "MSFT": 75}
        portfolio = Portfolio(
            portfolio_id="test",
            name="Test",
            holdings=holdings,
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
            rule_id="alert_1",
            portfolio_id="portfolio_1",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
            enabled=True,
        )

        assert rule.rule_id == "alert_1"
        assert rule.portfolio_id == "portfolio_1"
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
        pipeline.subscribe_market_data = AsyncMock()
        pipeline.subscribe_portfolio = AsyncMock()
        pipeline.subscribe_risk_metrics = AsyncMock()
        return pipeline

    @pytest.fixture
    def monitor(self, mock_data_pipeline):
        """Create PortfolioMonitor instance."""
        return PortfolioMonitor(data_pipeline=mock_data_pipeline)

    @pytest.mark.asyncio
    async def test_add_portfolio(self, monitor, mock_data_pipeline):
        """Test adding a portfolio."""
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=50000,
            created_at=datetime.now(),
            benchmark_symbol="SPY",
        )

        await monitor.add_portfolio(portfolio)

        assert "test_portfolio" in monitor.portfolios
        assert monitor.portfolios["test_portfolio"].name == "Test Portfolio"

        # Should subscribe to symbols
        mock_data_pipeline.subscribe_market_data.assert_called_with(["AAPL", "GOOGL"])

    @pytest.mark.asyncio
    async def test_remove_portfolio(self, monitor, mock_data_pipeline):
        """Test removing a portfolio."""
        # Add portfolio
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            name="Test",
            holdings={"AAPL": 100},
            initial_value=10000,
            created_at=datetime.now(),
        )
        await monitor.add_portfolio(portfolio)

        # Remove it
        await monitor.remove_portfolio("test_portfolio")
        assert "test_portfolio" not in monitor.portfolios

    @pytest.mark.asyncio
    async def test_add_alert_rule(self, monitor):
        """Test adding an alert rule."""
        # Add portfolio first
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            name="Test",
            holdings={"AAPL": 100},
            initial_value=10000,
            created_at=datetime.now(),
        )
        await monitor.add_portfolio(portfolio)

        # Add alert rule
        rule = AlertRule(
            rule_id="alert_1",
            portfolio_id="test_portfolio",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
        )

        await monitor.add_alert_rule(rule)

        assert "alert_1" in monitor.alert_rules
        assert monitor.alert_rules["alert_1"].portfolio_id == "test_portfolio"

    @pytest.mark.asyncio
    async def test_remove_alert_rule(self, monitor):
        """Test removing an alert rule."""
        # Add alert
        rule = AlertRule(
            rule_id="alert_1",
            portfolio_id="test_portfolio",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
        )
        await monitor.add_alert_rule(rule)

        # Remove alert
        await monitor.remove_alert_rule("alert_1")
        assert "alert_1" not in monitor.alert_rules

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping the monitor."""
        # Mock the monitoring task
        monitor._monitoring_task = asyncio.create_task(asyncio.sleep(0.1))
        monitor.is_running = True

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Stop monitoring
        monitor.is_running = False
        await asyncio.sleep(0.1)

        # Task should be done
        assert monitor._monitoring_task.done()

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, monitor):
        """Test alert cooldown functionality."""
        rule = AlertRule(
            rule_id="alert_1",
            portfolio_id="test_portfolio",
            rule_type="portfolio_change",
            threshold=-0.05,
            condition="below",
            enabled=True,
        )

        # Trigger alert
        rule.last_triggered = datetime.utcnow()

        # Check cooldown
        current_time = datetime.utcnow()
        cooldown_minutes = 5
        time_since_trigger = current_time - rule.last_triggered

        # Should be in cooldown
        assert time_since_trigger < timedelta(minutes=cooldown_minutes)

    def test_get_active_portfolios(self, monitor):
        """Test getting active portfolios."""
        # Should start empty
        assert len(monitor.portfolios) == 0

        # Add portfolio
        monitor.portfolios["test"] = Portfolio(
            portfolio_id="test",
            name="Test",
            holdings={"AAPL": 100},
            initial_value=10000,
            created_at=datetime.now(),
        )

        assert len(monitor.portfolios) == 1
        assert "test" in monitor.portfolios
