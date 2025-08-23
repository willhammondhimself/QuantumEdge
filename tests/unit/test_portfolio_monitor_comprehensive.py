"""Comprehensive tests for portfolio monitor module."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from collections import deque

from src.streaming.portfolio_monitor import Portfolio, AlertRule, PortfolioMonitor
from src.streaming.data_pipeline import (
    DataPipeline,
    MarketDataUpdate,
    PortfolioUpdate,
    RiskUpdate,
)


class TestPortfolio:
    """Test Portfolio dataclass."""

    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            portfolio_id="port_123",
            name="Growth Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            initial_value=50000.0,
            created_at=datetime.now(timezone.utc),
            benchmark_symbol="SPY",
            risk_free_rate=0.02,
        )

        assert portfolio.portfolio_id == "port_123"
        assert portfolio.name == "Growth Portfolio"
        assert len(portfolio.holdings) == 3
        assert portfolio.holdings["AAPL"] == 100
        assert portfolio.initial_value == 50000.0
        assert portfolio.benchmark_symbol == "SPY"
        assert portfolio.risk_free_rate == 0.02

    def test_get_symbols(self):
        """Test getting portfolio symbols."""
        portfolio = Portfolio(
            portfolio_id="port_123",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=30000.0,
            created_at=datetime.now(timezone.utc),
        )

        symbols = portfolio.get_symbols()
        assert len(symbols) == 2
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_get_total_shares(self):
        """Test getting total shares."""
        portfolio = Portfolio(
            portfolio_id="port_123",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            initial_value=30000.0,
            created_at=datetime.now(timezone.utc),
        )

        total_shares = portfolio.get_total_shares()
        assert total_shares == 225  # 100 + 50 + 75

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        portfolio = Portfolio(
            portfolio_id="port_empty",
            name="Empty Portfolio",
            holdings={},
            initial_value=10000.0,
            created_at=datetime.now(timezone.utc),
        )

        assert portfolio.get_symbols() == []
        assert portfolio.get_total_shares() == 0


class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            rule_id="rule_123",
            portfolio_id="port_123",
            rule_type="portfolio_change",
            threshold=0.05,
            condition="change_up",
            enabled=True,
            cooldown_minutes=15,
        )

        assert rule.rule_id == "rule_123"
        assert rule.portfolio_id == "port_123"
        assert rule.rule_type == "portfolio_change"
        assert rule.threshold == 0.05
        assert rule.condition == "change_up"
        assert rule.enabled is True
        assert rule.last_triggered is None
        assert rule.cooldown_minutes == 15

    def test_alert_rule_with_last_triggered(self):
        """Test alert rule with last triggered time."""
        trigger_time = datetime.now(timezone.utc)
        rule = AlertRule(
            rule_id="rule_123",
            portfolio_id="port_123",
            rule_type="risk_threshold",
            threshold=0.20,
            condition="volatility_above",
            last_triggered=trigger_time,
        )

        assert rule.last_triggered == trigger_time


class TestPortfolioMonitor:
    """Test PortfolioMonitor class."""

    @pytest.fixture
    def data_pipeline(self):
        """Create mock data pipeline."""
        pipeline = Mock(spec=DataPipeline)
        pipeline.register_portfolio_calculator = Mock()
        pipeline.register_risk_calculator = Mock()
        pipeline.subscribe_market_data = AsyncMock()
        pipeline.subscribe_portfolio = AsyncMock()
        pipeline.subscribe_risk_metrics = AsyncMock()
        pipeline.send_alert = AsyncMock()
        return pipeline

    @pytest.fixture
    def portfolio_monitor(self, data_pipeline):
        """Create portfolio monitor instance."""
        return PortfolioMonitor(data_pipeline)

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return Portfolio(
            portfolio_id="port_test",
            name="Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            initial_value=50000.0,
            created_at=datetime.now(timezone.utc),
        )

    def test_initialization(self, portfolio_monitor, data_pipeline):
        """Test portfolio monitor initialization."""
        assert portfolio_monitor.data_pipeline == data_pipeline
        assert len(portfolio_monitor.portfolios) == 0
        assert len(portfolio_monitor.alert_rules) == 0
        assert portfolio_monitor.max_history_length == 252
        assert isinstance(portfolio_monitor._risk_cache_duration, timedelta)

    @pytest.mark.asyncio
    async def test_add_portfolio(
        self, portfolio_monitor, sample_portfolio, data_pipeline
    ):
        """Test adding a portfolio."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Check portfolio was added
        assert sample_portfolio.portfolio_id in portfolio_monitor.portfolios
        assert (
            portfolio_monitor.portfolios[sample_portfolio.portfolio_id]
            == sample_portfolio
        )

        # Check price history initialized
        for symbol in sample_portfolio.get_symbols():
            assert symbol in portfolio_monitor.price_history
            assert isinstance(portfolio_monitor.price_history[symbol], deque)

        # Check portfolio history initialized
        assert sample_portfolio.portfolio_id in portfolio_monitor.portfolio_history

        # Check calculators registered
        assert data_pipeline.register_portfolio_calculator.called
        assert data_pipeline.register_risk_calculator.called

        # Check subscriptions
        data_pipeline.subscribe_market_data.assert_called_once_with(
            sample_portfolio.get_symbols()
        )
        data_pipeline.subscribe_portfolio.assert_called_once_with(
            sample_portfolio.portfolio_id
        )
        data_pipeline.subscribe_risk_metrics.assert_called_once_with(
            sample_portfolio.portfolio_id
        )

    @pytest.mark.asyncio
    async def test_remove_portfolio(self, portfolio_monitor, sample_portfolio):
        """Test removing a portfolio."""
        # First add the portfolio
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add some alert rules
        rule = AlertRule(
            rule_id="rule_1",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="portfolio_change",
            threshold=0.05,
            condition="change_up",
        )
        await portfolio_monitor.add_alert_rule(rule)

        # Remove portfolio
        await portfolio_monitor.remove_portfolio(sample_portfolio.portfolio_id)

        # Check portfolio removed
        assert sample_portfolio.portfolio_id not in portfolio_monitor.portfolios
        assert sample_portfolio.portfolio_id not in portfolio_monitor.portfolio_history

        # Check alert rules removed
        assert rule.rule_id not in portfolio_monitor.alert_rules

    @pytest.mark.asyncio
    async def test_portfolio_calculator(self, portfolio_monitor, sample_portfolio):
        """Test portfolio value calculator."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Get the calculator function
        calculator = portfolio_monitor._create_portfolio_calculator(sample_portfolio)

        # Create market data
        market_data = {
            "AAPL": MarketDataUpdate(
                symbol="AAPL",
                price=150.0,
                change=1.0,
                change_percent=0.0067,
                volume=1000000,
                timestamp=datetime.now(timezone.utc),
            ),
            "GOOGL": MarketDataUpdate(
                symbol="GOOGL",
                price=2500.0,
                change=10.0,
                change_percent=0.004,
                volume=500000,
                timestamp=datetime.now(timezone.utc),
            ),
            "MSFT": MarketDataUpdate(
                symbol="MSFT",
                price=300.0,
                change=2.0,
                change_percent=0.0067,
                volume=800000,
                timestamp=datetime.now(timezone.utc),
            ),
        }

        # Calculate portfolio value
        update = await calculator(market_data)

        assert isinstance(update, PortfolioUpdate)
        assert update.portfolio_id == sample_portfolio.portfolio_id

        # Check calculations
        # AAPL: 100 * 150 = 15,000
        # GOOGL: 50 * 2500 = 125,000
        # MSFT: 75 * 300 = 22,500
        # Total: 162,500
        assert update.total_value == 162500.0
        assert update.change == 112500.0  # 162,500 - 50,000
        assert update.change_percent == pytest.approx(2.25)  # 112,500 / 50,000

        # Check positions
        assert len(update.positions) == 3
        assert update.positions["AAPL"]["value"] == 15000.0
        assert update.positions["GOOGL"]["value"] == 125000.0
        assert update.positions["MSFT"]["value"] == 22500.0

        # Check weights
        assert update.positions["AAPL"]["weight"] == pytest.approx(15000.0 / 162500.0)
        assert update.positions["GOOGL"]["weight"] == pytest.approx(125000.0 / 162500.0)
        assert update.positions["MSFT"]["weight"] == pytest.approx(22500.0 / 162500.0)

    @pytest.mark.asyncio
    async def test_portfolio_calculator_missing_data(
        self, portfolio_monitor, sample_portfolio
    ):
        """Test portfolio calculator with missing market data."""
        await portfolio_monitor.add_portfolio(sample_portfolio)
        calculator = portfolio_monitor._create_portfolio_calculator(sample_portfolio)

        # Market data missing MSFT
        market_data = {
            "AAPL": MarketDataUpdate(
                symbol="AAPL",
                price=150.0,
                change=1.0,
                change_percent=0.0067,
                volume=1000000,
                timestamp=datetime.now(timezone.utc),
            ),
            "GOOGL": MarketDataUpdate(
                symbol="GOOGL",
                price=2500.0,
                change=10.0,
                change_percent=0.004,
                volume=500000,
                timestamp=datetime.now(timezone.utc),
            ),
        }

        update = await calculator(market_data)

        # Should still calculate with available data
        assert update is not None
        assert update.total_value == 140000.0  # 15,000 + 125,000 (no MSFT)
        assert len(update.positions) == 2

    @pytest.mark.asyncio
    async def test_risk_calculator(self, portfolio_monitor, sample_portfolio):
        """Test risk metrics calculator."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add sufficient portfolio history
        portfolio_history = portfolio_monitor.portfolio_history[
            sample_portfolio.portfolio_id
        ]
        # Simulate 50 days of portfolio values with some volatility
        np.random.seed(42)
        base_value = 50000.0
        for i in range(50):
            value = base_value * (1 + np.random.normal(0.0005, 0.02))
            portfolio_history.append(value)
            base_value = value

        # Get the calculator
        calculator = portfolio_monitor._create_risk_calculator(sample_portfolio)

        # Calculate risk metrics
        risk_update = await calculator({}, None)

        assert isinstance(risk_update, RiskUpdate)
        assert risk_update.portfolio_id == sample_portfolio.portfolio_id
        assert risk_update.volatility > 0
        assert risk_update.var_95 < 0  # VaR should be negative
        assert (
            risk_update.cvar_95 <= risk_update.var_95
        )  # CVaR should be worse than VaR
        assert risk_update.sharpe_ratio != 0
        assert risk_update.max_drawdown <= 0
        assert risk_update.beta == 1.0  # Default beta

    @pytest.mark.asyncio
    async def test_risk_calculator_insufficient_data(
        self, portfolio_monitor, sample_portfolio
    ):
        """Test risk calculator with insufficient data."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add only a few data points
        portfolio_history = portfolio_monitor.portfolio_history[
            sample_portfolio.portfolio_id
        ]
        for i in range(10):  # Less than required 30
            portfolio_history.append(50000.0 + i * 100)

        calculator = portfolio_monitor._create_risk_calculator(sample_portfolio)
        risk_update = await calculator({}, None)

        assert risk_update is None  # Should return None with insufficient data

    @pytest.mark.asyncio
    async def test_risk_calculator_caching(self, portfolio_monitor, sample_portfolio):
        """Test risk calculator caching."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add portfolio history
        portfolio_history = portfolio_monitor.portfolio_history[
            sample_portfolio.portfolio_id
        ]
        for i in range(50):
            portfolio_history.append(50000.0 + i * 100)

        calculator = portfolio_monitor._create_risk_calculator(sample_portfolio)

        # First call
        risk_update1 = await calculator({}, None)
        assert risk_update1 is not None

        # Second call should return cached result
        risk_update2 = await calculator({}, None)
        assert risk_update2 == risk_update1  # Same instance from cache

    @pytest.mark.asyncio
    async def test_add_alert_rule(self, portfolio_monitor):
        """Test adding alert rules."""
        rule = AlertRule(
            rule_id="rule_1",
            portfolio_id="port_123",
            rule_type="portfolio_change",
            threshold=0.05,
            condition="change_up",
        )

        await portfolio_monitor.add_alert_rule(rule)

        assert rule.rule_id in portfolio_monitor.alert_rules
        assert portfolio_monitor.alert_rules[rule.rule_id] == rule

    @pytest.mark.asyncio
    async def test_remove_alert_rule(self, portfolio_monitor):
        """Test removing alert rules."""
        rule = AlertRule(
            rule_id="rule_1",
            portfolio_id="port_123",
            rule_type="portfolio_change",
            threshold=0.05,
            condition="change_up",
        )

        await portfolio_monitor.add_alert_rule(rule)
        await portfolio_monitor.remove_alert_rule(rule.rule_id)

        assert rule.rule_id not in portfolio_monitor.alert_rules

    @pytest.mark.asyncio
    async def test_portfolio_alerts(
        self, portfolio_monitor, sample_portfolio, data_pipeline
    ):
        """Test portfolio alert checking."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add alert rules
        rule_up = AlertRule(
            rule_id="rule_up",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="portfolio_change",
            threshold=0.02,  # 2% threshold
            condition="change_up",
        )
        rule_down = AlertRule(
            rule_id="rule_down",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="portfolio_change",
            threshold=0.03,  # 3% threshold
            condition="change_down",
        )

        await portfolio_monitor.add_alert_rule(rule_up)
        await portfolio_monitor.add_alert_rule(rule_down)

        # Test upward change alert
        update_up = PortfolioUpdate(
            portfolio_id=sample_portfolio.portfolio_id,
            total_value=51500.0,
            change=1500.0,
            change_percent=0.03,  # 3% up
            positions={},
            timestamp=datetime.now(timezone.utc),
        )

        await portfolio_monitor._check_portfolio_alerts(sample_portfolio, update_up)

        # Should trigger upward alert
        data_pipeline.send_alert.assert_called_once()
        call_args = data_pipeline.send_alert.call_args[1]
        assert call_args["alert_type"] == "portfolio_alert"
        assert "increased" in call_args["message"]

        # Reset
        data_pipeline.send_alert.reset_mock()

        # Test downward change alert
        update_down = PortfolioUpdate(
            portfolio_id=sample_portfolio.portfolio_id,
            total_value=48000.0,
            change=-2000.0,
            change_percent=-0.04,  # 4% down
            positions={},
            timestamp=datetime.now(timezone.utc),
        )

        await portfolio_monitor._check_portfolio_alerts(sample_portfolio, update_down)

        # Should trigger downward alert
        data_pipeline.send_alert.assert_called_once()
        call_args = data_pipeline.send_alert.call_args[1]
        assert "decreased" in call_args["message"]

    @pytest.mark.asyncio
    async def test_alert_cooldown(
        self, portfolio_monitor, sample_portfolio, data_pipeline
    ):
        """Test alert cooldown mechanism."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        rule = AlertRule(
            rule_id="rule_1",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="portfolio_change",
            threshold=0.02,
            condition="change_up",
            cooldown_minutes=15,
        )

        await portfolio_monitor.add_alert_rule(rule)

        # First alert
        update = PortfolioUpdate(
            portfolio_id=sample_portfolio.portfolio_id,
            total_value=51500.0,
            change=1500.0,
            change_percent=0.03,
            positions={},
            timestamp=datetime.now(timezone.utc),
        )

        await portfolio_monitor._check_portfolio_alerts(sample_portfolio, update)
        assert data_pipeline.send_alert.called

        # Reset and try again immediately
        data_pipeline.send_alert.reset_mock()
        await portfolio_monitor._check_portfolio_alerts(sample_portfolio, update)

        # Should not trigger due to cooldown
        assert not data_pipeline.send_alert.called

    @pytest.mark.asyncio
    async def test_risk_alerts(
        self, portfolio_monitor, sample_portfolio, data_pipeline
    ):
        """Test risk-based alerts."""
        await portfolio_monitor.add_portfolio(sample_portfolio)

        # Add risk alert rules
        rule_vol = AlertRule(
            rule_id="rule_vol",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="risk_threshold",
            threshold=0.20,  # 20% volatility
            condition="volatility_above",
        )
        rule_dd = AlertRule(
            rule_id="rule_dd",
            portfolio_id=sample_portfolio.portfolio_id,
            rule_type="risk_threshold",
            threshold=0.10,  # 10% drawdown
            condition="drawdown_below",
        )

        await portfolio_monitor.add_alert_rule(rule_vol)
        await portfolio_monitor.add_alert_rule(rule_dd)

        # Test volatility alert
        risk_update = RiskUpdate(
            portfolio_id=sample_portfolio.portfolio_id,
            volatility=0.25,  # 25% volatility
            var_95=-0.05,
            cvar_95=-0.08,
            beta=1.0,
            sharpe_ratio=0.5,
            max_drawdown=-0.15,  # 15% drawdown
            timestamp=datetime.now(timezone.utc),
        )

        await portfolio_monitor._check_risk_alerts(sample_portfolio, risk_update)

        # Should trigger both alerts
        assert data_pipeline.send_alert.call_count == 2

    def test_get_portfolio_summary(self, portfolio_monitor, sample_portfolio):
        """Test getting portfolio summary."""
        # Add portfolio
        asyncio.run(portfolio_monitor.add_portfolio(sample_portfolio))

        # Add some history
        portfolio_history = portfolio_monitor.portfolio_history[
            sample_portfolio.portfolio_id
        ]
        portfolio_history.extend([50000, 51000, 52000, 53000])

        # Get summary
        summary = portfolio_monitor.get_portfolio_summary(sample_portfolio.portfolio_id)

        assert summary is not None
        assert summary["portfolio_id"] == sample_portfolio.portfolio_id
        assert summary["name"] == sample_portfolio.name
        assert summary["holdings"] == sample_portfolio.holdings
        assert summary["initial_value"] == sample_portfolio.initial_value
        assert summary["current_value"] == 53000
        assert summary["total_return"] == pytest.approx(0.06)  # (53000 - 50000) / 50000
        assert summary["data_points"] == 4
        assert summary["status"] == "active"

    def test_get_portfolio_summary_no_history(
        self, portfolio_monitor, sample_portfolio
    ):
        """Test getting portfolio summary with no history."""
        asyncio.run(portfolio_monitor.add_portfolio(sample_portfolio))

        summary = portfolio_monitor.get_portfolio_summary(sample_portfolio.portfolio_id)

        assert summary is not None
        assert summary["current_value"] == sample_portfolio.initial_value
        assert summary["total_return"] == 0.0
        assert summary["status"] == "initializing"

    def test_get_portfolio_summary_nonexistent(self, portfolio_monitor):
        """Test getting summary for nonexistent portfolio."""
        summary = portfolio_monitor.get_portfolio_summary("nonexistent")
        assert summary is None

    def test_get_all_portfolios_summary(self, portfolio_monitor):
        """Test getting all portfolios summary."""
        # Add multiple portfolios
        portfolio1 = Portfolio(
            portfolio_id="port_1",
            name="Portfolio 1",
            holdings={"AAPL": 100},
            initial_value=15000.0,
            created_at=datetime.now(timezone.utc),
        )
        portfolio2 = Portfolio(
            portfolio_id="port_2",
            name="Portfolio 2",
            holdings={"GOOGL": 50},
            initial_value=125000.0,
            created_at=datetime.now(timezone.utc),
        )

        asyncio.run(portfolio_monitor.add_portfolio(portfolio1))
        asyncio.run(portfolio_monitor.add_portfolio(portfolio2))

        summaries = portfolio_monitor.get_all_portfolios_summary()

        assert len(summaries) == 2
        assert any(s["portfolio_id"] == "port_1" for s in summaries)
        assert any(s["portfolio_id"] == "port_2" for s in summaries)


class TestPortfolioMonitorIntegration:
    """Integration tests for portfolio monitor."""

    @pytest.mark.asyncio
    async def test_full_portfolio_monitoring_flow(self):
        """Test complete portfolio monitoring flow."""
        # Create mock data pipeline
        pipeline = Mock(spec=DataPipeline)
        pipeline._portfolio_calculators = {}
        pipeline._risk_calculators = {}
        pipeline.register_portfolio_calculator = Mock(
            side_effect=lambda k, v: pipeline._portfolio_calculators.update({k: v})
        )
        pipeline.register_risk_calculator = Mock(
            side_effect=lambda k, v: pipeline._risk_calculators.update({k: v})
        )
        pipeline.subscribe_market_data = AsyncMock()
        pipeline.subscribe_portfolio = AsyncMock()
        pipeline.subscribe_risk_metrics = AsyncMock()
        pipeline.send_alert = AsyncMock()
        monitor = PortfolioMonitor(pipeline)

        # Create portfolio
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            name="Integration Test Portfolio",
            holdings={"AAPL": 100, "GOOGL": 50},
            initial_value=40000.0,
            created_at=datetime.now(timezone.utc),
        )

        # Add portfolio
        await monitor.add_portfolio(portfolio)

        # Add alert rule
        rule = AlertRule(
            rule_id="test_rule",
            portfolio_id=portfolio.portfolio_id,
            rule_type="portfolio_change",
            threshold=0.01,
            condition="change_up",
        )
        await monitor.add_alert_rule(rule)

        # Get the registered calculator
        calculator_key = portfolio.portfolio_id
        assert calculator_key in pipeline._portfolio_calculators

        # Simulate market data
        market_data = {
            "AAPL": MarketDataUpdate(
                symbol="AAPL",
                price=155.0,
                change=5.0,
                change_percent=0.033,
                volume=1000000,
                timestamp=datetime.now(timezone.utc),
            ),
            "GOOGL": MarketDataUpdate(
                symbol="GOOGL",
                price=2600.0,
                change=100.0,
                change_percent=0.04,
                volume=500000,
                timestamp=datetime.now(timezone.utc),
            ),
        }

        # Calculate portfolio value through the calculator
        calculator = pipeline._portfolio_calculators[calculator_key]
        portfolio_update = await calculator(market_data)

        assert portfolio_update is not None
        assert portfolio_update.total_value > portfolio.initial_value

        # Check summary
        summary = monitor.get_portfolio_summary(portfolio.portfolio_id)
        assert summary is not None
        assert summary["data_points"] > 0

    @pytest.mark.asyncio
    async def test_multiple_portfolios_monitoring(self):
        """Test monitoring multiple portfolios."""
        # Create mock data pipeline
        pipeline = Mock(spec=DataPipeline)
        pipeline.register_portfolio_calculator = Mock()
        pipeline.register_risk_calculator = Mock()
        pipeline.subscribe_market_data = AsyncMock()
        pipeline.subscribe_portfolio = AsyncMock()
        pipeline.subscribe_risk_metrics = AsyncMock()
        monitor = PortfolioMonitor(pipeline)

        # Create multiple portfolios
        portfolios = []
        for i in range(3):
            portfolio = Portfolio(
                portfolio_id=f"port_{i}",
                name=f"Portfolio {i}",
                holdings={"AAPL": 100 * (i + 1), "GOOGL": 50 * (i + 1)},
                initial_value=50000.0 * (i + 1),
                created_at=datetime.now(timezone.utc),
            )
            portfolios.append(portfolio)
            await monitor.add_portfolio(portfolio)

        # Verify all portfolios are tracked
        assert len(monitor.portfolios) == 3
        assert len(monitor.get_all_portfolios_summary()) == 3

        # Remove one portfolio
        await monitor.remove_portfolio("port_1")
        assert len(monitor.portfolios) == 2
        assert "port_1" not in monitor.portfolios
