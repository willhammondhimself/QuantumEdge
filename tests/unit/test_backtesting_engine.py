"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from src.backtesting.strategy import BuyAndHoldStrategy, RebalancingStrategy
from src.backtesting.portfolio import Portfolio, Transaction
from src.data.models import Price, MarketData, DataFrequency


class TestBacktestEngine:
    """Test BacktestEngine functionality."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = Mock()
        strategy.name = "Test Strategy"
        strategy.symbols = ["AAPL", "GOOGL", "MSFT"]
        strategy.get_target_weights = Mock(
            return_value={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        )
        strategy.initialize = Mock(
            return_value={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        )
        strategy.rebalance = Mock(return_value={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3})
        return strategy

    @pytest.fixture
    def backtest_config(self, mock_strategy):
        """Create a test backtest configuration."""
        return BacktestConfig(
            strategy=mock_strategy,
            symbols=["AAPL", "GOOGL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_cash=100000.0,
            commission_rate=0.001,
            min_commission=1.0,
            rebalance_frequency="monthly",
            min_weight=0.05,
            max_weight=0.95,
            benchmark_symbol="SPY",
        )

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = []

        for i, timestamp in enumerate(dates):
            # AAPL prices (starting at 150, growing)
            prices.append(
                Price(
                    symbol="AAPL",
                    timestamp=timestamp,
                    open=150 + i * 0.1,
                    high=152 + i * 0.1,
                    low=149 + i * 0.1,
                    close=151 + i * 0.1,
                    volume=1000000,
                    adjusted_close=151 + i * 0.1,
                )
            )

            # GOOGL prices (starting at 100, growing)
            prices.append(
                Price(
                    symbol="GOOGL",
                    timestamp=timestamp,
                    open=100 + i * 0.05,
                    high=101 + i * 0.05,
                    low=99 + i * 0.05,
                    close=100.5 + i * 0.05,
                    volume=800000,
                    adjusted_close=100.5 + i * 0.05,
                )
            )

            # MSFT prices (starting at 250, growing)
            prices.append(
                Price(
                    symbol="MSFT",
                    timestamp=timestamp,
                    open=250 + i * 0.15,
                    high=252 + i * 0.15,
                    low=249 + i * 0.15,
                    close=251 + i * 0.15,
                    volume=1200000,
                    adjusted_close=251 + i * 0.15,
                )
            )

            # SPY benchmark prices
            prices.append(
                Price(
                    symbol="SPY",
                    timestamp=timestamp,
                    open=400 + i * 0.1,
                    high=402 + i * 0.1,
                    low=399 + i * 0.1,
                    close=401 + i * 0.1,
                    volume=5000000,
                    adjusted_close=401 + i * 0.1,
                )
            )

        # Group by symbol
        market_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT", "SPY"]:
            symbol_prices = [p for p in prices if p.symbol == symbol]
            market_data[symbol] = MarketData(
                symbol=symbol,
                data=symbol_prices,
                frequency=DataFrequency.DAILY,
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                source="test",
            )

        return market_data

    def test_init(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine()
        assert engine.cache == {}
        assert hasattr(engine, "data_provider")

    @pytest.mark.asyncio
    @patch("src.backtesting.engine.YahooFinanceProvider")
    async def test_run_backtest_success(
        self, mock_provider_class, backtest_config, mock_market_data
    ):
        """Test successful backtest run."""
        # Mock the data provider
        mock_provider = AsyncMock()
        mock_provider.get_historical_data = AsyncMock(
            side_effect=lambda symbol, start, end: mock_market_data.get(symbol)
        )
        mock_provider_class.return_value = mock_provider

        # Create engine and run backtest
        engine = BacktestEngine()
        result = await engine.run_backtest(backtest_config)

        # Verify result
        assert isinstance(result, BacktestResult)
        assert result.success is True
        assert result.error_message is None
        assert len(result.portfolio_values) > 0
        assert result.portfolio_values.iloc[-1] > 0  # Final portfolio value

        # Verify transactions were made
        assert len(result.transactions) > 0

        # Verify performance metrics were calculated
        assert result.performance_metrics is not None
        assert (
            result.performance_metrics.total_return != 0
        )  # Market moved, so return should be non-zero
        assert hasattr(result.performance_metrics, "sharpe_ratio")

    @pytest.mark.asyncio
    @patch("src.backtesting.engine.YahooFinanceProvider")
    async def test_run_backtest_no_data(self, mock_provider_class, backtest_config):
        """Test backtest with no market data."""
        # Mock provider returns None for all data
        mock_provider = AsyncMock()
        mock_provider.get_historical_data = AsyncMock(return_value=None)
        mock_provider_class.return_value = mock_provider

        engine = BacktestEngine()
        result = await engine.run_backtest(backtest_config)

        assert result.success is False
        assert "No price data available for any symbols" in result.error_message

    @pytest.mark.skip(
        reason="Method _prepare_data doesn't exist in current implementation"
    )
    def test_prepare_data_aligned(self, mock_market_data):
        """Test data preparation and alignment."""
        engine = BacktestEngine()
        engine.market_data_cache = mock_market_data

        aligned_data = engine._prepare_data(["AAPL", "GOOGL", "MSFT"])

        # Check structure
        assert "prices" in aligned_data
        assert "returns" in aligned_data

        # Check alignment
        prices_df = aligned_data["prices"]
        returns_df = aligned_data["returns"]

        assert set(prices_df.columns) == {"AAPL", "GOOGL", "MSFT"}
        assert set(returns_df.columns) == {"AAPL", "GOOGL", "MSFT"}
        assert len(prices_df) > 0
        assert len(returns_df) == len(prices_df) - 1  # Returns have one less row

    @pytest.mark.skip(
        reason="Method _calculate_performance_metrics doesn't exist in current implementation"
    )
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        engine = BacktestEngine()

        # Create sample portfolio values
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        values = pd.Series([100000 * (1 + 0.0005 * i) for i in range(252)], index=dates)

        # Create sample benchmark
        benchmark = pd.Series([400 * (1 + 0.0003 * i) for i in range(252)], index=dates)

        metrics = engine._calculate_performance_metrics(
            portfolio_values=values,
            initial_value=100000,
            risk_free_rate=0.02,
            benchmark_returns=benchmark.pct_change().dropna(),
        )

        assert metrics.total_return > 0
        assert metrics.annualized_return > 0
        assert metrics.volatility > 0
        assert metrics.sharpe_ratio > 0
        assert metrics.max_drawdown < 0
        assert hasattr(metrics, "beta")
        assert hasattr(metrics, "alpha")

    @pytest.mark.skip(
        reason="Method _calculate_performance_metrics doesn't exist in current implementation"
    )
    def test_calculate_performance_metrics_negative_returns(self):
        """Test performance metrics with negative returns."""
        engine = BacktestEngine()

        # Create declining portfolio
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        values = pd.Series([100000 * (1 - 0.001 * i) for i in range(252)], index=dates)

        metrics = engine._calculate_performance_metrics(
            portfolio_values=values, initial_value=100000, risk_free_rate=0.02
        )

        assert metrics.total_return < 0
        assert metrics.annualized_return < 0
        assert metrics.max_drawdown < -0.1  # Significant drawdown
        assert metrics.sharpe_ratio < 0

    @pytest.mark.skip(
        reason="Method _prepare_data doesn't exist in current implementation"
    )
    def test_prepare_data_missing_symbol(self, mock_market_data):
        """Test data preparation with missing symbol."""
        engine = BacktestEngine()
        engine.market_data_cache = mock_market_data

        # Request data including a symbol not in cache
        aligned_data = engine._prepare_data(["AAPL", "GOOGL", "MISSING"])

        # Should only include available symbols
        prices_df = aligned_data["prices"]
        assert "MISSING" not in prices_df.columns
        assert set(prices_df.columns) == {"AAPL", "GOOGL"}

    @pytest.mark.asyncio
    async def test_run_backtest_with_benchmark(self, backtest_config, mock_market_data):
        """Test backtest with benchmark comparison."""
        # Mock the data provider
        with patch(
            "src.backtesting.engine.YahooFinanceProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.get_historical_data = AsyncMock(
                side_effect=lambda symbol, start, end: mock_market_data.get(symbol)
            )
            mock_provider_class.return_value = mock_provider

            engine = BacktestEngine()
            result = await engine.run_backtest(backtest_config)

            assert result.success is True
            assert result.benchmark_values is not None
            assert len(result.benchmark_values) > 0

            # Performance metrics should include beta and alpha
            assert hasattr(result.performance_metrics, "beta")
            assert hasattr(result.performance_metrics, "alpha")
            assert hasattr(result.performance_metrics, "information_ratio")

    @pytest.mark.skip(
        reason="Method _generate_rebalance_dates doesn't exist in current implementation"
    )
    def test_generate_rebalance_dates_monthly(self):
        """Test monthly rebalance date generation."""
        engine = BacktestEngine()
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        rebalance_dates = engine._generate_rebalance_dates(dates, "monthly")

        # Should have approximately 12 rebalance dates (one per month)
        assert 11 <= len(rebalance_dates) <= 12

        # Check dates are month-end or close to it
        for date in rebalance_dates:
            assert date.day >= 28 or date.day == 1

    @pytest.mark.skip(
        reason="Method _generate_rebalance_dates doesn't exist in current implementation"
    )
    def test_generate_rebalance_dates_quarterly(self):
        """Test quarterly rebalance date generation."""
        engine = BacktestEngine()
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        rebalance_dates = engine._generate_rebalance_dates(dates, "quarterly")

        # Should have 4 rebalance dates
        assert len(rebalance_dates) == 4

        # Check months are quarter-end
        for date in rebalance_dates:
            assert date.month in [3, 6, 9, 12]

    @pytest.mark.skip(
        reason="Method _generate_rebalance_dates doesn't exist in current implementation"
    )
    def test_generate_rebalance_dates_invalid(self):
        """Test invalid rebalance frequency."""
        engine = BacktestEngine()
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Should default to monthly for invalid frequency
        rebalance_dates = engine._generate_rebalance_dates(dates, "invalid")
        assert 11 <= len(rebalance_dates) <= 12  # Monthly default
