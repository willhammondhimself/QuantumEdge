"""Comprehensive tests for market data source module."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock
import time
import numpy as np

from src.streaming.market_data_source import (
    MarketDataSource,
    SimulatedMarketDataSource,
    create_market_data_source,
)
from src.streaming.data_pipeline import MarketDataUpdate
from src.data.models import Price


class TestMarketDataSource:
    """Test MarketDataSource class."""

    @pytest.fixture
    def market_source(self):
        """Create market data source instance."""
        return MarketDataSource(rate_limit_per_minute=30)

    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data."""
        return {
            "AAPL": Price(
                symbol="AAPL",
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000,
                timestamp=datetime.now(timezone.utc),
            ),
            "GOOGL": Price(
                symbol="GOOGL",
                open=2500.0,
                high=2520.0,
                low=2490.0,
                close=2510.0,
                volume=500000,
                timestamp=datetime.now(timezone.utc),
            ),
        }

    def test_initialization(self, market_source):
        """Test market data source initialization."""
        assert market_source.rate_limit_per_minute == 30
        assert market_source._error_count == 0
        assert market_source._max_consecutive_errors == 5
        assert isinstance(market_source._last_prices, dict)
        assert len(market_source._request_times) == 0

    @pytest.mark.asyncio
    async def test_get_market_updates_success(self, market_source, mock_price_data):
        """Test successful market data updates."""
        with patch.object(
            market_source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            mock_get_prices.return_value = mock_price_data

            symbols = ["AAPL", "GOOGL"]
            updates = await market_source.get_market_updates(symbols)

            assert len(updates) == 2
            assert all(isinstance(u, MarketDataUpdate) for u in updates)

            # Check AAPL update
            aapl_update = next(u for u in updates if u.symbol == "AAPL")
            assert aapl_update.price == 151.0
            assert aapl_update.volume == 1000000

            # Check GOOGL update
            googl_update = next(u for u in updates if u.symbol == "GOOGL")
            assert googl_update.price == 2510.0
            assert googl_update.volume == 500000

    @pytest.mark.asyncio
    async def test_get_market_updates_with_price_changes(
        self, market_source, mock_price_data
    ):
        """Test market updates with price changes."""
        # Set initial prices
        market_source._last_prices["AAPL"] = 150.0
        market_source._last_prices["GOOGL"] = 2500.0

        with patch.object(
            market_source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            mock_get_prices.return_value = mock_price_data

            updates = await market_source.get_market_updates(["AAPL", "GOOGL"])

            # Check price changes
            aapl_update = next(u for u in updates if u.symbol == "AAPL")
            assert aapl_update.change == 1.0  # 151 - 150
            assert aapl_update.change_percent == pytest.approx(1.0 / 150.0)

            googl_update = next(u for u in updates if u.symbol == "GOOGL")
            assert googl_update.change == 10.0  # 2510 - 2500
            assert googl_update.change_percent == pytest.approx(10.0 / 2500.0)

    @pytest.mark.asyncio
    async def test_get_market_updates_empty_symbols(self, market_source):
        """Test market updates with empty symbol list."""
        updates = await market_source.get_market_updates([])
        assert updates == []

    @pytest.mark.asyncio
    async def test_get_market_updates_rate_limited(self, market_source):
        """Test rate limiting behavior."""
        # Fill up rate limit
        current_time = time.time()
        market_source._request_times = [current_time - i for i in range(30)]

        updates = await market_source.get_market_updates(["AAPL"])
        assert updates == []  # Should be rate limited

    @pytest.mark.asyncio
    async def test_get_market_updates_error_handling(self, market_source):
        """Test error handling in market updates."""
        with patch.object(
            market_source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            mock_get_prices.side_effect = Exception("API error")

            updates = await market_source.get_market_updates(["AAPL"])

            assert updates == []
            assert market_source._error_count == 1
            assert market_source._last_error_time > 0

    def test_can_make_request(self, market_source):
        """Test rate limit checking."""
        # Initially should allow requests
        assert market_source._can_make_request() is True

        # Add recent requests
        current_time = time.time()
        market_source._request_times = [current_time - i for i in range(25)]
        assert market_source._can_make_request() is True

        # Exceed rate limit
        market_source._request_times = [current_time - i for i in range(30)]
        assert market_source._can_make_request() is False

        # Old requests should be removed
        market_source._request_times = [current_time - 70]  # Old request
        assert market_source._can_make_request() is True

    def test_is_healthy(self, market_source):
        """Test health check."""
        # Initially healthy
        assert market_source.is_healthy() is True

        # Too many errors
        market_source._error_count = 5
        assert market_source.is_healthy() is False

        # Recent error
        market_source._error_count = 1
        market_source._last_error_time = time.time()
        assert market_source.is_healthy() is False

        # Old error should be okay
        market_source._error_count = 1
        market_source._last_error_time = time.time() - 400  # > 5 minutes ago
        assert market_source.is_healthy() is True

    def test_get_stats(self, market_source):
        """Test statistics retrieval."""
        # Add some data
        market_source._last_prices["AAPL"] = 150.0
        market_source._request_times = [time.time() - 10]
        market_source._error_count = 2

        with patch.object(
            market_source.provider, "get_cache_stats"
        ) as mock_cache_stats:
            mock_cache_stats.return_value = {"hits": 100, "misses": 20}

            stats = market_source.get_stats()

            assert stats["provider"] == market_source.provider.name
            assert stats["healthy"] == market_source.is_healthy()
            assert stats["rate_limit_per_minute"] == 30
            assert stats["requests_last_minute"] == 1
            assert stats["error_count"] == 2
            assert stats["symbols_tracked"] == 1
            assert stats["cache_stats"]["hits"] == 100

    @pytest.mark.asyncio
    async def test_missing_price_data(self, market_source):
        """Test handling of missing price data."""
        with patch.object(
            market_source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            # Return data for only one symbol
            mock_get_prices.return_value = {
                "AAPL": Price(
                    symbol="AAPL",
                    open=150.0,
                    high=152.0,
                    low=149.0,
                    close=151.0,
                    volume=1000000,
                    timestamp=datetime.now(timezone.utc),
                )
            }

            updates = await market_source.get_market_updates(["AAPL", "GOOGL", "MSFT"])

            # Should only have update for AAPL
            assert len(updates) == 1
            assert updates[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_zero_price_handling(self, market_source):
        """Test handling of zero last price."""
        market_source._last_prices["AAPL"] = 0.0

        with patch.object(
            market_source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            mock_get_prices.return_value = {
                "AAPL": Price(
                    symbol="AAPL",
                    open=150.0,
                    high=152.0,
                    low=149.0,
                    close=151.0,
                    volume=1000000,
                    timestamp=datetime.now(timezone.utc),
                )
            }

            updates = await market_source.get_market_updates(["AAPL"])

            # Should handle zero price gracefully
            assert len(updates) == 1
            assert updates[0].change_percent == 0.0  # Division by zero handled


class TestSimulatedMarketDataSource:
    """Test SimulatedMarketDataSource class."""

    @pytest.fixture
    def sim_source(self):
        """Create simulated data source."""
        base_prices = {"AAPL": 150.0, "GOOGL": 2500.0}
        return SimulatedMarketDataSource(base_prices)

    def test_initialization(self, sim_source):
        """Test simulated source initialization."""
        assert sim_source.base_prices["AAPL"] == 150.0
        assert sim_source.base_prices["GOOGL"] == 2500.0
        assert sim_source.current_prices == sim_source.base_prices
        assert sim_source._volatility == 0.02
        assert sim_source._drift == 0.0001

    @pytest.mark.asyncio
    async def test_get_market_updates(self, sim_source):
        """Test simulated market updates."""
        updates = await sim_source.get_market_updates(["AAPL", "GOOGL"])

        assert len(updates) == 2

        # Check AAPL update
        aapl_update = next(u for u in updates if u.symbol == "AAPL")
        assert isinstance(aapl_update, MarketDataUpdate)
        assert aapl_update.price > 0
        assert abs(aapl_update.change_percent) < 0.1  # Reasonable change
        assert aapl_update.volume > 0

        # Check price was updated
        assert sim_source.current_prices["AAPL"] == aapl_update.price

    @pytest.mark.asyncio
    async def test_new_symbol_initialization(self, sim_source):
        """Test initialization of new symbols."""
        updates = await sim_source.get_market_updates(["TSLA"])

        assert len(updates) == 1
        assert updates[0].symbol == "TSLA"
        assert 50 <= updates[0].price <= 500  # Random initial price range
        assert "TSLA" in sim_source.current_prices

    @pytest.mark.asyncio
    async def test_price_bounds(self, sim_source):
        """Test price bounds enforcement."""
        # Set extreme prices
        sim_source.current_prices["AAPL"] = 0.1
        sim_source.current_prices["GOOGL"] = 9999.0

        # Mock random to produce extreme movements
        with patch("random.gauss") as mock_gauss:
            # First call for AAPL (negative), second for GOOGL (positive)
            mock_gauss.side_effect = [-10.0, 10.0]

            updates = await sim_source.get_market_updates(["AAPL", "GOOGL"])

            # Prices should be bounded
            aapl_update = next(u for u in updates if u.symbol == "AAPL")
            assert aapl_update.price >= 1.0

            googl_update = next(u for u in updates if u.symbol == "GOOGL")
            assert googl_update.price <= 10000.0

    def test_is_healthy(self, sim_source):
        """Test simulated source health check."""
        assert sim_source.is_healthy() is True

    def test_get_stats(self, sim_source):
        """Test simulated source statistics."""
        stats = sim_source.get_stats()

        assert stats["provider"] == "Simulated"
        assert stats["healthy"] is True
        assert stats["symbols_tracked"] == 2
        assert stats["base_volatility"] == 0.02
        assert stats["drift"] == 0.0001

    @pytest.mark.asyncio
    async def test_consistent_price_evolution(self, sim_source):
        """Test that prices evolve consistently."""
        initial_price = sim_source.current_prices["AAPL"]

        # Run multiple updates
        prices = [initial_price]
        for _ in range(10):
            updates = await sim_source.get_market_updates(["AAPL"])
            prices.append(updates[0].price)

        # Check prices changed
        assert len(set(prices)) > 1

        # Check prices stay reasonable
        price_changes = [
            abs(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)
        ]
        assert all(change < 0.1 for change in price_changes)  # < 10% per update


class TestMarketDataSourceFactory:
    """Test market data source factory function."""

    def test_create_real_source(self):
        """Test creating real market data source."""
        source = create_market_data_source(use_simulation=False)
        assert isinstance(source, MarketDataSource)
        assert hasattr(source, "provider")

    def test_create_simulated_source(self):
        """Test creating simulated market data source."""
        source = create_market_data_source(use_simulation=True)
        assert isinstance(source, SimulatedMarketDataSource)
        assert "AAPL" in source.base_prices
        assert "GOOGL" in source.base_prices
        assert source.base_prices["AAPL"] == 150.0

    def test_default_is_real_source(self):
        """Test default source is real."""
        source = create_market_data_source()
        assert isinstance(source, MarketDataSource)


class TestMarketDataSourceIntegration:
    """Integration tests for market data source."""

    @pytest.mark.asyncio
    async def test_multiple_updates_with_rate_limiting(self):
        """Test multiple updates respecting rate limits."""
        source = MarketDataSource(rate_limit_per_minute=2)  # Very low limit

        with patch.object(
            source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            mock_get_prices.return_value = {
                "AAPL": Price(
                    symbol="AAPL",
                    open=150.0,
                    high=152.0,
                    low=149.0,
                    close=151.0,
                    volume=1000000,
                    timestamp=datetime.now(timezone.utc),
                )
            }

            # First two requests should succeed
            updates1 = await source.get_market_updates(["AAPL"])
            assert len(updates1) == 1

            updates2 = await source.get_market_updates(["AAPL"])
            assert len(updates2) == 1

            # Third request should be rate limited
            updates3 = await source.get_market_updates(["AAPL"])
            assert len(updates3) == 0

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test recovery from errors."""
        source = MarketDataSource()

        with patch.object(
            source.provider, "get_current_prices", new_callable=AsyncMock
        ) as mock_get_prices:
            # First call fails
            mock_get_prices.side_effect = Exception("API error")
            updates1 = await source.get_market_updates(["AAPL"])
            assert len(updates1) == 0
            assert source._error_count == 1

            # Second call succeeds
            mock_get_prices.side_effect = None
            mock_get_prices.return_value = {
                "AAPL": Price(
                    symbol="AAPL",
                    open=150.0,
                    high=152.0,
                    low=149.0,
                    close=151.0,
                    volume=1000000,
                    timestamp=datetime.now(timezone.utc),
                )
            }
            updates2 = await source.get_market_updates(["AAPL"])
            assert len(updates2) == 1
            assert source._error_count == 0  # Reset on success
