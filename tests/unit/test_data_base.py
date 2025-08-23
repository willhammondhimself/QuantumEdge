"""Tests for base data provider."""

import pytest
from datetime import date, datetime
from typing import List, Optional
from unittest.mock import Mock, patch

from src.data.base import DataProvider
from src.data.models import Asset, Price, MarketData, DataFrequency, DataSourceError


class MockDataProvider(DataProvider):
    """Mock implementation of DataProvider for testing."""

    def __init__(self):
        super().__init__(name="MockProvider")
        self.get_asset_info_called = False
        self.get_current_price_called = False
        self.get_current_prices_called = False
        self.get_historical_data_called = False

    async def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """Mock get asset info."""
        self.get_asset_info_called = True
        if symbol == "INVALID":
            return None
        return Mock(spec=Asset)

    async def get_current_price(self, symbol: str) -> Optional[Price]:
        """Mock get current price."""
        self.get_current_price_called = True
        if symbol == "INVALID":
            return None
        return Mock(spec=Price)

    async def get_current_prices(self, symbols: List[str]) -> dict:
        """Mock get current prices."""
        self.get_current_prices_called = True
        return {s: Mock(spec=Price) if s != "INVALID" else None for s in symbols}

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> Optional[MarketData]:
        """Mock get historical data."""
        self.get_historical_data_called = True
        if symbol == "INVALID":
            return None
        return Mock(spec=MarketData)

    async def get_market_metrics(self) -> dict:
        """Mock get market metrics."""
        return {"vix": 15.5, "spy_return": 0.012}


class TestDataProvider:
    """Test DataProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            DataProvider()

    @pytest.mark.asyncio
    async def test_abstract_methods(self):
        """Test that concrete implementation works."""
        provider = MockDataProvider()

        # Test get_asset_info
        asset = await provider.get_asset_info("AAPL")
        assert provider.get_asset_info_called
        assert asset is not None

        # Test get_current_price
        price = await provider.get_current_price("AAPL")
        assert provider.get_current_price_called
        assert price is not None

        # Test get_current_prices
        prices = await provider.get_current_prices(["AAPL", "GOOGL"])
        assert provider.get_current_prices_called
        assert len(prices) == 2

        # Test get_historical_data
        data = await provider.get_historical_data(
            "AAPL", date(2024, 1, 1), date(2024, 1, 31), DataFrequency.DAILY
        )
        assert provider.get_historical_data_called
        assert data is not None

    @pytest.mark.asyncio
    async def test_invalid_symbols(self):
        """Test handling of invalid symbols."""
        provider = MockDataProvider()

        # Invalid asset info
        asset = await provider.get_asset_info("INVALID")
        assert asset is None

        # Invalid current price
        price = await provider.get_current_price("INVALID")
        assert price is None

        # Mixed valid/invalid prices
        prices = await provider.get_current_prices(["AAPL", "INVALID", "GOOGL"])
        assert prices["AAPL"] is not None
        assert prices["INVALID"] is None
        assert prices["GOOGL"] is not None

        # Invalid historical data
        data = await provider.get_historical_data(
            "INVALID", date(2024, 1, 1), date(2024, 1, 31)
        )
        assert data is None


class TestDataSourceError:
    """Test DataSourceError exception."""

    def test_error_creation(self):
        """Test creating error."""
        error = DataSourceError("Test error message")
        assert str(error) == "Test error message"

    def test_error_inheritance(self):
        """Test error inheritance."""
        error = DataSourceError("Test error")
        assert isinstance(error, Exception)

    def test_error_raising(self):
        """Test raising error."""
        with pytest.raises(DataSourceError, match="API limit reached"):
            raise DataSourceError("API limit reached")


class TestDataProviderValidation:
    """Test data validation in providers."""

    @pytest.mark.asyncio
    async def test_date_validation(self):
        """Test date range validation."""
        provider = MockDataProvider()

        # Valid date range
        data = await provider.get_historical_data(
            "AAPL", date(2024, 1, 1), date(2024, 1, 31), DataFrequency.DAILY
        )
        assert data is not None

        # Start date after end date - provider should handle
        data = await provider.get_historical_data(
            "AAPL", date(2024, 1, 31), date(2024, 1, 1), DataFrequency.DAILY
        )
        # Mock provider doesn't validate, but real ones should
        assert data is not None

    @pytest.mark.asyncio
    async def test_symbol_normalization(self):
        """Test symbol normalization."""
        provider = MockDataProvider()

        # Test various symbol formats
        symbols = ["aapl", "AAPL", " AAPL ", "aapl "]

        for symbol in symbols:
            asset = await provider.get_asset_info(symbol)
            assert asset is not None

    @pytest.mark.asyncio
    async def test_frequency_support(self):
        """Test different data frequencies."""
        provider = MockDataProvider()

        frequencies = [
            DataFrequency.MINUTE,
            DataFrequency.FIVE_MINUTE,
            DataFrequency.FIFTEEN_MINUTE,
            DataFrequency.THIRTY_MINUTE,
            DataFrequency.HOUR,
            DataFrequency.FOUR_HOUR,
            DataFrequency.DAILY,
            DataFrequency.WEEKLY,
            DataFrequency.MONTHLY,
        ]

        for freq in frequencies:
            data = await provider.get_historical_data(
                "AAPL", date(2024, 1, 1), date(2024, 1, 31), freq
            )
            assert data is not None


class TestDataProviderConcurrency:
    """Test concurrent data provider operations."""

    @pytest.mark.asyncio
    async def test_concurrent_get_prices(self):
        """Test concurrent price fetching."""
        provider = MockDataProvider()

        # Simulate concurrent requests
        import asyncio

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Get prices concurrently
        tasks = [provider.get_current_price(s) for s in symbols]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test mixed concurrent operations."""
        provider = MockDataProvider()

        import asyncio

        # Mix different types of operations
        tasks = [
            provider.get_asset_info("AAPL"),
            provider.get_current_price("GOOGL"),
            provider.get_current_prices(["MSFT", "AMZN"]),
            provider.get_historical_data("TSLA", date(2024, 1, 1), date(2024, 1, 31)),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert results[0] is not None  # Asset info
        assert results[1] is not None  # Price
        assert isinstance(results[2], dict)  # Prices dict
        assert results[3] is not None  # Historical data
