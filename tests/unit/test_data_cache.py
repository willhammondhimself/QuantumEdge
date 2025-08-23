"""Tests for data cache module."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.data.cache import DataCache, MarketDataCache
from src.data.models import Price, Asset, AssetType


class TestDataCache:
    """Test DataCache functionality."""

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_init_with_redis(self, mock_redis):
        """Test initialization with Redis available."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        cache = DataCache()

        assert cache.available is True
        assert cache.default_ttl == 300
        mock_redis.from_url.assert_called_once()
        mock_client.ping.assert_called_once()

    @patch("src.data.cache.REDIS_AVAILABLE", False)
    def test_init_without_redis(self):
        """Test initialization without Redis."""
        cache = DataCache()

        assert cache.available is False
        assert cache.default_ttl == 300

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_set_with_redis(self, mock_redis):
        """Test setting value with Redis."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        cache = DataCache()

        # Set value
        cache.set("test_key", {"data": "test_value"})

        # Should call setex
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == "quantumedge:data:test_key"
        assert call_args[0][1] == 300  # TTL

    @patch("src.data.cache.REDIS_AVAILABLE", False)
    def test_set_without_redis(self):
        """Test setting value without Redis (no-op)."""
        cache = DataCache()

        # Should not raise error
        cache.set("test_key", {"data": "test_value"})

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_get_with_redis(self, mock_redis):
        """Test getting value with Redis."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Mock stored data
        stored_data = json.dumps({"data": "test_value"})
        mock_client.get.return_value = stored_data.encode()

        cache = DataCache()

        # Get value
        result = cache.get("test_key")

        assert result == {"data": "test_value"}
        mock_client.get.assert_called_once_with("quantumedge:data:test_key")

    @patch("src.data.cache.REDIS_AVAILABLE", False)
    def test_get_without_redis(self):
        """Test getting value without Redis."""
        cache = DataCache()

        # Should return None
        result = cache.get("test_key")
        assert result is None

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_delete_with_redis(self, mock_redis):
        """Test deleting value with Redis."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        cache = DataCache()

        # Delete value
        cache.delete("test_key")

        mock_client.delete.assert_called_once_with("quantumedge:data:test_key")

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_clear_pattern_with_redis(self, mock_redis):
        """Test clearing cache pattern with Redis."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Mock keys
        mock_client.keys.return_value = [
            b"quantumedge:data:key1",
            b"quantumedge:data:key2",
        ]
        mock_client.delete.return_value = 2

        cache = DataCache()

        # Clear pattern
        count = cache.clear_pattern("*")

        mock_client.keys.assert_called_once_with("quantumedge:data:*")
        mock_client.delete.assert_called_once_with(
            b"quantumedge:data:key1", b"quantumedge:data:key2"
        )
        assert count == 2


class TestMarketDataCache:
    """Test MarketDataCache functionality."""

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_set_asset_info(self, mock_redis):
        """Test setting asset info."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Create DataCache first
        data_cache = DataCache()
        # Create MarketDataCache with DataCache instance
        cache = MarketDataCache(data_cache)

        # Create asset
        asset = Asset(
            symbol="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            exchange="NASDAQ",
            currency="USD",
        )

        # Set asset info
        cache.cache_asset_info(asset)

        # Should call parent set method
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        key = call_args[0][0]
        assert "asset:AAPL" in key

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_get_asset_info(self, mock_redis):
        """Test getting asset info."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Mock stored data
        asset_data = {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_type": "stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        }
        mock_client.get.return_value = json.dumps(asset_data).encode()

        # Create DataCache first
        data_cache = DataCache()
        # Create MarketDataCache with DataCache instance
        cache = MarketDataCache(data_cache)

        # Get asset info
        asset = cache.get_cached_asset_info("AAPL")

        assert asset is not None
        assert asset["symbol"] == "AAPL"
        assert asset["name"] == "Apple Inc."

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_set_current_price(self, mock_redis):
        """Test setting current price."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Create DataCache first
        data_cache = DataCache()
        # Create MarketDataCache with DataCache instance
        cache = MarketDataCache(data_cache)

        # Create price
        price = Price(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=175.0,
            high=178.0,
            low=174.0,
            close=177.0,
            volume=50000000,
        )

        # Set price
        cache.cache_price(price)

        # Should call parent set method with short TTL
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        key = call_args[0][0]
        ttl = call_args[0][1]
        assert "price:AAPL:current" in key
        assert ttl == 60  # Short TTL for current prices

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_set_historical_data(self, mock_redis):
        """Test setting historical data."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Create DataCache first
        data_cache = DataCache()
        # Create MarketDataCache with DataCache instance
        cache = MarketDataCache(data_cache)

        # Mock historical data
        from src.data.models import DataFrequency

        historical_data = Mock()
        historical_data.symbol = "AAPL"
        historical_data.start_date = "2024-01-01"
        historical_data.end_date = "2024-01-31"
        historical_data.frequency = "daily"

        # Set historical data with date range
        cache.cache_historical_data(historical_data)

        # Should call parent set method with longer TTL
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        key = call_args[0][0]
        ttl = call_args[0][1]
        assert "historical:AAPL:2024-01-01:2024-01-31:daily" in key
        assert ttl == 3600  # Longer TTL for historical data

    @patch("src.data.cache.redis")
    @patch("src.data.cache.REDIS_AVAILABLE", True)
    def test_clear_symbol_data(self, mock_redis):
        """Test clearing data for specific symbol."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.from_url.return_value = mock_client

        # Mock keys for symbol
        mock_client.keys.side_effect = [
            [b"quantumedge:data:price:AAPL:current"],
            [b"quantumedge:data:historical:AAPL:2024-01-01:2024-01-31:daily"],
            [b"quantumedge:data:asset:AAPL"],
        ]

        # Mock delete to return number of deleted keys
        mock_client.delete.side_effect = [1, 1, 1]

        # Create DataCache first
        data_cache = DataCache()
        # Create MarketDataCache with DataCache instance
        cache = MarketDataCache(data_cache)

        # Clear symbol data
        count = cache.clear_symbol_data("AAPL")

        # Should find and delete all AAPL keys
        assert mock_client.keys.call_count == 3  # price, historical, asset patterns
        assert mock_client.delete.call_count == 3
        assert count == 3
