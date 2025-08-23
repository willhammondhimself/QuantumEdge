"""Tests for Yahoo Finance data provider."""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
import yfinance as yf

from src.data.yahoo_finance import YahooFinanceProvider
from src.data.models import Asset, AssetType, Price, MarketData, DataFrequency


class TestYahooFinanceProvider:
    """Test Yahoo Finance data provider."""

    @pytest.fixture
    def provider(self):
        """Create YahooFinanceProvider instance."""
        return YahooFinanceProvider()

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_asset_info(self, mock_ticker_class, provider):
        """Test getting asset information."""
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Mock info data
        mock_ticker.info = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "quoteType": "EQUITY",
            "exchange": "NMS",
            "currency": "USD",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "longBusinessSummary": "Apple Inc. designs and manufactures smartphones...",
            "marketCap": 3000000000000,
        }

        # Get asset info
        asset = await provider.get_asset_info("AAPL")

        assert asset is not None
        assert asset.symbol == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == AssetType.STOCK
        assert asset.exchange == "NMS"
        assert asset.currency == "USD"
        assert asset.sector == "Technology"
        assert asset.industry == "Consumer Electronics"
        assert asset.market_cap == 3000000000000

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_asset_info_etf(self, mock_ticker_class, provider):
        """Test getting ETF asset information."""
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Mock ETF info
        mock_ticker.info = {
            "symbol": "SPY",
            "shortName": "SPDR S&P 500 ETF Trust",
            "quoteType": "ETF",
            "exchange": "NYSEArca",
            "currency": "USD",
        }

        # Get asset info
        asset = await provider.get_asset_info("SPY")

        assert asset is not None
        assert asset.symbol == "SPY"
        assert asset.asset_type == AssetType.ETF

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_asset_info_not_found(self, mock_ticker_class, provider):
        """Test asset not found."""
        # Mock ticker with no info
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {}

        # Should return None
        asset = await provider.get_asset_info("INVALID")
        assert asset is None

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_current_price(self, mock_ticker_class, provider):
        """Test getting current price."""
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Mock history data (since get_current_price uses history)
        mock_history = pd.DataFrame(
            {
                "Open": [175.0],
                "High": [178.0],
                "Low": [174.0],
                "Close": [177.0],
                "Volume": [50000000],
            },
            index=pd.to_datetime(["2024-01-01 15:59:00"]),
        )

        mock_ticker.history.return_value = mock_history

        # Get current price
        price = await provider.get_current_price("AAPL")

        assert price is not None
        assert price.symbol == "AAPL"
        assert price.open == 175.0
        assert price.high == 178.0
        assert price.low == 174.0
        assert price.close == 177.0
        assert price.volume == 50000000
        assert isinstance(price.timestamp, datetime)

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_current_prices(self, mock_ticker_class, provider):
        """Test getting multiple current prices."""

        # Mock ticker data for different symbols
        def create_mock_ticker(symbol, price):
            mock_ticker = Mock()
            # Mock history data
            mock_history = pd.DataFrame(
                {
                    "Open": [price - 5],
                    "High": [price + 5],
                    "Low": [price - 10],
                    "Close": [price],
                    "Volume": [1000000],
                },
                index=pd.to_datetime(["2024-01-01 15:59:00"]),
            )
            mock_ticker.history.return_value = mock_history
            return mock_ticker

        # Mock different tickers
        mock_ticker_class.side_effect = [
            create_mock_ticker("AAPL", 177.0),
            create_mock_ticker("GOOGL", 2850.0),
            Mock(history=Mock(return_value=pd.DataFrame())),  # Invalid ticker
        ]

        # Get multiple prices
        symbols = ["AAPL", "GOOGL", "INVALID"]
        prices = await provider.get_current_prices(symbols)

        assert len(prices) == 3
        assert prices["AAPL"] is not None
        assert prices["AAPL"].close == 177.0
        assert prices["GOOGL"] is not None
        assert prices["GOOGL"].close == 2850.0
        assert prices["INVALID"] is None

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_historical_data(self, mock_ticker_class, provider):
        """Test getting historical data."""
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Create mock historical data
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        mock_history = pd.DataFrame(
            {
                "Open": [175.0, 176.0, 177.0, 178.0, 179.0],
                "High": [178.0, 179.0, 180.0, 181.0, 182.0],
                "Low": [174.0, 175.0, 176.0, 177.0, 178.0],
                "Close": [177.0, 178.0, 179.0, 180.0, 181.0],
                "Volume": [50000000, 51000000, 52000000, 53000000, 54000000],
                "Adj Close": [177.0, 178.0, 179.0, 180.0, 181.0],
            },
            index=dates,
        )

        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"symbol": "AAPL"}

        # Get historical data
        market_data = await provider.get_historical_data(
            "AAPL", date(2024, 1, 1), date(2024, 1, 5), DataFrequency.DAILY
        )

        assert market_data is not None
        assert market_data.symbol == "AAPL"
        assert len(market_data.data) == 5
        assert market_data.frequency == DataFrequency.DAILY
        assert market_data.source == "Yahoo Finance"

        # Check first price point
        first_price = market_data.data[0]
        assert first_price.open == 175.0
        assert first_price.high == 178.0
        assert first_price.low == 174.0
        assert first_price.close == 177.0
        assert first_price.volume == 50000000

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_historical_data_empty(self, mock_ticker_class, provider):
        """Test getting historical data with no results."""
        # Mock ticker with empty history
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker.info = {"symbol": "AAPL"}

        # Should return None
        market_data = await provider.get_historical_data(
            "AAPL", date(2024, 1, 1), date(2024, 1, 5), DataFrequency.DAILY
        )

        assert market_data is None

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_get_historical_data_error(self, mock_ticker_class, provider):
        """Test handling errors in historical data."""
        # Mock ticker that raises exception
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("Network error")

        # Should raise DataSourceError
        from src.data.models import DataSourceError

        with pytest.raises(DataSourceError, match="Failed to fetch historical data"):
            await provider.get_historical_data(
                "AAPL", date(2024, 1, 1), date(2024, 1, 5), DataFrequency.DAILY
            )

    # Remove these tests as these methods don't exist in the implementation
