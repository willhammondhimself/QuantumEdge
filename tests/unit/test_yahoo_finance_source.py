"""Comprehensive tests for Yahoo Finance data source."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np
import logging

from src.streaming.yahoo_finance_source import YahooFinanceDataSource


class TestYahooFinanceDataSource:
    """Test YahooFinanceDataSource class."""

    @pytest.fixture
    def data_source(self):
        """Create Yahoo Finance data source instance."""
        return YahooFinanceDataSource(update_interval=30.0)

    @pytest.fixture
    def mock_ticker(self):
        """Create mock yfinance ticker."""
        ticker = Mock()

        # Mock info
        ticker.info = {
            "regularMarketPrice": 150.00,
            "regularMarketDayHigh": 152.00,
            "regularMarketDayLow": 149.00,
            "regularMarketVolume": 1000000,
            "regularMarketPreviousClose": 148.00,
        }

        # Mock history data
        history_data = pd.DataFrame(
            {
                "Open": [148.0, 150.0],
                "High": [151.0, 152.0],
                "Low": [147.0, 149.0],
                "Close": [149.0, 151.0],
                "Volume": [900000, 1000000],
            },
            index=pd.to_datetime(["2024-01-08", "2024-01-09"]),
        )

        ticker.history = Mock(return_value=history_data)

        return ticker

    @pytest.fixture
    def mock_ticker_insufficient_data(self):
        """Create mock ticker with insufficient history."""
        ticker = Mock()
        ticker.info = {}

        # Only one day of data
        history_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [152.0],
                "Low": [149.0],
                "Close": [151.0],
                "Volume": [1000000],
            },
            index=pd.to_datetime(["2024-01-09"]),
        )

        ticker.history = Mock(return_value=history_data)

        return ticker

    def test_initialization(self, data_source):
        """Test data source initialization."""
        assert data_source.update_interval == 30.0
        assert data_source.symbols == [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
        ]
        assert data_source._running is False
        assert isinstance(data_source._symbols, set)

    @pytest.mark.asyncio
    async def test_start_stop(self, data_source):
        """Test starting and stopping the data source."""
        # Start
        await data_source.start()
        assert data_source._running is True

        # Stop
        await data_source.stop()
        assert data_source._running is False

    @pytest.mark.asyncio
    async def test_fetch_real_price_success(self, data_source, mock_ticker):
        """Test fetching real-time price successfully."""
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await data_source.fetch_real_price("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 151.0
        assert result["change_percent"] == pytest.approx((151.0 - 149.0) / 149.0 * 100)
        assert result["volume"] == 1000000
        assert result["source"] == "yahoo_finance"
        assert "timestamp" in result

        # Check yfinance was called correctly
        mock_ticker.history.assert_called_once_with(period="2d")

    @pytest.mark.asyncio
    async def test_fetch_real_price_insufficient_data(
        self, data_source, mock_ticker_insufficient_data
    ):
        """Test handling insufficient history data."""
        with patch("yfinance.Ticker", return_value=mock_ticker_insufficient_data):
            result = await data_source.fetch_real_price("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_real_price_exception(self, data_source):
        """Test exception handling during price fetch."""
        with patch("yfinance.Ticker", side_effect=Exception("Network error")):
            with patch("logging.Logger.error") as mock_logger:
                result = await data_source.fetch_real_price("AAPL")

                assert result is None
                mock_logger.assert_called_once()
                assert "Error fetching price for AAPL" in str(mock_logger.call_args)

    @pytest.mark.asyncio
    async def test_fetch_real_price_zero_previous(self, data_source):
        """Test handling zero previous price."""
        ticker = Mock()
        ticker.info = {}

        # Zero previous price
        history_data = pd.DataFrame(
            {
                "Open": [0.0, 150.0],
                "High": [0.0, 152.0],
                "Low": [0.0, 149.0],
                "Close": [0.0, 151.0],
                "Volume": [0, 1000000],
            },
            index=pd.to_datetime(["2024-01-08", "2024-01-09"]),
        )

        ticker.history = Mock(return_value=history_data)

        with patch("yfinance.Ticker", return_value=ticker):
            result = await data_source.fetch_real_price("AAPL")

        # Should handle division by zero gracefully
        assert result is not None
        # When dividing by zero, Python returns inf
        assert np.isinf(result["change_percent"])

    @pytest.mark.asyncio
    async def test_get_market_data_success(self, data_source, mock_ticker):
        """Test getting market data for multiple symbols."""
        # Mock Tickers class
        mock_tickers = Mock()
        mock_tickers.tickers = {"AAPL": mock_ticker, "GOOGL": mock_ticker}

        data_source.symbols = ["AAPL", "GOOGL"]

        with patch("yfinance.Tickers", return_value=mock_tickers):
            results = await data_source.get_market_data()

        assert len(results) == 2
        assert all(r["symbol"] in ["AAPL", "GOOGL"] for r in results)
        assert all("price" in r for r in results)
        assert all("change_percent" in r for r in results)
        assert all(r["source"] == "yahoo_finance" for r in results)

    @pytest.mark.asyncio
    async def test_get_market_data_partial_failure(
        self, data_source, mock_ticker, mock_ticker_insufficient_data
    ):
        """Test getting market data with some symbols failing."""
        # Mock Tickers with mixed results
        mock_tickers = Mock()
        mock_tickers.tickers = {
            "AAPL": mock_ticker,  # Good data
            "GOOGL": mock_ticker_insufficient_data,  # Insufficient data
        }

        data_source.symbols = ["AAPL", "GOOGL"]

        with patch("yfinance.Tickers", return_value=mock_tickers):
            results = await data_source.get_market_data()

        # Should only have data for AAPL
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_market_data_exception(self, data_source):
        """Test exception handling in market data fetch."""
        with patch("yfinance.Tickers", side_effect=Exception("API error")):
            with patch("logging.Logger.error") as mock_logger:
                results = await data_source.get_market_data()

                assert results == []
                mock_logger.assert_called_once()
                assert "Error fetching market data" in str(mock_logger.call_args)

    @pytest.mark.asyncio
    async def test_get_market_data_missing_volume(self, data_source):
        """Test handling missing volume data."""
        ticker = Mock()

        # History without Volume column
        history_data = pd.DataFrame(
            {
                "Open": [148.0, 150.0],
                "High": [151.0, 152.0],
                "Low": [147.0, 149.0],
                "Close": [149.0, 151.0],
            },
            index=pd.to_datetime(["2024-01-08", "2024-01-09"]),
        )

        ticker.history = Mock(return_value=history_data)

        mock_tickers = Mock()
        mock_tickers.tickers = {"AAPL": ticker}

        data_source.symbols = ["AAPL"]

        with patch("yfinance.Tickers", return_value=mock_tickers):
            results = await data_source.get_market_data()

        assert len(results) == 1
        assert results[0]["volume"] == 0  # Default when Volume is missing

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, data_source):
        """Test getting historical data successfully."""
        # Create mock historical data
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        history_data = pd.DataFrame(
            {
                "Open": np.random.uniform(140, 160, 252),
                "High": np.random.uniform(145, 165, 252),
                "Low": np.random.uniform(135, 155, 252),
                "Close": np.random.uniform(140, 160, 252),
                "Volume": np.random.randint(500000, 2000000, 252),
            },
            index=dates,
        )

        ticker = Mock()
        ticker.history = Mock(return_value=history_data)

        with patch("yfinance.Ticker", return_value=ticker):
            results = await data_source.get_historical_data("AAPL", "1y")

        assert len(results) == 252
        assert all(r["symbol"] == "AAPL" for r in results)

        # Check first entry structure
        first = results[0]
        assert "date" in first
        assert "open" in first
        assert "high" in first
        assert "low" in first
        assert "close" in first
        assert "volume" in first

        # Check data types
        assert isinstance(first["open"], float)
        assert isinstance(first["volume"], int)

        # Check yfinance was called correctly
        ticker.history.assert_called_once_with(period="1y")

    @pytest.mark.asyncio
    async def test_get_historical_data_exception(self, data_source):
        """Test exception handling in historical data fetch."""
        with patch("yfinance.Ticker", side_effect=Exception("Connection error")):
            with patch("logging.Logger.error") as mock_logger:
                results = await data_source.get_historical_data("AAPL")

                assert results == []
                mock_logger.assert_called_once()
                assert "Error fetching historical data for AAPL" in str(
                    mock_logger.call_args
                )

    @pytest.mark.asyncio
    async def test_get_historical_data_empty_response(self, data_source):
        """Test handling empty historical data response."""
        ticker = Mock()
        ticker.history = Mock(return_value=pd.DataFrame())  # Empty dataframe

        with patch("yfinance.Ticker", return_value=ticker):
            results = await data_source.get_historical_data("AAPL")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_historical_data_custom_period(self, data_source):
        """Test historical data with custom period."""
        # Create mock data for 1 month
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        history_data = pd.DataFrame(
            {
                "Open": [150.0] * 30,
                "High": [152.0] * 30,
                "Low": [148.0] * 30,
                "Close": [151.0] * 30,
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

        ticker = Mock()
        ticker.history = Mock(return_value=history_data)

        with patch("yfinance.Ticker", return_value=ticker):
            results = await data_source.get_historical_data("AAPL", "1mo")

        assert len(results) == 30
        ticker.history.assert_called_once_with(period="1mo")


class TestYahooFinanceIntegration:
    """Integration tests for Yahoo Finance data source."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full lifecycle of data source."""
        data_source = YahooFinanceDataSource(update_interval=10.0)

        # Start
        await data_source.start()
        assert data_source._running is True

        # Mock successful market data fetch
        with patch.object(data_source, "get_market_data") as mock_get_data:
            mock_get_data.return_value = [
                {
                    "symbol": "AAPL",
                    "price": 150.0,
                    "change_percent": 1.0,
                    "volume": 1000000,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "yahoo_finance",
                }
            ]

            results = await data_source.get_market_data()
            assert len(results) == 1

        # Stop
        await data_source.stop()
        assert data_source._running is False

    @pytest.mark.asyncio
    async def test_batch_efficiency(self):
        """Test that batch requests are used efficiently."""
        data_source = YahooFinanceDataSource()
        data_source.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # Mock ticker data
        mock_ticker = Mock()
        history_data = pd.DataFrame(
            {
                "Open": [148.0, 150.0],
                "High": [151.0, 152.0],
                "Low": [147.0, 149.0],
                "Close": [149.0, 151.0],
                "Volume": [900000, 1000000],
            },
            index=pd.to_datetime(["2024-01-08", "2024-01-09"]),
        )
        mock_ticker.history = Mock(return_value=history_data)

        # Mock Tickers
        mock_tickers = Mock()
        mock_tickers.tickers = {s: mock_ticker for s in data_source.symbols}

        with patch("yfinance.Tickers", return_value=mock_tickers) as mock_tickers_class:
            results = await data_source.get_market_data()

            # Check that Tickers was called with all symbols at once
            mock_tickers_class.assert_called_once()
            call_args = mock_tickers_class.call_args[0][0]
            assert "AAPL" in call_args
            assert "GOOGL" in call_args
            assert "MSFT" in call_args
            assert "AMZN" in call_args

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        data_source = YahooFinanceDataSource()

        # Mock ticker
        ticker = Mock()
        history_data = pd.DataFrame(
            {
                "Open": [148.0, 150.0],
                "High": [151.0, 152.0],
                "Low": [147.0, 149.0],
                "Close": [149.0, 151.0],
                "Volume": [900000, 1000000],
            },
            index=pd.to_datetime(["2024-01-08", "2024-01-09"]),
        )
        ticker.history = Mock(return_value=history_data)

        with patch("yfinance.Ticker", return_value=ticker):
            # Launch concurrent requests
            tasks = [
                data_source.fetch_real_price("AAPL"),
                data_source.fetch_real_price("GOOGL"),
                data_source.fetch_real_price("MSFT"),
            ]

            results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r is not None for r in results)
