"""Comprehensive tests for Alpha Vantage data source."""

import pytest
import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging

from src.streaming.alpha_vantage_source import AlphaVantageDataSource


class TestAlphaVantageDataSource:
    """Test AlphaVantageDataSource class."""

    @pytest.fixture
    def api_key(self):
        """Test API key."""
        return "TEST_API_KEY_123"

    @pytest.fixture
    def data_source(self, api_key):
        """Create Alpha Vantage data source instance."""
        return AlphaVantageDataSource(api_key=api_key, update_interval=60.0)

    @pytest.fixture
    def mock_response_global_quote(self):
        """Mock response for GLOBAL_QUOTE endpoint."""
        return {
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "150.00",
                "03. high": "152.00",
                "04. low": "149.00",
                "05. price": "151.50",
                "06. volume": "1000000",
                "07. latest trading day": "2024-01-09",
                "08. previous close": "150.00",
                "09. change": "1.50",
                "10. change percent": "1.00%",
            }
        }

    @pytest.fixture
    def mock_response_time_series(self):
        """Mock response for TIME_SERIES_DAILY endpoint."""
        return {
            "Meta Data": {
                "1. Information": "Daily Prices (full size)",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2024-01-09",
                "4. Output Size": "Full size",
                "5. Time Zone": "US/Eastern",
            },
            "Time Series (Daily)": {
                "2024-01-09": {
                    "1. open": "150.00",
                    "2. high": "152.00",
                    "3. low": "149.00",
                    "4. close": "151.50",
                    "5. volume": "1000000",
                },
                "2024-01-08": {
                    "1. open": "149.00",
                    "2. high": "151.00",
                    "3. low": "148.00",
                    "4. close": "150.00",
                    "5. volume": "900000",
                },
                "2024-01-05": {
                    "1. open": "148.00",
                    "2. high": "150.00",
                    "3. low": "147.00",
                    "4. close": "149.00",
                    "5. volume": "850000",
                },
            },
        }

    def test_initialization(self, data_source, api_key):
        """Test data source initialization."""
        assert data_source.api_key == api_key
        assert data_source.update_interval == 60.0
        assert data_source.base_url == "https://www.alphavantage.co/query"
        assert data_source.session is None
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
        assert isinstance(data_source.session, aiohttp.ClientSession)

        # Stop
        await data_source.stop()
        assert data_source._running is False

    @pytest.mark.asyncio
    async def test_fetch_real_price_success(
        self, data_source, mock_response_global_quote
    ):
        """Test fetching real-time price successfully."""
        # Create mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_global_quote)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        # Fetch price
        result = await data_source.fetch_real_price("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 151.50
        assert result["change_percent"] == 1.0  # (151.50 - 150.00) / 150.00 * 100
        assert result["volume"] == 1000000
        assert result["source"] == "alphavantage"
        assert "timestamp" in result

        # Check API call
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[0][0] == data_source.base_url
        assert call_args[1]["params"]["function"] == "GLOBAL_QUOTE"
        assert call_args[1]["params"]["symbol"] == "AAPL"
        assert call_args[1]["params"]["apikey"] == "TEST_API_KEY_123"

    @pytest.mark.asyncio
    async def test_fetch_real_price_no_session(self, data_source):
        """Test fetching price without active session."""
        result = await data_source.fetch_real_price("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_real_price_api_error(self, data_source):
        """Test handling API error responses."""
        # Create mock session with error response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        result = await data_source.fetch_real_price("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_real_price_empty_quote(self, data_source):
        """Test handling empty quote response."""
        # Mock response without Global Quote
        mock_response_empty = {"Note": "API call frequency limit reached"}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_empty)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        result = await data_source.fetch_real_price("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_real_price_exception(self, data_source):
        """Test exception handling during price fetch."""
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=aiohttp.ClientError("Network error"))

        data_source.session = mock_session

        with patch("logging.Logger.error") as mock_logger:
            result = await data_source.fetch_real_price("AAPL")
            assert result is None
            mock_logger.assert_called_once()
            assert "Error fetching price for AAPL" in str(mock_logger.call_args)

    @pytest.mark.asyncio
    async def test_fetch_real_price_zero_previous_close(self, data_source):
        """Test handling zero previous close price."""
        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "151.50",
                "06. volume": "1000000",
                "08. previous close": "0.00",
            }
        }

        mock_session = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        result = await data_source.fetch_real_price("AAPL")
        assert result["change_percent"] == 0  # Should handle division by zero

    @pytest.mark.asyncio
    async def test_get_market_data(self, data_source, mock_response_global_quote):
        """Test getting market data for multiple symbols."""
        # Mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_global_quote)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        # Override symbols for testing
        data_source.symbols = ["AAPL", "GOOGL"]

        # Get market data
        with patch("asyncio.sleep") as mock_sleep:
            results = await data_source.get_market_data()

        assert len(results) == 2
        assert all(r["symbol"] in ["AAPL", "GOOGL"] for r in results)

        # Check rate limiting sleep was called
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(12)  # 60 seconds / 5 requests

    @pytest.mark.asyncio
    async def test_get_market_data_partial_failure(self, data_source):
        """Test getting market data with some failures."""
        # Mock session with alternating success/failure
        mock_session = AsyncMock()

        # First call succeeds
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(
            return_value={
                "Global Quote": {
                    "01. symbol": "AAPL",
                    "05. price": "151.50",
                    "06. volume": "1000000",
                    "08. previous close": "150.00",
                }
            }
        )

        # Second call fails
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500

        # Alternate responses
        # Create context managers for each response
        mock_context_success = AsyncMock()
        mock_context_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_context_success.__aexit__ = AsyncMock(return_value=None)

        mock_context_fail = AsyncMock()
        mock_context_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_context_fail.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(side_effect=[mock_context_success, mock_context_fail])

        data_source.session = mock_session
        data_source.symbols = ["AAPL", "GOOGL"]

        with patch("asyncio.sleep"):
            results = await data_source.get_market_data()

        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_historical_data_success(
        self, data_source, mock_response_time_series
    ):
        """Test getting historical data successfully."""
        # Mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_time_series)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        # Get historical data
        results = await data_source.get_historical_data("AAPL", "1year")

        assert len(results) == 3
        assert all(r["symbol"] == "AAPL" for r in results)

        # Check data is sorted by date
        dates = [r["date"] for r in results]
        assert dates == sorted(dates)

        # Check first entry
        assert results[0]["date"] == "2024-01-05"
        assert results[0]["open"] == 148.0
        assert results[0]["close"] == 149.0
        assert results[0]["volume"] == 850000

        # Check API call
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args[1]["params"]["function"] == "TIME_SERIES_DAILY"
        assert call_args[1]["params"]["symbol"] == "AAPL"
        assert call_args[1]["params"]["outputsize"] == "full"

    @pytest.mark.asyncio
    async def test_get_historical_data_no_session(self, data_source):
        """Test getting historical data without session."""
        results = await data_source.get_historical_data("AAPL")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_historical_data_api_error(self, data_source):
        """Test handling API error for historical data."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401  # Unauthorized

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        results = await data_source.get_historical_data("AAPL")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_historical_data_empty_time_series(self, data_source):
        """Test handling empty time series response."""
        mock_response = {
            "Meta Data": {"2. Symbol": "AAPL"},
            "Error Message": "Invalid API call",
        }

        mock_session = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        results = await data_source.get_historical_data("AAPL")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_historical_data_exception(self, data_source):
        """Test exception handling for historical data."""
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=Exception("Connection timeout"))

        data_source.session = mock_session

        with patch("logging.Logger.error") as mock_logger:
            results = await data_source.get_historical_data("AAPL")
            assert results == []
            mock_logger.assert_called_once()
            assert "Error fetching historical data for AAPL" in str(
                mock_logger.call_args
            )

    @pytest.mark.asyncio
    async def test_get_historical_data_limit_252_days(self, data_source):
        """Test that historical data is limited to last 252 trading days."""
        # Create mock response with 500 days of data
        time_series = {}
        for i in range(500):
            date = f"2024-{(12 - i // 30):02d}-{(30 - i % 30):02d}"
            time_series[date] = {
                "1. open": "150.00",
                "2. high": "152.00",
                "3. low": "149.00",
                "4. close": "151.00",
                "5. volume": "1000000",
            }

        mock_response = {
            "Meta Data": {"2. Symbol": "AAPL"},
            "Time Series (Daily)": time_series,
        }

        mock_session = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        results = await data_source.get_historical_data("AAPL")
        assert len(results) == 252  # Limited to last year


class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage data source."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full lifecycle of data source."""
        data_source = AlphaVantageDataSource(api_key="TEST_KEY", update_interval=30.0)

        # Start
        await data_source.start()
        assert data_source._running is True
        assert data_source.session is not None

        # Mock successful price fetch
        with patch.object(data_source, "fetch_real_price") as mock_fetch:
            mock_fetch.return_value = {
                "symbol": "AAPL",
                "price": 150.0,
                "change_percent": 1.0,
                "volume": 1000000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "alphavantage",
            }

            with patch("asyncio.sleep"):
                results = await data_source.get_market_data()
                assert len(results) == len(data_source.symbols)

        # Stop
        await data_source.stop()
        assert data_source._running is False

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test rate limiting is properly enforced."""
        data_source = AlphaVantageDataSource(api_key="TEST_KEY")
        data_source.symbols = ["AAPL", "GOOGL", "MSFT"]

        # Mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "Global Quote": {
                    "01. symbol": "TEST",
                    "05. price": "100.00",
                    "06. volume": "1000",
                    "08. previous close": "99.00",
                }
            }
        )

        # Create context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = Mock(return_value=mock_context)

        data_source.session = mock_session

        # Track sleep calls
        sleep_calls = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("asyncio.sleep", mock_sleep):
            await data_source.get_market_data()

        # Should have slept between each API call (3 symbols = 3 sleeps)
        assert len(sleep_calls) == 3
        assert all(s == 12 for s in sleep_calls)  # 60/5 = 12 seconds between calls
