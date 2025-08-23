"""
Yahoo Finance real market data source (free)
Uses yfinance library for real-time data with enhanced reliability
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
from collections import deque
from ..api.config import settings
from .market_data_source import MarketDataSource

logger = logging.getLogger(__name__)


class YahooFinanceDataSource(MarketDataSource):
    """Real-time market data from Yahoo Finance with enhanced reliability"""

    def __init__(self, update_interval: float = 30.0):
        """
        Initialize Yahoo Finance data source

        Args:
            update_interval: How often to fetch data (seconds)
        """
        self.update_interval = update_interval
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        self._running = False
        self._symbols = set()  # For test compatibility

        # Rate limiting
        self.rate_limit = settings.market_data_rate_limit
        self.request_times = deque(maxlen=self.rate_limit)

        # Circuit breaker
        self.failure_count = 0
        self.failure_threshold = settings.market_data_failure_threshold
        self.circuit_breaker_timeout = settings.market_data_circuit_breaker_timeout
        self.circuit_open_time = None

        # Retry configuration
        self.retry_attempts = settings.market_data_retry_attempts
        self.retry_delay = settings.market_data_retry_delay

        # Health monitoring
        self.last_successful_request = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0

    async def start(self):
        """Start the data source"""
        self._running = True
        logger.info(
            f"Started Yahoo Finance data source with {len(self.symbols)} symbols"
        )

    async def stop(self):
        """Stop the data source"""
        self._running = False

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()

        # Remove requests older than 1 minute
        while self.request_times and self.request_times[0] < current_time - 60:
            self.request_times.popleft()

        # If we're at the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open_time is None:
            return False

        if time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            # Circuit breaker timeout expired, reset
            self.circuit_open_time = None
            self.failure_count = 0
            logger.info("Circuit breaker reset after timeout")
            return False

        return True

    def _record_success(self):
        """Record a successful request"""
        self.successful_requests += 1
        self.last_successful_request = datetime.now()
        self.failure_count = 0
        if self.circuit_open_time:
            self.circuit_open_time = None
            logger.info("Circuit breaker closed after successful request")

    def _record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.circuit_open_time = time.time()
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} consecutive failures"
            )

    async def _make_request_with_retry(self, request_func, *args, **kwargs):
        """Make a request with retry logic and circuit breaker"""
        self.total_requests += 1

        if self._is_circuit_open():
            logger.warning("Circuit breaker is open, skipping request")
            return None

        await self._check_rate_limit()

        for attempt in range(self.retry_attempts):
            try:
                result = await request_func(*args, **kwargs)
                if result is not None:
                    self._record_success()
                    return result
                else:
                    logger.warning(
                        f"Request returned None (attempt {attempt + 1}/{self.retry_attempts})"
                    )

            except Exception as e:
                logger.error(
                    f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )

                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    await asyncio.sleep(delay)

        # All attempts failed
        self._record_failure()
        return None

    def get_health_status(self) -> Dict:
        """Get health status of the data source"""
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )
        time_since_last_success = (
            datetime.now() - self.last_successful_request
        ).total_seconds()

        return {
            "provider": "yahoo_finance",
            "status": "circuit_open" if self._is_circuit_open() else "healthy",
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "failure_count": self.failure_count,
            "time_since_last_success": round(time_since_last_success, 2),
            "circuit_breaker_open": self._is_circuit_open(),
            "last_successful_request": self.last_successful_request.isoformat(),
        }

    async def fetch_real_price(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time price for a symbol"""
        return await self._make_request_with_retry(self._fetch_price_internal, symbol)

    async def _fetch_price_internal(self, symbol: str) -> Optional[Dict]:
        """Internal method to fetch price data"""
        ticker = yf.Ticker(symbol)

        # Get current data
        info = ticker.info
        history = ticker.history(period="2d")

        if len(history) >= 2:
            current_price = history["Close"].iloc[-1]
            previous_price = history["Close"].iloc[-2]
            change_percent = (current_price - previous_price) / previous_price * 100

            return {
                "symbol": symbol,
                "price": float(current_price),
                "change_percent": float(change_percent),
                "timestamp": datetime.now().isoformat(),
                "volume": int(history["Volume"].iloc[-1]) if "Volume" in history else 0,
                "source": "yahoo_finance",
            }

        return None

    async def get_market_data(self) -> List[Dict]:
        """Get market data for all symbols"""
        return await self._make_request_with_retry(self._get_market_data_internal) or []

    async def _get_market_data_internal(self) -> List[Dict]:
        """Internal method to fetch market data"""
        market_data = []

        # Yahoo Finance allows batch requests
        symbols_str = " ".join(self.symbols)
        tickers = yf.Tickers(symbols_str)

        for symbol in self.symbols:
            try:
                ticker = tickers.tickers[symbol]
                history = ticker.history(period="2d")

                if len(history) >= 2:
                    current_price = history["Close"].iloc[-1]
                    previous_price = history["Close"].iloc[-2]
                    change_percent = (
                        (current_price - previous_price) / previous_price * 100
                    )

                    market_data.append(
                        {
                            "symbol": symbol,
                            "price": float(current_price),
                            "change_percent": float(change_percent),
                            "timestamp": datetime.now().isoformat(),
                            "volume": (
                                int(history["Volume"].iloc[-1])
                                if "Volume" in history
                                else 0
                            ),
                            "source": "yahoo_finance",
                        }
                    )
            except Exception as e:
                logger.warning(f"Error fetching data for symbol {symbol}: {e}")
                continue

        return market_data

    async def get_historical_data(self, symbol: str, period: str = "1y") -> List[Dict]:
        """Get historical data for backtesting"""
        return (
            await self._make_request_with_retry(
                self._get_historical_data_internal, symbol, period
            )
            or []
        )

    async def _get_historical_data_internal(
        self, symbol: str, period: str = "1y"
    ) -> List[Dict]:
        """Internal method to fetch historical data"""
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)

        historical_data = []
        for date, row in history.iterrows():
            historical_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )

        return historical_data
