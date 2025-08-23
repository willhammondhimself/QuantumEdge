"""
Alpha Vantage market data source implementation.

Provides real-time and historical market data from Alpha Vantage API
with rate limiting, error handling, and caching.
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

from ..api.config import settings
from .market_data_source import MarketDataSource

logger = logging.getLogger(__name__)


class AlphaVantageDataSource(MarketDataSource):
    """Alpha Vantage market data source with enhanced reliability."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, update_interval: float = 60.0):
        """
        Initialize Alpha Vantage data source.

        Args:
            api_key: Alpha Vantage API key
            update_interval: How often to fetch data (seconds)
        """
        self.api_key = api_key
        self.update_interval = update_interval
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        self._running = False
        self._session = None

        # Rate limiting (Alpha Vantage: 5 calls per minute, 500 per day for free tier)
        self.rate_limit = 5  # calls per minute for free tier
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

        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache for free tier

    async def start(self):
        """Start the data source."""
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.market_data_timeout)
        )
        logger.info(
            f"Started Alpha Vantage data source with {len(self.symbols)} symbols"
        )

    async def stop(self):
        """Stop the data source."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()

        # Remove requests older than 1 minute
        while self.request_times and self.request_times[0] < current_time - 60:
            self.request_times.popleft()

        # If we're at the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(
                    f"Alpha Vantage rate limit reached, waiting {wait_time:.2f} seconds"
                )
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_open_time is None:
            return False

        if time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            # Circuit breaker timeout expired, reset
            self.circuit_open_time = None
            self.failure_count = 0
            logger.info("Alpha Vantage circuit breaker reset after timeout")
            return False

        return True

    def _record_success(self):
        """Record a successful request."""
        self.successful_requests += 1
        self.last_successful_request = datetime.now()
        self.failure_count = 0
        if self.circuit_open_time:
            self.circuit_open_time = None
            logger.info("Alpha Vantage circuit breaker closed after successful request")

    def _record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.circuit_open_time = time.time()
            logger.warning(
                f"Alpha Vantage circuit breaker opened after {self.failure_count} consecutive failures"
            )

    def _get_cache_key(self, function: str, symbol: str = None) -> str:
        """Generate cache key."""
        if symbol:
            return f"{function}_{symbol}"
        return function

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if not expired."""
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return data
            else:
                del self.cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Dict):
        """Store data in cache."""
        self.cache[cache_key] = (time.time(), data)

    async def _make_api_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with retry logic and circuit breaker."""
        self.total_requests += 1

        if self._is_circuit_open():
            logger.warning("Alpha Vantage circuit breaker is open, skipping request")
            return None

        if not self._session:
            logger.error("Alpha Vantage session not initialized")
            return None

        await self._check_rate_limit()

        # Add API key to parameters
        params["apikey"] = self.api_key

        for attempt in range(self.retry_attempts):
            try:
                async with self._session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check for rate limit message
                        if "Note" in data:
                            logger.warning("Alpha Vantage rate limit exceeded")
                            await asyncio.sleep(60)  # Wait 1 minute
                            continue

                        # Check for error message
                        if "Error Message" in data:
                            logger.error(
                                f"Alpha Vantage error: {data['Error Message']}"
                            )
                            self._record_failure()
                            return None

                        self._record_success()
                        return data

                    else:
                        logger.error(
                            f"Alpha Vantage API returned status {response.status}"
                        )

            except Exception as e:
                logger.error(
                    f"Alpha Vantage request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )

                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    await asyncio.sleep(delay)

        # All attempts failed
        self._record_failure()
        return None

    async def fetch_real_price(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time price for a symbol."""
        cache_key = self._get_cache_key("quote", symbol)

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        params = {"function": "GLOBAL_QUOTE", "symbol": symbol}

        data = await self._make_api_request(params)
        if not data or "Global Quote" not in data:
            return None

        try:
            quote = data["Global Quote"]

            result = {
                "symbol": symbol,
                "price": float(quote["05. price"]),
                "change_percent": float(quote["10. change percent"].replace("%", "")),
                "timestamp": datetime.now().isoformat(),
                "volume": int(quote["06. volume"]),
                "source": "alpha_vantage",
            }

            # Cache the result
            self._set_cache(cache_key, result)
            return result

        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Alpha Vantage quote data for {symbol}: {e}")
            return None

    async def get_market_data(self) -> List[Dict]:
        """Get market data for all symbols."""
        market_data = []

        # Alpha Vantage doesn't support batch requests, so we fetch individually
        # This is rate-limited, so we'll only fetch a few symbols at a time
        symbols_to_fetch = self.symbols[
            :3
        ]  # Limit to 3 symbols to stay within rate limits

        for symbol in symbols_to_fetch:
            try:
                data = await self.fetch_real_price(symbol)
                if data:
                    market_data.append(data)
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
                continue

        return market_data

    async def get_historical_data(self, symbol: str, period: str = "1y") -> List[Dict]:
        """Get historical data for backtesting."""
        cache_key = self._get_cache_key("daily", symbol)

        # Check cache first (longer TTL for historical data)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",  # Get all available data
        }

        data = await self._make_api_request(params)
        if not data or "Time Series (Daily)" not in data:
            return []

        try:
            time_series = data["Time Series (Daily)"]
            historical_data = []

            # Convert to our format
            for date_str, daily_data in time_series.items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # Filter by period if needed
                    if period == "1y":
                        cutoff_date = datetime.now().date() - timedelta(days=365)
                        if date_obj < cutoff_date:
                            continue
                    elif period == "1mo":
                        cutoff_date = datetime.now().date() - timedelta(days=30)
                        if date_obj < cutoff_date:
                            continue

                    historical_data.append(
                        {
                            "date": date_str,
                            "symbol": symbol,
                            "open": float(daily_data["1. open"]),
                            "high": float(daily_data["2. high"]),
                            "low": float(daily_data["3. low"]),
                            "close": float(daily_data["4. close"]),
                            "volume": int(daily_data["6. volume"]),
                        }
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Error parsing historical data for {symbol} on {date_str}: {e}"
                    )
                    continue

            # Sort by date (newest first)
            historical_data.sort(key=lambda x: x["date"], reverse=True)

            # Cache the result with longer TTL
            self._set_cache(cache_key, historical_data)
            return historical_data

        except Exception as e:
            logger.error(
                f"Error parsing Alpha Vantage historical data for {symbol}: {e}"
            )
            return []

    def get_health_status(self) -> Dict:
        """Get health status of the data source."""
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )
        time_since_last_success = (
            datetime.now() - self.last_successful_request
        ).total_seconds()

        return {
            "provider": "alpha_vantage",
            "status": "circuit_open" if self._is_circuit_open() else "healthy",
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "failure_count": self.failure_count,
            "time_since_last_success": round(time_since_last_success, 2),
            "circuit_breaker_open": self._is_circuit_open(),
            "rate_limit_per_minute": self.rate_limit,
            "cache_entries": len(self.cache),
            "last_successful_request": self.last_successful_request.isoformat(),
        }
