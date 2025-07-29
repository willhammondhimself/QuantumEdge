"""
Abstract base classes for data providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import asyncio
import logging

from .models import Asset, Price, MarketData, DataFrequency, MarketMetrics, DataSourceError

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for market data providers."""
    
    def __init__(self, name: str, rate_limit_per_minute: int = 60):
        """
        Initialize data provider.
        
        Args:
            name: Provider name
            rate_limit_per_minute: API rate limit per minute
        """
        self.name = name
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_request_times: List[datetime] = []
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self._last_request_times = [
            t for t in self._last_request_times 
            if (now - t).total_seconds() < 60
        ]
        
        # Check if we're at the rate limit
        if len(self._last_request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self._last_request_times[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached for {self.name}, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self._last_request_times.append(now)
    
    @abstractmethod
    async def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """Get asset information."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Price]:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[Price]]:
        """Get current prices for multiple symbols."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> Optional[MarketData]:
        """Get historical market data."""
        pass
    
    @abstractmethod
    async def get_market_metrics(self) -> Optional[MarketMetrics]:
        """Get current market-wide metrics."""
        pass
    
    async def get_multiple_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
        max_concurrent: int = 5
    ) -> Dict[str, Optional[MarketData]]:
        """Get historical data for multiple symbols with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_data(symbol: str) -> tuple[str, Optional[MarketData]]:
            async with semaphore:
                try:
                    data = await self.get_historical_data(symbol, start_date, end_date, frequency)
                    return symbol, data
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return symbol, None
        
        tasks = [fetch_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            
            symbol, data = result
            data_dict[symbol] = data
        
        return data_dict
    
    def validate_symbol(self, symbol: str) -> str:
        """Validate and normalize symbol."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        return symbol.upper().strip()
    
    def validate_date_range(self, start_date: date, end_date: date) -> None:
        """Validate date range."""
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > date.today():
            raise ValueError("End date cannot be in the future")
    
    async def health_check(self) -> bool:
        """Check if the data provider is healthy."""
        try:
            # Try to get data for a common symbol
            test_data = await self.get_current_price("AAPL")
            return test_data is not None
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return False


class CachedDataProvider(DataProvider):
    """Data provider with caching capabilities."""
    
    def __init__(self, name: str, cache_ttl_seconds: int = 300, rate_limit_per_minute: int = 60):
        """
        Initialize cached data provider.
        
        Args:
            name: Provider name
            cache_ttl_seconds: Cache time-to-live in seconds
            rate_limit_per_minute: API rate limit per minute
        """
        super().__init__(name, rate_limit_per_minute)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, tuple[datetime, Any]] = {}
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key."""
        key_parts = [method] + [str(arg) for arg in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self._cache:
            return None
        
        timestamp, value = self._cache[key]
        if (datetime.now() - timestamp).total_seconds() > self.cache_ttl_seconds:
            del self._cache[key]
            return None
        
        return value
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set item in cache."""
        self._cache[key] = (datetime.now(), value)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        now = datetime.now()
        expired_count = 0
        
        for timestamp, _ in self._cache.values():
            if (now - timestamp).total_seconds() > self.cache_ttl_seconds:
                expired_count += 1
        
        return {
            'total_items': len(self._cache),
            'expired_items': expired_count,
            'active_items': len(self._cache) - expired_count
        }