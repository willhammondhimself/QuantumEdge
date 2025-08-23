"""
Data caching system using Redis.
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import logging

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .models import Asset, Price, MarketData, MarketMetrics, DataValidationError

logger = logging.getLogger(__name__)


class DataCache:
    """Data caching system with Redis backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 300,
        key_prefix: str = "quantumedge:data:",
    ):
        """
        Initialize data cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                self.available = True
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.redis_client = None
                self.available = False
                self._memory_cache: Dict[str, tuple[datetime, Any]] = {}
        else:
            logger.info("Redis not available. Using in-memory cache.")
            self.redis_client = None
            self.available = False
            self._memory_cache: Dict[str, tuple[datetime, Any]] = {}

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        full_key = self._make_key(key)

        try:
            if self.redis_client and self.available:
                # Use Redis
                serialized = self._serialize_value(value)
                return self.redis_client.setex(full_key, ttl, serialized)
            else:
                # Use in-memory cache
                expiry = datetime.now() + timedelta(seconds=ttl)
                self._memory_cache[full_key] = (expiry, value)
                return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._make_key(key)

        try:
            if self.redis_client and self.available:
                # Use Redis
                serialized = self.redis_client.get(full_key)
                if serialized:
                    return self._deserialize_value(serialized)
                return None
            else:
                # Use in-memory cache
                if full_key in self._memory_cache:
                    expiry, value = self._memory_cache[full_key]
                    if datetime.now() < expiry:
                        return value
                    else:
                        # Expired, remove it
                        del self._memory_cache[full_key]
                return None
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        full_key = self._make_key(key)

        try:
            if self.redis_client and self.available:
                return bool(self.redis_client.delete(full_key))
            else:
                if full_key in self._memory_cache:
                    del self._memory_cache[full_key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = self._make_key(key)

        try:
            if self.redis_client and self.available:
                return bool(self.redis_client.exists(full_key))
            else:
                if full_key in self._memory_cache:
                    expiry, _ = self._memory_cache[full_key]
                    if datetime.now() < expiry:
                        return True
                    else:
                        del self._memory_cache[full_key]
                return False
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        full_pattern = self._make_key(pattern)

        try:
            if self.redis_client and self.available:
                keys = self.redis_client.keys(full_pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # For in-memory cache, do simple pattern matching
                count = 0
                keys_to_delete = []
                for key in self._memory_cache.keys():
                    if pattern.replace("*", "") in key:
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    del self._memory_cache[key]
                    count += 1

                return count
        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client and self.available:
                info = self.redis_client.info()
                return {
                    "backend": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            else:
                # In-memory cache stats
                now = datetime.now()
                active_keys = 0
                expired_keys = 0

                for full_key, (expiry, _) in self._memory_cache.items():
                    if now < expiry:
                        active_keys += 1
                    else:
                        expired_keys += 1

                return {
                    "backend": "memory",
                    "active_keys": active_keys,
                    "expired_keys": expired_keys,
                    "total_keys": len(self._memory_cache),
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"backend": "error", "error": str(e)}

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if isinstance(value, (Asset, Price, MarketData, MarketMetrics)):
            # Use pickle for complex objects
            return pickle.dumps(value)
        else:
            # Use JSON for simple types
            return json.dumps(value, default=str).encode()

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try pickle first
            return pickle.loads(data)
        except (pickle.PickleError, pickle.UnpicklingError):
            try:
                # Fall back to JSON
                return json.loads(data.decode())
            except json.JSONDecodeError:
                logger.error("Failed to deserialize cached data")
                return None


class MarketDataCache:
    """Specialized cache for market data with convenience methods."""

    def __init__(self, cache: DataCache):
        """Initialize with data cache instance."""
        self.cache = cache

    def cache_price(self, price: Price, ttl: int = 60) -> bool:
        """Cache current price with short TTL."""
        key = f"price:{price.symbol}:current"
        return self.cache.set(key, price, ttl)

    def get_cached_price(self, symbol: str) -> Optional[Price]:
        """Get cached current price."""
        key = f"price:{symbol}:current"
        return self.cache.get(key)

    def cache_historical_data(self, data: MarketData, ttl: int = 3600) -> bool:
        """Cache historical data with longer TTL."""
        key = f"historical:{data.symbol}:{data.start_date}:{data.end_date}:{data.frequency}"
        return self.cache.set(key, data, ttl)

    def get_cached_historical_data(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> Optional[MarketData]:
        """Get cached historical data."""
        key = f"historical:{symbol}:{start_date}:{end_date}:{frequency}"
        return self.cache.get(key)

    def cache_asset_info(self, asset: Asset, ttl: int = 86400) -> bool:
        """Cache asset info with long TTL (24 hours)."""
        key = f"asset:{asset.symbol}"
        return self.cache.set(key, asset, ttl)

    def get_cached_asset_info(self, symbol: str) -> Optional[Asset]:
        """Get cached asset info."""
        key = f"asset:{symbol}"
        return self.cache.get(key)

    def cache_market_metrics(self, metrics: MarketMetrics, ttl: int = 300) -> bool:
        """Cache market metrics."""
        key = "market_metrics:current"
        return self.cache.set(key, metrics, ttl)

    def get_cached_market_metrics(self) -> Optional[MarketMetrics]:
        """Get cached market metrics."""
        key = "market_metrics:current"
        return self.cache.get(key)

    def clear_symbol_data(self, symbol: str) -> int:
        """Clear all cached data for a symbol."""
        patterns = [f"price:{symbol}:*", f"historical:{symbol}:*", f"asset:{symbol}"]

        count = 0
        for pattern in patterns:
            count += self.cache.clear_pattern(pattern)

        return count

    def warm_cache(self, symbols: List[str], data_provider) -> Dict[str, bool]:
        """Warm cache with current prices for symbols."""
        results = {}

        for symbol in symbols:
            try:
                # This would typically be called by the data provider
                # Just mark as attempted for now
                results[symbol] = True
            except Exception as e:
                logger.error(f"Failed to warm cache for {symbol}: {e}")
                results[symbol] = False

        return results


# Global cache instance
_global_cache: Optional[DataCache] = None
_global_market_cache: Optional[MarketDataCache] = None


def get_data_cache(redis_url: str = "redis://localhost:6379/0") -> DataCache:
    """Get global data cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache(redis_url)
    return _global_cache


def get_market_data_cache(
    redis_url: str = "redis://localhost:6379/0",
) -> MarketDataCache:
    """Get global market data cache instance."""
    global _global_market_cache
    if _global_market_cache is None:
        cache = get_data_cache(redis_url)
        _global_market_cache = MarketDataCache(cache)
    return _global_market_cache
