"""
Market data feeds and data management for QuantumEdge.

This module provides real-time and historical market data from various sources
including Yahoo Finance, Alpha Vantage, and other financial data providers.
"""

from typing import List

# Data models
from .models import (
    Asset, Price, MarketData, PortfolioSnapshot, MarketMetrics,
    AssetType, DataFrequency,
    DataValidationError, DataSourceError, RateLimitError
)

# Data providers
from .base import DataProvider, CachedDataProvider
from .yahoo_finance import YahooFinanceProvider, get_yahoo_price, get_yahoo_historical

# Caching system
from .cache import DataCache, MarketDataCache, get_data_cache, get_market_data_cache

__all__: List[str] = [
    # Models
    "Asset", "Price", "MarketData", "PortfolioSnapshot", "MarketMetrics",
    "AssetType", "DataFrequency",
    "DataValidationError", "DataSourceError", "RateLimitError",
    
    # Providers
    "DataProvider", "CachedDataProvider", "YahooFinanceProvider",
    "get_yahoo_price", "get_yahoo_historical",
    
    # Cache
    "DataCache", "MarketDataCache", "get_data_cache", "get_market_data_cache"
]