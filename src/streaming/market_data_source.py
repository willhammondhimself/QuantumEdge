"""
Market data source for real-time data pipeline.

This module integrates with Yahoo Finance and other data providers
to supply real-time market data for WebSocket streaming.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
from collections import defaultdict

from .data_pipeline import MarketDataUpdate
from ..data.yahoo_finance import YahooFinanceProvider

logger = logging.getLogger(__name__)


class MarketDataSource:
    """
    Real-time market data source using Yahoo Finance.
    
    Provides market data updates for the data pipeline with
    rate limiting and error handling.
    """
    
    def __init__(self, rate_limit_per_minute: int = 60):
        """
        Initialize market data source.
        
        Args:
            rate_limit_per_minute: Maximum API calls per minute
        """
        self.provider = YahooFinanceProvider()
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Rate limiting
        self._request_times: List[float] = []
        self._last_prices: Dict[str, float] = {}
        
        # Error tracking
        self._error_count = 0
        self._last_error_time = 0
        self._max_consecutive_errors = 5
        
        logger.info(f"Initialized market data source with {rate_limit_per_minute} requests/minute limit")
    
    async def get_market_updates(self, symbols: List[str]) -> List[MarketDataUpdate]:
        """
        Get market data updates for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch
            
        Returns:
            List of market data updates
        """
        if not symbols:
            return []
        
        # Check rate limit
        if not self._can_make_request():
            logger.debug("Rate limit reached, skipping market data request")
            return []
        
        try:
            # Get current prices
            current_time = time.time()
            prices = await self.provider.get_current_prices(symbols)
            
            updates = []
            for symbol in symbols:
                if symbol in prices and prices[symbol]:
                    price_data = prices[symbol]
                    current_price = price_data.close
                    
                    # Calculate change from last known price
                    last_price = self._last_prices.get(symbol, current_price)
                    change = current_price - last_price
                    change_percent = (change / last_price) if last_price != 0 else 0.0
                    
                    # Create update
                    update = MarketDataUpdate(
                        symbol=symbol,
                        price=current_price,
                        change=change,
                        change_percent=change_percent,
                        volume=price_data.volume or 0,
                        timestamp=price_data.timestamp
                    )
                    
                    updates.append(update)
                    
                    # Update last price
                    self._last_prices[symbol] = current_price
            
            # Track successful request
            self._request_times.append(current_time)
            self._error_count = 0  # Reset error count on success
            
            logger.debug(f"Fetched market data for {len(updates)} symbols")
            return updates
            
        except Exception as e:
            self._error_count += 1
            self._last_error_time = time.time()
            
            logger.error(f"Error fetching market data: {e}")
            
            # Return empty list on error, but don't crash the pipeline
            return []
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self._request_times = [t for t in self._request_times if t > cutoff_time]
        
        # Check if we're under the rate limit
        return len(self._request_times) < self.rate_limit_per_minute
    
    def is_healthy(self) -> bool:
        """Check if the data source is healthy."""
        current_time = time.time()
        
        # Consider unhealthy if too many consecutive errors
        if self._error_count >= self._max_consecutive_errors:
            return False
        
        # Consider unhealthy if recent error
        if current_time - self._last_error_time < 300:  # 5 minutes
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data source statistics."""
        current_time = time.time()
        
        # Count recent requests
        recent_requests = [t for t in self._request_times if t > current_time - 60]
        
        return {
            "provider": self.provider.name,
            "healthy": self.is_healthy(),
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "requests_last_minute": len(recent_requests),
            "error_count": self._error_count,
            "last_error_time": self._last_error_time,
            "symbols_tracked": len(self._last_prices),
            "cache_stats": self.provider.get_cache_stats()
        }


class SimulatedMarketDataSource:
    """
    Simulated market data source for testing and development.
    
    Generates realistic price movements for testing the streaming system.
    """
    
    def __init__(self, base_prices: Optional[Dict[str, float]] = None):
        """
        Initialize simulated data source.
        
        Args:
            base_prices: Base prices for symbols (defaults generated if not provided)
        """
        self.base_prices = base_prices or {}
        self.current_prices = self.base_prices.copy()
        
        # Simulation parameters
        self._volatility = 0.02  # 2% daily volatility
        self._drift = 0.0001     # Slight upward drift
        
        logger.info("Initialized simulated market data source")
    
    async def get_market_updates(self, symbols: List[str]) -> List[MarketDataUpdate]:
        """
        Generate simulated market data updates.
        
        Args:
            symbols: List of symbols to simulate
            
        Returns:
            List of simulated market updates
        """
        import random
        import math
        
        updates = []
        current_time = datetime.utcnow()
        
        for symbol in symbols:
            # Initialize base price if not set
            if symbol not in self.current_prices:
                self.current_prices[symbol] = random.uniform(50, 500)
            
            # Generate price movement using geometric Brownian motion
            last_price = self.current_prices[symbol]
            
            # Random walk with drift
            dt = 1/252  # Assume daily timestep
            random_shock = random.gauss(0, 1)
            
            price_change = (self._drift * dt + 
                           self._volatility * math.sqrt(dt) * random_shock)
            
            new_price = last_price * math.exp(price_change)
            
            # Ensure price stays reasonable
            new_price = max(1.0, min(new_price, 10000.0))
            
            change = new_price - last_price
            change_percent = change / last_price
            
            # Generate random volume
            volume = random.randint(100000, 1000000)
            
            update = MarketDataUpdate(
                symbol=symbol,
                price=new_price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                timestamp=current_time
            )
            
            updates.append(update)
            self.current_prices[symbol] = new_price
        
        # Add small delay to simulate network latency
        await asyncio.sleep(0.1)
        
        return updates
    
    def is_healthy(self) -> bool:
        """Simulated source is always healthy."""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulated data source statistics."""
        return {
            "provider": "Simulated",
            "healthy": True,
            "symbols_tracked": len(self.current_prices),
            "base_volatility": self._volatility,
            "drift": self._drift
        }


# Factory function to create appropriate data source
def create_market_data_source(use_simulation: bool = False) -> Any:
    """
    Create a market data source.
    
    Args:
        use_simulation: Whether to use simulated data
        
    Returns:
        Market data source instance
    """
    if use_simulation:
        # Provide some realistic base prices for common symbols
        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "MSFT": 300.0,
            "AMZN": 3200.0,
            "TSLA": 800.0,
            "SPY": 400.0,
            "QQQ": 350.0,
            "IVV": 450.0
        }
        return SimulatedMarketDataSource(base_prices)
    else:
        return MarketDataSource()