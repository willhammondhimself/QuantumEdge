"""
Yahoo Finance data provider implementation.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import logging

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
    pd = None

from .base import CachedDataProvider
from .models import (
    Asset, Price, MarketData, DataFrequency, MarketMetrics, AssetType,
    DataSourceError, DataValidationError
)

logger = logging.getLogger(__name__)


class YahooFinanceProvider(CachedDataProvider):
    """Yahoo Finance data provider."""
    
    # Yahoo Finance frequency mapping
    YF_FREQUENCY_MAP = {
        DataFrequency.MINUTE: "1m",
        DataFrequency.FIVE_MINUTE: "5m",
        DataFrequency.FIFTEEN_MINUTE: "15m",
        DataFrequency.THIRTY_MINUTE: "30m",
        DataFrequency.HOUR: "1h",
        DataFrequency.DAILY: "1d",
        DataFrequency.WEEKLY: "1wk",
        DataFrequency.MONTHLY: "1mo"
    }
    
    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize Yahoo Finance provider.
        
        Args:
            cache_ttl_seconds: Cache TTL in seconds
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required for YahooFinanceProvider. Install with: pip install yfinance")
        
        super().__init__("Yahoo Finance", cache_ttl_seconds, rate_limit_per_minute=120)
    
    async def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """Get asset information from Yahoo Finance."""
        await self._check_rate_limit()
        
        symbol = self.validate_symbol(symbol)
        cache_key = self._get_cache_key("asset_info", symbol)
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Run yfinance call in thread pool to avoid blocking
            ticker = yf.Ticker(symbol)
            info = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ticker.info
            )
            
            if not info or 'symbol' not in info:
                logger.warning(f"No info found for symbol {symbol}")
                return None
            
            # Determine asset type
            asset_type = self._determine_asset_type(info)
            
            asset = Asset(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                asset_type=asset_type,
                exchange=info.get('exchange'),
                currency=info.get('currency', 'USD'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                description=info.get('longBusinessSummary'),
                market_cap=info.get('marketCap')
            )
            
            self._set_cache(cache_key, asset)
            return asset
            
        except Exception as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            raise DataSourceError(f"Failed to fetch asset info: {e}")
    
    async def get_current_price(self, symbol: str) -> Optional[Price]:
        """Get current price from Yahoo Finance."""
        await self._check_rate_limit()
        
        symbol = self.validate_symbol(symbol)
        cache_key = self._get_cache_key("current_price", symbol)
        
        # Check cache first (short TTL for prices)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current data
            hist = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ticker.history(period="1d", interval="1m")
            )
            
            if hist.empty:
                logger.warning(f"No price data found for symbol {symbol}")
                return None
            
            # Get most recent price
            latest = hist.iloc[-1]
            
            price = Price(
                symbol=symbol,
                timestamp=datetime.now(),
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                volume=int(latest['Volume']),
                adjusted_close=float(latest['Close'])  # For intraday, close = adjusted
            )
            
            # Cache with shorter TTL for current prices
            self._cache[cache_key] = (datetime.now(), price)
            return price
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise DataSourceError(f"Failed to fetch current price: {e}")
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[Price]]:
        """Get current prices for multiple symbols."""
        if not symbols:
            return {}
        
        # For now, fetch individually (can be optimized later)
        results = {}
        for symbol in symbols:
            try:
                price = await self.get_current_price(symbol)
                results[symbol] = price
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> Optional[MarketData]:
        """Get historical data from Yahoo Finance."""
        await self._check_rate_limit()
        
        symbol = self.validate_symbol(symbol)
        self.validate_date_range(start_date, end_date)
        
        cache_key = self._get_cache_key("historical", symbol, start_date, end_date, frequency)
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            yf_freq = self.YF_FREQUENCY_MAP.get(frequency, "1d")
            
            # Get historical data
            hist = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval=yf_freq,
                    auto_adjust=True,
                    prepost=False
                )
            )
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Convert to Price objects
            prices = []
            for timestamp, row in hist.iterrows():
                try:
                    price = Price(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']),
                        adjusted_close=float(row['Close'])  # Yahoo auto-adjusts
                    )
                    prices.append(price)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid price data for {symbol} at {timestamp}: {e}")
                    continue
            
            if not prices:
                return None
            
            market_data = MarketData(
                symbol=symbol,
                data=prices,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                source="Yahoo Finance"
            )
            
            self._set_cache(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise DataSourceError(f"Failed to fetch historical data: {e}")
    
    async def get_market_metrics(self) -> Optional[MarketMetrics]:
        """Get market-wide metrics."""
        await self._check_rate_limit()
        
        cache_key = self._get_cache_key("market_metrics")
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Define symbols for market metrics
            symbols = {
                'spy': '^GSPC',    # S&P 500
                'vix': '^VIX',     # Volatility Index
                'dxy': 'DX-Y.NYB', # US Dollar Index
                'gold': 'GC=F',    # Gold futures
                'oil': 'CL=F',     # Crude oil futures
                'tnx': '^TNX'      # 10-year treasury yield
            }
            
            metrics_data = {}
            
            # Fetch current prices for all symbols
            for key, symbol in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = await asyncio.get_event_loop().run_in_executor(
                        None, lambda t=ticker: t.history(period="1d")
                    )
                    
                    if not hist.empty:
                        latest_close = float(hist['Close'].iloc[-1])
                        metrics_data[key] = latest_close
                
                except Exception as e:
                    logger.warning(f"Failed to get data for {key} ({symbol}): {e}")
                    metrics_data[key] = None
            
            # Calculate S&P 500 return (1-day)
            spy_return = None
            if 'spy' in metrics_data and metrics_data['spy']:
                try:
                    spy_ticker = yf.Ticker('^GSPC')
                    spy_hist = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: spy_ticker.history(period="2d")
                    )
                    if len(spy_hist) >= 2:
                        today_close = float(spy_hist['Close'].iloc[-1])
                        yesterday_close = float(spy_hist['Close'].iloc[-2])
                        spy_return = (today_close - yesterday_close) / yesterday_close
                except Exception as e:
                    logger.warning(f"Failed to calculate SPY return: {e}")
            
            metrics = MarketMetrics(
                timestamp=datetime.now(),
                vix=metrics_data.get('vix'),
                spy_return=spy_return,
                bond_yield_10y=metrics_data.get('tnx'),
                dxy=metrics_data.get('dxy'),
                gold_price=metrics_data.get('gold'),
                oil_price=metrics_data.get('oil')
            )
            
            self._set_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get market metrics: {e}")
            raise DataSourceError(f"Failed to fetch market metrics: {e}")
    
    def _determine_asset_type(self, info: Dict[str, Any]) -> AssetType:
        """Determine asset type from Yahoo Finance info."""
        quote_type = info.get('quoteType', '').lower()
        
        if quote_type in ['equity', 'stock']:
            return AssetType.STOCK
        elif quote_type in ['etf', 'fund']:
            return AssetType.ETF
        elif quote_type in ['bond', 'fixed_income']:
            return AssetType.BOND
        elif quote_type in ['cryptocurrency', 'crypto']:
            return AssetType.CRYPTO
        elif quote_type in ['commodity', 'future']:
            return AssetType.COMMODITY
        elif quote_type in ['currency', 'forex']:
            return AssetType.CURRENCY
        else:
            # Default to stock if uncertain
            return AssetType.STOCK


# Convenience function for quick access
async def get_yahoo_price(symbol: str) -> Optional[Price]:
    """Quick function to get current price from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not available")
    
    provider = YahooFinanceProvider()
    return await provider.get_current_price(symbol)


async def get_yahoo_historical(
    symbol: str,
    start_date: date,
    end_date: date,
    frequency: DataFrequency = DataFrequency.DAILY
) -> Optional[MarketData]:
    """Quick function to get historical data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not available")
    
    provider = YahooFinanceProvider()
    return await provider.get_historical_data(symbol, start_date, end_date, frequency)