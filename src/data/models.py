"""
Data models for market data and financial instruments.
"""

import numpy as np
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class AssetType(str, Enum):
    """Asset type classification."""

    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    CURRENCY = "currency"


class DataFrequency(str, Enum):
    """Data frequency options."""

    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


@dataclass
class Asset:
    """Financial asset information."""

    symbol: str
    name: str
    asset_type: AssetType
    exchange: Optional[str] = None
    currency: str = "USD"
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    market_cap: Optional[float] = None

    def __post_init__(self):
        """Validate asset data."""
        self.symbol = self.symbol.upper().strip()
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")


@dataclass
class Price:
    """Price data point."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    def __post_init__(self):
        """Validate price data."""
        if self.high < max(self.open, self.close):
            raise ValueError("High price must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low price must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

    @property
    def ohlc(self) -> tuple:
        """Get OHLC tuple."""
        return (self.open, self.high, self.low, self.close)

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def weighted_price(self) -> float:
        """Calculate weighted price (OHLC/4)."""
        return (self.open + self.high + self.low + self.close) / 4.0


@dataclass
class MarketData:
    """Historical market data for an asset."""

    symbol: str
    data: List[Price]
    frequency: DataFrequency
    start_date: date
    end_date: date
    source: str

    def __post_init__(self):
        """Validate market data."""
        if not self.data:
            raise ValueError("Market data cannot be empty")

        # Sort data by timestamp
        self.data.sort(key=lambda p: p.timestamp)

        # Validate date range
        actual_start = self.data[0].timestamp.date()
        actual_end = self.data[-1].timestamp.date()

        if actual_start > self.start_date:
            self.start_date = actual_start
        if actual_end < self.end_date:
            self.end_date = actual_end

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = []
        for price in self.data:
            records.append(
                {
                    "timestamp": price.timestamp,
                    "open": price.open,
                    "high": price.high,
                    "low": price.low,
                    "close": price.close,
                    "volume": price.volume,
                    "adjusted_close": price.adjusted_close,
                }
            )

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df

    def get_returns(self, price_column: str = "close") -> np.ndarray:
        """Calculate returns."""
        df = self.to_dataframe()
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")

        prices = df[price_column].values
        returns = np.diff(np.log(prices))
        return returns

    def get_volatility(
        self, price_column: str = "close", annualize: bool = True
    ) -> float:
        """Calculate volatility."""
        returns = self.get_returns(price_column)
        vol = np.std(returns)

        if annualize:
            # Annualization factors
            factors = {
                DataFrequency.MINUTE: np.sqrt(252 * 24 * 60),
                DataFrequency.FIVE_MINUTE: np.sqrt(252 * 24 * 12),
                DataFrequency.FIFTEEN_MINUTE: np.sqrt(252 * 24 * 4),
                DataFrequency.THIRTY_MINUTE: np.sqrt(252 * 24 * 2),
                DataFrequency.HOUR: np.sqrt(252 * 24),
                DataFrequency.FOUR_HOUR: np.sqrt(252 * 6),
                DataFrequency.DAILY: np.sqrt(252),
                DataFrequency.WEEKLY: np.sqrt(52),
                DataFrequency.MONTHLY: np.sqrt(12),
            }
            vol *= factors.get(self.frequency, 1.0)

        return vol

    def get_price_at_date(self, target_date: datetime) -> Optional[Price]:
        """Get price closest to target date."""
        if not self.data:
            return None

        closest_price = min(
            self.data, key=lambda p: abs((p.timestamp - target_date).total_seconds())
        )

        # Only return if within reasonable time window (e.g., 1 day for daily data)
        time_diff = abs((closest_price.timestamp - target_date).total_seconds())
        max_diff = 24 * 3600  # 1 day in seconds

        if time_diff <= max_diff:
            return closest_price
        return None

    def get_price_range(self, start: datetime, end: datetime) -> List[Price]:
        """Get prices within date range."""
        return [p for p in self.data if start <= p.timestamp <= end]


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time."""

    timestamp: datetime
    symbols: List[str]
    weights: List[float]
    values: List[float]
    total_value: float

    def __post_init__(self):
        """Validate portfolio snapshot."""
        if len(self.symbols) != len(self.weights) != len(self.values):
            raise ValueError("Symbols, weights, and values must have same length")

        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        if abs(sum(self.values) - self.total_value) > 1e-6:
            raise ValueError("Sum of values must equal total value")


@dataclass
class MarketMetrics:
    """Market-wide metrics and indicators."""

    timestamp: datetime
    vix: Optional[float] = None  # Volatility index
    spy_return: Optional[float] = None  # S&P 500 return
    bond_yield_10y: Optional[float] = None  # 10-year treasury yield
    dxy: Optional[float] = None  # US Dollar index
    gold_price: Optional[float] = None  # Gold price
    oil_price: Optional[float] = None  # Oil price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "vix": self.vix,
            "spy_return": self.spy_return,
            "bond_yield_10y": self.bond_yield_10y,
            "dxy": self.dxy,
            "gold_price": self.gold_price,
            "oil_price": self.oil_price,
        }


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


class DataSourceError(Exception):
    """Exception raised for data source errors."""

    pass


class RateLimitError(Exception):
    """Exception raised when API rate limits are exceeded."""

    pass
