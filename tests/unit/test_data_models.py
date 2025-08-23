"""Tests for data models."""

import pytest
from datetime import datetime, date
import numpy as np
import pandas as pd

from src.data.models import (
    Asset,
    AssetType,
    Price,
    MarketData,
    PortfolioSnapshot,
    MarketMetrics,
    DataFrequency,
    DataValidationError,
)


class TestAsset:
    """Test Asset dataclass."""

    def test_asset_creation(self):
        """Test creating asset."""
        asset = Asset(
            symbol="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            exchange="NASDAQ",
            currency="USD",
            sector="Technology",
            industry="Consumer Electronics",
            description="Apple Inc. designs and manufactures smartphones...",
            market_cap=3000000000000,
        )

        assert asset.symbol == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == AssetType.STOCK
        assert asset.exchange == "NASDAQ"
        assert asset.currency == "USD"
        assert asset.sector == "Technology"
        assert asset.industry == "Consumer Electronics"
        assert asset.market_cap == 3000000000000

    def test_asset_symbol_normalization(self):
        """Test symbol normalization."""
        asset = Asset(symbol="  aapl  ", name="Apple Inc.", asset_type=AssetType.STOCK)

        # Symbol should be uppercase and stripped
        assert asset.symbol == "AAPL"

    def test_asset_empty_symbol(self):
        """Test empty symbol validation."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            Asset(symbol="", name="Invalid Asset", asset_type=AssetType.STOCK)

    def test_asset_types(self):
        """Test different asset types."""
        # Stock
        stock = Asset(symbol="AAPL", name="Apple", asset_type=AssetType.STOCK)
        assert stock.asset_type == AssetType.STOCK

        # ETF
        etf = Asset(symbol="SPY", name="SPDR S&P 500", asset_type=AssetType.ETF)
        assert etf.asset_type == AssetType.ETF

        # Bond
        bond = Asset(symbol="TLT", name="Treasury Bond", asset_type=AssetType.BOND)
        assert bond.asset_type == AssetType.BOND

        # Crypto
        crypto = Asset(symbol="BTC", name="Bitcoin", asset_type=AssetType.CRYPTO)
        assert crypto.asset_type == AssetType.CRYPTO


class TestPrice:
    """Test Price dataclass."""

    def test_price_creation(self):
        """Test creating price."""
        timestamp = datetime.now()
        price = Price(
            symbol="AAPL",
            timestamp=timestamp,
            open=175.0,
            high=178.0,
            low=174.0,
            close=177.0,
            volume=50000000,
            adjusted_close=177.0,
        )

        assert price.symbol == "AAPL"
        assert price.timestamp == timestamp
        assert price.open == 175.0
        assert price.high == 178.0
        assert price.low == 174.0
        assert price.close == 177.0
        assert price.volume == 50000000
        assert price.adjusted_close == 177.0

    def test_price_validation_high(self):
        """Test high price validation."""
        with pytest.raises(ValueError, match="High price must be >= max"):
            Price(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=180.0,
                high=175.0,  # Less than open
                low=170.0,
                close=177.0,
                volume=50000000,
            )

    def test_price_validation_low(self):
        """Test low price validation."""
        with pytest.raises(ValueError, match="Low price must be <= min"):
            Price(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=175.0,
                high=180.0,
                low=178.0,  # Greater than close
                close=177.0,
                volume=50000000,
            )

    def test_price_validation_volume(self):
        """Test volume validation."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Price(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=175.0,
                high=178.0,
                low=174.0,
                close=177.0,
                volume=-1000,
            )

    def test_price_ohlc_property(self):
        """Test OHLC property."""
        price = Price(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=175.0,
            high=178.0,
            low=174.0,
            close=177.0,
            volume=50000000,
        )

        ohlc = price.ohlc
        assert ohlc == (175.0, 178.0, 174.0, 177.0)

    def test_price_typical_price(self):
        """Test typical price calculation."""
        price = Price(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=175.0,
            high=180.0,
            low=170.0,
            close=175.0,
            volume=50000000,
        )

        # Typical price = (H + L + C) / 3
        expected = (180.0 + 170.0 + 175.0) / 3.0
        assert price.typical_price == expected

    def test_price_weighted_price(self):
        """Test weighted price calculation."""
        price = Price(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=175.0,
            high=180.0,
            low=170.0,
            close=177.0,
            volume=50000000,
        )

        # Weighted price = (O + H + L + C) / 4
        expected = (175.0 + 180.0 + 170.0 + 177.0) / 4.0
        assert price.weighted_price == expected


class TestMarketData:
    """Test MarketData dataclass."""

    def test_market_data_creation(self):
        """Test creating market data."""
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, i, 9, 30),
                open=175.0 + i,
                high=178.0 + i,
                low=174.0 + i,
                close=177.0 + i,
                volume=50000000 + i * 1000000,
            )
            for i in range(1, 6)
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            source="yahoo_finance",
        )

        assert market_data.symbol == "AAPL"
        assert len(market_data.data) == 5
        assert market_data.frequency == DataFrequency.DAILY
        assert market_data.source == "yahoo_finance"

    def test_market_data_empty_validation(self):
        """Test empty data validation."""
        with pytest.raises(ValueError, match="Market data cannot be empty"):
            MarketData(
                symbol="AAPL",
                data=[],
                frequency=DataFrequency.DAILY,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 5),
                source="yahoo_finance",
            )

    def test_market_data_sorting(self):
        """Test data sorting by timestamp."""
        # Create unsorted prices
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 3, 9, 30),
                open=177.0,
                high=180.0,
                low=176.0,
                close=179.0,
                volume=52000000,
            ),
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1, 9, 30),
                open=175.0,
                high=178.0,
                low=174.0,
                close=177.0,
                volume=50000000,
            ),
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 2, 9, 30),
                open=176.0,
                high=179.0,
                low=175.0,
                close=178.0,
                volume=51000000,
            ),
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            source="yahoo_finance",
        )

        # Should be sorted by timestamp
        timestamps = [p.timestamp for p in market_data.data]
        assert timestamps[0] < timestamps[1] < timestamps[2]

    def test_market_data_to_dataframe(self):
        """Test conversion to DataFrame."""
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, i, 9, 30),
                open=175.0 + i,
                high=178.0 + i,
                low=174.0 + i,
                close=177.0 + i,
                volume=50000000 + i * 1000000,
            )
            for i in range(1, 4)
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            source="yahoo_finance",
        )

        df = market_data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.name == "timestamp"

    def test_market_data_get_returns(self):
        """Test returns calculation."""
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, i, 9, 30),
                open=100.0 * (1.01**i),
                high=100.0 * (1.01**i) * 1.02,  # 2% above close
                low=100.0 * (1.01**i) * 0.98,  # 2% below close
                close=100.0 * (1.01**i),  # 1% daily return
                volume=50000000,
            )
            for i in range(1, 6)
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            source="yahoo_finance",
        )

        returns = market_data.get_returns()

        assert isinstance(returns, np.ndarray)
        assert len(returns) == 4  # One less than prices
        # Each return should be approximately ln(1.01)
        assert np.allclose(returns, np.log(1.01), rtol=1e-10)

    def test_market_data_get_volatility(self):
        """Test volatility calculation."""
        # Create prices with known volatility
        np.random.seed(42)
        prices = []
        price = 100.0

        for i in range(252):  # One year of daily data
            # Daily return with 1% std dev
            daily_return = 1 + np.random.normal(0, 0.01)
            price *= daily_return

            prices.append(
                Price(
                    symbol="AAPL",
                    timestamp=datetime(2024, 1, 1) + pd.Timedelta(days=i),
                    open=price * 0.99,
                    high=price * 1.01,
                    low=price * 0.98,
                    close=price,
                    volume=50000000,
                )
            )

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            source="test",
        )

        # Annualized volatility
        vol_annual = market_data.get_volatility(annualize=True)
        # Daily volatility
        vol_daily = market_data.get_volatility(annualize=False)

        # Annualized should be approximately daily * sqrt(252)
        assert vol_annual > vol_daily
        assert vol_annual / vol_daily > 10  # sqrt(252) ≈ 15.87

    def test_market_data_get_price_at_date(self):
        """Test getting price at specific date."""
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, i, 9, 30),
                open=175.0 + i,
                high=178.0 + i,
                low=174.0 + i,
                close=177.0 + i,
                volume=50000000,
            )
            for i in range(1, 6)
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            source="yahoo_finance",
        )

        # Get exact match
        target_date = datetime(2024, 1, 3, 9, 30)
        price = market_data.get_price_at_date(target_date)
        assert price is not None
        assert price.close == 180.0

        # Get closest match (within tolerance)
        target_date = datetime(2024, 1, 3, 12, 0)  # Different time
        price = market_data.get_price_at_date(target_date)
        assert price is not None
        assert price.close == 180.0

        # Too far away
        target_date = datetime(2024, 2, 1, 9, 30)
        price = market_data.get_price_at_date(target_date)
        assert price is None

    def test_market_data_get_price_range(self):
        """Test getting price range."""
        prices = [
            Price(
                symbol="AAPL",
                timestamp=datetime(2024, 1, i, 9, 30),
                open=175.0 + i,
                high=178.0 + i,
                low=174.0 + i,
                close=177.0 + i,
                volume=50000000,
            )
            for i in range(1, 11)
        ]

        market_data = MarketData(
            symbol="AAPL",
            data=prices,
            frequency=DataFrequency.DAILY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            source="yahoo_finance",
        )

        # Get subset
        start = datetime(2024, 1, 3, 0, 0)
        end = datetime(2024, 1, 7, 23, 59)

        subset = market_data.get_price_range(start, end)

        assert len(subset) == 5
        assert subset[0].close == 180.0  # Jan 3
        assert subset[-1].close == 184.0  # Jan 7


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot dataclass."""

    def test_portfolio_snapshot_creation(self):
        """Test creating portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            symbols=["AAPL", "GOOGL", "MSFT"],
            weights=[0.4, 0.3, 0.3],
            values=[40000.0, 30000.0, 30000.0],
            total_value=100000.0,
        )

        assert len(snapshot.symbols) == 3
        assert sum(snapshot.weights) == 1.0
        assert sum(snapshot.values) == snapshot.total_value

    def test_portfolio_snapshot_validation_length(self):
        """Test length validation."""
        with pytest.raises(ValueError, match="same length"):
            PortfolioSnapshot(
                timestamp=datetime.now(),
                symbols=["AAPL", "GOOGL"],
                weights=[0.5, 0.3, 0.2],  # Wrong length
                values=[50000.0, 50000.0],
                total_value=100000.0,
            )

    def test_portfolio_snapshot_validation_weights(self):
        """Test weights sum validation."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            PortfolioSnapshot(
                timestamp=datetime.now(),
                symbols=["AAPL", "GOOGL"],
                weights=[0.6, 0.3],  # Sum to 0.9
                values=[60000.0, 40000.0],
                total_value=100000.0,
            )

    def test_portfolio_snapshot_validation_values(self):
        """Test values sum validation."""
        with pytest.raises(ValueError, match="Sum of values must equal total value"):
            PortfolioSnapshot(
                timestamp=datetime.now(),
                symbols=["AAPL", "GOOGL"],
                weights=[0.5, 0.5],
                values=[50000.0, 40000.0],  # Sum to 90000
                total_value=100000.0,
            )


class TestMarketMetrics:
    """Test MarketMetrics dataclass."""

    def test_market_metrics_creation(self):
        """Test creating market metrics."""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            vix=15.5,
            spy_return=0.012,
            bond_yield_10y=0.045,
            dxy=98.5,
            gold_price=1950.0,
            oil_price=75.50,
        )

        assert metrics.vix == 15.5
        assert metrics.spy_return == 0.012
        assert metrics.bond_yield_10y == 0.045
        assert metrics.dxy == 98.5
        assert metrics.gold_price == 1950.0
        assert metrics.oil_price == 75.50

    def test_market_metrics_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        metrics = MarketMetrics(timestamp=timestamp, vix=15.5, spy_return=0.012)

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert data["timestamp"] == timestamp
        assert data["vix"] == 15.5
        assert data["spy_return"] == 0.012
        assert data["bond_yield_10y"] is None
