"""
Main backtesting engine for portfolio strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging

from .portfolio import Portfolio, PortfolioState
from .strategy import PortfolioStrategy, BuyAndHoldStrategy
from .metrics import PerformanceMetrics, RiskMetrics, calculate_all_metrics
from ..data.models import MarketData, Price
from ..data.yahoo_finance import YahooFinanceProvider

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Strategy and data
    strategy: PortfolioStrategy
    symbols: List[str]
    start_date: date
    end_date: date

    # Portfolio settings
    initial_cash: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    min_commission: float = 1.0

    # Rebalancing settings
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    min_weight: float = 0.0
    max_weight: float = 1.0

    # Benchmark
    benchmark_symbol: Optional[str] = "SPY"
    risk_free_rate: float = 0.02

    # Advanced settings
    lookback_period: int = 252  # For mean estimation
    allow_short_selling: bool = False
    transaction_cost_model: str = "fixed"  # fixed, market_impact


@dataclass
class BacktestResult:
    """Results from backtesting."""

    # Configuration
    config: BacktestConfig

    # Portfolio data
    portfolio_values: pd.Series
    portfolio_weights: pd.DataFrame
    benchmark_values: Optional[pd.Series] = None

    # Performance metrics
    performance_metrics: Optional[PerformanceMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None

    # Additional data
    transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_dates: List[datetime] = field(default_factory=list)

    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, data_provider: Optional[YahooFinanceProvider] = None):
        """
        Initialize backtesting engine.

        Args:
            data_provider: Data provider for market data
        """
        self.data_provider = data_provider or YahooFinanceProvider()
        self.cache = {}  # Simple caching for price data

    async def get_price_data(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Get historical price data for symbols.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with prices (symbols as columns, dates as index)
        """
        cache_key = (tuple(symbols), start_date, end_date)

        if cache_key in self.cache:
            return self.cache[cache_key]

        price_data = {}

        for symbol in symbols:
            try:
                market_data = await self.data_provider.get_historical_data(
                    symbol, start_date, end_date
                )

                if market_data and market_data.data:
                    prices = []
                    dates = []

                    for price in market_data.data:
                        dates.append(price.timestamp.date())
                        prices.append(price.close)

                    price_series = pd.Series(prices, index=pd.to_datetime(dates))
                    price_data[symbol] = price_series

            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                continue

        if not price_data:
            raise ValueError("No price data available for any symbols")

        # Combine into DataFrame and forward-fill missing values
        df = pd.DataFrame(price_data)
        df = df.fillna(method="ffill").dropna()

        self.cache[cache_key] = df
        return df

    def get_rebalance_dates(
        self, start_date: datetime, end_date: datetime, frequency: str
    ) -> List[datetime]:
        """
        Generate rebalancing dates based on frequency.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Rebalancing frequency

        Returns:
            List of rebalancing dates
        """
        dates = []
        current = start_date

        if frequency == "daily":
            while current <= end_date:
                dates.append(current)
                current += timedelta(days=1)

        elif frequency == "weekly":
            # Rebalance on Mondays
            while current <= end_date:
                if current.weekday() == 0:  # Monday
                    dates.append(current)
                current += timedelta(days=1)

        elif frequency == "monthly":
            # Rebalance on first trading day of month
            while current <= end_date:
                if (
                    current.day == 1
                    or (current - timedelta(days=1)).month != current.month
                ):
                    dates.append(current)
                current += timedelta(days=1)

        elif frequency == "quarterly":
            # Rebalance on first trading day of quarter
            while current <= end_date:
                if current.month in [1, 4, 7, 10] and (
                    current.day == 1
                    or (current - timedelta(days=1)).month != current.month
                ):
                    dates.append(current)
                current += timedelta(days=1)

        else:
            # Default to monthly
            dates = self.get_rebalance_dates(start_date, end_date, "monthly")

        return dates

    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run backtesting simulation.

        Args:
            config: Backtest configuration

        Returns:
            Backtest results
        """
        start_time = datetime.now()

        try:
            # Get price data
            logger.info(f"Getting price data for {len(config.symbols)} symbols")
            prices = await self.get_price_data(
                config.symbols, config.start_date, config.end_date
            )

            if prices.empty:
                return BacktestResult(
                    config=config,
                    portfolio_values=pd.Series(),
                    portfolio_weights=pd.DataFrame(),
                    success=False,
                    error_message="No price data available",
                )

            # Get benchmark data if specified
            benchmark_values = None
            if config.benchmark_symbol:
                try:
                    benchmark_data = await self.get_price_data(
                        [config.benchmark_symbol], config.start_date, config.end_date
                    )
                    if not benchmark_data.empty:
                        benchmark_values = benchmark_data[config.benchmark_symbol]
                except Exception as e:
                    logger.warning(f"Could not get benchmark data: {e}")

            # Initialize portfolio
            portfolio = Portfolio(
                initial_cash=config.initial_cash,
                commission_rate=config.commission_rate,
                min_commission=config.min_commission,
                symbols=config.symbols,
            )

            # Initialize strategy
            initial_weights = config.strategy.initialize(prices)

            # Track results
            portfolio_values = []
            portfolio_weights = []
            rebalance_dates = []

            # Get trading dates
            trading_dates = prices.index
            last_rebalance = None

            logger.info(
                f"Running backtest from {config.start_date} to {config.end_date}"
            )

            for i, current_date in enumerate(trading_dates):
                current_datetime = pd.to_datetime(current_date)

                # Get current prices
                current_prices = prices.loc[current_date].to_dict()

                # Check if we should rebalance
                should_rebalance = False

                if last_rebalance is None:
                    # First day - always rebalance
                    should_rebalance = True
                elif hasattr(config.strategy, "should_rebalance"):
                    should_rebalance = config.strategy.should_rebalance(
                        current_datetime, last_rebalance
                    )
                else:
                    # Default rebalancing logic based on frequency
                    if config.rebalance_frequency == "daily":
                        should_rebalance = True
                    elif config.rebalance_frequency == "weekly":
                        days_since = (current_datetime - last_rebalance).days
                        should_rebalance = days_since >= 7
                    elif config.rebalance_frequency == "monthly":
                        should_rebalance = (
                            current_datetime.month != last_rebalance.month
                        )
                    elif config.rebalance_frequency == "quarterly":
                        current_quarter = (current_datetime.month - 1) // 3
                        last_quarter = (last_rebalance.month - 1) // 3
                        should_rebalance = (
                            current_quarter != last_quarter
                            or current_datetime.year != last_rebalance.year
                        )

                # Rebalance if needed
                if should_rebalance:
                    # Get current weights
                    current_weights = portfolio.get_weights(current_prices)

                    # Get target weights from strategy
                    if last_rebalance is None:
                        target_weights = initial_weights
                    else:
                        # Get historical data up to current date
                        historical_data = prices.iloc[: i + 1]
                        target_weights = config.strategy.rebalance(
                            current_datetime, historical_data, current_weights
                        )

                    # Apply weight constraints
                    target_weights = self.apply_weight_constraints(
                        target_weights,
                        config.min_weight,
                        config.max_weight,
                        config.allow_short_selling,
                    )

                    # Execute rebalancing
                    transactions = portfolio.rebalance_to_weights(
                        current_datetime, target_weights, current_prices
                    )

                    if transactions:
                        rebalance_dates.append(current_datetime)
                        last_rebalance = current_datetime

                # Update portfolio state
                state = portfolio.update_state(current_datetime, current_prices)

                # Record portfolio value and weights
                portfolio_values.append(state.total_value)

                weight_row = {
                    symbol: state.weights.get(symbol, 0.0) for symbol in config.symbols
                }
                weight_row["date"] = current_date
                portfolio_weights.append(weight_row)

            # Convert results to pandas objects
            portfolio_values_series = pd.Series(
                portfolio_values, index=trading_dates, name="Portfolio Value"
            )

            portfolio_weights_df = pd.DataFrame(portfolio_weights)
            portfolio_weights_df.set_index("date", inplace=True)

            # Calculate performance metrics
            performance_metrics, risk_metrics = calculate_all_metrics(
                portfolio_values_series, benchmark_values, config.risk_free_rate
            )

            # Get transaction history
            transaction_history = portfolio.get_transaction_history()

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Backtest completed in {execution_time:.2f} seconds")

            return BacktestResult(
                config=config,
                portfolio_values=portfolio_values_series,
                portfolio_weights=portfolio_weights_df,
                benchmark_values=benchmark_values,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                transactions=transaction_history,
                rebalance_dates=rebalance_dates,
                success=True,
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return BacktestResult(
                config=config,
                portfolio_values=pd.Series(),
                portfolio_weights=pd.DataFrame(),
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )

    def apply_weight_constraints(
        self,
        weights: Dict[str, float],
        min_weight: float,
        max_weight: float,
        allow_short: bool,
    ) -> Dict[str, float]:
        """
        Apply weight constraints to portfolio weights.

        Args:
            weights: Target weights
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            allow_short: Whether to allow short selling

        Returns:
            Constrained weights
        """
        constrained_weights = {}

        for symbol, weight in weights.items():
            # Apply bounds
            if not allow_short:
                weight = max(0.0, weight)

            weight = max(min_weight, min(max_weight, weight))
            constrained_weights[symbol] = weight

        # Normalize to sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for symbol in constrained_weights:
                constrained_weights[symbol] /= total_weight

        return constrained_weights

    def compare_strategies(
        self,
        strategies: List[PortfolioStrategy],
        symbols: List[str],
        start_date: date,
        end_date: date,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies using the same data.

        Args:
            strategies: List of strategies to compare
            symbols: List of symbols to trade
            start_date: Start date
            end_date: End date
            **kwargs: Additional backtest configuration parameters

        Returns:
            Dictionary of strategy name -> backtest result
        """
        results = {}

        for strategy in strategies:
            config = BacktestConfig(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )

            # Note: This is a synchronous wrapper - in practice you'd want async
            import asyncio

            result = asyncio.run(self.run_backtest(config))
            results[strategy.name] = result

        return results

    def generate_report(
        self, results: Union[BacktestResult, Dict[str, BacktestResult]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive backtest report.

        Args:
            results: Single result or dictionary of results

        Returns:
            Report dictionary
        """
        if isinstance(results, BacktestResult):
            results = {"Strategy": results}

        report = {
            "summary": {},
            "performance_comparison": {},
            "risk_analysis": {},
            "transaction_costs": {},
        }

        # Performance comparison
        performance_data = []
        for name, result in results.items():
            if result.success and result.performance_metrics:
                metrics = result.performance_metrics
                performance_data.append(
                    {
                        "Strategy": name,
                        "Total Return": f"{metrics.total_return:.2%}",
                        "CAGR": f"{metrics.cagr:.2%}",
                        "Volatility": f"{metrics.annualized_volatility:.2%}",
                        "Sharpe Ratio": f"{metrics.sharpe_ratio:.3f}",
                        "Max Drawdown": f"{metrics.max_drawdown:.2%}",
                        "Calmar Ratio": f"{metrics.calmar_ratio:.3f}",
                    }
                )

        report["performance_comparison"] = performance_data

        # Risk analysis
        for name, result in results.items():
            if result.success and result.risk_metrics:
                report["risk_analysis"][name] = {
                    "VaR_95": f"{result.performance_metrics.var_95:.2%}",
                    "CVaR_95": f"{result.performance_metrics.cvar_95:.2%}",
                    "Skewness": f"{result.performance_metrics.skewness:.3f}",
                    "Kurtosis": f"{result.performance_metrics.kurtosis:.3f}",
                }

        # Transaction costs
        for name, result in results.items():
            if result.success and not result.transactions.empty:
                total_commissions = result.transactions["commission"].sum()
                num_transactions = len(result.transactions)

                report["transaction_costs"][name] = {
                    "Total Commissions": f"${total_commissions:.2f}",
                    "Number of Transactions": num_transactions,
                    "Average Commission": (
                        f"${total_commissions/num_transactions:.2f}"
                        if num_transactions > 0
                        else "N/A"
                    ),
                }

        return report
