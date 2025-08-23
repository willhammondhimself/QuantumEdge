"""
Portfolio monitoring service for real-time portfolio tracking.

This module provides portfolio value calculations, risk monitoring,
and alert generation for real-time portfolio updates.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque

from .data_pipeline import DataPipeline, MarketDataUpdate, PortfolioUpdate, RiskUpdate

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Portfolio definition with holdings and metadata."""

    portfolio_id: str
    name: str
    holdings: Dict[str, float]  # symbol -> shares
    initial_value: float
    created_at: datetime
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02

    def get_symbols(self) -> List[str]:
        """Get all symbols in the portfolio."""
        return list(self.holdings.keys())

    def get_total_shares(self) -> float:
        """Get total number of shares across all holdings."""
        return sum(self.holdings.values())


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    portfolio_id: str
    rule_type: str  # "price_change", "portfolio_change", "risk_threshold"
    threshold: float
    condition: str  # "above", "below", "change_up", "change_down"
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 15  # Minimum time between alerts


class PortfolioMonitor:
    """
    Real-time portfolio monitoring service.

    Calculates portfolio values, risk metrics, and generates alerts
    based on market data updates.
    """

    def __init__(self, data_pipeline: DataPipeline):
        """
        Initialize portfolio monitor.

        Args:
            data_pipeline: Data pipeline for receiving market updates
        """
        self.data_pipeline = data_pipeline

        # Portfolio storage
        self.portfolios: Dict[str, Portfolio] = {}
        self.alert_rules: Dict[str, AlertRule] = {}

        # Historical data for calculations
        self.price_history: Dict[str, deque] = {}  # symbol -> price history
        self.portfolio_history: Dict[str, deque] = {}  # portfolio_id -> value history
        self.max_history_length = 252  # ~1 year of daily data

        # Risk calculation cache
        self._risk_calculation_cache: Dict[str, Tuple[datetime, RiskUpdate]] = {}
        self._risk_cache_duration = timedelta(minutes=5)  # Cache risk calculations

        # Register calculators with data pipeline
        self._register_calculators()

    def _register_calculators(self):
        """Register portfolio and risk calculators with the data pipeline."""
        # We'll dynamically register calculators when portfolios are added
        pass

    async def add_portfolio(self, portfolio: Portfolio):
        """
        Add a portfolio to monitor.

        Args:
            portfolio: Portfolio to monitor
        """
        self.portfolios[portfolio.portfolio_id] = portfolio

        # Initialize price history for portfolio symbols
        for symbol in portfolio.get_symbols():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.max_history_length)

        # Initialize portfolio history
        self.portfolio_history[portfolio.portfolio_id] = deque(
            maxlen=self.max_history_length
        )

        # Register calculators
        self.data_pipeline.register_portfolio_calculator(
            portfolio.portfolio_id, self._create_portfolio_calculator(portfolio)
        )

        self.data_pipeline.register_risk_calculator(
            portfolio.portfolio_id, self._create_risk_calculator(portfolio)
        )

        # Subscribe to market data for portfolio symbols
        await self.data_pipeline.subscribe_market_data(portfolio.get_symbols())
        await self.data_pipeline.subscribe_portfolio(portfolio.portfolio_id)
        await self.data_pipeline.subscribe_risk_metrics(portfolio.portfolio_id)

        logger.info(f"Added portfolio to monitor: {portfolio.portfolio_id}")

    async def remove_portfolio(self, portfolio_id: str):
        """
        Remove a portfolio from monitoring.

        Args:
            portfolio_id: Portfolio ID to remove
        """
        if portfolio_id in self.portfolios:
            del self.portfolios[portfolio_id]

            if portfolio_id in self.portfolio_history:
                del self.portfolio_history[portfolio_id]

            if portfolio_id in self._risk_calculation_cache:
                del self._risk_calculation_cache[portfolio_id]

            # Remove alert rules for this portfolio
            rules_to_remove = [
                rule_id
                for rule_id, rule in self.alert_rules.items()
                if rule.portfolio_id == portfolio_id
            ]
            for rule_id in rules_to_remove:
                del self.alert_rules[rule_id]

            logger.info(f"Removed portfolio from monitor: {portfolio_id}")

    def _create_portfolio_calculator(self, portfolio: Portfolio):
        """Create portfolio value calculator function."""

        async def calculate_portfolio_value(
            market_data: Dict[str, MarketDataUpdate]
        ) -> Optional[PortfolioUpdate]:
            """Calculate current portfolio value and metrics."""

            try:
                current_time = datetime.utcnow()
                total_value = 0.0
                positions = {}

                # Calculate current value for each holding
                for symbol, shares in portfolio.holdings.items():
                    if symbol in market_data:
                        current_price = market_data[symbol].price
                        position_value = shares * current_price
                        total_value += position_value

                        positions[symbol] = {
                            "shares": shares,
                            "price": current_price,
                            "value": position_value,
                            "weight": 0.0,  # Will calculate after total
                        }

                        # Update price history
                        if symbol in self.price_history:
                            self.price_history[symbol].append(current_price)

                # Calculate position weights
                if total_value > 0:
                    for symbol in positions:
                        positions[symbol]["weight"] = (
                            positions[symbol]["value"] / total_value
                        )

                # Calculate change from initial value
                change = total_value - portfolio.initial_value
                change_percent = (
                    change / portfolio.initial_value
                    if portfolio.initial_value != 0
                    else 0.0
                )

                # Update portfolio history
                self.portfolio_history[portfolio.portfolio_id].append(total_value)

                # Create update
                update = PortfolioUpdate(
                    portfolio_id=portfolio.portfolio_id,
                    total_value=total_value,
                    change=change,
                    change_percent=change_percent,
                    positions=positions,
                    timestamp=current_time,
                )

                # Check alert rules
                await self._check_portfolio_alerts(portfolio, update)

                return update

            except Exception as e:
                logger.error(
                    f"Error calculating portfolio value for {portfolio.portfolio_id}: {e}"
                )
                return None

        return calculate_portfolio_value

    def _create_risk_calculator(self, portfolio: Portfolio):
        """Create risk metrics calculator function."""

        async def calculate_risk_metrics(
            market_data: Dict[str, MarketDataUpdate],
            portfolio_update: Optional[PortfolioUpdate],
        ) -> Optional[RiskUpdate]:
            """Calculate risk metrics for the portfolio."""

            try:
                current_time = datetime.utcnow()
                portfolio_id = portfolio.portfolio_id

                # Check cache first
                if portfolio_id in self._risk_calculation_cache:
                    cached_time, cached_update = self._risk_calculation_cache[
                        portfolio_id
                    ]
                    if current_time - cached_time < self._risk_cache_duration:
                        return cached_update

                # Need sufficient history for risk calculations
                portfolio_values = list(self.portfolio_history[portfolio_id])
                if len(portfolio_values) < 30:  # Need at least 30 data points
                    return None

                # Calculate returns
                returns = np.diff(portfolio_values) / portfolio_values[:-1]

                if len(returns) == 0:
                    return None

                # Calculate risk metrics
                volatility = float(np.std(returns))
                annualized_volatility = volatility * np.sqrt(252)  # Assume daily data

                # VaR and CVaR (95% confidence)
                var_95 = float(np.percentile(returns, 5))
                cvar_95 = float(np.mean(returns[returns <= var_95]))

                # Beta calculation (if we have benchmark data)
                beta = 1.0  # Default beta
                # TODO: Calculate beta against benchmark

                # Sharpe ratio
                mean_return = float(np.mean(returns))
                excess_return = mean_return - (
                    portfolio.risk_free_rate / 252
                )  # Daily risk-free rate
                sharpe_ratio = excess_return / volatility if volatility != 0 else 0.0

                # Maximum drawdown
                portfolio_values_array = np.array(portfolio_values)
                running_max = np.maximum.accumulate(portfolio_values_array)
                drawdowns = (portfolio_values_array - running_max) / running_max
                max_drawdown = float(np.min(drawdowns))

                # Create risk update
                update = RiskUpdate(
                    portfolio_id=portfolio_id,
                    volatility=annualized_volatility,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    beta=beta,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    timestamp=current_time,
                )

                # Cache the result
                self._risk_calculation_cache[portfolio_id] = (current_time, update)

                # Check risk alert rules
                await self._check_risk_alerts(portfolio, update)

                return update

            except Exception as e:
                logger.error(
                    f"Error calculating risk metrics for {portfolio.portfolio_id}: {e}"
                )
                return None

        return calculate_risk_metrics

    async def add_alert_rule(self, rule: AlertRule):
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.rule_id] = rule
        logger.info(
            f"Added alert rule: {rule.rule_id} for portfolio {rule.portfolio_id}"
        )

    async def remove_alert_rule(self, rule_id: str):
        """
        Remove an alert rule.

        Args:
            rule_id: Rule ID to remove
        """
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    async def _check_portfolio_alerts(
        self, portfolio: Portfolio, update: PortfolioUpdate
    ):
        """Check portfolio-based alert rules."""

        current_time = datetime.utcnow()

        for rule in self.alert_rules.values():
            if (
                rule.portfolio_id != portfolio.portfolio_id
                or not rule.enabled
                or rule.rule_type not in ["portfolio_change"]
            ):
                continue

            # Check cooldown
            if rule.last_triggered and current_time - rule.last_triggered < timedelta(
                minutes=rule.cooldown_minutes
            ):
                continue

            should_trigger = False
            message = ""

            if rule.rule_type == "portfolio_change":
                if (
                    rule.condition == "change_up"
                    and update.change_percent >= rule.threshold
                ):
                    should_trigger = True
                    message = f"Portfolio {portfolio.name} increased by {update.change_percent:.2%}"
                elif (
                    rule.condition == "change_down"
                    and update.change_percent <= -rule.threshold
                ):
                    should_trigger = True
                    message = f"Portfolio {portfolio.name} decreased by {abs(update.change_percent):.2%}"
                elif rule.condition == "above" and update.total_value >= rule.threshold:
                    should_trigger = True
                    message = (
                        f"Portfolio {portfolio.name} value above ${rule.threshold:,.2f}"
                    )
                elif rule.condition == "below" and update.total_value <= rule.threshold:
                    should_trigger = True
                    message = (
                        f"Portfolio {portfolio.name} value below ${rule.threshold:,.2f}"
                    )

            if should_trigger:
                await self.data_pipeline.send_alert(
                    alert_type="portfolio_alert",
                    message=message,
                    severity="info" if abs(update.change_percent) < 0.05 else "warning",
                )
                rule.last_triggered = current_time

    async def _check_risk_alerts(self, portfolio: Portfolio, update: RiskUpdate):
        """Check risk-based alert rules."""

        current_time = datetime.utcnow()

        for rule in self.alert_rules.values():
            if (
                rule.portfolio_id != portfolio.portfolio_id
                or not rule.enabled
                or rule.rule_type != "risk_threshold"
            ):
                continue

            # Check cooldown
            if rule.last_triggered and current_time - rule.last_triggered < timedelta(
                minutes=rule.cooldown_minutes
            ):
                continue

            should_trigger = False
            message = ""
            severity = "warning"

            # Check various risk thresholds
            if (
                rule.condition == "volatility_above"
                and update.volatility >= rule.threshold
            ):
                should_trigger = True
                message = f"Portfolio {portfolio.name} volatility {update.volatility:.2%} exceeds threshold"
                severity = "warning"
            elif (
                rule.condition == "drawdown_below"
                and update.max_drawdown <= -rule.threshold
            ):
                should_trigger = True
                message = f"Portfolio {portfolio.name} drawdown {update.max_drawdown:.2%} exceeds threshold"
                severity = "error"
            elif rule.condition == "var_below" and update.var_95 <= -rule.threshold:
                should_trigger = True
                message = f"Portfolio {portfolio.name} VaR {update.var_95:.2%} exceeds threshold"
                severity = "warning"

            if should_trigger:
                await self.data_pipeline.send_alert(
                    alert_type="risk_alert", message=message, severity=severity
                )
                rule.last_triggered = current_time

    def get_portfolio_summary(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio summary with latest data."""

        if portfolio_id not in self.portfolios:
            return None

        portfolio = self.portfolios[portfolio_id]
        portfolio_values = list(self.portfolio_history[portfolio_id])

        if not portfolio_values:
            return {
                "portfolio_id": portfolio_id,
                "name": portfolio.name,
                "current_value": portfolio.initial_value,
                "total_return": 0.0,
                "status": "initializing",
            }

        current_value = portfolio_values[-1]
        total_return = (
            current_value - portfolio.initial_value
        ) / portfolio.initial_value

        # Get cached risk metrics
        risk_metrics = None
        if portfolio_id in self._risk_calculation_cache:
            _, risk_update = self._risk_calculation_cache[portfolio_id]
            risk_metrics = risk_update.to_dict()

        return {
            "portfolio_id": portfolio_id,
            "name": portfolio.name,
            "holdings": portfolio.holdings,
            "initial_value": portfolio.initial_value,
            "current_value": current_value,
            "total_return": total_return,
            "benchmark_symbol": portfolio.benchmark_symbol,
            "risk_metrics": risk_metrics,
            "data_points": len(portfolio_values),
            "created_at": portfolio.created_at.isoformat(),
            "status": "active",
        }

    def get_all_portfolios_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all monitored portfolios."""
        summaries = []
        for portfolio_id in self.portfolios.keys():
            summary = self.get_portfolio_summary(portfolio_id)
            if summary:
                summaries.append(summary)
        return summaries
