"""
Real-time data pipeline for portfolio monitoring.

This module provides real-time market data streaming and portfolio
value calculations for WebSocket clients.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass
import numpy as np

from .websocket import WebSocketConnectionManager, WebSocketMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class MarketDataUpdate:
    """Market data update structure."""

    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioUpdate:
    """Portfolio value update structure."""

    portfolio_id: str
    total_value: float
    change: float
    change_percent: float
    positions: Dict[str, Dict[str, float]]  # symbol -> {shares, value, weight}
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "portfolio_id": self.portfolio_id,
            "total_value": self.total_value,
            "change": self.change,
            "change_percent": self.change_percent,
            "positions": self.positions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskUpdate:
    """Risk metrics update structure."""

    portfolio_id: str
    volatility: float
    var_95: float
    cvar_95: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "portfolio_id": self.portfolio_id,
            "volatility": self.volatility,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "beta": self.beta,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "timestamp": self.timestamp.isoformat(),
        }


class DataPipeline:
    """
    Real-time data pipeline for market data and portfolio updates.

    Manages data sources, processing, and distribution to WebSocket clients.
    """

    def __init__(
        self,
        connection_manager: WebSocketConnectionManager,
        update_interval: int = 1,  # seconds
        redis_client=None,
    ):
        """
        Initialize data pipeline.

        Args:
            connection_manager: WebSocket connection manager
            update_interval: Update frequency in seconds
            redis_client: Optional Redis client for caching
        """
        self.connection_manager = connection_manager
        self.update_interval = update_interval
        self.redis_client = redis_client

        # Data sources and subscribers
        self.market_data_sources: Dict[str, Callable] = {}
        self.portfolio_calculators: Dict[str, Callable] = {}
        self.risk_calculators: Dict[str, Callable] = {}

        # Active subscriptions
        self.market_data_subscribers: Set[str] = set()  # symbols
        self.portfolio_subscribers: Set[str] = set()  # portfolio_ids
        self.risk_subscribers: Set[str] = set()  # portfolio_ids

        # Data caches
        self.market_data_cache: Dict[str, MarketDataUpdate] = {}
        self.portfolio_cache: Dict[str, PortfolioUpdate] = {}
        self.risk_cache: Dict[str, RiskUpdate] = {}

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the data pipeline."""
        if self._running:
            return

        self._running = True
        logger.info("Starting real-time data pipeline")

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._market_data_loop()))
        self._tasks.append(asyncio.create_task(self._portfolio_update_loop()))
        self._tasks.append(asyncio.create_task(self._risk_update_loop()))

    async def stop(self):
        """Stop the data pipeline."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping real-time data pipeline")

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    def register_market_data_source(self, source_name: str, source_func: Callable):
        """
        Register a market data source.

        Args:
            source_name: Name of the data source
            source_func: Async function that returns market data updates
        """
        self.market_data_sources[source_name] = source_func
        logger.info(f"Registered market data source: {source_name}")

    def register_portfolio_calculator(
        self, portfolio_id: str, calculator_func: Callable
    ):
        """
        Register a portfolio value calculator.

        Args:
            portfolio_id: Portfolio identifier
            calculator_func: Async function that calculates portfolio values
        """
        self.portfolio_calculators[portfolio_id] = calculator_func
        logger.info(f"Registered portfolio calculator: {portfolio_id}")

    def register_risk_calculator(self, portfolio_id: str, calculator_func: Callable):
        """
        Register a risk metrics calculator.

        Args:
            portfolio_id: Portfolio identifier
            calculator_func: Async function that calculates risk metrics
        """
        self.risk_calculators[portfolio_id] = calculator_func
        logger.info(f"Registered risk calculator: {portfolio_id}")

    async def subscribe_market_data(self, symbols: List[str]):
        """
        Subscribe to market data for symbols.

        Args:
            symbols: List of stock symbols to monitor
        """
        self.market_data_subscribers.update(symbols)
        logger.info(f"Subscribed to market data: {symbols}")

    async def subscribe_portfolio(self, portfolio_id: str):
        """
        Subscribe to portfolio updates.

        Args:
            portfolio_id: Portfolio to monitor
        """
        self.portfolio_subscribers.add(portfolio_id)
        logger.info(f"Subscribed to portfolio: {portfolio_id}")

    async def subscribe_risk_metrics(self, portfolio_id: str):
        """
        Subscribe to risk metrics updates.

        Args:
            portfolio_id: Portfolio to monitor for risk
        """
        self.risk_subscribers.add(portfolio_id)
        logger.info(f"Subscribed to risk metrics: {portfolio_id}")

    async def _market_data_loop(self):
        """Background loop for market data updates."""
        while self._running:
            try:
                if not self.market_data_subscribers:
                    await asyncio.sleep(self.update_interval)
                    continue

                # Collect market data from all sources
                updates = []
                for source_name, source_func in self.market_data_sources.items():
                    try:
                        source_updates = await source_func(
                            list(self.market_data_subscribers)
                        )
                        if source_updates:
                            updates.extend(source_updates)
                    except Exception as e:
                        logger.error(f"Error in market data source {source_name}: {e}")

                # Process and broadcast updates
                for update in updates:
                    if isinstance(update, MarketDataUpdate):
                        # Cache the update
                        self.market_data_cache[update.symbol] = update

                        # Broadcast to subscribers
                        await self.connection_manager.broadcast_to_topic(
                            f"market_data:{update.symbol}",
                            WebSocketMessage(
                                message_type=MessageType.PRICE_UPDATE,
                                data=update.to_dict(),
                            ),
                        )

                        # Also broadcast to general market data topic
                        await self.connection_manager.broadcast_to_topic(
                            "market_data:all",
                            WebSocketMessage(
                                message_type=MessageType.PRICE_UPDATE,
                                data=update.to_dict(),
                            ),
                        )

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _portfolio_update_loop(self):
        """Background loop for portfolio value updates."""
        while self._running:
            try:
                if not self.portfolio_subscribers:
                    await asyncio.sleep(
                        self.update_interval * 5
                    )  # Less frequent if no subscribers
                    continue

                # Calculate portfolio updates
                for portfolio_id in self.portfolio_subscribers.copy():
                    if portfolio_id not in self.portfolio_calculators:
                        continue

                    try:
                        calculator = self.portfolio_calculators[portfolio_id]
                        update = await calculator(self.market_data_cache)

                        if isinstance(update, PortfolioUpdate):
                            # Cache the update
                            self.portfolio_cache[portfolio_id] = update

                            # Broadcast to subscribers
                            await self.connection_manager.broadcast_to_topic(
                                f"portfolio:{portfolio_id}",
                                WebSocketMessage(
                                    message_type=MessageType.PORTFOLIO_UPDATE,
                                    data=update.to_dict(),
                                ),
                            )

                    except Exception as e:
                        logger.error(f"Error calculating portfolio {portfolio_id}: {e}")

                await asyncio.sleep(
                    self.update_interval * 5
                )  # Portfolio updates less frequent

            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(self.update_interval * 5)

    async def _risk_update_loop(self):
        """Background loop for risk metrics updates."""
        while self._running:
            try:
                if not self.risk_subscribers:
                    await asyncio.sleep(
                        self.update_interval * 30
                    )  # Much less frequent if no subscribers
                    continue

                # Calculate risk updates
                for portfolio_id in self.risk_subscribers.copy():
                    if portfolio_id not in self.risk_calculators:
                        continue

                    try:
                        calculator = self.risk_calculators[portfolio_id]
                        update = await calculator(
                            self.market_data_cache,
                            self.portfolio_cache.get(portfolio_id),
                        )

                        if isinstance(update, RiskUpdate):
                            # Cache the update
                            self.risk_cache[portfolio_id] = update

                            # Broadcast to subscribers
                            await self.connection_manager.broadcast_to_topic(
                                f"risk:{portfolio_id}",
                                WebSocketMessage(
                                    message_type=MessageType.RISK_UPDATE,
                                    data=update.to_dict(),
                                ),
                            )

                    except Exception as e:
                        logger.error(f"Error calculating risk for {portfolio_id}: {e}")

                await asyncio.sleep(
                    self.update_interval * 30
                )  # Risk updates much less frequent

            except Exception as e:
                logger.error(f"Error in risk update loop: {e}")
                await asyncio.sleep(self.update_interval * 30)

    async def send_alert(self, alert_type: str, message: str, severity: str = "info"):
        """
        Send alert to all connected clients.

        Args:
            alert_type: Type of alert (e.g., "price_alert", "risk_alert")
            message: Alert message
            severity: Alert severity ("info", "warning", "error")
        """
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.connection_manager.broadcast_to_topic(
            "alerts", WebSocketMessage(message_type=MessageType.ALERT, data=alert_data)
        )

        logger.info(f"Sent alert: {alert_type} - {message}")

    def get_cached_data(self) -> Dict[str, Any]:
        """Get all cached data for debugging/monitoring."""
        return {
            "market_data": {
                symbol: update.to_dict()
                for symbol, update in self.market_data_cache.items()
            },
            "portfolios": {
                portfolio_id: update.to_dict()
                for portfolio_id, update in self.portfolio_cache.items()
            },
            "risk_metrics": {
                portfolio_id: update.to_dict()
                for portfolio_id, update in self.risk_cache.items()
            },
            "subscribers": {
                "market_data": list(self.market_data_subscribers),
                "portfolios": list(self.portfolio_subscribers),
                "risk_metrics": list(self.risk_subscribers),
            },
            "running": self._running,
        }
