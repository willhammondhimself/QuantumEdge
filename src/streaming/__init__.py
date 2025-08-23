"""
Real-time streaming module for QuantumEdge.

This module provides WebSocket-based real-time data streaming for
portfolio monitoring, market data, and risk metrics.
"""

from .websocket import (
    WebSocketConnectionManager,
    WebSocketMessage,
    MessageType,
    connection_manager,
)
from .data_pipeline import DataPipeline, MarketDataUpdate, PortfolioUpdate, RiskUpdate
from .portfolio_monitor import PortfolioMonitor, Portfolio, AlertRule
from .market_data_source import (
    MarketDataSource,
    SimulatedMarketDataSource,
    create_market_data_source,
)

__all__ = [
    # WebSocket components
    "WebSocketConnectionManager",
    "WebSocketMessage",
    "MessageType",
    "connection_manager",
    # Data pipeline
    "DataPipeline",
    "MarketDataUpdate",
    "PortfolioUpdate",
    "RiskUpdate",
    # Portfolio monitoring
    "PortfolioMonitor",
    "Portfolio",
    "AlertRule",
    # Market data sources
    "MarketDataSource",
    "SimulatedMarketDataSource",
    "create_market_data_source",
]
