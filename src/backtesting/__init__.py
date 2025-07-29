"""
Backtesting engine for portfolio optimization strategies.

This module provides comprehensive backtesting capabilities including:
- Historical simulation with realistic transaction costs
- Performance metrics calculation 
- Risk attribution and decomposition
- Benchmark comparison
- Walk-forward analysis
"""

from typing import List

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .strategy import PortfolioStrategy, BuyAndHoldStrategy, RebalancingStrategy
from .metrics import PerformanceMetrics, RiskMetrics, calculate_all_metrics
from .portfolio import Portfolio, Transaction, PortfolioState

__all__: List[str] = [
    # Core engine
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    
    # Strategies
    "PortfolioStrategy",
    "BuyAndHoldStrategy",
    "RebalancingStrategy",
    
    # Performance analysis
    "PerformanceMetrics",
    "RiskMetrics", 
    "calculate_all_metrics",
    
    # Portfolio management
    "Portfolio",
    "Transaction",
    "PortfolioState"
]