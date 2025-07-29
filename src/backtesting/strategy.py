"""
Portfolio strategies for backtesting.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

from ..optimization import MeanVarianceOptimizer, CVXPY_AVAILABLE
from ..quantum_algorithms import QuantumVQE, PortfolioQAOA


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class PortfolioStrategy(ABC):
    """Abstract base class for portfolio strategies."""
    
    def __init__(self, name: str, symbols: List[str]):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
        """
        self.name = name
        self.symbols = symbols
        self.initialized = False
    
    @abstractmethod
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Initialize strategy and return initial weights.
        
        Args:
            prices: Historical price data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of initial weights
        """
        pass
    
    @abstractmethod
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Determine new portfolio weights.
        
        Args:
            current_date: Current date
            prices: Historical price data up to current date
            current_weights: Current portfolio weights
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of new weights
        """
        pass
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """
        Determine if rebalancing should occur.
        
        Args:
            current_date: Current date
            last_rebalance: Last rebalancing date
            
        Returns:
            True if should rebalance
        """
        return True  # Default: always rebalance when called


class BuyAndHoldStrategy(PortfolioStrategy):
    """Simple buy-and-hold strategy with fixed weights."""
    
    def __init__(
        self, 
        name: str = "Buy and Hold",
        symbols: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize buy-and-hold strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            weights: Fixed weights (if None, use equal weights)
        """
        if symbols is None:
            symbols = []
        
        super().__init__(name, symbols)
        self.target_weights = weights
        
        # If no weights provided, use equal weights
        if self.target_weights is None and self.symbols:
            weight = 1.0 / len(self.symbols)
            self.target_weights = {symbol: weight for symbol in self.symbols}
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with target weights."""
        self.initialized = True
        return self.target_weights.copy() if self.target_weights else {}
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Return target weights (no rebalancing)."""
        return self.target_weights.copy() if self.target_weights else current_weights


class RebalancingStrategy(PortfolioStrategy):
    """Strategy that rebalances to target weights periodically."""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        target_weights: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.05  # Rebalance if weight drifts >5%
    ):
        """
        Initialize rebalancing strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            frequency: Rebalancing frequency
            target_weights: Target weights (if None, use equal weights)
            drift_threshold: Threshold for drift-based rebalancing
        """
        super().__init__(name, symbols)
        self.frequency = frequency
        self.drift_threshold = drift_threshold
        
        if target_weights is None:
            weight = 1.0 / len(symbols)
            self.target_weights = {symbol: weight for symbol in symbols}
        else:
            self.target_weights = target_weights
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with target weights."""
        self.initialized = True
        return self.target_weights.copy()
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check if rebalancing should occur based on frequency."""
        if self.frequency == RebalanceFrequency.DAILY:
            return True
        elif self.frequency == RebalanceFrequency.WEEKLY:
            return (current_date - last_rebalance).days >= 7
        elif self.frequency == RebalanceFrequency.MONTHLY:
            return current_date.month != last_rebalance.month
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_date.year != last_rebalance.year
        elif self.frequency == RebalanceFrequency.ANNUALLY:
            return current_date.year != last_rebalance.year
        
        return False
    
    def check_drift(self, current_weights: Dict[str, float]) -> bool:
        """Check if weights have drifted beyond threshold."""
        for symbol in self.symbols:
            target = self.target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            
            if abs(current - target) > self.drift_threshold:
                return True
        
        return False
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Return target weights."""
        return self.target_weights.copy()


class MeanVarianceStrategy(PortfolioStrategy):
    """Strategy using mean-variance optimization."""
    
    def __init__(
        self,
        name: str = "Mean-Variance",
        symbols: Optional[List[str]] = None,
        lookback_period: int = 252,  # 1 year
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize mean-variance strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            lookback_period: Lookback period for estimation (trading days)
            frequency: Rebalancing frequency
            risk_aversion: Risk aversion parameter
            risk_free_rate: Risk-free rate
        """
        if symbols is None:
            symbols = []
            
        super().__init__(name, symbols)
        self.lookback_period = lookback_period
        self.frequency = frequency
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.last_rebalance = None
        
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for mean-variance optimization")
        
        self.optimizer = MeanVarianceOptimizer(risk_free_rate=risk_free_rate)
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with equal weights."""
        self.initialized = True
        weight = 1.0 / len(self.symbols)
        return {symbol: weight for symbol in self.symbols}
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check rebalancing based on frequency."""
        if self.frequency == RebalanceFrequency.DAILY:
            return True
        elif self.frequency == RebalanceFrequency.WEEKLY:
            return (current_date - last_rebalance).days >= 7
        elif self.frequency == RebalanceFrequency.MONTHLY:
            return current_date.month != last_rebalance.month
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_date.year != last_rebalance.year
        
        return False
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Optimize portfolio using mean-variance."""
        try:
            # Get recent price data
            end_idx = prices.index.get_loc(current_date)
            start_idx = max(0, end_idx - self.lookback_period + 1)
            
            recent_prices = prices.iloc[start_idx:end_idx + 1][self.symbols]
            
            if len(recent_prices) < 20:  # Need minimum data
                return current_weights
            
            # Calculate returns
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) < 10:
                return current_weights
            
            # Estimate expected returns and covariance
            expected_returns = returns.mean().values * 252  # Annualize
            covariance_matrix = returns.cov().values * 252  # Annualize
            
            # Optimize portfolio
            result = self.optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                objective='max_sharpe',
                risk_aversion=self.risk_aversion
            )
            
            if result.success and result.weights is not None:
                new_weights = {}
                for i, symbol in enumerate(self.symbols):
                    new_weights[symbol] = float(result.weights[i])
                return new_weights
            else:
                return current_weights
                
        except Exception as e:
            print(f"Error in mean-variance optimization: {e}")
            return current_weights


class VQEStrategy(PortfolioStrategy):
    """Strategy using Variational Quantum Eigensolver."""
    
    def __init__(
        self,
        name: str = "VQE",
        symbols: Optional[List[str]] = None,
        lookback_period: int = 252,
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        depth: int = 3,
        max_iterations: int = 100
    ):
        """
        Initialize VQE strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            lookback_period: Lookback period for covariance estimation
            frequency: Rebalancing frequency
            depth: Circuit depth
            max_iterations: Maximum optimization iterations
        """
        if symbols is None:
            symbols = []
            
        super().__init__(name, symbols)
        self.lookback_period = lookback_period
        self.frequency = frequency
        self.depth = depth
        self.max_iterations = max_iterations
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with equal weights."""
        self.initialized = True
        weight = 1.0 / len(self.symbols)
        return {symbol: weight for symbol in self.symbols}
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check rebalancing based on frequency."""
        if self.frequency == RebalanceFrequency.MONTHLY:
            return current_date.month != last_rebalance.month
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_date.year != last_rebalance.year
        
        return False
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Optimize using VQE."""
        try:
            # Get recent price data
            end_idx = prices.index.get_loc(current_date)
            start_idx = max(0, end_idx - self.lookback_period + 1)
            
            recent_prices = prices.iloc[start_idx:end_idx + 1][self.symbols]
            
            if len(recent_prices) < 20:
                return current_weights
            
            # Calculate returns and covariance
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) < 10:
                return current_weights
            
            covariance_matrix = returns.cov().values * 252  # Annualize
            
            # Run VQE optimization
            vqe = QuantumVQE(
                num_assets=len(self.symbols),
                depth=self.depth,
                max_iterations=self.max_iterations
            )
            
            result = vqe.solve_eigenportfolio(covariance_matrix)
            
            if result.success and result.eigenvector is not None:
                # Convert eigenvector to weights (normalize to sum to 1)
                weights = np.abs(result.eigenvector)
                weights = weights / np.sum(weights)
                
                new_weights = {}
                for i, symbol in enumerate(self.symbols):
                    new_weights[symbol] = float(weights[i])
                return new_weights
            else:
                return current_weights
                
        except Exception as e:
            print(f"Error in VQE optimization: {e}")
            return current_weights


class QAOAStrategy(PortfolioStrategy):
    """Strategy using Quantum Approximate Optimization Algorithm."""
    
    def __init__(
        self,
        name: str = "QAOA",
        symbols: Optional[List[str]] = None,
        lookback_period: int = 252,
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        num_layers: int = 3,
        risk_aversion: float = 1.0,
        cardinality_constraint: Optional[int] = None
    ):
        """
        Initialize QAOA strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            lookback_period: Lookback period for estimation
            frequency: Rebalancing frequency
            num_layers: QAOA layers
            risk_aversion: Risk aversion parameter
            cardinality_constraint: Maximum number of assets
        """
        if symbols is None:
            symbols = []
            
        super().__init__(name, symbols)
        self.lookback_period = lookback_period
        self.frequency = frequency
        self.num_layers = num_layers
        self.risk_aversion = risk_aversion
        self.cardinality_constraint = cardinality_constraint or len(symbols)
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with equal weights."""
        self.initialized = True
        weight = 1.0 / len(self.symbols)
        return {symbol: weight for symbol in self.symbols}
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check rebalancing based on frequency."""
        if self.frequency == RebalanceFrequency.MONTHLY:
            return current_date.month != last_rebalance.month
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_date.year != last_rebalance.year
        
        return False
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Optimize using QAOA."""
        try:
            # Get recent price data
            end_idx = prices.index.get_loc(current_date)
            start_idx = max(0, end_idx - self.lookback_period + 1)
            
            recent_prices = prices.iloc[start_idx:end_idx + 1][self.symbols]
            
            if len(recent_prices) < 20:
                return current_weights
            
            # Calculate returns, expected returns, and covariance
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) < 10:
                return current_weights
            
            expected_returns = returns.mean().values * 252  # Annualize
            covariance_matrix = returns.cov().values * 252  # Annualize
            
            # Run QAOA optimization
            qaoa = PortfolioQAOA(
                num_assets=len(self.symbols),
                num_layers=self.num_layers
            )
            
            result = qaoa.solve_portfolio_selection(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_aversion=self.risk_aversion,
                cardinality_constraint=self.cardinality_constraint
            )
            
            if result.success and result.optimal_portfolio is not None:
                # Normalize to get weights
                weights = result.optimal_portfolio.astype(float)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                
                new_weights = {}
                for i, symbol in enumerate(self.symbols):
                    new_weights[symbol] = float(weights[i])
                return new_weights
            else:
                return current_weights
                
        except Exception as e:
            print(f"Error in QAOA optimization: {e}")
            return current_weights


class CustomStrategy(PortfolioStrategy):
    """Custom strategy with user-defined rebalancing function."""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        rebalance_func: Callable,
        should_rebalance_func: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize custom strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols
            rebalance_func: Function that returns new weights
            should_rebalance_func: Function that determines if rebalancing needed
            **kwargs: Additional parameters passed to functions
        """
        super().__init__(name, symbols)
        self.rebalance_func = rebalance_func
        self.should_rebalance_func = should_rebalance_func
        self.kwargs = kwargs
    
    def initialize(self, prices: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Initialize with equal weights."""
        self.initialized = True
        weight = 1.0 / len(self.symbols)
        return {symbol: weight for symbol in self.symbols}
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Use custom rebalancing logic if provided."""
        if self.should_rebalance_func:
            return self.should_rebalance_func(current_date, last_rebalance, **self.kwargs)
        return True  # Default: always rebalance
    
    def rebalance(
        self, 
        current_date: datetime, 
        prices: pd.DataFrame,
        current_weights: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Use custom rebalancing function."""
        merged_kwargs = {**self.kwargs, **kwargs}
        return self.rebalance_func(
            current_date, 
            prices, 
            current_weights, 
            **merged_kwargs
        )