"""
Portfolio state management and transaction tracking for backtesting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TransactionType(Enum):
    """Transaction types."""
    BUY = "buy"
    SELL = "sell"
    REBALANCE = "rebalance"
    DIVIDEND = "dividend"


@dataclass
class Transaction:
    """Individual transaction record."""
    timestamp: datetime
    symbol: str
    transaction_type: TransactionType
    quantity: float
    price: float
    value: float
    commission: float = 0.0
    
    @property
    def net_value(self) -> float:
        """Net transaction value after commission."""
        if self.transaction_type == TransactionType.BUY:
            return -(self.value + self.commission)
        else:
            return self.value - self.commission


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""
    timestamp: datetime
    positions: Dict[str, float]  # symbol -> quantity
    cash: float
    market_values: Dict[str, float]  # symbol -> market value
    total_value: float
    weights: Dict[str, float]  # symbol -> weight
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get asset allocation as percentages."""
        if self.total_value <= 0:
            return {}
        
        return {
            symbol: value / self.total_value 
            for symbol, value in self.market_values.items()
            if value > 0
        }


class Portfolio:
    """Portfolio manager for backtesting with transaction tracking."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% commission
        min_commission: float = 1.0,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            commission_rate: Commission as percentage of trade value
            min_commission: Minimum commission per trade
            symbols: List of symbols to track
        """
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.symbols = symbols or []
        
        # Current state
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # symbol -> shares
        self.transactions: List[Transaction] = []
        self.states: List[PortfolioState] = []
        
        # Initialize positions
        for symbol in self.symbols:
            self.positions[symbol] = 0.0
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        commission = abs(trade_value) * self.commission_rate
        return max(commission, self.min_commission)
    
    def get_market_values(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current market values for all positions."""
        market_values = {}
        for symbol in self.symbols:
            if symbol in prices and symbol in self.positions:
                market_values[symbol] = self.positions[symbol] * prices[symbol]
            else:
                market_values[symbol] = 0.0
        return market_values
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        market_values = self.get_market_values(prices)
        return self.cash + sum(market_values.values())
    
    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        total_value = self.get_total_value(prices)
        if total_value <= 0:
            return {symbol: 0.0 for symbol in self.symbols}
        
        market_values = self.get_market_values(prices)
        return {
            symbol: value / total_value 
            for symbol, value in market_values.items()
        }
    
    def update_state(self, timestamp: datetime, prices: Dict[str, float]) -> PortfolioState:
        """Update and record portfolio state."""
        market_values = self.get_market_values(prices)
        total_value = self.get_total_value(prices)
        weights = self.get_weights(prices)
        
        state = PortfolioState(
            timestamp=timestamp,
            positions=self.positions.copy(),
            cash=self.cash,
            market_values=market_values,
            total_value=total_value,
            weights=weights
        )
        
        self.states.append(state)
        return state
    
    def rebalance_to_weights(
        self,
        timestamp: datetime,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> List[Transaction]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            timestamp: Rebalancing timestamp
            target_weights: Target weights for each symbol
            prices: Current prices for each symbol
            
        Returns:
            List of transactions executed
        """
        current_total = self.get_total_value(prices)
        current_weights = self.get_weights(prices)
        transactions = []
        
        # Calculate required trades
        for symbol in self.symbols:
            if symbol not in prices:
                continue
                
            target_weight = target_weights.get(symbol, 0.0)
            current_weight = current_weights.get(symbol, 0.0)
            
            target_value = target_weight * current_total
            current_value = current_weight * current_total
            
            trade_value = target_value - current_value
            
            if abs(trade_value) < self.min_commission:
                continue  # Skip small trades
            
            price = prices[symbol]
            shares_to_trade = trade_value / price
            
            # Execute trade
            commission = self.calculate_commission(abs(trade_value))
            
            if trade_value > 0:  # Buy
                # Check if we have enough cash
                total_cost = abs(trade_value) + commission
                if total_cost > self.cash:
                    # Scale down the trade
                    available_cash = self.cash - commission
                    if available_cash > 0:
                        trade_value = available_cash
                        shares_to_trade = trade_value / price
                    else:
                        continue  # Can't afford even minimum trade
                
                transaction = Transaction(
                    timestamp=timestamp,
                    symbol=symbol,
                    transaction_type=TransactionType.BUY,
                    quantity=shares_to_trade,
                    price=price,
                    value=trade_value,
                    commission=commission
                )
                
                self.positions[symbol] += shares_to_trade
                self.cash -= (trade_value + commission)
                
            else:  # Sell
                shares_to_sell = abs(shares_to_trade)
                
                # Check if we have enough shares
                if shares_to_sell > self.positions[symbol]:
                    shares_to_sell = self.positions[symbol]
                    trade_value = -(shares_to_sell * price)
                
                if shares_to_sell == 0:
                    continue
                
                transaction = Transaction(
                    timestamp=timestamp,
                    symbol=symbol,
                    transaction_type=TransactionType.SELL,
                    quantity=shares_to_sell,
                    price=price,
                    value=abs(trade_value),
                    commission=commission
                )
                
                self.positions[symbol] -= shares_to_sell
                self.cash += (abs(trade_value) - commission)
            
            transactions.append(transaction)
            self.transactions.append(transaction)
        
        return transactions
    
    def add_dividend(
        self,
        timestamp: datetime,
        symbol: str,
        dividend_per_share: float
    ) -> Optional[Transaction]:
        """Add dividend payment."""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return None
        
        dividend_amount = self.positions[symbol] * dividend_per_share
        
        transaction = Transaction(
            timestamp=timestamp,
            symbol=symbol,
            transaction_type=TransactionType.DIVIDEND,
            quantity=self.positions[symbol],
            price=dividend_per_share,
            value=dividend_amount,
            commission=0.0
        )
        
        self.cash += dividend_amount
        self.transactions.append(transaction)
        
        return transaction
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame."""
        if not self.transactions:
            return pd.DataFrame()
        
        data = []
        for txn in self.transactions:
            data.append({
                'timestamp': txn.timestamp,
                'symbol': txn.symbol,
                'type': txn.transaction_type.value,
                'quantity': txn.quantity,
                'price': txn.price,
                'value': txn.value,
                'commission': txn.commission,
                'net_value': txn.net_value
            })
        
        return pd.DataFrame(data)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio state history as DataFrame."""
        if not self.states:
            return pd.DataFrame()
        
        data = []
        for state in self.states:
            row = {
                'timestamp': state.timestamp,
                'cash': state.cash,
                'total_value': state.total_value
            }
            
            # Add individual asset values and weights
            for symbol in self.symbols:
                row[f'{symbol}_value'] = state.market_values.get(symbol, 0.0)
                row[f'{symbol}_weight'] = state.weights.get(symbol, 0.0)
                row[f'{symbol}_shares'] = state.positions.get(symbol, 0.0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get basic performance summary."""
        if not self.states:
            return {}
        
        initial_value = self.initial_cash
        final_value = self.states[-1].total_value
        
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate transaction costs
        total_commissions = sum(txn.commission for txn in self.transactions)
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_commissions': total_commissions,
            'num_transactions': len(self.transactions)
        }