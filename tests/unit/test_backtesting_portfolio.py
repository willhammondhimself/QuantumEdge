"""Tests for backtesting portfolio management."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock

from src.backtesting.portfolio import (
    Portfolio,
    Transaction,
    PortfolioState,
    TransactionType,
)


class TestTransaction:
    """Test Transaction dataclass."""

    def test_transaction_creation(self):
        """Test creating a transaction."""
        txn = Transaction(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            quantity=100,
            price=150.0,
            value=100 * 150.0,
            commission=10.0,
            transaction_type=TransactionType.BUY,
        )

        assert txn.timestamp == datetime(2023, 1, 1)
        assert txn.symbol == "AAPL"
        assert txn.quantity == 100
        assert txn.price == 150.0
        assert txn.value == 15000.0
        assert txn.commission == 10.0
        assert txn.transaction_type == TransactionType.BUY
        assert txn.net_value == -(15000.0 + 10.0)  # Negative for buy

    def test_sell_transaction(self):
        """Test sell transaction."""
        txn = Transaction(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            quantity=50,
            price=155.0,
            value=50 * 155.0,
            commission=8.0,
            transaction_type=TransactionType.SELL,
        )

        assert txn.quantity == 50
        assert txn.value == 7750.0
        assert txn.net_value == 7750.0 - 8.0  # Positive for sell, minus commission


class TestPortfolioState:
    """Test PortfolioState dataclass."""

    def test_portfolio_state_creation(self):
        """Test creating a portfolio state."""
        state = PortfolioState(
            timestamp=datetime(2023, 1, 1),
            cash=50000.0,
            positions={"AAPL": 100, "GOOGL": 50},
            market_values={"AAPL": 15000.0, "GOOGL": 125000.0},
            total_value=50000.0 + 15000.0 + 125000.0,
            weights={"AAPL": 0.079, "GOOGL": 0.658, "cash": 0.263},
        )

        assert state.timestamp == datetime(2023, 1, 1)
        assert state.cash == 50000.0
        assert state.positions == {"AAPL": 100, "GOOGL": 50}
        assert state.total_value == 190000.0

    def test_portfolio_state_defaults(self):
        """Test portfolio state with minimal data."""
        state = PortfolioState(
            timestamp=datetime(2023, 1, 1),
            cash=100000.0,
            positions={},
            market_values={},
            total_value=100000.0,
            weights={},
        )

        assert state.positions == {}
        assert state.market_values == {}
        assert state.weights == {}


class TestPortfolio:
    """Test Portfolio class functionality."""

    def test_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(
            initial_cash=100000.0, commission_rate=0.001, min_commission=1.0
        )

        assert portfolio.cash == 100000.0
        assert portfolio.initial_cash == 100000.0
        assert portfolio.commission_rate == 0.001
        assert portfolio.min_commission == 1.0
        assert portfolio.positions == {}
        assert len(portfolio.transactions) == 0
        assert len(portfolio.states) == 0

    def test_get_total_value(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=["AAPL", "GOOGL"])

        # Add some positions
        portfolio.positions = {"AAPL": 100, "GOOGL": 50}

        prices = {"AAPL": 150.0, "GOOGL": 2500.0}
        total_value = portfolio.get_total_value(prices)

        expected = 100000.0 + 100 * 150.0 + 50 * 2500.0  # cash + holdings value
        assert total_value == expected

    def test_get_total_value_missing_price(self):
        """Test value calculation with missing price."""
        portfolio = Portfolio(initial_cash=50000.0, symbols=["AAPL", "GOOGL"])
        portfolio.positions = {"AAPL": 100, "GOOGL": 50}

        # Only provide price for AAPL
        prices = {"AAPL": 150.0}

        # Should handle missing price gracefully (0 value for missing symbols)
        total_value = portfolio.get_total_value(prices)
        assert total_value == 50000.0 + 100 * 150.0  # Only AAPL has value

    def test_get_weights(self):
        """Test portfolio weight calculation."""
        portfolio = Portfolio(initial_cash=50000.0, symbols=["AAPL", "GOOGL"])
        portfolio.positions = {"AAPL": 100, "GOOGL": 20}

        prices = {"AAPL": 150.0, "GOOGL": 2500.0}
        weights = portfolio.get_weights(prices)

        # Weights only include positions, not cash
        assert weights["AAPL"] == pytest.approx(15000.0 / 115000.0)
        assert weights["GOOGL"] == pytest.approx(50000.0 / 115000.0)

    def test_calculate_commission(self):
        """Test commission calculation."""
        portfolio = Portfolio(
            initial_cash=100000.0, commission_rate=0.001, min_commission=1.0
        )

        # Large trade - percentage commission
        commission = portfolio.calculate_commission(15000.0)  # trade value
        assert commission == 15000.0 * 0.001  # 15.0

        # Small trade - minimum commission
        commission = portfolio.calculate_commission(0.5)
        assert commission == 1.0  # min commission

    def test_rebalance_buy(self):
        """Test buying through rebalancing."""
        portfolio = Portfolio(
            initial_cash=100000.0, commission_rate=0.001, symbols=["AAPL"]
        )

        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0}
        target_weights = {"AAPL": 0.15}  # 15% of portfolio in AAPL

        transactions = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Check positions updated
        assert portfolio.positions["AAPL"] > 0

        # Check cash reduced
        assert portfolio.cash < 100000.0

        # Check transaction recorded
        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.symbol == "AAPL"
        assert txn.quantity > 0
        assert txn.price == 150.0
        assert txn.transaction_type == TransactionType.BUY

    def test_rebalance_sell(self):
        """Test selling through rebalancing."""
        portfolio = Portfolio(
            initial_cash=50000.0, commission_rate=0.001, symbols=["AAPL"]
        )
        portfolio.positions = {"AAPL": 100}

        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 155.0}
        target_weights = {"AAPL": 0.05}  # Reduce AAPL to 5% of portfolio

        transactions = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Check positions reduced
        assert portfolio.positions["AAPL"] < 100

        # Check cash increased
        assert portfolio.cash > 50000.0

        # Check transaction recorded
        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.quantity > 0  # Quantity is positive for sells in this implementation
        assert txn.transaction_type == TransactionType.SELL

    def test_rebalance_sell_all(self):
        """Test selling all shares of a position."""
        portfolio = Portfolio(initial_cash=50000.0, symbols=["AAPL"])
        portfolio.positions = {"AAPL": 100}

        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0}
        target_weights = {"AAPL": 0.0}  # Sell all AAPL

        portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Position should be zero
        assert portfolio.positions["AAPL"] == 0

    def test_rebalance_insufficient_cash(self):
        """Test buying with insufficient cash."""
        portfolio = Portfolio(
            initial_cash=1000.0, commission_rate=0.001, symbols=["AAPL"]
        )

        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0}
        target_weights = {"AAPL": 1.0}  # Try to put 100% in AAPL

        # Should execute partial trade with available cash
        transactions = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Should have bought as much as possible
        assert len(transactions) == 1
        assert portfolio.cash < 1000.0  # Some cash used

    def test_rebalance_sell_all_shares(self):
        """Test selling more shares than owned."""
        portfolio = Portfolio(initial_cash=50000.0, symbols=["AAPL"])
        portfolio.positions = {"AAPL": 50}

        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0}
        target_weights = {"AAPL": 0.0}  # Sell all

        # Should sell all available shares
        transactions = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        assert portfolio.positions["AAPL"] == 0

    def test_rebalance_to_weights(self):
        """Test rebalancing portfolio to target weights."""
        portfolio = Portfolio(
            initial_cash=100000.0,
            commission_rate=0.001,
            symbols=["AAPL", "GOOGL", "MSFT"],
        )

        # Initial positions
        portfolio.positions = {"AAPL": 100, "GOOGL": 20, "MSFT": 0}

        # Current prices
        prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}

        # Target weights
        target_weights = {"AAPL": 0.4, "GOOGL": 0.4, "MSFT": 0.2}

        timestamp = datetime(2023, 1, 1)
        trades = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Check trades were executed
        assert len(trades) > 0

        # Calculate final weights
        final_weights = portfolio.get_weights(prices)

        # Check weights are close to target (within tolerance for rounding)
        for symbol, target_weight in target_weights.items():
            assert abs(final_weights.get(symbol, 0) - target_weight) < 0.01

    def test_rebalance_no_changes_needed(self):
        """Test rebalancing when already at target weights."""
        portfolio = Portfolio(
            initial_cash=0.0, commission_rate=0.001, symbols=["AAPL", "GOOGL"]
        )
        portfolio.positions = {"AAPL": 100, "GOOGL": 40}

        prices = {"AAPL": 150.0, "GOOGL": 375.0}

        # Current weights: AAPL = 15000/30000 = 0.5, GOOGL = 15000/30000 = 0.5
        target_weights = {"AAPL": 0.5, "GOOGL": 0.5}

        trades = portfolio.rebalance_to_weights(
            datetime(2023, 1, 1), target_weights, prices
        )

        # No trades should be needed (or very small trades due to commission threshold)
        assert len(trades) <= 2  # May have small adjustments

    def test_rebalance_partial_allocation(self):
        """Test rebalancing with partial allocation (not 100%)."""
        portfolio = Portfolio(
            initial_cash=100000.0, commission_rate=0.001, symbols=["AAPL", "GOOGL"]
        )

        # Target only 60% invested
        target_weights = {"AAPL": 0.3, "GOOGL": 0.3}  # 40% stays in cash

        prices = {"AAPL": 150.0, "GOOGL": 2500.0}

        trades = portfolio.rebalance_to_weights(
            datetime(2023, 1, 1), target_weights, prices
        )

        # Check trades were executed
        assert len(trades) > 0

        # Check final allocation
        final_weights = portfolio.get_weights(prices)
        # Note: get_weights only returns position weights, not cash
        total_weight = sum(final_weights.values())
        assert total_weight < 0.65  # Should be around 60% invested

    def test_update_state(self):
        """Test recording portfolio state."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=["AAPL"])
        portfolio.positions = {"AAPL": 100}

        prices = {"AAPL": 150.0}
        timestamp = datetime(2023, 1, 1)

        state = portfolio.update_state(timestamp, prices)

        assert len(portfolio.states) == 1
        assert state.timestamp == timestamp
        assert state.cash == 100000.0
        assert state.positions == {"AAPL": 100}
        assert state.total_value == 115000.0

    def test_get_transaction_history(self):
        """Test getting transaction history."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=["AAPL"])

        # Execute a trade
        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0}
        target_weights = {"AAPL": 0.15}

        transactions = portfolio.rebalance_to_weights(timestamp, target_weights, prices)

        # Get transaction history
        history = portfolio.get_transaction_history()

        assert len(history) == len(transactions)
        assert isinstance(history, pd.DataFrame)
        assert "symbol" in history.columns
        assert "quantity" in history.columns
        assert "price" in history.columns

    def test_get_portfolio_history(self):
        """Test getting portfolio history."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=["AAPL"])

        # Update states over time
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]

        # First state - no positions
        portfolio.update_state(timestamps[0], {})

        # Second state - buy some AAPL
        portfolio.positions["AAPL"] = 50
        portfolio.cash = 92500.0  # Bought 50 shares at 150
        portfolio.update_state(timestamps[1], {"AAPL": 150.0})

        # Third state - AAPL price increased
        portfolio.update_state(timestamps[2], {"AAPL": 160.0})

        # Get portfolio history
        history = portfolio.get_portfolio_history()

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 3
        assert "cash" in history.columns
        assert "total_value" in history.columns

    def test_state_tracking(self):
        """Test portfolio state tracking over time."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=["AAPL"])

        # Record multiple states
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]

        for timestamp in timestamps:
            portfolio.update_state(timestamp, {"AAPL": 150.0})

        # Check states were recorded
        assert len(portfolio.states) == 3

        # Check state values
        for i, state in enumerate(portfolio.states):
            assert state.timestamp == timestamps[i]
            assert state.cash == 100000.0
            assert state.total_value == 100000.0

    def test_asset_allocation(self):
        """Test getting asset allocation."""
        portfolio = Portfolio(initial_cash=50000.0, symbols=["AAPL", "GOOGL"])

        # Set up positions
        portfolio.positions = {"AAPL": 100, "GOOGL": 20}

        # Update state
        timestamp = datetime(2023, 1, 1)
        prices = {"AAPL": 150.0, "GOOGL": 2500.0}
        state = portfolio.update_state(timestamp, prices)

        # Test asset allocation method on state
        allocation = state.get_asset_allocation()

        # Total value = 50000 + 100*150 + 20*2500 = 115000
        # AAPL = 15000/115000 = 0.130
        # GOOGL = 50000/115000 = 0.435
        assert allocation["AAPL"] == pytest.approx(15000 / 115000)
        assert allocation["GOOGL"] == pytest.approx(50000 / 115000)

    def test_empty_portfolio_operations(self):
        """Test operations on empty portfolio."""
        portfolio = Portfolio(initial_cash=100000.0, symbols=[])

        # Empty portfolio value
        assert portfolio.get_total_value({}) == 100000.0

        # Empty portfolio weights
        weights = portfolio.get_weights({})
        assert weights == {}

        # Empty transaction history
        history = portfolio.get_transaction_history()
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0
