"""Tests for backtesting performance metrics."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtesting.metrics import (
    PerformanceMetrics,
    RiskMetrics,
    calculate_returns,
    calculate_drawdown,
    calculate_var_cvar,
    calculate_downside_deviation,
    calculate_omega_ratio,
    calculate_beta_alpha,
    calculate_rolling_metrics,
    calculate_all_metrics,
    compare_strategies,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.14,
            cagr=0.14,
            volatility=0.02,
            annualized_volatility=0.18,
            max_drawdown=-0.12,
            max_drawdown_duration=30,
            sharpe_ratio=0.78,
            sortino_ratio=0.95,
            calmar_ratio=1.17,
            omega_ratio=1.2,
            downside_deviation=0.015,
            var_95=-0.05,
            cvar_95=-0.08,
            skewness=0.1,
            kurtosis=0.2,
            beta=0.9,
            alpha=0.02,
            information_ratio=0.5,
            tracking_error=0.05,
        )

        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 0.78
        assert metrics.beta == 0.9


class TestRiskMetrics:
    """Test RiskMetrics dataclass."""

    def test_risk_metrics_creation(self):
        """Test creating risk metrics."""
        drawdown_series = pd.Series([0, -0.05, -0.10, -0.08, 0])
        rolling_vol = pd.Series([0.15, 0.16, 0.17, 0.18, 0.19])

        metrics = RiskMetrics(
            drawdown_series=drawdown_series,
            drawdown_periods=[(datetime(2023, 1, 1), datetime(2023, 1, 15), -0.10)],
            rolling_volatility=rolling_vol,
            rolling_sharpe=pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
            var_series={0.95: -0.05, 0.99: -0.08},
            cvar_series={0.95: -0.08, 0.99: -0.12},
            rolling_beta=pd.Series([0.8, 0.85, 0.9, 0.95, 1.0]),
        )

        assert len(metrics.drawdown_series) == 5
        assert metrics.var_series[0.95] == -0.05


class TestCalculateReturns:
    """Test return calculation functions."""

    def test_simple_returns(self):
        """Test simple return calculation."""
        prices = pd.Series([100, 105, 110, 108, 115])
        returns = calculate_returns(prices, method="simple")

        assert len(returns) == 4
        assert returns.iloc[0] == pytest.approx(0.05)  # (105-100)/100
        assert returns.iloc[1] == pytest.approx(0.047619, rel=1e-5)  # (110-105)/105

    def test_log_returns(self):
        """Test log return calculation."""
        prices = pd.Series([100, 105, 110])
        returns = calculate_returns(prices, method="log")

        assert len(returns) == 2
        assert returns.iloc[0] == pytest.approx(np.log(105 / 100))
        assert returns.iloc[1] == pytest.approx(np.log(110 / 105))

    def test_invalid_method(self):
        """Test invalid return calculation method."""
        prices = pd.Series([100, 105])

        with pytest.raises(ValueError):
            calculate_returns(prices, method="invalid")


class TestCalculateDrawdown:
    """Test drawdown calculation."""

    def test_drawdown_calculation(self):
        """Test basic drawdown calculation."""
        # Create cumulative returns with a drawdown
        cumulative_returns = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9, 1.05, 1.15])

        drawdown_series, max_dd, max_duration = calculate_drawdown(cumulative_returns)

        # Maximum should be at 1.2, drawdown to 0.9
        assert max_dd == pytest.approx(-0.25)  # (1.2 - 0.9) / 1.2 as negative
        assert max_duration > 0

        # Check drawdown series
        assert drawdown_series.iloc[0] == 0  # No drawdown at start
        assert drawdown_series.iloc[4] == pytest.approx(-0.25)  # Maximum drawdown

    def test_no_drawdown(self):
        """Test drawdown with monotonically increasing values."""
        cumulative_returns = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])

        drawdown_series, max_dd, max_duration = calculate_drawdown(cumulative_returns)

        assert max_dd == 0.0
        assert max_duration == 0
        assert all(drawdown_series == 0)


class TestCalculateVaRCVaR:
    """Test Value at Risk and Conditional Value at Risk calculations."""

    def test_var_cvar_normal_distribution(self):
        """Test VaR/CVaR on normally distributed returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.02, 1000))

        results = calculate_var_cvar(returns, confidence_levels=[0.95, 0.99])

        # Check structure
        assert "var" in results
        assert "cvar" in results
        assert 0.95 in results["var"]
        assert 0.99 in results["var"]

        # VaR should be positive (we take absolute value)
        assert results["var"][0.95] > 0
        assert results["var"][0.99] > 0

        # CVaR should be greater than VaR
        assert results["cvar"][0.95] >= results["var"][0.95]
        assert results["cvar"][0.99] >= results["var"][0.99]

    def test_var_cvar_no_tail_losses(self):
        """Test VaR/CVaR with no losses beyond VaR threshold."""
        # All returns are positive
        returns = pd.Series(np.random.uniform(0.01, 0.02, 100))

        results = calculate_var_cvar(returns, confidence_levels=[0.95])

        # VaR should be the 5th percentile (negative of return)
        # CVaR should be close to VaR (average of worst 5%)
        assert (
            results["var"][0.95] > 0
        )  # Returns are positive, so "losses" are negative returns
        assert results["cvar"][0.95] > 0
        # CVaR should be close to VaR but not exactly equal
        assert abs(results["cvar"][0.95] - results["var"][0.95]) < 0.005


class TestCalculateDownsideDeviation:
    """Test downside deviation calculation."""

    def test_downside_deviation_with_negative_returns(self):
        """Test downside deviation with mix of positive and negative returns."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.015])

        downside_dev = calculate_downside_deviation(returns, mar=0.0)

        # Should only consider negative returns
        negative_returns = returns[returns < 0]
        expected = np.sqrt(np.mean(negative_returns**2))

        assert downside_dev == pytest.approx(expected)

    def test_downside_deviation_no_negative_returns(self):
        """Test downside deviation with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])

        downside_dev = calculate_downside_deviation(returns, mar=0.0)

        assert downside_dev == 0.0

    def test_downside_deviation_with_mar(self):
        """Test downside deviation with minimum acceptable return."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        mar = 0.025  # Returns below this are considered downside

        downside_dev = calculate_downside_deviation(returns, mar=mar)

        # Returns 0.01 and 0.02 are below MAR
        assert downside_dev > 0


class TestCalculateOmegaRatio:
    """Test Omega ratio calculation."""

    def test_omega_ratio_positive_returns(self):
        """Test Omega ratio with mostly positive returns."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.04])

        omega = calculate_omega_ratio(returns, mar=0.0)

        # Sum of gains > sum of losses
        assert omega > 1.0

    def test_omega_ratio_negative_returns(self):
        """Test Omega ratio with mostly negative returns."""
        returns = pd.Series([-0.02, -0.01, 0.005, -0.03, -0.01])

        omega = calculate_omega_ratio(returns, mar=0.0)

        # Sum of losses > sum of gains
        assert omega < 1.0

    def test_omega_ratio_no_losses(self):
        """Test Omega ratio with no losses."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])

        omega = calculate_omega_ratio(returns, mar=0.0)

        # No losses, should be infinity
        assert omega == np.inf

    def test_omega_ratio_no_gains(self):
        """Test Omega ratio with no gains."""
        returns = pd.Series([-0.01, -0.02, -0.03])

        omega = calculate_omega_ratio(returns, mar=0.0)

        # No gains, ratio should be 0
        assert omega == 0.0


class TestCalculateBetaAlpha:
    """Test beta and alpha calculation."""

    def test_beta_alpha_calculation(self):
        """Test beta and alpha with correlated returns."""
        # Create correlated returns
        np.random.seed(42)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        portfolio_returns = 0.8 * benchmark_returns + np.random.normal(
            0.0002, 0.005, 252
        )

        beta, alpha = calculate_beta_alpha(
            portfolio_returns, benchmark_returns, risk_free_rate=0.02
        )

        # Beta should be close to 0.8 (the correlation factor)
        assert 0.7 < beta < 0.9

        # Alpha should be small but positive
        assert alpha > 0

    def test_beta_alpha_no_correlation(self):
        """Test beta with uncorrelated returns."""
        np.random.seed(42)
        benchmark_returns = pd.Series(np.random.normal(0, 0.01, 252))
        portfolio_returns = pd.Series(np.random.normal(0, 0.01, 252))

        beta, alpha = calculate_beta_alpha(portfolio_returns, benchmark_returns)

        # Beta should be close to 0
        assert abs(beta) < 0.3

    def test_beta_alpha_insufficient_data(self):
        """Test beta/alpha with insufficient data."""
        benchmark_returns = pd.Series([0.01])
        portfolio_returns = pd.Series([0.02])

        beta, alpha = calculate_beta_alpha(portfolio_returns, benchmark_returns)

        assert beta == 0.0
        assert alpha == 0.0

    def test_beta_alpha_zero_variance_benchmark(self):
        """Test beta with zero variance benchmark."""
        benchmark_returns = pd.Series([0.01] * 10)  # Constant returns
        portfolio_returns = pd.Series(np.random.normal(0.01, 0.02, 10))

        beta, alpha = calculate_beta_alpha(portfolio_returns, benchmark_returns)

        assert beta == 0.0  # Undefined beta defaults to 0


class TestCalculateRollingMetrics:
    """Test rolling metrics calculation."""

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 500))

        rolling = calculate_rolling_metrics(returns, window=252)

        assert "volatility" in rolling
        assert len(rolling["volatility"]) == len(returns)

        # Check annualization
        # Rolling volatility should be around 0.01 * sqrt(252)
        non_nan_vol = rolling["volatility"].dropna()
        assert non_nan_vol.mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.2)

    def test_rolling_sharpe(self):
        """Test rolling Sharpe ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 500))

        rolling = calculate_rolling_metrics(returns, window=252, risk_free_rate=0.02)

        assert "sharpe" in rolling
        assert len(rolling["sharpe"]) == len(returns)

        # Check that Sharpe ratio is reasonable
        non_nan_sharpe = rolling["sharpe"].dropna()
        assert len(non_nan_sharpe) > 0

    def test_rolling_beta(self):
        """Test rolling beta calculation."""
        np.random.seed(42)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
        portfolio_returns = 0.8 * benchmark_returns + np.random.normal(0, 0.005, 500)

        rolling = calculate_rolling_metrics(
            portfolio_returns, window=252, benchmark_returns=benchmark_returns
        )

        assert "beta" in rolling
        assert rolling["beta"] is not None

        # Average beta should be close to 0.8
        non_nan_beta = rolling["beta"].dropna()
        assert 0.7 < non_nan_beta.mean() < 0.9


class TestCalculateAllMetrics:
    """Test comprehensive metrics calculation."""

    def test_calculate_all_metrics_basic(self):
        """Test calculating all metrics with basic portfolio."""
        # Create simple portfolio value series
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        values = pd.Series(
            100000 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(), index=dates
        )

        perf_metrics, risk_metrics = calculate_all_metrics(values)

        # Check performance metrics
        assert isinstance(perf_metrics, PerformanceMetrics)
        assert perf_metrics.total_return != 0
        assert perf_metrics.annualized_return != 0
        assert perf_metrics.volatility > 0
        assert perf_metrics.max_drawdown <= 0

        # Check risk metrics
        assert isinstance(risk_metrics, RiskMetrics)
        assert len(risk_metrics.drawdown_series) > 0
        assert len(risk_metrics.rolling_volatility) > 0

    def test_calculate_all_metrics_with_benchmark(self):
        """Test calculating metrics with benchmark comparison."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

        # Create portfolio that outperforms benchmark
        benchmark_values = pd.Series(
            100000 * (1 + np.random.normal(0.0003, 0.008, 252)).cumprod(), index=dates
        )
        portfolio_values = pd.Series(
            100000 * (1 + np.random.normal(0.0007, 0.01, 252)).cumprod(), index=dates
        )

        perf_metrics, risk_metrics = calculate_all_metrics(
            portfolio_values, benchmark_values=benchmark_values
        )

        # Should have benchmark-relative metrics
        assert perf_metrics.beta is not None
        assert perf_metrics.alpha is not None
        assert perf_metrics.information_ratio is not None
        assert perf_metrics.tracking_error is not None

    def test_calculate_all_metrics_short_series(self):
        """Test metrics calculation with short time series."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        values = pd.Series([100000 * (1 + 0.01 * i) for i in range(10)], index=dates)

        perf_metrics, risk_metrics = calculate_all_metrics(values, periods_per_year=252)

        # Should still calculate basic metrics
        assert perf_metrics.total_return > 0
        assert perf_metrics.volatility >= 0

    def test_calculate_all_metrics_no_returns(self):
        """Test metrics calculation with single data point."""
        values = pd.Series([100000], index=[pd.Timestamp("2023-01-01")])

        with pytest.raises(ValueError, match="No return data available"):
            calculate_all_metrics(values)


class TestCompareStrategies:
    """Test strategy comparison functionality."""

    def test_compare_strategies_basic(self):
        """Test comparing multiple strategies."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

        # Create different strategy performances
        strategies = {
            "Buy and Hold": pd.Series(
                100000 * (1 + np.random.normal(0.0003, 0.01, 252)).cumprod(),
                index=dates,
            ),
            "Mean-Variance": pd.Series(
                100000 * (1 + np.random.normal(0.0005, 0.008, 252)).cumprod(),
                index=dates,
            ),
            "VQE": pd.Series(
                100000 * (1 + np.random.normal(0.0007, 0.012, 252)).cumprod(),
                index=dates,
            ),
        }

        comparison_df = compare_strategies(strategies)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert "Strategy" in comparison_df.columns
        assert "Total Return" in comparison_df.columns
        assert "Sharpe Ratio" in comparison_df.columns

    def test_compare_strategies_with_benchmark(self):
        """Test comparing strategies with benchmark."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

        benchmark = pd.Series(
            100000 * (1 + np.random.normal(0.0004, 0.009, 252)).cumprod(), index=dates
        )

        strategies = {
            "Strategy A": pd.Series(
                100000 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(),
                index=dates,
            )
        }

        comparison_df = compare_strategies(strategies, benchmark=benchmark)

        # Should include beta and alpha columns
        assert "Beta" in comparison_df.columns
        assert "Alpha" in comparison_df.columns

    def test_compare_strategies_error_handling(self):
        """Test strategy comparison with calculation errors."""
        strategies = {
            "Good Strategy": pd.Series([100000, 105000, 110000]),
            "Bad Strategy": pd.Series([100000]),  # Too short
        }

        # Should handle error gracefully and only return good strategy
        comparison_df = compare_strategies(strategies)

        assert len(comparison_df) == 1
        assert comparison_df.iloc[0]["Strategy"] == "Good Strategy"
