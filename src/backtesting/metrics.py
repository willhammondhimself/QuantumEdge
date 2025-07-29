"""
Performance and risk metrics calculation for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Downside risk metrics
    downside_deviation: float
    var_95: float  # Value at Risk at 95%
    cvar_95: float  # Conditional Value at Risk at 95%
    
    # Additional metrics
    skewness: float
    kurtosis: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass 
class RiskMetrics:
    """Detailed risk analysis metrics."""
    # Drawdown analysis
    drawdown_series: pd.Series
    drawdown_periods: List[Tuple[datetime, datetime, float]]
    
    # Rolling metrics
    rolling_volatility: pd.Series
    rolling_sharpe: pd.Series
    rolling_beta: Optional[pd.Series] = None
    
    # Tail risk
    var_series: Dict[float, float]  # confidence -> VaR
    cvar_series: Dict[float, float]  # confidence -> CVaR
    
    # Risk decomposition
    asset_contribution: Optional[Dict[str, float]] = None
    factor_exposure: Optional[Dict[str, float]] = None


def calculate_returns(prices: Union[pd.Series, pd.DataFrame], method: str = 'simple') -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series or DataFrame
        method: 'simple' or 'log'
    
    Returns:
        Returns series or DataFrame
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("method must be 'simple' or 'log'")


def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series and maximum drawdown.
    
    Args:
        cumulative_returns: Cumulative return series
    
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_duration)
    """
    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    # Calculate maximum drawdown duration
    max_duration = 0
    current_duration = 0
    
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return drawdown, abs(max_dd), max_duration


def calculate_var_cvar(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Dict[float, float]]:
    """
    Calculate Value at Risk and Conditional Value at Risk.
    
    Args:
        returns: Return series
        confidence_levels: List of confidence levels
    
    Returns:
        Dictionary with VaR and CVaR for each confidence level
    """
    var_dict = {}
    cvar_dict = {}
    
    for confidence in confidence_levels:
        # Calculate VaR (percentile of loss distribution)
        var = np.percentile(returns, (1 - confidence) * 100)
        var_dict[confidence] = abs(var)
        
        # Calculate CVaR (mean of losses beyond VaR)
        tail_losses = returns[returns <= var]
        if len(tail_losses) > 0:
            cvar = tail_losses.mean()
            cvar_dict[confidence] = abs(cvar)
        else:
            cvar_dict[confidence] = abs(var)
    
    return {'var': var_dict, 'cvar': cvar_dict}


def calculate_downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Calculate downside deviation.
    
    Args:
        returns: Return series
        mar: Minimum acceptable return (default 0)
    
    Returns:
        Downside deviation
    """
    downside_returns = returns[returns < mar]
    if len(downside_returns) == 0:
        return 0.0
    
    return np.sqrt(np.mean((downside_returns - mar) ** 2))


def calculate_omega_ratio(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Return series
        mar: Minimum acceptable return threshold
    
    Returns:
        Omega ratio
    """
    excess_returns = returns - mar
    gains = excess_returns[excess_returns > 0].sum()
    losses = abs(excess_returns[excess_returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    
    return gains / losses


def calculate_beta_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> Tuple[float, float]:
    """
    Calculate beta and alpha relative to benchmark.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Tuple of (beta, alpha)
    """
    # Align series
    aligned_data = pd.DataFrame({
        'portfolio': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return 0.0, 0.0
    
    portfolio_returns = aligned_data['portfolio']
    benchmark_returns = aligned_data['benchmark']
    
    # Calculate beta using covariance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns, ddof=1)
    
    if benchmark_variance == 0:
        beta = 0.0
    else:
        beta = covariance / benchmark_variance
    
    # Calculate alpha
    portfolio_mean = portfolio_returns.mean()
    benchmark_mean = benchmark_returns.mean()
    
    # Convert risk-free rate to same frequency as returns
    periods_per_year = 252  # Assume daily returns
    rf_period = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    
    alpha = portfolio_mean - rf_period - beta * (benchmark_mean - rf_period)
    
    return beta, alpha


def calculate_rolling_metrics(
    returns: pd.Series, 
    window: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, pd.Series]:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Return series
        window: Rolling window size
        benchmark_returns: Optional benchmark for beta calculation
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Dictionary of rolling metrics
    """
    # Convert risk-free rate to period rate
    periods_per_year = 252
    rf_period = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    
    rolling_metrics = {}
    
    # Rolling volatility (annualized)
    rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(periods_per_year)
    
    # Rolling Sharpe ratio
    excess_returns = returns - rf_period
    rolling_metrics['sharpe'] = (
        excess_returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(periods_per_year)
    )
    
    # Rolling beta (if benchmark provided)
    if benchmark_returns is not None:
        rolling_betas = []
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        for i in range(window, len(aligned_data) + 1):
            window_data = aligned_data.iloc[i-window:i]
            if len(window_data) >= window:
                portfolio_window = window_data['portfolio']
                benchmark_window = window_data['benchmark']
                
                covariance = np.cov(portfolio_window, benchmark_window)[0, 1]
                benchmark_variance = np.var(benchmark_window, ddof=1)
                
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                else:
                    beta = 0.0
                
                rolling_betas.append(beta)
            else:
                rolling_betas.append(np.nan)
        
        # Pad with NaNs for alignment
        rolling_betas = [np.nan] * (window - 1) + rolling_betas
        rolling_metrics['beta'] = pd.Series(rolling_betas, index=aligned_data.index)
    
    return rolling_metrics


def calculate_all_metrics(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Tuple[PerformanceMetrics, RiskMetrics]:
    """
    Calculate comprehensive performance and risk metrics.
    
    Args:
        portfolio_values: Portfolio value time series
        benchmark_values: Optional benchmark value series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Tuple of (PerformanceMetrics, RiskMetrics)
    """
    # Calculate returns
    returns = calculate_returns(portfolio_values, method='simple')
    
    if len(returns) == 0:
        raise ValueError("No return data available")
    
    # Basic return metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    # Annualized return
    total_periods = len(portfolio_values)
    years = total_periods / periods_per_year
    
    if years > 0:
        cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years) - 1
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    else:
        cagr = 0.0
        annualized_return = 0.0
    
    # Risk metrics
    volatility = returns.std()
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    drawdown_series, max_drawdown, max_dd_duration = calculate_drawdown(cumulative_returns)
    
    # Risk-adjusted returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if volatility > 0:
        sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0.0
    
    # Sortino ratio
    downside_deviation = calculate_downside_deviation(returns)
    if downside_deviation > 0:
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
    else:
        sortino_ratio = 0.0
    
    # Calmar ratio
    if max_drawdown > 0:
        calmar_ratio = cagr / max_drawdown
    else:
        calmar_ratio = 0.0
    
    # Omega ratio
    omega_ratio = calculate_omega_ratio(returns)
    
    # VaR and CVaR
    risk_metrics_dict = calculate_var_cvar(returns)
    var_95 = risk_metrics_dict['var'].get(0.95, 0.0)
    cvar_95 = risk_metrics_dict['cvar'].get(0.95, 0.0)
    
    # Higher moments
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Benchmark-relative metrics
    beta = None
    alpha = None
    information_ratio = None
    tracking_error = None
    
    if benchmark_values is not None:
        benchmark_returns = calculate_returns(benchmark_values, method='simple')
        
        # Align returns
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) > 1:
            portfolio_aligned = aligned_data['portfolio']
            benchmark_aligned = aligned_data['benchmark']
            
            # Beta and alpha
            beta, alpha = calculate_beta_alpha(
                portfolio_aligned, 
                benchmark_aligned, 
                risk_free_rate
            )
            
            # Tracking error and information ratio
            active_returns = portfolio_aligned - benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(periods_per_year)
            
            if tracking_error > 0:
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year)
            else:
                information_ratio = 0.0
    
    # Create performance metrics object
    performance_metrics = PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        cagr=cagr,
        volatility=volatility,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        omega_ratio=omega_ratio,
        downside_deviation=downside_deviation,
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skewness,
        kurtosis=kurtosis,
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
        tracking_error=tracking_error
    )
    
    # Calculate rolling metrics
    rolling_metrics = calculate_rolling_metrics(
        returns, 
        window=min(252, len(returns)//4),  # 1 year or 1/4 of data
        benchmark_returns=benchmark_returns if benchmark_values is not None else None,
        risk_free_rate=risk_free_rate
    )
    
    # Create risk metrics object
    risk_metrics = RiskMetrics(
        drawdown_series=drawdown_series,
        drawdown_periods=[],  # TODO: Extract drawdown periods
        rolling_volatility=rolling_metrics.get('volatility', pd.Series()),
        rolling_sharpe=rolling_metrics.get('sharpe', pd.Series()),
        rolling_beta=rolling_metrics.get('beta'),
        var_series=risk_metrics_dict['var'],
        cvar_series=risk_metrics_dict['cvar']
    )
    
    return performance_metrics, risk_metrics


def compare_strategies(
    strategies: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compare multiple strategies using key metrics.
    
    Args:
        strategies: Dictionary of strategy name -> value series
        benchmark: Optional benchmark series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, values in strategies.items():
        try:
            performance, risk = calculate_all_metrics(
                values, 
                benchmark, 
                risk_free_rate
            )
            
            results.append({
                'Strategy': name,
                'Total Return': f"{performance.total_return:.2%}",
                'CAGR': f"{performance.cagr:.2%}",
                'Volatility': f"{performance.annualized_volatility:.2%}",
                'Sharpe Ratio': f"{performance.sharpe_ratio:.3f}",
                'Sortino Ratio': f"{performance.sortino_ratio:.3f}",
                'Calmar Ratio': f"{performance.calmar_ratio:.3f}",
                'Max Drawdown': f"{performance.max_drawdown:.2%}",
                'VaR (95%)': f"{performance.var_95:.2%}",
                'CVaR (95%)': f"{performance.cvar_95:.2%}",
                'Beta': f"{performance.beta:.3f}" if performance.beta is not None else 'N/A',
                'Alpha': f"{performance.alpha:.3f}" if performance.alpha is not None else 'N/A'
            })
            
        except Exception as e:
            print(f"Error calculating metrics for {name}: {e}")
            continue
    
    return pd.DataFrame(results)