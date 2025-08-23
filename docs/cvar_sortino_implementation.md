# CVaR and Sortino Ratio Implementation

## Overview

We have successfully implemented Conditional Value at Risk (CVaR) and Sortino ratio objectives in the QuantumEdge portfolio optimization framework. These risk-adjusted objectives provide more sophisticated portfolio optimization capabilities beyond the traditional Sharpe ratio.

## Key Features

### 1. CVaR (Conditional Value at Risk) Optimization

CVaR measures the expected loss in the worst α% of cases. For example, 5% CVaR represents the average loss in the worst 5% of scenarios.

**Implementation Details:**
- Uses historical returns data to calculate tail risk
- Falls back to normal distribution approximation if no historical data provided
- Minimizes expected tail losses for more conservative portfolios
- API endpoint supports configurable confidence levels (1-10%)

### 2. Sortino Ratio Optimization

The Sortino ratio is similar to the Sharpe ratio but only considers downside volatility, making it more appropriate for non-symmetric return distributions.

**Implementation Details:**
- Focuses on downside deviation below target return (default: risk-free rate)
- Uses historical returns data for accurate downside risk calculation
- Maximizes return relative to downside risk only
- Better for portfolios with asymmetric return profiles

## API Usage

### Mean-Variance Optimizer Endpoint

```python
POST /api/v1/optimize/mean-variance

{
    "expected_returns": [0.08, 0.10, 0.12, 0.07, 0.09],
    "covariance_matrix": [[...], ...],
    "objective": "minimize_cvar",  # or "maximize_sortino"
    "returns_data": [[...], ...],  # Historical returns (optional)
    "cvar_confidence": 0.05,       # 5% CVaR (95% confidence)
    "constraints": {
        "long_only": true,
        "sum_to_one": true
    }
}
```

### Classical Optimizer Endpoint

```python
POST /api/v1/optimize/classical?method=genetic_algorithm

{
    "expected_returns": [...],
    "covariance_matrix": [[...], ...],
    "objective": "minimize_cvar",
    "returns_data": [[...], ...],
    "constraints": {...}
}
```

## Python Usage

### Using MeanVarianceOptimizer

```python
from src.optimization.mean_variance import MeanVarianceOptimizer, ObjectiveType, PortfolioConstraints

optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

# CVaR optimization
result = optimizer.optimize_portfolio(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    objective=ObjectiveType.MINIMIZE_CVAR,
    constraints=constraints,
    returns_data=historical_returns,  # n_periods x n_assets
    cvar_confidence=0.05  # 5% CVaR
)

# Sortino optimization
result = optimizer.optimize_portfolio(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    objective=ObjectiveType.MAXIMIZE_SORTINO,
    constraints=constraints,
    returns_data=historical_returns
)
```

### Using Classical Solvers

```python
from src.optimization.classical_solvers import (
    GeneticAlgorithmOptimizer, OptimizerParameters, ObjectiveType
)

params = OptimizerParameters(max_iterations=1000, population_size=50)
optimizer = GeneticAlgorithmOptimizer(params)

result = optimizer.optimize(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    constraints=constraints,
    objective=ObjectiveType.MINIMIZE_CVAR,
    returns_data=historical_returns
)
```

## Implementation Details

### CVaR Calculation

The CVaR implementation uses the following approach:

1. **With Historical Data**: Calculates actual portfolio returns, finds the VaR threshold at the specified confidence level, and computes the average of returns below this threshold.

2. **Without Historical Data**: Uses a normal distribution approximation based on the Cornish-Fisher expansion.

### Sortino Ratio Calculation

The Sortino ratio implementation:

1. **With Historical Data**: Calculates actual downside deviations from target return (default: risk-free rate).

2. **Without Historical Data**: Approximates downside deviation as a fraction of total volatility.

## Testing

Comprehensive test suites have been added:

- **Unit Tests**: `tests/unit/test_cvar_sortino.py`
- **Integration Tests**: `tests/integration/test_cvar_sortino_api.py`

All tests verify:
- Correct objective function calculations
- API endpoint functionality
- Classical solver support
- Synthetic data generation for missing historical returns

## Benefits

1. **Better Risk Management**: CVaR provides a more comprehensive view of tail risk than VaR alone.

2. **Asymmetric Risk Consideration**: Sortino ratio is more appropriate for strategies with non-normal return distributions.

3. **Flexible Implementation**: Works with both CVXPY optimizer and classical metaheuristics (GA, SA, PSO).

4. **Backward Compatible**: Existing code continues to work with new optional parameters.

## Future Enhancements

1. **Robust CVaR**: Incorporate parameter uncertainty in CVaR calculations
2. **Multi-Period CVaR**: Extend to multi-period risk measures
3. **Conditional Sortino**: Combine CVaR concepts with Sortino ratio
4. **Risk Parity with CVaR**: Implement risk parity using CVaR as risk measure