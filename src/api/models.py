"""
Pydantic models for API request/response validation.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class OptimizationType(str, Enum):
    """Portfolio optimization types."""

    MEAN_VARIANCE = "mean_variance"
    ROBUST = "robust"
    VQE = "quantum_vqe"
    QAOA = "quantum_qaoa"


class ObjectiveFunction(str, Enum):
    """Optimization objective functions."""

    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_CVAR = "minimize_cvar"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MAXIMIZE_SORTINO = "maximize_sortino"


class OptimizationStatus(str, Enum):
    """Optimization status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request Models


class PortfolioConstraintsModel(BaseModel):
    """Portfolio constraints specification."""

    long_only: bool = True
    sum_to_one: bool = True
    min_weight: Optional[float] = Field(None, ge=0, le=1)
    max_weight: Optional[float] = Field(None, ge=0, le=1)
    max_assets: Optional[int] = Field(None, ge=1)
    min_assets: Optional[int] = Field(None, ge=1)
    forbidden_assets: Optional[List[int]] = None
    required_assets: Optional[List[int]] = None
    max_variance: Optional[float] = Field(None, ge=0)
    min_return: Optional[float] = None
    target_return: Optional[float] = None
    max_turnover: Optional[float] = Field(None, ge=0, le=2)
    current_weights: Optional[List[float]] = None

    @validator("max_weight")
    def max_weight_valid(cls, v, values):
        if v is not None and values.get("min_weight") is not None:
            if v < values["min_weight"]:
                raise ValueError("max_weight must be >= min_weight")
        return v


class MeanVarianceRequest(BaseModel):
    """Mean-variance optimization request."""

    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_SHARPE
    risk_aversion: float = Field(1.0, ge=0)
    risk_free_rate: Optional[float] = Field(None, ge=0, le=1)
    constraints: Optional[PortfolioConstraintsModel] = None

    # Advanced objective parameters
    cvar_confidence: float = Field(0.05, ge=0.01, le=0.1)  # 5% CVaR by default
    returns_data: Optional[List[List[float]]] = (
        None  # Historical returns for CVaR/Sortino/Calmar
    )
    lookback_periods: Optional[int] = Field(
        252, ge=50, le=1000
    )  # For Calmar ratio calculation

    @validator("covariance_matrix")
    def covariance_matrix_valid(cls, v, values):
        n = len(values.get("expected_returns", []))
        if len(v) != n:
            raise ValueError(f"Covariance matrix must be {n}x{n}")
        for row in v:
            if len(row) != n:
                raise ValueError(f"Covariance matrix must be {n}x{n}")
        return v


class RobustOptimizationRequest(BaseModel):
    """Robust optimization request."""

    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    return_uncertainty: Optional[List[float]] = None
    uncertainty_level: float = Field(0.1, ge=0, le=1)
    risk_free_rate: Optional[float] = Field(None, ge=0, le=1)
    constraints: Optional[PortfolioConstraintsModel] = None


class VQERequest(BaseModel):
    """VQE optimization request."""

    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    depth: int = Field(2, ge=1, le=10)
    optimizer: str = Field("COBYLA", pattern="^(COBYLA|gradient|L-BFGS-B)$")
    max_iterations: int = Field(1000, ge=10, le=10000)
    num_eigenportfolios: int = Field(1, ge=1, le=5)
    num_random_starts: int = Field(3, ge=1, le=10)


class QAOARequest(BaseModel):
    """QAOA optimization request."""

    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: Optional[List[List[float]]] = Field(None)
    risk_aversion: float = Field(1.0, ge=0)
    # Accept legacy alias p_layers and map to num_layers
    num_layers: int = Field(1, ge=1, le=5)
    p_layers: Optional[int] = Field(None, ge=1, le=5)
    # Accept legacy risk_tolerance and map to risk_aversion
    risk_tolerance: Optional[float] = Field(None, ge=0)
    optimizer: str = Field("COBYLA", pattern="^(COBYLA|L-BFGS-B)$")
    max_iterations: int = Field(1000, ge=10, le=5000)
    cardinality_constraint: Optional[int] = Field(None, ge=1)

    @validator("num_layers", always=True)
    def map_p_layers(cls, v, values):
        p = values.get("p_layers")
        if p is not None:
            return p
        return v

    @validator("risk_aversion", always=True)
    def map_risk_tolerance(cls, v, values):
        rt = values.get("risk_tolerance")
        if rt is not None:
            return rt
        return v


class EfficientFrontierRequest(BaseModel):
    """Efficient frontier computation request."""

    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    num_points: int = Field(50, ge=10, le=200)
    risk_free_rate: Optional[float] = Field(None, ge=0, le=1)
    constraints: Optional[PortfolioConstraintsModel] = None


# Response Models


class PortfolioResult(BaseModel):
    """Portfolio optimization result."""

    weights: List[float]
    expected_return: float
    expected_variance: float
    volatility: float
    sharpe_ratio: float
    objective_value: float

    @validator("weights")
    def weights_valid(cls, v):
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to approximately 1.0")
        return v


class OptimizationResponse(BaseModel):
    """Base optimization response."""

    optimization_id: str
    status: OptimizationStatus
    optimization_type: OptimizationType
    solve_time: float
    success: bool
    message: Optional[str] = None
    portfolio: Optional[PortfolioResult] = None


class VQEResponse(OptimizationResponse):
    """VQE optimization response."""

    eigenvalue: Optional[float] = None
    optimization_history: Optional[List[float]] = None
    num_iterations: Optional[int] = None
    eigenportfolios: Optional[List[PortfolioResult]] = None


class QAOAResponse(OptimizationResponse):
    """QAOA optimization response."""

    optimal_value: Optional[float] = None
    optimization_history: Optional[List[float]] = None
    num_iterations: Optional[int] = None
    probability_distribution: Optional[List[float]] = None


class EfficientFrontierResponse(BaseModel):
    """Efficient frontier response."""

    optimization_id: str
    status: OptimizationStatus
    returns: List[float]
    risks: List[float]
    sharpe_ratios: List[float]
    portfolios: List[List[float]]
    solve_time: float
    success: bool
    message: Optional[str] = None


class ComparisonRequest(BaseModel):
    """Request for comparing multiple optimization methods."""

    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    methods: List[OptimizationType] = Field(..., min_items=2)
    risk_aversion: float = Field(1.0, ge=0)
    constraints: Optional[PortfolioConstraintsModel] = None


class ComparisonResponse(BaseModel):
    """Comparison of multiple optimization methods."""

    optimization_id: str
    status: OptimizationStatus
    results: Dict[str, OptimizationResponse]
    best_method: Optional[str] = None
    comparison_metrics: Optional[Dict[str, Dict[str, float]]] = None
    solve_time: float
    success: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    services: Dict[str, str]
    active_optimizations: int


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    optimization_id: Optional[str] = None


# Backtesting Models


class BacktestStrategy(str, Enum):
    """Backtesting strategy types."""

    BUY_AND_HOLD = "buy_and_hold"
    REBALANCING = "rebalancing"
    MEAN_VARIANCE = "mean_variance"
    VQE = "vqe"
    QAOA = "qaoa"


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class BacktestRequest(BaseModel):
    """Backtest configuration request."""

    # Strategy configuration
    strategy_type: BacktestStrategy
    symbols: List[str] = Field(..., min_items=2, max_items=20)
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

    # Portfolio settings
    initial_cash: float = Field(100000.0, gt=0)
    commission_rate: float = Field(0.001, ge=0, le=0.1)
    min_commission: float = Field(1.0, ge=0)

    # Strategy-specific parameters
    rebalance_frequency: Optional[RebalanceFrequency] = RebalanceFrequency.MONTHLY
    target_weights: Optional[Dict[str, float]] = None
    risk_aversion: Optional[float] = Field(1.0, gt=0)
    lookback_period: Optional[int] = Field(252, ge=20, le=1000)

    # VQE/QAOA specific
    depth: Optional[int] = Field(3, ge=1, le=10)
    num_layers: Optional[int] = Field(3, ge=1, le=10)
    max_iterations: Optional[int] = Field(100, ge=10, le=1000)
    cardinality_constraint: Optional[int] = None

    # Risk management
    min_weight: float = Field(0.0, ge=0, le=1)
    max_weight: float = Field(1.0, ge=0, le=1)
    allow_short_selling: bool = False

    # Benchmark
    benchmark_symbol: Optional[str] = "SPY"
    risk_free_rate: float = Field(0.02, ge=0, le=0.2)

    @validator("target_weights")
    def validate_target_weights(cls, v, values):
        if v is not None:
            symbols = values.get("symbols", [])
            if set(v.keys()) != set(symbols):
                raise ValueError("target_weights keys must match symbols")
            if abs(sum(v.values()) - 1.0) > 1e-6:
                raise ValueError("target_weights must sum to 1.0")
        return v


class PerformanceMetricsModel(BaseModel):
    """Performance metrics model."""

    total_return: float
    annualized_return: float
    cagr: float
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    downside_deviation: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class BacktestSummary(BaseModel):
    """Summary statistics for backtest."""

    initial_value: float
    final_value: float
    total_return: float
    num_rebalances: int
    total_commissions: float


class BacktestResponse(BaseModel):
    """Backtest execution response."""

    backtest_id: str
    success: bool
    execution_time: float

    # Results (if successful)
    performance_metrics: Optional[PerformanceMetricsModel] = None
    summary: Optional[BacktestSummary] = None

    # Data arrays (optional, for charts)
    portfolio_values: Optional[List[Dict[str, Any]]] = None
    portfolio_weights: Optional[List[Dict[str, Any]]] = None
    benchmark_values: Optional[List[Dict[str, Any]]] = None

    # Error information
    error_message: Optional[str] = None

    # Metadata
    config: Optional[Dict[str, Any]] = None


class CompareStrategiesRequest(BaseModel):
    """Request to compare multiple strategies."""

    strategies: List[BacktestRequest] = Field(..., min_items=2, max_items=5)
    strategy_names: List[str] = Field(..., min_items=2, max_items=5)

    @validator("strategy_names", allow_reuse=True)
    def validate_strategy_names(cls, v, values):
        strategies = values.get("strategies", [])
        if len(v) != len(strategies):
            raise ValueError("strategy_names length must match strategies length")
        return v


class StrategyComparisonResult(BaseModel):
    """Individual strategy result in comparison."""

    name: str
    success: bool
    performance_metrics: Optional[PerformanceMetricsModel] = None
    summary: Optional[BacktestSummary] = None
    error_message: Optional[str] = None


class CompareStrategiesResponse(BaseModel):
    """Response for strategy comparison."""

    comparison_id: str
    success: bool
    execution_time: float

    # Individual results
    results: List[StrategyComparisonResult] = []

    # Comparison data
    performance_comparison: Optional[List[Dict[str, Any]]] = None
    portfolio_values: Optional[Dict[str, List[Dict[str, Any]]]] = None

    # Error information
    error_message: Optional[str] = None
