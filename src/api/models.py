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
    VQE = "vqe"
    QAOA = "qaoa"


class ObjectiveFunction(str, Enum):
    """Optimization objective functions."""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_UTILITY = "maximize_utility"


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
    
    @validator('max_weight')
    def max_weight_valid(cls, v, values):
        if v is not None and values.get('min_weight') is not None:
            if v < values['min_weight']:
                raise ValueError('max_weight must be >= min_weight')
        return v


class MeanVarianceRequest(BaseModel):
    """Mean-variance optimization request."""
    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_SHARPE
    risk_aversion: float = Field(1.0, ge=0)
    risk_free_rate: Optional[float] = Field(None, ge=0, le=1)
    constraints: Optional[PortfolioConstraintsModel] = None
    
    @validator('covariance_matrix')
    def covariance_matrix_valid(cls, v, values):
        n = len(values.get('expected_returns', []))
        if len(v) != n:
            raise ValueError(f'Covariance matrix must be {n}x{n}')
        for row in v:
            if len(row) != n:
                raise ValueError(f'Covariance matrix must be {n}x{n}')
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
    optimizer: str = Field("COBYLA", regex="^(COBYLA|gradient|L-BFGS-B)$")
    max_iterations: int = Field(1000, ge=10, le=10000)
    num_eigenportfolios: int = Field(1, ge=1, le=5)
    num_random_starts: int = Field(3, ge=1, le=10)


class QAOARequest(BaseModel):
    """QAOA optimization request."""
    expected_returns: List[float] = Field(..., min_items=2)
    covariance_matrix: List[List[float]] = Field(..., min_items=2)
    risk_aversion: float = Field(1.0, ge=0)
    num_layers: int = Field(1, ge=1, le=5)
    optimizer: str = Field("COBYLA", regex="^(COBYLA|L-BFGS-B)$")
    max_iterations: int = Field(1000, ge=10, le=5000)
    cardinality_constraint: Optional[int] = Field(None, ge=1)


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
    
    @validator('weights')
    def weights_valid(cls, v):
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError('Weights must sum to approximately 1.0')
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