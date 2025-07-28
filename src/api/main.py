"""
QuantumEdge FastAPI application.

Main application entry point providing REST API endpoints for
quantum-inspired portfolio optimization.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .deps import get_redis_client, get_logger, get_optimization_manager, OptimizationManager
from .models import (
    HealthResponse, ErrorResponse, OptimizationResponse, 
    MeanVarianceRequest, RobustOptimizationRequest, VQERequest, QAOARequest,
    EfficientFrontierRequest, EfficientFrontierResponse,
    ComparisonRequest, ComparisonResponse,
    OptimizationStatus, OptimizationType
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Invalid input",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred" if not settings.debug else str(exc)
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(
    redis_client=Depends(get_redis_client),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Health check endpoint."""
    services = {}
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            services["redis"] = "healthy"
        except Exception:
            services["redis"] = "unhealthy"
    else:
        services["redis"] = "unavailable"
    
    # Check optimization algorithms
    try:
        from ..quantum_algorithms import QuantumCircuit
        services["quantum_algorithms"] = "available"
    except ImportError:
        services["quantum_algorithms"] = "unavailable"
    
    try:
        from ..optimization import CVXPY_AVAILABLE
        services["cvxpy"] = "available" if CVXPY_AVAILABLE else "unavailable"
    except ImportError:
        services["cvxpy"] = "unavailable"
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
        services=services,
        active_optimizations=len(optimization_manager.running_optimizations)
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "docs": f"{settings.docs_url}",
        "health": "/health"
    }


# Portfolio optimization endpoints
@app.post(f"{settings.api_prefix}/optimize/mean-variance", response_model=OptimizationResponse)
async def optimize_mean_variance(
    request: MeanVarianceRequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Optimize portfolio using mean-variance framework."""
    optimization_id = str(uuid.uuid4())
    
    # Check if we can start new optimization
    if not optimization_manager.can_start_optimization():
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent optimizations reached. Please try again later."
        )
    
    # Start optimization tracking
    optimization_manager.start_optimization(optimization_id)
    
    try:
        # Import optimizer
        from ..optimization import MeanVarianceOptimizer, CVXPY_AVAILABLE
        
        if not CVXPY_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="CVXPY not available. Mean-variance optimization requires CVXPY."
            )
        
        # Create optimizer
        optimizer = MeanVarianceOptimizer(
            risk_free_rate=request.risk_free_rate or settings.default_risk_free_rate
        )
        
        # Convert inputs
        import numpy as np
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Convert constraints if provided
        constraints = None
        if request.constraints:
            from ..optimization import PortfolioConstraints
            constraints = PortfolioConstraints(
                long_only=request.constraints.long_only,
                sum_to_one=request.constraints.sum_to_one,
                min_weight=request.constraints.min_weight,
                max_weight=request.constraints.max_weight,
                max_assets=request.constraints.max_assets,
                min_assets=request.constraints.min_assets,
                forbidden_assets=request.constraints.forbidden_assets,
                required_assets=request.constraints.required_assets,
                max_variance=request.constraints.max_variance,
                min_return=request.constraints.min_return,
                target_return=request.constraints.target_return,
                max_turnover=request.constraints.max_turnover,
                current_weights=np.array(request.constraints.current_weights) if request.constraints.current_weights else None
            )
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objective=request.objective,
            constraints=constraints,
            risk_aversion=request.risk_aversion
        )
        solve_time = time.time() - start_time
        
        # Create response
        portfolio_result = None
        if result.success and result.weights is not None:
            from .models import PortfolioResult
            portfolio_result = PortfolioResult(
                weights=result.weights.tolist(),
                expected_return=result.expected_return,
                expected_variance=result.expected_variance,
                volatility=np.sqrt(result.expected_variance),
                sharpe_ratio=result.sharpe_ratio,
                objective_value=result.objective_value
            )
        
        response = OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.COMPLETED if result.success else OptimizationStatus.FAILED,
            optimization_type=OptimizationType.MEAN_VARIANCE,
            solve_time=solve_time,
            success=result.success,
            message=f"Optimization {result.status}" if not result.success else None,
            portfolio=portfolio_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Mean-variance optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    finally:
        # Always clean up
        optimization_manager.finish_optimization(optimization_id)


@app.post(f"{settings.api_prefix}/optimize/robust", response_model=OptimizationResponse)
async def optimize_robust(
    request: RobustOptimizationRequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Optimize portfolio using robust optimization."""
    optimization_id = str(uuid.uuid4())
    
    if not optimization_manager.can_start_optimization():
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent optimizations reached. Please try again later."
        )
    
    optimization_manager.start_optimization(optimization_id)
    
    try:
        from ..optimization import RobustOptimizer, CVXPY_AVAILABLE
        
        if not CVXPY_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="CVXPY not available. Robust optimization requires CVXPY."
            )
        
        # Create optimizer
        optimizer = RobustOptimizer(
            risk_free_rate=request.risk_free_rate or settings.default_risk_free_rate,
            uncertainty_level=request.uncertainty_level
        )
        
        # Convert inputs
        import numpy as np
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        return_uncertainty = np.array(request.return_uncertainty) if request.return_uncertainty else None
        
        # Convert constraints if provided
        constraints = None
        if request.constraints:
            from ..optimization import PortfolioConstraints
            constraints = PortfolioConstraints(
                long_only=request.constraints.long_only,
                sum_to_one=request.constraints.sum_to_one,
                min_weight=request.constraints.min_weight,
                max_weight=request.constraints.max_weight,
                min_return=request.constraints.min_return,
                max_variance=request.constraints.max_variance
            )
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize_robust_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            return_uncertainty=return_uncertainty,
            constraints=constraints
        )
        solve_time = time.time() - start_time
        
        # Create response
        portfolio_result = None
        if result.success and result.weights is not None:
            from .models import PortfolioResult
            portfolio_result = PortfolioResult(
                weights=result.weights.tolist(),
                expected_return=result.expected_return,
                expected_variance=result.expected_variance,
                volatility=np.sqrt(result.expected_variance),
                sharpe_ratio=result.sharpe_ratio,
                objective_value=result.objective_value
            )
        
        response = OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.COMPLETED if result.success else OptimizationStatus.FAILED,
            optimization_type=OptimizationType.ROBUST,
            solve_time=solve_time,
            success=result.success,
            message=f"Optimization {result.status}" if not result.success else None,
            portfolio=portfolio_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Robust optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    finally:
        optimization_manager.finish_optimization(optimization_id)


@app.post(f"{settings.api_prefix}/quantum/vqe", response_model=OptimizationResponse)
async def optimize_vqe(
    request: VQERequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Optimize portfolio using Variational Quantum Eigensolver."""
    optimization_id = str(uuid.uuid4())
    
    if not optimization_manager.can_start_optimization():
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent optimizations reached. Please try again later."
        )
    
    optimization_manager.start_optimization(optimization_id)
    
    try:
        from ..quantum_algorithms import QuantumVQE
        
        # Create VQE optimizer
        num_assets = len(request.covariance_matrix)
        vqe = QuantumVQE(
            num_assets=num_assets,
            depth=request.depth,
            optimizer=request.optimizer,
            max_iterations=request.max_iterations
        )
        
        # Convert inputs
        import numpy as np
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Run optimization
        start_time = time.time()
        if request.num_eigenportfolios > 1:
            results = vqe.compute_multiple_eigenportfolios(
                covariance_matrix,
                num_eigenportfolios=request.num_eigenportfolios,
                num_random_starts=request.num_random_starts
            )
            result = results[0] if results else None  # Best result
        else:
            result = vqe.solve_eigenportfolio(covariance_matrix)
        
        solve_time = time.time() - start_time
        
        # Create response
        portfolio_result = None
        if result and result.success and result.eigenvector is not None:
            from .models import PortfolioResult
            # Calculate portfolio metrics for eigenvector
            expected_return = 0.0  # VQE doesn't optimize for return directly
            expected_variance = float(result.eigenvalue)
            
            portfolio_result = PortfolioResult(
                weights=result.eigenvector.tolist(),
                expected_return=expected_return,
                expected_variance=expected_variance,
                volatility=np.sqrt(expected_variance),
                sharpe_ratio=0.0,  # Not applicable for eigenportfolios
                objective_value=result.eigenvalue
            )
        
        response = OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.COMPLETED if result and result.success else OptimizationStatus.FAILED,
            optimization_type=OptimizationType.VQE,
            solve_time=solve_time,
            success=result.success if result else False,
            message="VQE eigenportfolio optimization completed" if result and result.success else "VQE optimization failed",
            portfolio=portfolio_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"VQE optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"VQE optimization failed: {str(e)}")
    
    finally:
        optimization_manager.finish_optimization(optimization_id)


@app.post(f"{settings.api_prefix}/quantum/qaoa", response_model=OptimizationResponse) 
async def optimize_qaoa(
    request: QAOARequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Optimize portfolio using Quantum Approximate Optimization Algorithm."""
    optimization_id = str(uuid.uuid4())
    
    if not optimization_manager.can_start_optimization():
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent optimizations reached. Please try again later."
        )
    
    optimization_manager.start_optimization(optimization_id)
    
    try:
        from ..quantum_algorithms import PortfolioQAOA
        
        # Create QAOA optimizer
        qaoa = PortfolioQAOA(
            num_assets=len(request.expected_returns),
            num_layers=request.num_layers,
            optimizer=request.optimizer,
            max_iterations=request.max_iterations
        )
        
        # Convert inputs
        import numpy as np
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Run optimization
        start_time = time.time()
        result = qaoa.solve_portfolio_selection(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_aversion=request.risk_aversion,
            cardinality_constraint=request.cardinality_constraint
        )
        solve_time = time.time() - start_time
        
        # Create response
        portfolio_result = None
        if result.success and result.optimal_portfolio is not None:
            from .models import PortfolioResult
            # Calculate portfolio metrics
            weights = result.optimal_portfolio.astype(float)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)  # Normalize
            
            expected_return = float(np.dot(weights, expected_returns))
            expected_variance = float(weights.T @ covariance_matrix @ weights)
            
            portfolio_result = PortfolioResult(
                weights=weights.tolist(),
                expected_return=expected_return,
                expected_variance=expected_variance,
                volatility=np.sqrt(expected_variance),
                sharpe_ratio=(expected_return - settings.default_risk_free_rate) / np.sqrt(expected_variance) if expected_variance > 0 else 0.0,
                objective_value=result.optimal_value
            )
        
        response = OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.COMPLETED if result.success else OptimizationStatus.FAILED,
            optimization_type=OptimizationType.QAOA,
            solve_time=solve_time,
            success=result.success,
            message="QAOA portfolio selection completed" if result.success else "QAOA optimization failed",
            portfolio=portfolio_result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"QAOA optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"QAOA optimization failed: {str(e)}")
    
    finally:
        optimization_manager.finish_optimization(optimization_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )