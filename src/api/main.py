"""
QuantumEdge FastAPI application.

Main application entry point providing REST API endpoints for
quantum-inspired portfolio optimization.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .deps import get_redis_client, get_logger, get_optimization_manager, OptimizationManager
from .models import (
    HealthResponse, ErrorResponse, OptimizationResponse, 
    MeanVarianceRequest, RobustOptimizationRequest, VQERequest, QAOARequest,
    EfficientFrontierRequest, EfficientFrontierResponse,
    ComparisonRequest, ComparisonResponse,
    OptimizationStatus, OptimizationType,
    # Backtesting models
    BacktestRequest, BacktestResponse, CompareStrategiesRequest, CompareStrategiesResponse,
    BacktestStrategy, PerformanceMetricsModel, BacktestSummary
)
from ..data.models import DataFrequency
from ..data.yahoo_finance import YahooFinanceProvider

# WebSocket imports
from ..streaming.websocket import connection_manager, WebSocketMessage, MessageType
from ..streaming.data_pipeline import DataPipeline
from ..streaming.portfolio_monitor import PortfolioMonitor, Portfolio, AlertRule
from ..streaming.market_data_source import create_market_data_source

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

# Initialize WebSocket components
data_pipeline = DataPipeline(connection_manager)
portfolio_monitor = PortfolioMonitor(data_pipeline)

# Initialize market data source
market_data_source = create_market_data_source(use_simulation=settings.environment == "development")
data_pipeline.register_market_data_source("primary", market_data_source.get_market_updates)

# Start data pipeline on app startup
@app.on_event("startup")
async def startup_event():
    """Initialize real-time data pipeline on startup."""
    # Initialize WebSocket background tasks
    await connection_manager._setup_background_tasks()
    
    # Start data pipeline
    await data_pipeline.start()
    logger.info("Real-time data pipeline started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on app shutdown."""
    await data_pipeline.stop()
    await connection_manager.cleanup()
    logger.info("Real-time services shut down")


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
        
        # Convert returns data if provided for advanced objectives
        returns_data = None
        if request.returns_data:
            returns_data = np.array(request.returns_data)
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objective=request.objective,
            constraints=constraints,
            risk_aversion=request.risk_aversion,
            returns_data=returns_data,
            cvar_confidence=request.cvar_confidence,
            lookback_periods=request.lookback_periods
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


# Market Data endpoints
@app.get(f"{settings.api_prefix}/market/asset/{{symbol}}")
async def get_asset_info(symbol: str):
    """Get asset information."""
    try:
        provider = YahooFinanceProvider()
        asset = await provider.get_asset_info(symbol)
        
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset {symbol} not found")
        
        return {
            "symbol": asset.symbol,
            "name": asset.name,
            "asset_type": asset.asset_type,
            "exchange": asset.exchange,
            "currency": asset.currency,
            "sector": asset.sector,
            "industry": asset.industry,
            "description": asset.description,
            "market_cap": asset.market_cap
        }
    except Exception as e:
        logger.error(f"Failed to get asset info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch asset info: {str(e)}")


@app.get(f"{settings.api_prefix}/market/price/{{symbol}}")
async def get_current_price(symbol: str):
    """Get current price for a symbol."""
    try:
        provider = YahooFinanceProvider()
        price = await provider.get_current_price(symbol)
        
        if not price:
            raise HTTPException(status_code=404, detail=f"Price data for {symbol} not found")
        
        return {
            "symbol": price.symbol,
            "timestamp": price.timestamp.isoformat(),
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume,
            "adjusted_close": price.adjusted_close
        }
    except Exception as e:
        logger.error(f"Failed to get current price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch current price: {str(e)}")


@app.post(f"{settings.api_prefix}/market/prices")
async def get_current_prices(symbols: List[str]):
    """Get current prices for multiple symbols."""
    try:
        provider = YahooFinanceProvider()
        prices = await provider.get_current_prices(symbols)
        
        result = {}
        for symbol, price in prices.items():
            if price:
                result[symbol] = {
                    "symbol": price.symbol,
                    "timestamp": price.timestamp.isoformat(),
                    "open": price.open,
                    "high": price.high,
                    "low": price.low,
                    "close": price.close,
                    "volume": price.volume,
                    "adjusted_close": price.adjusted_close
                }
            else:
                result[symbol] = None
        
        return result
    except Exception as e:
        logger.error(f"Failed to get current prices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch current prices: {str(e)}")


@app.get(f"{settings.api_prefix}/market/history/{{symbol}}")
async def get_historical_data(
    symbol: str,
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    frequency: DataFrequency = Query(DataFrequency.DAILY, description="Data frequency")
):
    """Get historical market data for a symbol."""
    try:
        provider = YahooFinanceProvider()
        market_data = await provider.get_historical_data(symbol, start_date, end_date, frequency)
        
        if not market_data:
            raise HTTPException(status_code=404, detail=f"Historical data for {symbol} not found")
        
        # Convert price data to dictionary format
        price_data = []
        for price in market_data.data:
            price_data.append({
                "timestamp": price.timestamp.isoformat(),
                "open": price.open,
                "high": price.high,
                "low": price.low,
                "close": price.close,
                "volume": price.volume,
                "adjusted_close": price.adjusted_close
            })
        
        return {
            "symbol": market_data.symbol,
            "frequency": market_data.frequency,
            "start_date": market_data.start_date.isoformat(),
            "end_date": market_data.end_date.isoformat(),
            "source": market_data.source,
            "data": price_data,
            "count": len(price_data)
        }
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")


@app.get(f"{settings.api_prefix}/market/metrics")
async def get_market_metrics():
    """Get current market-wide metrics."""
    try:
        provider = YahooFinanceProvider()
        metrics = await provider.get_market_metrics()
        
        if not metrics:
            raise HTTPException(status_code=500, detail="Failed to retrieve market metrics")
        
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "vix": metrics.vix,
            "spy_return": metrics.spy_return,
            "bond_yield_10y": metrics.bond_yield_10y,
            "dxy": metrics.dxy,
            "gold_price": metrics.gold_price,
            "oil_price": metrics.oil_price
        }
    except Exception as e:
        logger.error(f"Failed to get market metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market metrics: {str(e)}")


@app.get(f"{settings.api_prefix}/market/health")
async def check_market_data_health():
    """Check health of market data providers."""
    try:
        provider = YahooFinanceProvider()
        is_healthy = await provider.health_check()
        
        return {
            "provider": provider.name,
            "healthy": is_healthy,
            "rate_limit_per_minute": provider.rate_limit_per_minute,
            "cache_stats": provider.get_cache_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Market data health check failed: {e}")
        return {
            "provider": "Yahoo Finance",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Backtesting endpoints
@app.post(f"{settings.api_prefix}/backtest/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Run a portfolio backtesting simulation."""
    backtest_id = str(uuid.uuid4())
    
    try:
        # Import backtesting components
        from ..backtesting import (
            BacktestEngine, BacktestConfig, BuyAndHoldStrategy, 
            RebalancingStrategy, MeanVarianceStrategy, VQEStrategy, QAOAStrategy
        )
        from datetime import datetime
        
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
        
        # Create strategy based on type
        if request.strategy_type == BacktestStrategy.BUY_AND_HOLD:
            strategy = BuyAndHoldStrategy(
                name="Buy and Hold",
                symbols=request.symbols,
                weights=request.target_weights
            )
        elif request.strategy_type == BacktestStrategy.REBALANCING:
            strategy = RebalancingStrategy(
                name="Rebalancing",
                symbols=request.symbols,
                frequency=request.rebalance_frequency,
                target_weights=request.target_weights
            )
        elif request.strategy_type == BacktestStrategy.MEAN_VARIANCE:
            strategy = MeanVarianceStrategy(
                name="Mean-Variance",
                symbols=request.symbols,
                lookback_period=request.lookback_period,
                frequency=request.rebalance_frequency,
                risk_aversion=request.risk_aversion,
                risk_free_rate=request.risk_free_rate
            )
        elif request.strategy_type == BacktestStrategy.VQE:
            strategy = VQEStrategy(
                name="VQE",
                symbols=request.symbols,
                lookback_period=request.lookback_period,
                frequency=request.rebalance_frequency,
                depth=request.depth,
                max_iterations=request.max_iterations
            )
        elif request.strategy_type == BacktestStrategy.QAOA:
            strategy = QAOAStrategy(
                name="QAOA",
                symbols=request.symbols,
                lookback_period=request.lookback_period,
                frequency=request.rebalance_frequency,
                num_layers=request.num_layers,
                risk_aversion=request.risk_aversion,
                cardinality_constraint=request.cardinality_constraint
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported strategy type: {request.strategy_type}"
            )
        
        # Create backtest configuration
        config = BacktestConfig(
            strategy=strategy,
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=request.initial_cash,
            commission_rate=request.commission_rate,
            min_commission=request.min_commission,
            rebalance_frequency=request.rebalance_frequency.value,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            benchmark_symbol=request.benchmark_symbol,
            risk_free_rate=request.risk_free_rate,
            lookback_period=request.lookback_period,
            allow_short_selling=request.allow_short_selling
        )
        
        # Create backtesting engine
        engine = BacktestEngine()
        
        # Run backtest
        start_time = time.time()
        result = await engine.run_backtest(config)
        execution_time = time.time() - start_time
        
        if not result.success:
            return BacktestResponse(
                backtest_id=backtest_id,
                success=False,
                execution_time=execution_time,
                error_message=result.error_message
            )
        
        # Convert performance metrics
        performance_metrics = None
        if result.performance_metrics:
            performance_metrics = PerformanceMetricsModel(
                total_return=result.performance_metrics.total_return,
                annualized_return=result.performance_metrics.annualized_return,
                cagr=result.performance_metrics.cagr,
                volatility=result.performance_metrics.volatility,
                annualized_volatility=result.performance_metrics.annualized_volatility,
                max_drawdown=result.performance_metrics.max_drawdown,
                max_drawdown_duration=result.performance_metrics.max_drawdown_duration,
                sharpe_ratio=result.performance_metrics.sharpe_ratio,
                sortino_ratio=result.performance_metrics.sortino_ratio,
                calmar_ratio=result.performance_metrics.calmar_ratio,
                omega_ratio=result.performance_metrics.omega_ratio,
                downside_deviation=result.performance_metrics.downside_deviation,
                var_95=result.performance_metrics.var_95,
                cvar_95=result.performance_metrics.cvar_95,
                skewness=result.performance_metrics.skewness,
                kurtosis=result.performance_metrics.kurtosis,
                beta=result.performance_metrics.beta,
                alpha=result.performance_metrics.alpha,
                information_ratio=result.performance_metrics.information_ratio,
                tracking_error=result.performance_metrics.tracking_error
            )
        
        # Create summary
        summary = BacktestSummary(
            initial_value=request.initial_cash,
            final_value=result.portfolio_values.iloc[-1] if not result.portfolio_values.empty else request.initial_cash,
            total_return=result.performance_metrics.total_return if result.performance_metrics else 0.0,
            num_rebalances=len(result.rebalance_dates),
            total_commissions=result.transactions['commission'].sum() if not result.transactions.empty else 0.0
        )
        
        # Convert time series data for charts
        portfolio_values = None
        portfolio_weights = None
        benchmark_values = None
        
        if not result.portfolio_values.empty:
            portfolio_values = [
                {"date": date.isoformat(), "value": float(value)}
                for date, value in result.portfolio_values.items()
            ]
        
        if not result.portfolio_weights.empty:
            portfolio_weights = []
            for date, row in result.portfolio_weights.iterrows():
                weight_data = {"date": date.isoformat()}
                for symbol in request.symbols:
                    weight_data[symbol] = float(row.get(symbol, 0.0))
                portfolio_weights.append(weight_data)
        
        if result.benchmark_values is not None and not result.benchmark_values.empty:
            benchmark_values = [
                {"date": date.isoformat(), "value": float(value)}
                for date, value in result.benchmark_values.items()
            ]
        
        return BacktestResponse(
            backtest_id=backtest_id,
            success=True,
            execution_time=execution_time,
            performance_metrics=performance_metrics,
            summary=summary,
            portfolio_values=portfolio_values,
            portfolio_weights=portfolio_weights,
            benchmark_values=benchmark_values,
            config=request.dict()
        )
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return BacktestResponse(
            backtest_id=backtest_id,
            success=False,
            execution_time=0.0,
            error_message=str(e)
        )


@app.post(f"{settings.api_prefix}/backtest/compare", response_model=CompareStrategiesResponse)
async def compare_strategies(
    request: CompareStrategiesRequest,
    background_tasks: BackgroundTasks,
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Compare multiple portfolio strategies."""
    comparison_id = str(uuid.uuid4())
    
    try:
        from ..backtesting import BacktestEngine
        
        # Run each strategy
        engine = BacktestEngine()
        results = []
        portfolio_values = {}
        
        start_time = time.time()
        
        for i, (strategy_request, strategy_name) in enumerate(zip(request.strategies, request.strategy_names)):
            try:
                # Use the same logic as single backtest
                single_result = await run_backtest(strategy_request, background_tasks, optimization_manager)
                
                if single_result.success:
                    results.append({
                        "name": strategy_name,
                        "success": True,
                        "performance_metrics": single_result.performance_metrics,
                        "summary": single_result.summary
                    })
                    
                    # Store portfolio values for comparison chart
                    if single_result.portfolio_values:
                        portfolio_values[strategy_name] = single_result.portfolio_values
                else:
                    results.append({
                        "name": strategy_name,
                        "success": False,
                        "error_message": single_result.error_message
                    })
                    
            except Exception as e:
                results.append({
                    "name": strategy_name,
                    "success": False,
                    "error_message": str(e)
                })
        
        execution_time = time.time() - start_time
        
        # Create performance comparison table
        performance_comparison = []
        for result in results:
            if result["success"] and result.get("performance_metrics"):
                metrics = result["performance_metrics"]
                performance_comparison.append({
                    "Strategy": result["name"],
                    "Total Return": f"{metrics.total_return:.2%}",
                    "CAGR": f"{metrics.cagr:.2%}",
                    "Volatility": f"{metrics.annualized_volatility:.2%}",
                    "Sharpe Ratio": f"{metrics.sharpe_ratio:.3f}",
                    "Max Drawdown": f"{metrics.max_drawdown:.2%}",
                    "Calmar Ratio": f"{metrics.calmar_ratio:.3f}"
                })
        
        return CompareStrategiesResponse(
            comparison_id=comparison_id,
            success=True,
            execution_time=execution_time,
            results=results,
            performance_comparison=performance_comparison,
            portfolio_values=portfolio_values
        )
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        return CompareStrategiesResponse(
            comparison_id=comparison_id,
            success=False,
            execution_time=0.0,
            error_message=str(e)
        )


# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data streaming."""
    client_id = None
    
    try:
        # Accept connection and get client ID
        client_id = await connection_manager.connect(websocket)
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                await connection_manager.handle_message(client_id, data)
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected normally")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                # Send error message to client
                await connection_manager.send_to_client(
                    client_id,
                    WebSocketMessage(
                        message_type=MessageType.ERROR,
                        data={"error": "Message processing failed", "detail": str(e)}
                    )
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected during setup")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if client_id:
            await connection_manager.disconnect(client_id)


# Portfolio monitoring endpoints
@app.post(f"{settings.api_prefix}/streaming/portfolio")
async def add_portfolio_monitoring(portfolio_data: Dict[str, Any]):
    """Add a portfolio for real-time monitoring."""
    try:
        # Extract portfolio data
        portfolio_id = portfolio_data.get("portfolio_id", str(uuid.uuid4()))
        name = portfolio_data.get("name", "Unnamed Portfolio")
        holdings = portfolio_data.get("holdings", {})  # symbol -> shares
        initial_value = portfolio_data.get("initial_value", 0.0)
        benchmark_symbol = portfolio_data.get("benchmark_symbol", "SPY")
        risk_free_rate = portfolio_data.get("risk_free_rate", settings.default_risk_free_rate)
        
        if not holdings:
            raise HTTPException(status_code=400, detail="Portfolio holdings cannot be empty")
        
        # Create portfolio object
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            name=name,
            holdings=holdings,
            initial_value=initial_value,
            created_at=datetime.utcnow(),
            benchmark_symbol=benchmark_symbol,
            risk_free_rate=risk_free_rate
        )
        
        # Add to monitor
        await portfolio_monitor.add_portfolio(portfolio)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "message": f"Portfolio {name} added for monitoring",
            "symbols": portfolio.get_symbols()
        }
        
    except Exception as e:
        logger.error(f"Error adding portfolio monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add portfolio monitoring: {str(e)}")


@app.delete(f"{settings.api_prefix}/streaming/portfolio/{{portfolio_id}}")
async def remove_portfolio_monitoring(portfolio_id: str):
    """Remove a portfolio from real-time monitoring."""
    try:
        await portfolio_monitor.remove_portfolio(portfolio_id)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "message": "Portfolio removed from monitoring"
        }
        
    except Exception as e:
        logger.error(f"Error removing portfolio monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove portfolio monitoring: {str(e)}")


@app.get(f"{settings.api_prefix}/streaming/portfolios")
async def get_monitored_portfolios():
    """Get all portfolios currently being monitored."""
    try:
        portfolios = portfolio_monitor.get_all_portfolios_summary()
        return {
            "success": True,
            "portfolios": portfolios,
            "count": len(portfolios)
        }
        
    except Exception as e:
        logger.error(f"Error getting monitored portfolios: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitored portfolios: {str(e)}")


@app.get(f"{settings.api_prefix}/streaming/portfolio/{{portfolio_id}}")
async def get_portfolio_summary(portfolio_id: str):
    """Get detailed summary of a monitored portfolio."""
    try:
        summary = portfolio_monitor.get_portfolio_summary(portfolio_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
        
        return {
            "success": True,
            "portfolio": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio summary: {str(e)}")


@app.post(f"{settings.api_prefix}/streaming/alert-rule")
async def add_alert_rule(rule_data: Dict[str, Any]):
    """Add an alert rule for portfolio monitoring."""
    try:
        rule = AlertRule(
            rule_id=rule_data.get("rule_id", str(uuid.uuid4())),
            portfolio_id=rule_data["portfolio_id"],
            rule_type=rule_data["rule_type"],
            threshold=rule_data["threshold"],
            condition=rule_data["condition"],
            enabled=rule_data.get("enabled", True),
            cooldown_minutes=rule_data.get("cooldown_minutes", 15)
        )
        
        await portfolio_monitor.add_alert_rule(rule)
        
        return {
            "success": True,
            "rule_id": rule.rule_id,
            "message": "Alert rule added successfully"
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error adding alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add alert rule: {str(e)}")


@app.delete(f"{settings.api_prefix}/streaming/alert-rule/{{rule_id}}")
async def remove_alert_rule(rule_id: str):
    """Remove an alert rule."""
    try:
        await portfolio_monitor.remove_alert_rule(rule_id)
        
        return {
            "success": True,
            "rule_id": rule_id,
            "message": "Alert rule removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error removing alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove alert rule: {str(e)}")


# WebSocket status and debugging endpoints
@app.get(f"{settings.api_prefix}/streaming/status")
async def get_streaming_status():
    """Get status of real-time streaming services."""
    try:
        connection_stats = connection_manager.get_connection_stats()
        pipeline_data = data_pipeline.get_cached_data()
        market_source_stats = market_data_source.get_stats()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "websocket_connections": connection_stats,
            "data_pipeline": {
                "running": pipeline_data["running"],
                "subscribers": pipeline_data["subscribers"],
                "cached_symbols": len(pipeline_data["market_data"]),
                "cached_portfolios": len(pipeline_data["portfolios"])
            },
            "market_data_source": market_source_stats,
            "monitored_portfolios": len(portfolio_monitor.portfolios)
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get streaming status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )