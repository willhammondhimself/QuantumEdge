"""
Classical mean-variance portfolio optimization using CVXPY.

This module provides the baseline implementation of Markowitz mean-variance
optimization to compare against quantum-inspired approaches.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Portfolio optimization objective types."""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_CVAR = "minimize_cvar"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MAXIMIZE_SORTINO = "maximize_sortino"


@dataclass
class OptimizationResult:
    """Result from portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    expected_variance: float
    sharpe_ratio: float
    objective_value: float
    status: str
    solve_time: float
    success: bool


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    # Weight constraints
    long_only: bool = True
    sum_to_one: bool = True
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    
    # Asset selection constraints
    max_assets: Optional[int] = None
    min_assets: Optional[int] = None
    forbidden_assets: Optional[List[int]] = None
    required_assets: Optional[List[int]] = None
    
    # Risk constraints
    max_variance: Optional[float] = None
    min_return: Optional[float] = None
    target_return: Optional[float] = None
    
    # Turnover constraints
    max_turnover: Optional[float] = None
    current_weights: Optional[np.ndarray] = None


class MeanVarianceOptimizer:
    """
    Mean-variance portfolio optimizer using modern portfolio theory.
    
    Implements Markowitz mean-variance optimization with support for various
    objective functions and portfolio constraints.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        solver_preference: List[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            solver_preference: Preferred CVXPY solvers in order
            verbose: Enable verbose solver output
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for mean-variance optimization")
        
        self.risk_free_rate = risk_free_rate
        self.solver_preference = solver_preference or ['OSQP', 'ECOS', 'SCS']
        self.verbose = verbose
    
    def _get_available_solver(self) -> str:
        """Get the first available solver from preferences."""
        for solver in self.solver_preference:
            if hasattr(cp, solver):
                return solver
        return 'OSQP'  # Default fallback
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objective: ObjectiveType = ObjectiveType.MAXIMIZE_SHARPE,
        constraints: Optional[PortfolioConstraints] = None,
        risk_aversion: float = 1.0,
        returns_data: Optional[np.ndarray] = None,
        cvar_confidence: float = 0.05,
        lookback_periods: int = 252
    ) -> OptimizationResult:
        """
        Optimize portfolio using mean-variance framework with advanced objectives.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            objective: Optimization objective
            constraints: Portfolio constraints
            risk_aversion: Risk aversion parameter for utility maximization
            returns_data: Historical returns data for CVaR/Sortino/Calmar (n_periods x n_assets)
            cvar_confidence: Confidence level for CVaR (e.g., 0.05 for 5% CVaR)
            lookback_periods: Number of periods for Calmar ratio calculation
            
        Returns:
            Optimization result
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        # Create optimization variable
        w = cp.Variable(n_assets)
        
        # Portfolio return and variance
        portfolio_return = expected_returns.T @ w
        portfolio_variance = cp.quad_form(w, covariance_matrix)
        
        # Define objective function
        additional_constraints = []
        
        if objective == ObjectiveType.MAXIMIZE_SHARPE:
            # Use utility approximation for Sharpe ratio
            objective_func = cp.Maximize(portfolio_return - 0.5 * portfolio_variance)
        elif objective == ObjectiveType.MINIMIZE_VARIANCE:
            objective_func = cp.Minimize(portfolio_variance)
        elif objective == ObjectiveType.MAXIMIZE_RETURN:
            objective_func = cp.Maximize(portfolio_return)
        elif objective == ObjectiveType.MAXIMIZE_UTILITY:
            # Mean-variance utility: return - (risk_aversion/2) * variance
            objective_func = cp.Maximize(portfolio_return - (risk_aversion/2) * portfolio_variance)
        elif objective == ObjectiveType.MINIMIZE_CVAR:
            # CVaR optimization requires historical returns data
            if returns_data is None:
                raise ValueError("returns_data required for CVaR optimization")
            objective_func, cvar_constraints = self._create_cvar_objective(w, returns_data, cvar_confidence)
            additional_constraints.extend(cvar_constraints)
        elif objective == ObjectiveType.MAXIMIZE_SORTINO:
            # Sortino ratio optimization (return/downside deviation)
            if returns_data is None:
                raise ValueError("returns_data required for Sortino optimization")
            objective_func, sortino_constraints = self._create_sortino_objective(w, expected_returns, returns_data)
            additional_constraints.extend(sortino_constraints)
        elif objective == ObjectiveType.MAXIMIZE_CALMAR:
            # Calmar ratio optimization (return/max drawdown)
            if returns_data is None:
                raise ValueError("returns_data required for Calmar optimization")
            objective_func, calmar_constraints = self._create_calmar_objective(w, expected_returns, returns_data, lookback_periods)
            additional_constraints.extend(calmar_constraints)
        else:
            raise ValueError(f"Unknown objective type: {objective}")
        
        # Define constraints
        constraint_list = []
        
        # Weight sum constraint
        if constraints.sum_to_one:
            constraint_list.append(cp.sum(w) == 1)
        
        # Long-only constraint
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        # Weight bounds
        if constraints.min_weight is not None:
            constraint_list.append(w >= constraints.min_weight)
        if constraints.max_weight is not None:
            constraint_list.append(w <= constraints.max_weight)
        
        # Return constraints
        if constraints.min_return is not None:
            constraint_list.append(portfolio_return >= constraints.min_return)
        if constraints.target_return is not None:
            constraint_list.append(portfolio_return == constraints.target_return)
        
        # Risk constraints
        if constraints.max_variance is not None:
            constraint_list.append(portfolio_variance <= constraints.max_variance)
        
        # Asset selection constraints (approximate)
        if constraints.forbidden_assets is not None:
            for asset in constraints.forbidden_assets:
                constraint_list.append(w[asset] == 0)
        
        if constraints.required_assets is not None:
            for asset in constraints.required_assets:
                constraint_list.append(w[asset] >= 0.01)  # Minimum allocation
        
        # Turnover constraint
        if constraints.max_turnover is not None and constraints.current_weights is not None:
            turnover = cp.norm(w - constraints.current_weights, 1)
            constraint_list.append(turnover <= constraints.max_turnover)
        
        # Add objective-specific constraints
        constraint_list.extend(additional_constraints)
        
        # Create and solve problem
        problem = cp.Problem(objective_func, constraint_list)
        
        # Solve with preferred solver
        solver_name = self._get_available_solver()
        try:
            solve_time = problem.solve(solver=solver_name, verbose=self.verbose)
        except Exception as e:
            logger.warning(f"Solver {solver_name} failed: {e}. Trying backup solver.")
            solve_time = problem.solve(verbose=self.verbose)
        
        # Extract results
        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            expected_return = float(expected_returns.T @ weights)
            expected_variance = float(weights.T @ covariance_matrix @ weights)
            
            # Calculate Sharpe ratio
            if expected_variance > 0:
                sharpe_ratio = (expected_return - self.risk_free_rate) / np.sqrt(expected_variance)
            else:
                sharpe_ratio = 0.0
            
            success = True
        else:
            logger.error(f"Optimization failed with status: {problem.status}")
            weights = np.zeros(n_assets)
            expected_return = 0.0
            expected_variance = 0.0
            sharpe_ratio = 0.0
            success = False
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_variance=expected_variance,
            sharpe_ratio=sharpe_ratio,
            objective_value=problem.value if problem.value is not None else 0.0,
            status=problem.status,
            solve_time=solve_time if solve_time is not None else 0.0,
            success=success
        )
    
    def _create_cvar_objective(self, w, returns_data: np.ndarray, confidence: float):
        """
        Create CVaR (Conditional Value at Risk) optimization objective.
        
        CVaR is the expected value of losses that exceed the VaR threshold.
        We minimize CVaR to find portfolios with better tail risk properties.
        """
        n_scenarios, n_assets = returns_data.shape
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_data @ w
        
        # VaR auxiliary variable
        var = cp.Variable()
        
        # CVaR auxiliary variables (losses exceeding VaR)
        u = cp.Variable(n_scenarios, nonneg=True)
        
        # CVaR formulation: VaR + (1/confidence) * mean(u)
        # where u[i] = max(0, -portfolio_returns[i] - var)
        cvar = var + (1.0 / confidence) * cp.sum(u) / n_scenarios
        
        # Constraint: u[i] >= -portfolio_returns[i] - var for all scenarios
        constraints_cvar = [u[i] >= -portfolio_returns[i] - var for i in range(n_scenarios)]
        
        return cp.Minimize(cvar), constraints_cvar
    
    def _create_sortino_objective(self, w, expected_returns: np.ndarray, returns_data: np.ndarray):
        """
        Create Sortino ratio optimization objective.
        
        Sortino ratio = (expected_return - risk_free_rate) / downside_deviation
        We maximize this ratio focusing only on downside risk.
        """
        n_scenarios, n_assets = returns_data.shape
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_data @ w
        expected_portfolio_return = expected_returns.T @ w
        
        # Downside deviations (only negative returns relative to risk-free rate)
        downside_deviations = cp.Variable(n_scenarios, nonneg=True)
        
        # Constraint: downside_deviation[i] >= max(0, risk_free_rate - portfolio_returns[i])
        constraints_sortino = [
            downside_deviations[i] >= self.risk_free_rate - portfolio_returns[i] 
            for i in range(n_scenarios)
        ]
        
        # Downside variance (mean of squared downside deviations)
        downside_variance = cp.sum_squares(downside_deviations) / n_scenarios
        
        # Sortino ratio approximation: maximize excess return / sqrt(downside variance)
        # Using utility approximation: excess_return - 0.5 * downside_variance
        excess_return = expected_portfolio_return - self.risk_free_rate
        
        return cp.Maximize(excess_return - 0.5 * downside_variance), constraints_sortino
    
    def _create_calmar_objective(self, w, expected_returns: np.ndarray, returns_data: np.ndarray, lookback_periods: int):
        """
        Create Calmar ratio optimization objective.
        
        Calmar ratio = annualized_return / max_drawdown
        We maximize this ratio by optimizing return relative to maximum drawdown.
        """
        n_scenarios, n_assets = returns_data.shape
        lookback = min(lookback_periods, n_scenarios)
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_data[-lookback:] @ w  # Use recent data
        expected_portfolio_return = expected_returns.T @ w
        
        # Cumulative returns
        cumulative_returns = cp.Variable(lookback)
        cumulative_returns[0] = portfolio_returns[0]
        
        # Build cumulative return series
        constraints_cumulative = []
        for i in range(1, lookback):
            constraints_cumulative.append(cumulative_returns[i] == cumulative_returns[i-1] + portfolio_returns[i])
        
        # Running maximum (for drawdown calculation)
        running_max = cp.Variable(lookback)
        drawdowns = cp.Variable(lookback)
        
        # Initialize running maximum
        constraints_calmar = [running_max[0] == cumulative_returns[0]]
        constraints_calmar.append(drawdowns[0] == 0)  # No drawdown at start
        
        # Update running maximum and calculate drawdowns
        for i in range(1, lookback):
            constraints_calmar.append(running_max[i] >= running_max[i-1])
            constraints_calmar.append(running_max[i] >= cumulative_returns[i])
            constraints_calmar.append(drawdowns[i] == running_max[i] - cumulative_returns[i])
        
        # Maximum drawdown
        max_drawdown = cp.Variable()
        constraints_calmar.append(max_drawdown >= cp.max(drawdowns))
        
        # Calmar ratio approximation: maximize return - penalty * max_drawdown
        # Using penalty term to avoid division by zero
        penalty = 10.0  # Adjust based on typical drawdown magnitudes
        
        all_constraints = constraints_cumulative + constraints_calmar
        
        return cp.Maximize(expected_portfolio_return - penalty * max_drawdown), all_constraints

    def compute_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_points: int = 50,
        constraints: Optional[PortfolioConstraints] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute efficient frontier.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            num_points: Number of points on the frontier
            constraints: Portfolio constraints
            
        Returns:
            Tuple of (returns, risks, weights) for efficient portfolios
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        # Find minimum and maximum returns
        min_ret_result = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            ObjectiveType.MINIMIZE_VARIANCE,
            constraints
        )
        max_ret_result = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            ObjectiveType.MAXIMIZE_RETURN,
            constraints
        )
        
        min_return = min_ret_result.expected_return
        max_return = max_ret_result.expected_return
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_points)
        
        # Compute efficient portfolios
        efficient_returns = []
        efficient_risks = []
        efficient_weights = []
        
        for target_ret in target_returns:
            # Add target return constraint
            target_constraints = PortfolioConstraints(
                long_only=constraints.long_only,
                sum_to_one=constraints.sum_to_one,
                min_weight=constraints.min_weight,
                max_weight=constraints.max_weight,
                target_return=target_ret
            )
            
            result = self.optimize_portfolio(
                expected_returns, covariance_matrix,
                ObjectiveType.MINIMIZE_VARIANCE,
                target_constraints
            )
            
            if result.success:
                efficient_returns.append(result.expected_return)
                efficient_risks.append(np.sqrt(result.expected_variance))
                efficient_weights.append(result.weights)
        
        return (
            np.array(efficient_returns),
            np.array(efficient_risks),
            np.array(efficient_weights)
        )

    def optimize_maximum_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find maximum Sharpe ratio portfolio using quadratic programming.
        
        This method uses the analytical solution for maximum Sharpe ratio
        when possible, falling back to numerical optimization.
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        # Excess returns
        excess_returns = expected_returns - self.risk_free_rate
        
        # Create optimization variable (auxiliary variable y = w/kappa)
        y = cp.Variable(n_assets)
        kappa = cp.Variable()
        
        # Objective: maximize excess return
        objective = cp.Maximize(excess_returns.T @ y)
        
        # Constraints
        constraint_list = [
            cp.quad_form(y, covariance_matrix) <= kappa,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        
        # Additional constraints on y (which represent w/kappa)
        if constraints.long_only:
            constraint_list.append(y >= 0)
        
        if constraints.min_weight is not None:
            constraint_list.append(y >= constraints.min_weight * kappa)
        if constraints.max_weight is not None:
            constraint_list.append(y <= constraints.max_weight * kappa)
        
        # Solve problem
        problem = cp.Problem(objective, constraint_list)
        solver_name = self._get_available_solver()
        
        try:
            solve_time = problem.solve(solver=solver_name, verbose=self.verbose)
        except Exception as e:
            logger.warning(f"Solver {solver_name} failed: {e}. Trying backup solver.")
            solve_time = problem.solve(verbose=self.verbose)
        
        # Extract results
        if problem.status in ['optimal', 'optimal_inaccurate']:
            # Convert back to portfolio weights
            y_val = y.value
            kappa_val = kappa.value
            
            if kappa_val > 1e-8:
                weights = y_val / kappa_val
                expected_return = float(expected_returns.T @ weights)
                expected_variance = float(weights.T @ covariance_matrix @ weights)
                
                # Calculate Sharpe ratio
                if expected_variance > 0:
                    sharpe_ratio = (expected_return - self.risk_free_rate) / np.sqrt(expected_variance)
                else:
                    sharpe_ratio = 0.0
                
                success = True
            else:
                logger.error("Optimization resulted in zero portfolio value")
                weights = np.zeros(n_assets)
                expected_return = 0.0
                expected_variance = 0.0
                sharpe_ratio = 0.0
                success = False
        else:
            logger.error(f"Maximum Sharpe optimization failed: {problem.status}")
            weights = np.zeros(n_assets)
            expected_return = 0.0
            expected_variance = 0.0
            sharpe_ratio = 0.0
            success = False
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_variance=expected_variance,
            sharpe_ratio=sharpe_ratio,
            objective_value=problem.value if problem.value is not None else 0.0,
            status=problem.status,
            solve_time=solve_time if solve_time is not None else 0.0,
            success=success
        )


class RobustOptimizer:
    """
    Robust portfolio optimizer for handling parameter uncertainty.
    
    Implements robust optimization techniques to account for estimation
    errors in expected returns and covariance matrices.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        uncertainty_level: float = 0.1,
        solver_preference: List[str] = None,
        verbose: bool = False
    ):
        """
        Initialize robust optimizer.
        
        Args:
            risk_free_rate: Risk-free rate
            uncertainty_level: Level of parameter uncertainty (0-1)
            solver_preference: Preferred CVXPY solvers
            verbose: Enable verbose solver output
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for robust optimization")
        
        self.risk_free_rate = risk_free_rate
        self.uncertainty_level = uncertainty_level
        self.solver_preference = solver_preference or ['OSQP', 'ECOS', 'SCS']
        self.verbose = verbose
    
    def optimize_robust_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        return_uncertainty: Optional[np.ndarray] = None,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize robust portfolio under parameter uncertainty.
        
        Args:
            expected_returns: Expected returns (center of uncertainty set)
            covariance_matrix: Covariance matrix
            return_uncertainty: Uncertainty in expected returns
            constraints: Portfolio constraints
            
        Returns:
            Robust optimization result
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        if return_uncertainty is None:
            # Default uncertainty: proportional to return magnitude
            return_uncertainty = self.uncertainty_level * np.abs(expected_returns)
        
        # Create optimization variables
        w = cp.Variable(n_assets)
        t = cp.Variable()  # Auxiliary variable for robust constraint
        
        # Robust portfolio return (worst-case)
        # min_mu w^T μ subject to ||w^T δμ||_2 <= t
        # This gives: w^T μ - t as the worst-case return
        robust_return = expected_returns.T @ w - t
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(w, covariance_matrix)
        
        # Robust constraint: ||D w||_2 <= t where D = diag(return_uncertainty)
        uncertainty_constraint = cp.norm(cp.multiply(return_uncertainty, w), 2) <= t
        
        # Objective: maximize robust utility
        objective = cp.Maximize(robust_return - 0.5 * portfolio_variance)
        
        # Constraints
        constraint_list = [uncertainty_constraint]
        
        # Standard portfolio constraints
        if constraints.sum_to_one:
            constraint_list.append(cp.sum(w) == 1)
        
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        if constraints.min_weight is not None:
            constraint_list.append(w >= constraints.min_weight)
        if constraints.max_weight is not None:
            constraint_list.append(w <= constraints.max_weight)
        
        if constraints.min_return is not None:
            constraint_list.append(robust_return >= constraints.min_return)
        
        if constraints.max_variance is not None:
            constraint_list.append(portfolio_variance <= constraints.max_variance)
        
        # Solve problem
        problem = cp.Problem(objective, constraint_list)
        
        solver_name = self.solver_preference[0] if self.solver_preference else 'OSQP'
        try:
            solve_time = problem.solve(solver=solver_name, verbose=self.verbose)
        except Exception as e:
            logger.warning(f"Solver {solver_name} failed: {e}. Trying backup solver.")
            solve_time = problem.solve(verbose=self.verbose)
        
        # Extract results
        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            expected_return = float(expected_returns.T @ weights)
            expected_variance = float(weights.T @ covariance_matrix @ weights)
            
            # Calculate Sharpe ratio (using nominal expected return)
            if expected_variance > 0:
                sharpe_ratio = (expected_return - self.risk_free_rate) / np.sqrt(expected_variance)
            else:
                sharpe_ratio = 0.0
            
            success = True
        else:
            logger.error(f"Robust optimization failed: {problem.status}")
            weights = np.zeros(n_assets)
            expected_return = 0.0
            expected_variance = 0.0
            sharpe_ratio = 0.0
            success = False
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_variance=expected_variance,
            sharpe_ratio=sharpe_ratio,
            objective_value=problem.value if problem.value is not None else 0.0,
            status=problem.status,
            solve_time=solve_time if solve_time is not None else 0.0,
            success=success
        )