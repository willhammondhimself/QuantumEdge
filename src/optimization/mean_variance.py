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
    max_tracking_error: Optional[float] = None
    
    # Return constraints
    min_return: Optional[float] = None
    target_return: Optional[float] = None
    
    # Turnover constraints
    max_turnover: Optional[float] = None
    current_weights: Optional[np.ndarray] = None


class MeanVarianceOptimizer:
    """Classical mean-variance portfolio optimizer."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        solver: str = 'OSQP',
        verbose: bool = False
    ):
        """
        Initialize mean-variance optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            solver: CVXPY solver to use
            verbose: Whether to print solver output
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for MeanVarianceOptimizer. Install with: pip install cvxpy")
        
        self.risk_free_rate = risk_free_rate
        self.solver = solver
        self.verbose = verbose
        
        # Available solvers in order of preference
        self.solvers = ['OSQP', 'ECOS', 'SCS', 'CLARABEL']
    
    def _get_available_solver(self) -> str:
        """Get first available solver from preference list."""
        for solver in self.solvers:
            if hasattr(cp, solver):
                return solver
        return 'OSQP'  # Default fallback
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objective: ObjectiveType = ObjectiveType.MAXIMIZE_SHARPE,
        constraints: Optional[PortfolioConstraints] = None,
        risk_aversion: float = 1.0
    ) -> OptimizationResult:
        """
        Optimize portfolio using mean-variance framework.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            objective: Optimization objective
            constraints: Portfolio constraints
            risk_aversion: Risk aversion parameter for utility maximization
            
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
            num_points: Number of points on frontier
            constraints: Portfolio constraints
            
        Returns:
            Tuple of (returns, risks, weights) along frontier
        """
        logger.info(f"Computing efficient frontier with {num_points} points")
        
        # Find minimum and maximum return portfolios
        min_var_result = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            objective=ObjectiveType.MINIMIZE_VARIANCE,
            constraints=constraints
        )
        
        max_ret_result = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            objective=ObjectiveType.MAXIMIZE_RETURN,
            constraints=constraints
        )
        
        if not (min_var_result.success and max_ret_result.success):
            raise RuntimeError("Failed to find boundary portfolios for efficient frontier")
        
        # Create return targets
        min_return = min_var_result.expected_return
        max_return = max_ret_result.expected_return
        target_returns = np.linspace(min_return, max_return, num_points)
        
        # Compute portfolio for each target return
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Set target return constraint
            target_constraints = constraints or PortfolioConstraints()
            target_constraints.target_return = target_return
            
            result = self.optimize_portfolio(
                expected_returns, covariance_matrix,
                objective=ObjectiveType.MINIMIZE_VARIANCE,
                constraints=target_constraints
            )
            
            if result.success:
                frontier_returns.append(result.expected_return)
                frontier_risks.append(np.sqrt(result.expected_variance))
                frontier_weights.append(result.weights)
        
        return (
            np.array(frontier_returns),
            np.array(frontier_risks),
            np.array(frontier_weights)
        )
    
    def optimize_maximum_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find maximum Sharpe ratio portfolio using two-step approach.
        
        This solves the Sharpe ratio optimization exactly by transforming
        to a quadratic program.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            constraints: Portfolio constraints
            
        Returns:
            Optimization result for maximum Sharpe portfolio
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
        
        # Long-only constraint
        if constraints.long_only:
            constraint_list.append(y >= 0)
        
        # Additional constraints (scaled appropriately)
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
            # Convert back to weights
            y_val = y.value
            kappa_val = kappa.value
            
            if kappa_val > 1e-10:
                weights = y_val / kappa_val
            else:
                weights = np.zeros(n_assets)
            
            # Calculate portfolio metrics
            expected_return = float(expected_returns.T @ weights)
            expected_variance = float(weights.T @ covariance_matrix @ weights)
            
            if expected_variance > 0:
                sharpe_ratio = (expected_return - self.risk_free_rate) / np.sqrt(expected_variance)
            else:
                sharpe_ratio = 0.0
            
            success = True
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


class RobustOptimizer(MeanVarianceOptimizer):
    """Robust portfolio optimization with uncertainty sets."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        solver: str = 'OSQP',
        verbose: bool = False,
        uncertainty_level: float = 0.1
    ):
        """
        Initialize robust optimizer.
        
        Args:
            risk_free_rate: Risk-free rate
            solver: CVXPY solver
            verbose: Solver verbosity
            uncertainty_level: Level of uncertainty in parameters
        """
        super().__init__(risk_free_rate, solver, verbose)
        self.uncertainty_level = uncertainty_level
    
    def optimize_robust_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        return_uncertainty: Optional[np.ndarray] = None,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio with robust optimization under uncertainty.
        
        Args:
            expected_returns: Expected returns (point estimates)
            covariance_matrix: Covariance matrix
            return_uncertainty: Uncertainty in return estimates
            constraints: Portfolio constraints
            
        Returns:
            Robust optimization result
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        if return_uncertainty is None:
            # Default uncertainty based on return volatility
            return_uncertainty = self.uncertainty_level * np.sqrt(np.diag(covariance_matrix))
        
        # Create optimization variables
        w = cp.Variable(n_assets)
        
        # Worst-case portfolio return (robust counterpart)
        portfolio_return = expected_returns.T @ w
        uncertainty_term = cp.norm(cp.multiply(return_uncertainty, w), 2)
        robust_return = portfolio_return - uncertainty_term
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(w, covariance_matrix)
        
        # Robust utility maximization
        objective = cp.Maximize(robust_return - 0.5 * portfolio_variance)
        
        # Standard constraints
        constraint_list = []
        
        if constraints.sum_to_one:
            constraint_list.append(cp.sum(w) == 1)
        
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        if constraints.min_weight is not None:
            constraint_list.append(w >= constraints.min_weight)
        if constraints.max_weight is not None:
            constraint_list.append(w <= constraints.max_weight)
        
        # Solve robust problem
        problem = cp.Problem(objective, constraint_list)
        
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