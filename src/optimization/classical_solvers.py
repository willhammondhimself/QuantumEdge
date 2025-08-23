"""
Advanced classical optimization algorithms for portfolio optimization.

This module provides classical optimization methods to serve as baselines
for comparing against quantum-inspired approaches.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
from .mean_variance import OptimizationResult, PortfolioConstraints, ObjectiveType
import warnings

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Types of classical optimization methods."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GRADIENT_FREE = "gradient_free"
    HYBRID = "hybrid"


@dataclass
class OptimizerParameters:
    """Parameters for classical optimizers."""

    max_iterations: int = 1000
    population_size: int = 50
    tolerance: float = 1e-6
    seed: Optional[int] = None
    verbose: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 100

    # Algorithm-specific parameters
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    temperature: float = 1000.0
    cooling_rate: float = 0.95
    inertia_weight: float = 0.9
    cognitive_param: float = 2.0
    social_param: float = 2.0


class BaseOptimizer(ABC):
    """Abstract base class for classical portfolio optimizers."""

    def __init__(self, parameters: OptimizerParameters):
        """Initialize optimizer with parameters."""
        self.parameters = parameters
        self.history = []
        self.convergence_history = []

        if parameters.seed is not None:
            np.random.seed(parameters.seed)

    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        objective: ObjectiveType,
        returns_data: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize portfolio allocation."""
        pass

    def _objective_function(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        objective: ObjectiveType,
        risk_free_rate: float = 0.02,
        returns_data: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate objective function value for given weights.

        Args:
            weights: Portfolio weights
            expected_returns: Expected asset returns
            covariance_matrix: Asset covariance matrix
            objective: Optimization objective
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            Objective function value (to be minimized)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        if objective == ObjectiveType.MAXIMIZE_SHARPE:
            if portfolio_volatility < 1e-10:
                return -np.inf  # Invalid portfolio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Minimize negative Sharpe

        elif objective == ObjectiveType.MINIMIZE_VARIANCE:
            return portfolio_variance

        elif objective == ObjectiveType.MAXIMIZE_RETURN:
            return -portfolio_return  # Minimize negative return

        elif objective == ObjectiveType.MAXIMIZE_UTILITY:
            # Utility = Return - 0.5 * RiskAversion * Variance
            risk_aversion = 3.0  # Default moderate risk aversion
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility

        elif objective == ObjectiveType.MINIMIZE_CVAR:
            if returns_data is None:
                # If no historical returns, approximate using normal distribution
                # CVaR ≈ μ - σ * φ(Φ^(-1)(α)) / α
                from scipy import stats

                alpha = 1 - confidence_level
                z_alpha = stats.norm.ppf(alpha)
                phi_z = stats.norm.pdf(z_alpha)
                cvar_approx = -portfolio_return + portfolio_volatility * phi_z / alpha
                return cvar_approx
            else:
                # Calculate CVaR from historical returns
                portfolio_returns = returns_data @ weights
                var_threshold = np.percentile(
                    portfolio_returns, (1 - confidence_level) * 100
                )
                tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
                if len(tail_returns) == 0:
                    return -portfolio_return  # No tail risk, use expected return
                cvar = -np.mean(tail_returns)  # Negative because we minimize
                return cvar

        elif objective == ObjectiveType.MAXIMIZE_SORTINO:
            target_return = risk_free_rate
            if returns_data is None:
                # Approximate downside deviation using semi-variance
                downside_variance = portfolio_variance * 0.5  # Rough approximation
                downside_deviation = np.sqrt(downside_variance)
            else:
                # Calculate actual downside deviation from historical returns
                portfolio_returns = returns_data @ weights
                downside_returns = portfolio_returns[portfolio_returns < target_return]
                if len(downside_returns) == 0:
                    downside_deviation = 1e-10  # No downside, excellent!
                else:
                    downside_deviation = np.sqrt(
                        np.mean((downside_returns - target_return) ** 2)
                    )

            if downside_deviation < 1e-10:
                return -1000.0  # Very high Sortino ratio

            sortino_ratio = (portfolio_return - target_return) / downside_deviation
            return -sortino_ratio  # Minimize negative Sortino

        else:
            raise ValueError(f"Unsupported objective: {objective}")

    def _apply_constraints(
        self, weights: np.ndarray, constraints: PortfolioConstraints
    ) -> np.ndarray:
        """Apply portfolio constraints to weights."""
        # Normalize weights to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Apply weight bounds
        if constraints.min_weight is not None:
            weights = np.maximum(weights, constraints.min_weight)

        if constraints.max_weight is not None:
            weights = np.minimum(weights, constraints.max_weight)

        # Renormalize after bounds
        weights = weights / np.sum(weights)

        # Apply sector constraints (simplified)
        # NOTE: Sector constraints would require sector mappings
        # which are not implemented in the current PortfolioConstraints

        return weights

    def _check_convergence(self, current_value: float, iteration: int) -> bool:
        """Check if optimization has converged."""
        self.convergence_history.append(current_value)

        if not self.parameters.early_stopping:
            return False

        if len(self.convergence_history) < self.parameters.early_stopping_patience:
            return False

        # Check if improvement in last N iterations is below tolerance
        recent_values = self.convergence_history[
            -self.parameters.early_stopping_patience :
        ]
        improvement = abs(max(recent_values) - min(recent_values))

        return improvement < self.parameters.tolerance


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for portfolio optimization.

    Uses evolutionary principles to evolve a population of portfolio
    allocations towards better solutions.
    """

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        objective: ObjectiveType,
        returns_data: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        start_time = time.time()
        n_assets = len(expected_returns)

        logger.info(f"Starting genetic algorithm optimization: {n_assets} assets")

        # Generate synthetic returns data if needed for CVaR or Sortino
        if (
            objective in [ObjectiveType.MINIMIZE_CVAR, ObjectiveType.MAXIMIZE_SORTINO]
            and returns_data is None
        ):
            logger.info(
                "Generating synthetic returns data for CVaR/Sortino optimization"
            )
            # Generate 252 days of returns using multivariate normal
            # Use a separate random state to not affect main optimization randomness
            rng = np.random.RandomState(seed=42)  # Fixed seed for synthetic data
            returns_data = rng.multivariate_normal(
                expected_returns, covariance_matrix, size=252
            )

        # Initialize population
        population = self._initialize_population(n_assets, constraints)
        best_fitness = float("inf")
        best_weights = None

        for iteration in range(self.parameters.max_iterations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                fitness = self._objective_function(
                    individual,
                    expected_returns,
                    covariance_matrix,
                    objective,
                    returns_data=returns_data,
                )
                fitness_scores.append(fitness)

            # Track best solution
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_weights = population[current_best_idx].copy()

            # Check convergence
            if self._check_convergence(best_fitness, iteration):
                logger.info(f"GA converged at iteration {iteration}")
                break

            # Selection, crossover, and mutation
            new_population = []

            # Elite selection - keep best individuals
            elite_size = max(1, self.parameters.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate rest of population through crossover and mutation
            while len(new_population) < self.parameters.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.random() < self.parameters.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._mutate(child1, constraints)
                child2 = self._mutate(child2, constraints)

                new_population.extend([child1, child2])

            # Trim population to correct size
            population = new_population[: self.parameters.population_size]

            if self.parameters.verbose and iteration % 50 == 0:
                logger.debug(
                    f"GA Iteration {iteration}: Best fitness = {best_fitness:.6f}"
                )

        # Prepare result
        final_weights = self._apply_constraints(best_weights, constraints)
        portfolio_return = np.dot(final_weights, expected_returns)
        portfolio_variance = np.dot(
            final_weights, np.dot(covariance_matrix, final_weights)
        )

        execution_time = time.time() - start_time

        result = OptimizationResult(
            weights=final_weights,
            expected_return=portfolio_return,
            expected_variance=portfolio_variance,
            sharpe_ratio=(portfolio_return - 0.02) / np.sqrt(portfolio_variance),
            objective_value=(
                -best_fitness
                if objective
                in [
                    ObjectiveType.MAXIMIZE_SHARPE,
                    ObjectiveType.MAXIMIZE_RETURN,
                    ObjectiveType.MAXIMIZE_UTILITY,
                ]
                else best_fitness
            ),
            status=(
                "converged"
                if len(self.convergence_history) < self.parameters.max_iterations
                else "max_iter"
            ),
            solve_time=execution_time,
            success=True,
        )

        logger.info(
            f"GA optimization completed: return={portfolio_return:.4f}, "
            f"volatility={np.sqrt(portfolio_variance):.4f}"
        )

        return result

    def _initialize_population(
        self, n_assets: int, constraints: PortfolioConstraints
    ) -> List[np.ndarray]:
        """Initialize random population."""
        population = []

        for _ in range(self.parameters.population_size):
            # Generate random weights
            weights = np.random.uniform(0, 1, n_assets)
            weights = self._apply_constraints(weights, constraints)
            population.append(weights)

        return population

    def _tournament_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> np.ndarray:
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        # Uniform crossover
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def _mutate(
        self, individual: np.ndarray, constraints: PortfolioConstraints
    ) -> np.ndarray:
        """Apply mutation to individual."""
        if np.random.random() < self.parameters.mutation_rate:
            # Add gaussian noise
            noise = np.random.normal(0, 0.1, len(individual))
            individual = individual + noise
            individual = np.maximum(individual, 0)  # Ensure non-negative
            individual = self._apply_constraints(individual, constraints)

        return individual


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimizer for portfolio optimization.

    Uses the simulated annealing metaheuristic to find good solutions
    by gradually reducing randomness in the search process.
    """

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        objective: ObjectiveType,
        returns_data: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize using simulated annealing."""
        start_time = time.time()
        n_assets = len(expected_returns)

        logger.info(f"Starting simulated annealing optimization: {n_assets} assets")

        # Generate synthetic returns data if needed for CVaR or Sortino
        if (
            objective in [ObjectiveType.MINIMIZE_CVAR, ObjectiveType.MAXIMIZE_SORTINO]
            and returns_data is None
        ):
            logger.info(
                "Generating synthetic returns data for CVaR/Sortino optimization"
            )
            # Generate 252 days of returns using multivariate normal
            # Use a separate random state to not affect main optimization randomness
            rng = np.random.RandomState(seed=42)  # Fixed seed for synthetic data
            returns_data = rng.multivariate_normal(
                expected_returns, covariance_matrix, size=252
            )

        # Initialize solution
        current_weights = np.random.uniform(0, 1, n_assets)
        current_weights = self._apply_constraints(current_weights, constraints)
        current_fitness = self._objective_function(
            current_weights,
            expected_returns,
            covariance_matrix,
            objective,
            returns_data=returns_data,
        )

        # Best solution tracking
        best_weights = current_weights.copy()
        best_fitness = current_fitness

        temperature = self.parameters.temperature

        for iteration in range(self.parameters.max_iterations):
            # Generate neighbor solution
            neighbor_weights = self._generate_neighbor(current_weights, constraints)
            neighbor_fitness = self._objective_function(
                neighbor_weights,
                expected_returns,
                covariance_matrix,
                objective,
                returns_data=returns_data,
            )

            # Accept or reject neighbor
            if neighbor_fitness < current_fitness:
                # Better solution - accept
                current_weights = neighbor_weights
                current_fitness = neighbor_fitness

                if neighbor_fitness < best_fitness:
                    best_weights = neighbor_weights.copy()
                    best_fitness = neighbor_fitness
            else:
                # Worse solution - accept with probability based on temperature
                delta = neighbor_fitness - current_fitness
                probability = np.exp(-delta / temperature) if temperature > 0 else 0

                if np.random.random() < probability:
                    current_weights = neighbor_weights
                    current_fitness = neighbor_fitness

            # Cool down
            temperature *= self.parameters.cooling_rate

            # Check convergence
            if self._check_convergence(best_fitness, iteration):
                logger.info(f"SA converged at iteration {iteration}")
                break

            if self.parameters.verbose and iteration % 100 == 0:
                logger.debug(
                    f"SA Iteration {iteration}: Best fitness = {best_fitness:.6f}, "
                    f"Temperature = {temperature:.6f}"
                )

        # Prepare result
        final_weights = self._apply_constraints(best_weights, constraints)
        portfolio_return = np.dot(final_weights, expected_returns)
        portfolio_variance = np.dot(
            final_weights, np.dot(covariance_matrix, final_weights)
        )

        execution_time = time.time() - start_time

        result = OptimizationResult(
            weights=final_weights,
            expected_return=portfolio_return,
            expected_variance=portfolio_variance,
            sharpe_ratio=(portfolio_return - 0.02) / np.sqrt(portfolio_variance),
            objective_value=(
                -best_fitness
                if objective
                in [
                    ObjectiveType.MAXIMIZE_SHARPE,
                    ObjectiveType.MAXIMIZE_RETURN,
                    ObjectiveType.MAXIMIZE_UTILITY,
                ]
                else best_fitness
            ),
            status=(
                "converged"
                if len(self.convergence_history) < self.parameters.max_iterations
                else "max_iter"
            ),
            solve_time=execution_time,
            success=True,
        )

        logger.info(
            f"SA optimization completed: return={portfolio_return:.4f}, "
            f"volatility={np.sqrt(portfolio_variance):.4f}"
        )

        return result

    def _generate_neighbor(
        self, current_weights: np.ndarray, constraints: PortfolioConstraints
    ) -> np.ndarray:
        """Generate neighbor solution."""
        # Small random perturbation
        neighbor = current_weights.copy()

        # Choose random subset of weights to perturb
        n_perturb = max(1, len(current_weights) // 5)
        indices = np.random.choice(len(current_weights), n_perturb, replace=False)

        # Add gaussian noise
        for idx in indices:
            neighbor[idx] += np.random.normal(0, 0.05)

        # Ensure non-negative and apply constraints
        neighbor = np.maximum(neighbor, 0)
        neighbor = self._apply_constraints(neighbor, constraints)

        return neighbor


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization for portfolio optimization.

    Uses a swarm of particles moving through the solution space,
    influenced by their own best positions and the global best.
    """

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        objective: ObjectiveType,
        returns_data: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize using particle swarm optimization."""
        start_time = time.time()
        n_assets = len(expected_returns)

        logger.info(f"Starting particle swarm optimization: {n_assets} assets")

        # Generate synthetic returns data if needed for CVaR or Sortino
        if (
            objective in [ObjectiveType.MINIMIZE_CVAR, ObjectiveType.MAXIMIZE_SORTINO]
            and returns_data is None
        ):
            logger.info(
                "Generating synthetic returns data for CVaR/Sortino optimization"
            )
            # Generate 252 days of returns using multivariate normal
            # Use a separate random state to not affect main optimization randomness
            rng = np.random.RandomState(seed=42)  # Fixed seed for synthetic data
            returns_data = rng.multivariate_normal(
                expected_returns, covariance_matrix, size=252
            )

        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_fitness = []

        # Initialize swarm
        for _ in range(self.parameters.population_size):
            # Random initial position
            position = np.random.uniform(0, 1, n_assets)
            position = self._apply_constraints(position, constraints)

            # Random initial velocity
            velocity = np.random.uniform(-0.1, 0.1, n_assets)

            # Calculate initial fitness
            fitness = self._objective_function(
                position,
                expected_returns,
                covariance_matrix,
                objective,
                returns_data=returns_data,
            )

            particles.append(position)
            velocities.append(velocity)
            personal_best_positions.append(position.copy())
            personal_best_fitness.append(fitness)

        # Find global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        for iteration in range(self.parameters.max_iterations):
            for i in range(self.parameters.population_size):
                # Update velocity
                r1, r2 = np.random.random(), np.random.random()

                cognitive_component = (
                    self.parameters.cognitive_param
                    * r1
                    * (personal_best_positions[i] - particles[i])
                )
                social_component = (
                    self.parameters.social_param
                    * r2
                    * (global_best_position - particles[i])
                )

                velocities[i] = (
                    self.parameters.inertia_weight * velocities[i]
                    + cognitive_component
                    + social_component
                )

                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = self._apply_constraints(particles[i], constraints)

                # Evaluate fitness
                fitness = self._objective_function(
                    particles[i],
                    expected_returns,
                    covariance_matrix,
                    objective,
                    returns_data=returns_data,
                )

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness

                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_position = particles[i].copy()
                        global_best_fitness = fitness

            # Check convergence
            if self._check_convergence(global_best_fitness, iteration):
                logger.info(f"PSO converged at iteration {iteration}")
                break

            if self.parameters.verbose and iteration % 50 == 0:
                logger.debug(
                    f"PSO Iteration {iteration}: Best fitness = {global_best_fitness:.6f}"
                )

        # Prepare result
        final_weights = self._apply_constraints(global_best_position, constraints)
        portfolio_return = np.dot(final_weights, expected_returns)
        portfolio_variance = np.dot(
            final_weights, np.dot(covariance_matrix, final_weights)
        )

        execution_time = time.time() - start_time

        result = OptimizationResult(
            weights=final_weights,
            expected_return=portfolio_return,
            expected_variance=portfolio_variance,
            sharpe_ratio=(portfolio_return - 0.02) / np.sqrt(portfolio_variance),
            objective_value=(
                -global_best_fitness
                if objective
                in [
                    ObjectiveType.MAXIMIZE_SHARPE,
                    ObjectiveType.MAXIMIZE_RETURN,
                    ObjectiveType.MAXIMIZE_UTILITY,
                ]
                else global_best_fitness
            ),
            status=(
                "converged"
                if len(self.convergence_history) < self.parameters.max_iterations
                else "max_iter"
            ),
            solve_time=execution_time,
            success=True,
        )

        logger.info(
            f"PSO optimization completed: return={portfolio_return:.4f}, "
            f"volatility={np.sqrt(portfolio_variance):.4f}"
        )

        return result


class ClassicalOptimizerFactory:
    """Factory for creating classical optimization algorithms."""

    @staticmethod
    def create_optimizer(
        method: OptimizationMethod, parameters: Optional[OptimizerParameters] = None
    ) -> BaseOptimizer:
        """
        Create optimizer instance.

        Args:
            method: Type of optimization method
            parameters: Optimizer parameters

        Returns:
            Optimizer instance
        """
        if parameters is None:
            parameters = OptimizerParameters()

        if method == OptimizationMethod.GENETIC_ALGORITHM:
            return GeneticAlgorithmOptimizer(parameters)
        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            return SimulatedAnnealingOptimizer(parameters)
        elif method == OptimizationMethod.PARTICLE_SWARM:
            return ParticleSwarmOptimizer(parameters)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")

    @staticmethod
    def get_default_parameters(method: OptimizationMethod) -> OptimizerParameters:
        """Get default parameters for optimization method."""
        base_params = OptimizerParameters()

        if method == OptimizationMethod.GENETIC_ALGORITHM:
            base_params.population_size = 100
            base_params.max_iterations = 500
            base_params.mutation_rate = 0.1
            base_params.crossover_rate = 0.8

        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            base_params.max_iterations = 2000
            base_params.temperature = 1000.0
            base_params.cooling_rate = 0.95

        elif method == OptimizationMethod.PARTICLE_SWARM:
            base_params.population_size = 50
            base_params.max_iterations = 1000
            base_params.inertia_weight = 0.9
            base_params.cognitive_param = 2.0
            base_params.social_param = 2.0

        return base_params


def compare_classical_methods(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    objective: ObjectiveType = ObjectiveType.MAXIMIZE_SHARPE,
    methods: Optional[List[OptimizationMethod]] = None,
    returns_data: Optional[np.ndarray] = None,
) -> Dict[str, OptimizationResult]:
    """
    Compare multiple classical optimization methods.

    Args:
        expected_returns: Expected asset returns
        covariance_matrix: Asset covariance matrix
        constraints: Portfolio constraints
        objective: Optimization objective
        methods: List of methods to compare (None for all)

    Returns:
        Dictionary mapping method names to optimization results
    """
    if methods is None:
        methods = [
            OptimizationMethod.GENETIC_ALGORITHM,
            OptimizationMethod.SIMULATED_ANNEALING,
            OptimizationMethod.PARTICLE_SWARM,
        ]

    results = {}

    for method in methods:
        logger.info(f"Running {method.value} optimization...")

        try:
            params = ClassicalOptimizerFactory.get_default_parameters(method)
            params.verbose = False  # Reduce output for comparison

            optimizer = ClassicalOptimizerFactory.create_optimizer(method, params)
            result = optimizer.optimize(
                expected_returns,
                covariance_matrix,
                constraints,
                objective,
                returns_data,
            )

            results[method.value] = result

        except Exception as e:
            logger.error(f"Failed to run {method.value}: {e}")

    return results
