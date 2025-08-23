"""Example demonstrating CVaR and Sortino ratio optimization."""

import numpy as np
import matplotlib.pyplot as plt
from src.optimization.mean_variance import (
    MeanVarianceOptimizer,
    ObjectiveType,
    PortfolioConstraints,
)
from src.optimization.classical_solvers import (
    GeneticAlgorithmOptimizer,
    OptimizerParameters,
    compare_classical_methods,
)


def generate_sample_data(n_assets=5, n_periods=252):
    """Generate sample market data for demonstration."""
    np.random.seed(42)

    # Generate expected returns (annualized)
    expected_returns = np.array([0.08, 0.10, 0.12, 0.07, 0.09])

    # Generate covariance matrix
    correlations = np.array(
        [
            [1.0, 0.3, 0.2, 0.1, 0.2],
            [0.3, 1.0, 0.4, 0.2, 0.3],
            [0.2, 0.4, 1.0, 0.3, 0.2],
            [0.1, 0.2, 0.3, 1.0, 0.4],
            [0.2, 0.3, 0.2, 0.4, 1.0],
        ]
    )
    volatilities = np.array([0.15, 0.20, 0.25, 0.12, 0.18])
    covariance_matrix = np.outer(volatilities, volatilities) * correlations

    # Generate historical returns data
    returns_data = np.random.multivariate_normal(
        expected_returns / 252,  # Daily returns
        covariance_matrix / 252,  # Daily covariance
        size=n_periods,
    )

    return expected_returns, covariance_matrix, returns_data


def compare_objectives():
    """Compare different portfolio optimization objectives."""
    # Generate sample data
    expected_returns, covariance_matrix, returns_data = generate_sample_data()

    # Create optimizer
    optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
    constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

    # Define objectives to compare
    objectives = [
        ObjectiveType.MAXIMIZE_SHARPE,
        ObjectiveType.MINIMIZE_VARIANCE,
        ObjectiveType.MINIMIZE_CVAR,
        ObjectiveType.MAXIMIZE_SORTINO,
    ]

    results = {}

    for objective in objectives:
        print(f"\nOptimizing for {objective.value}...")

        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objective=objective,
            constraints=constraints,
            returns_data=returns_data,
            cvar_confidence=0.05,  # 5% CVaR (95% confidence)
        )

        if result.success:
            results[objective.value] = {
                "weights": result.weights,
                "return": result.expected_return,
                "volatility": np.sqrt(result.expected_variance),
                "sharpe": result.sharpe_ratio,
            }

            print(f"Expected Return: {result.expected_return:.4f}")
            print(f"Volatility: {np.sqrt(result.expected_variance):.4f}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print(f"Weights: {result.weights}")

            # Calculate additional metrics
            portfolio_returns = returns_data @ result.weights

            # CVaR (5%)
            var_threshold = np.percentile(portfolio_returns, 5)
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
            cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
            print(f"CVaR (5%): {cvar:.4f}")

            # Sortino ratio
            downside_returns = portfolio_returns[
                portfolio_returns < 0.02 / 252
            ]  # Daily risk-free rate
            downside_deviation = (
                np.sqrt(np.mean(downside_returns**2))
                if len(downside_returns) > 0
                else 0
            )
            sortino = (
                (result.expected_return - 0.02) / (downside_deviation * np.sqrt(252))
                if downside_deviation > 0
                else 0
            )
            print(f"Sortino Ratio: {sortino:.4f}")

    return results


def compare_classical_optimizers():
    """Compare classical optimization methods for CVaR objective."""
    # Generate sample data
    expected_returns, covariance_matrix, returns_data = generate_sample_data()

    # Compare methods
    constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

    print("\nComparing classical optimizers for CVaR minimization...")
    results = compare_classical_methods(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        constraints=constraints,
        objective=ObjectiveType.MINIMIZE_CVAR,
        returns_data=returns_data,
    )

    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Success: {result.success}")
        print(f"  Solve Time: {result.solve_time:.3f}s")
        print(f"  Expected Return: {result.expected_return:.4f}")
        print(f"  Volatility: {np.sqrt(result.expected_variance):.4f}")
        print(f"  Objective Value: {result.objective_value:.4f}")


def plot_efficient_frontiers():
    """Plot efficient frontiers for different objectives."""
    # Generate sample data
    expected_returns, covariance_matrix, returns_data = generate_sample_data()

    # Create optimizer
    optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
    constraints = PortfolioConstraints(long_only=True, sum_to_one=True)

    # Compute standard efficient frontier
    returns, risks, _ = optimizer.compute_efficient_frontier(
        expected_returns, covariance_matrix, num_points=50, constraints=constraints
    )

    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, "b-", label="Mean-Variance Frontier", linewidth=2)

    # Plot individual objective optimal portfolios
    objectives = {
        "Sharpe": ObjectiveType.MAXIMIZE_SHARPE,
        "Min Variance": ObjectiveType.MINIMIZE_VARIANCE,
        "CVaR": ObjectiveType.MINIMIZE_CVAR,
        "Sortino": ObjectiveType.MAXIMIZE_SORTINO,
    }

    colors = ["red", "green", "orange", "purple"]

    for (name, obj), color in zip(objectives.items(), colors):
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            objective=obj,
            constraints=constraints,
            returns_data=returns_data,
            cvar_confidence=0.05,
        )

        if result.success:
            plt.scatter(
                np.sqrt(result.expected_variance),
                result.expected_return,
                color=color,
                s=100,
                label=f"{name} Optimal",
                edgecolor="black",
                linewidth=2,
            )

    plt.xlabel("Risk (Volatility)", fontsize=12)
    plt.ylabel("Expected Return", fontsize=12)
    plt.title("Efficient Frontier with Different Optimization Objectives", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier_objectives.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    print("CVaR and Sortino Ratio Optimization Example")
    print("=" * 50)

    # Compare different objectives
    results = compare_objectives()

    # Skip the plotting for now (can be run separately)
    # plot_efficient_frontiers()

    print("\nExample completed successfully!")
