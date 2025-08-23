#!/usr/bin/env python3
"""
QuantumEdge: Noise Models, Quantum Volume, and Classical Comparison Demo

This example demonstrates the new quantum noise modeling, quantum volume
calculations, and comprehensive classical solver comparison capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import QuantumEdge modules
from src.quantum_algorithms import (
    QuantumCircuit,
    NoiseModel,
    QuantumVolumeProtocol,
    QuantumVQE,
    PortfolioQAOA,
)
from src.optimization.classical_solvers import (
    ClassicalOptimizerFactory,
    OptimizationMethod,
    compare_classical_methods,
)
from src.optimization.mean_variance import PortfolioConstraints, ObjectiveType
from src.benchmarking import run_comprehensive_benchmark, BenchmarkConfiguration


def demo_noise_models():
    """Demonstrate quantum noise modeling capabilities."""
    logger.info("=== Quantum Noise Modeling Demo ===")

    # Create a noise model with depolarizing noise
    noise_model = NoiseModel()
    noise_model.add_depolarizing_noise(
        error_probability=0.01,  # 1% error probability
        gate_types=["all"],  # Apply to all gates
        num_qubits=1,  # Single-qubit depolarizing noise
    )

    # Add two-qubit noise for entangling gates
    noise_model.add_depolarizing_noise(
        error_probability=0.02,  # 2% error probability for 2-qubit gates
        gate_types=["CNOT", "CZ"],
        num_qubits=2,
    )

    # Create quantum circuit with and without noise
    logger.info("Creating quantum circuits with and without noise...")

    # Noiseless circuit
    clean_circuit = QuantumCircuit(3)
    clean_circuit.h(0)
    clean_circuit.cnot(0, 1)
    clean_circuit.ry(1, np.pi / 4)
    clean_circuit.cnot(1, 2)
    clean_circuit.h(2)

    # Noisy circuit
    noisy_circuit = QuantumCircuit(3, noise_model=noise_model)
    noisy_circuit.h(0)
    noisy_circuit.cnot(0, 1)
    noisy_circuit.ry(1, np.pi / 4)
    noisy_circuit.cnot(1, 2)
    noisy_circuit.h(2)

    # Compare fidelities
    clean_fidelity = clean_circuit.get_estimated_fidelity()
    noisy_fidelity = noisy_circuit.get_estimated_fidelity()

    logger.info(f"Clean circuit fidelity: {clean_fidelity:.4f}")
    logger.info(f"Noisy circuit fidelity: {noisy_fidelity:.4f}")
    logger.info(f"Fidelity loss: {clean_fidelity - noisy_fidelity:.4f}")

    # Get noise statistics
    noise_stats = noisy_circuit.get_noise_statistics()
    logger.info(f"Noise applications: {noise_stats['noise_applications']}")
    logger.info(f"Noise rate: {noise_stats['noise_rate']:.2%}")

    return noise_model


def demo_quantum_volume():
    """Demonstrate quantum volume protocol."""
    logger.info("\n=== Quantum Volume Protocol Demo ===")

    # Create quantum volume protocol
    qv_protocol = QuantumVolumeProtocol()

    # Run quantum volume test on small system
    logger.info("Running quantum volume protocol for 3 qubits...")
    qv_result = qv_protocol.run_quantum_volume_protocol(
        num_qubits=3,
        num_circuits=20,  # Reduced for demo
        num_shots=256,  # Reduced for demo
    )

    logger.info(f"Quantum Volume: {qv_result.quantum_volume}")
    logger.info(f"Success rate: {qv_result.success_probability:.2%}")
    logger.info(f"Heavy output probability: {qv_result.heavy_output_probability:.3f}")
    logger.info(f"Circuit fidelity: {qv_result.circuit_fidelity_estimate:.3f}")
    logger.info(f"Execution time: {qv_result.execution_time:.2f}s")

    # Test quantum volume with noise
    logger.info("\nTesting quantum volume with noise...")
    noise_model = NoiseModel()
    noise_model.add_depolarizing_noise(error_probability=0.05)

    qv_noisy_result = qv_protocol.run_quantum_volume_protocol(
        num_qubits=3, num_circuits=10, num_shots=256, noise_model=noise_model
    )

    logger.info(f"Noisy QV: {qv_noisy_result.quantum_volume}")
    logger.info(f"Noisy success rate: {qv_noisy_result.success_probability:.2%}")
    logger.info(f"Noisy fidelity: {qv_noisy_result.circuit_fidelity_estimate:.3f}")

    # Analyze circuit complexity
    test_circuit = QuantumCircuit(4)
    test_circuit.h(0)
    test_circuit.cnot(0, 1)
    test_circuit.ry(1, np.pi / 3)
    test_circuit.cnot(1, 2)
    test_circuit.cz(2, 3)

    complexity = qv_protocol.analyze_circuit_complexity(test_circuit)
    logger.info(f"\nCircuit complexity analysis:")
    logger.info(f"  Total gates: {complexity['total_gates']}")
    logger.info(f"  Circuit depth: {complexity['circuit_depth']}")
    logger.info(f"  Two-qubit gates: {complexity['two_qubit_gates']}")
    logger.info(f"  Complexity score: {complexity['complexity_score']:.3f}")

    return qv_result


def demo_classical_solvers():
    """Demonstrate advanced classical optimization methods."""
    logger.info("\n=== Classical Optimization Solvers Demo ===")

    # Create test portfolio optimization problem
    n_assets = 8
    np.random.seed(42)

    # Generate synthetic market data
    expected_returns = np.random.uniform(0.08, 0.15, n_assets)
    volatilities = np.random.uniform(0.15, 0.30, n_assets)

    # Create correlation matrix
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            correlation = np.random.uniform(-0.3, 0.5)
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation

    # Convert to covariance matrix
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)
    covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    # Define constraints
    constraints = PortfolioConstraints(
        min_weight=0.05,  # Minimum 5% in each asset
        max_weight=0.30,  # Maximum 30% in each asset
        long_only=True,
        budget_constraint=1.0,
    )

    logger.info(f"Portfolio problem: {n_assets} assets")
    logger.info(
        f"Expected returns: {expected_returns.mean():.2%} ± {expected_returns.std():.2%}"
    )
    logger.info(f"Volatilities: {volatilities.mean():.2%} ± {volatilities.std():.2%}")

    # Compare all classical methods
    logger.info("\nRunning classical optimization methods...")
    classical_results = compare_classical_methods(
        expected_returns, covariance_matrix, constraints, ObjectiveType.MAXIMIZE_SHARPE
    )

    # Display results
    logger.info("\nClassical optimization results:")
    for method, result in classical_results.items():
        logger.info(f"{method}:")
        logger.info(f"  Sharpe ratio: {result.sharpe_ratio:.3f}")
        logger.info(f"  Expected return: {result.expected_return:.2%}")
        logger.info(f"  Volatility: {np.sqrt(result.expected_variance):.2%}")
        logger.info(f"  Execution time: {result.solve_time:.3f}s")
        logger.info(f"  Success: {result.success}")

    # Find best performing method
    best_method = max(
        classical_results.keys(), key=lambda x: classical_results[x].sharpe_ratio
    )
    best_sharpe = classical_results[best_method].sharpe_ratio

    logger.info(f"\nBest classical method: {best_method} (Sharpe: {best_sharpe:.3f})")

    return classical_results, expected_returns, covariance_matrix


def demo_comprehensive_benchmark():
    """Demonstrate comprehensive quantum vs classical benchmarking."""
    logger.info("\n=== Comprehensive Quantum vs Classical Benchmark ===")

    # Configure benchmark
    config = BenchmarkConfiguration(
        num_assets=6,  # Small for demo
        num_trials=2,  # Reduced for demo
        max_iterations=100,  # Reduced for demo
        noise_levels=[0.0, 0.01, 0.05],
        verbose=True,
        save_results=True,
        output_dir="demo_benchmark_results",
    )

    # Run comprehensive benchmark
    logger.info("Running comprehensive benchmark (this may take a few minutes)...")
    results = run_comprehensive_benchmark(config)

    # Display summary
    logger.info("\n=== Benchmark Results Summary ===")
    logger.info(f"Total runtime: {results.total_runtime:.1f}s")
    logger.info(f"Best quantum method: {results.best_quantum_method}")
    logger.info(f"Best classical method: {results.best_classical_method}")
    logger.info(
        f"Quantum advantage achieved: {'Yes' if results.quantum_advantage_achieved else 'No'}"
    )

    # Show detailed results
    logger.info("\nQuantum VQE Results:")
    for method, metrics in results.quantum_vqe_results.items():
        logger.info(
            f"  {method}: Sharpe={metrics.sharpe_ratio:.3f}, "
            f"Time={metrics.execution_time:.2f}s"
        )

    logger.info("\nQuantum QAOA Results:")
    for method, metrics in results.quantum_qaoa_results.items():
        logger.info(
            f"  {method}: Sharpe={metrics.sharpe_ratio:.3f}, "
            f"Time={metrics.execution_time:.2f}s"
        )

    logger.info("\nClassical Results:")
    for method, metrics in results.classical_results.items():
        logger.info(
            f"  {method}: Sharpe={metrics.sharpe_ratio:.3f}, "
            f"Time={metrics.execution_time:.2f}s"
        )

    # Noise analysis summary
    if results.noise_analysis:
        logger.info("\nNoise Analysis:")
        for noise_level, method_results in results.noise_analysis.items():
            logger.info(f"  Noise level {noise_level}:")
            if "vqe" in method_results:
                vqe_sharpe = method_results["vqe"].sharpe_ratio
                logger.info(f"    VQE Sharpe: {vqe_sharpe:.3f}")

    return results


def create_visualization(benchmark_results):
    """Create simple visualization of benchmark results."""
    logger.info("\n=== Creating Visualization ===")

    try:
        # Collect all methods and their performance
        all_methods = {}
        all_methods.update(benchmark_results.quantum_vqe_results)
        all_methods.update(benchmark_results.quantum_qaoa_results)
        all_methods.update(benchmark_results.classical_results)

        method_names = list(all_methods.keys())
        sharpe_ratios = [all_methods[name].sharpe_ratio for name in method_names]
        execution_times = [all_methods[name].execution_time for name in method_names]

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sharpe ratio comparison
        colors = []
        for name in method_names:
            if (
                name in benchmark_results.quantum_vqe_results
                or name in benchmark_results.quantum_qaoa_results
            ):
                colors.append("red" if "vqe" in name.lower() else "blue")
            else:
                colors.append("green")

        bars1 = ax1.bar(method_names, sharpe_ratios, color=colors, alpha=0.7)
        ax1.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Highlight best performer
        best_idx = np.argmax(sharpe_ratios)
        bars1[best_idx].set_color("gold")
        bars1[best_idx].set_edgecolor("black")
        bars1[best_idx].set_linewidth(2)

        # Execution time comparison
        bars2 = ax2.bar(method_names, execution_times, color=colors, alpha=0.7)
        ax2.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Time (seconds)")
        ax2.set_yscale("log")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.7, label="VQE"),
            Patch(facecolor="blue", alpha=0.7, label="QAOA"),
            Patch(facecolor="green", alpha=0.7, label="Classical"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.95)
        )

        plt.tight_layout()

        # Save plot
        output_dir = Path("demo_benchmark_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight"
        )

        logger.info(f"Visualization saved to {output_dir / 'benchmark_comparison.png'}")

        # Show if running interactively
        # plt.show()

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def main():
    """Run the complete demo."""
    logger.info(
        "QuantumEdge: Noise Models, Quantum Volume, and Classical Comparison Demo"
    )
    logger.info("=" * 80)

    try:
        # 1. Demonstrate noise models
        noise_model = demo_noise_models()

        # 2. Demonstrate quantum volume
        qv_result = demo_quantum_volume()

        # 3. Demonstrate classical solvers
        classical_results, expected_returns, covariance_matrix = (
            demo_classical_solvers()
        )

        # 4. Run comprehensive benchmark
        benchmark_results = demo_comprehensive_benchmark()

        # 5. Create visualization
        create_visualization(benchmark_results)

        logger.info("\n" + "=" * 80)
        logger.info("Demo completed successfully!")
        logger.info("Key achievements:")
        logger.info("✓ Quantum noise modeling with depolarizing channels")
        logger.info("✓ Quantum volume protocol implementation")
        logger.info("✓ Advanced classical optimization algorithms")
        logger.info("✓ Comprehensive quantum vs classical benchmarking")
        logger.info("✓ Performance visualization and analysis")

        # Final summary
        if benchmark_results.quantum_advantage_achieved:
            logger.info("\n🎉 QUANTUM ADVANTAGE DEMONSTRATED! 🎉")
        else:
            logger.info(
                f"\n📊 Classical methods outperformed quantum methods in this test."
            )
            logger.info(
                "This is expected for small problems and noisy quantum simulations."
            )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
