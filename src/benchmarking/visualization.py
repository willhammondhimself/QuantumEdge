"""
Visualization tools for quantum vs classical optimization benchmarks.

This module provides comprehensive visualization capabilities for analyzing
and presenting benchmark results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
import logging

from .quantum_classical_comparison import ComparisonResult, PerformanceMetrics
from ..quantum_algorithms import QuantumVolumeResult

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Comprehensive visualization tool for benchmark results."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_plots: bool = True,
        output_dir: str = "plots",
    ):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots
        """
        self.figsize = figsize
        self.save_plots = save_plots
        self.output_dir = output_dir

        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)

    def create_performance_comparison(self, result: ComparisonResult) -> plt.Figure:
        """Create comprehensive performance comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Quantum vs Classical Optimization Performance",
            fontsize=16,
            fontweight="bold",
        )

        # Collect all methods and their metrics
        all_methods = {}
        all_methods.update(result.quantum_vqe_results)
        all_methods.update(result.quantum_qaoa_results)
        all_methods.update(result.classical_results)

        method_names = list(all_methods.keys())

        # Extract metrics
        sharpe_ratios = [all_methods[name].sharpe_ratio for name in method_names]
        returns = [all_methods[name].expected_return for name in method_names]
        volatilities = [all_methods[name].volatility for name in method_names]
        execution_times = [all_methods[name].execution_time for name in method_names]

        # Color mapping
        colors = []
        for name in method_names:
            if (
                name in result.quantum_vqe_results
                or name in result.quantum_qaoa_results
            ):
                colors.append("red" if "vqe" in name.lower() else "blue")
            else:
                colors.append("green")

        # Plot 1: Sharpe Ratio Comparison
        axes[0, 0].bar(method_names, sharpe_ratios, color=colors, alpha=0.7)
        axes[0, 0].set_title("Sharpe Ratio Comparison", fontweight="bold")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Highlight best performers
        best_idx = np.argmax(sharpe_ratios)
        axes[0, 0].bar(
            method_names[best_idx],
            sharpe_ratios[best_idx],
            color="gold",
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )

        # Plot 2: Risk-Return Scatter
        axes[0, 1].scatter(volatilities, returns, c=colors, s=100, alpha=0.7)
        for i, name in enumerate(method_names):
            axes[0, 1].annotate(
                name,
                (volatilities[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axes[0, 1].set_title("Risk-Return Profile", fontweight="bold")
        axes[0, 1].set_xlabel("Volatility")
        axes[0, 1].set_ylabel("Expected Return")
        axes[0, 1].grid(True, alpha=0.3)

        # Add efficient frontier line (approximate)
        vol_range = np.linspace(min(volatilities), max(volatilities), 100)
        max_return = max(returns)
        min_vol = min(volatilities)
        efficient_returns = max_return * (vol_range / min_vol) ** 0.5
        axes[0, 1].plot(
            vol_range,
            efficient_returns,
            "k--",
            alpha=0.3,
            label="Approx. Efficient Frontier",
        )
        axes[0, 1].legend()

        # Plot 3: Execution Time Comparison
        axes[1, 0].bar(method_names, execution_times, color=colors, alpha=0.7)
        axes[1, 0].set_title("Execution Time Comparison", fontweight="bold")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")

        # Plot 4: Success Rate and Quantum Metrics
        success_rates = [all_methods[name].success_rate for name in method_names]
        bars = axes[1, 1].bar(method_names, success_rates, color=colors, alpha=0.7)
        axes[1, 1].set_title("Algorithm Success Rate", fontweight="bold")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Add quantum-specific annotations
        for i, name in enumerate(method_names):
            if (
                hasattr(all_methods[name], "circuit_depth")
                and all_methods[name].circuit_depth
            ):
                axes[1, 1].text(
                    i,
                    success_rates[i] + 0.05,
                    f"Depth: {all_methods[name].circuit_depth}",
                    ha="center",
                    fontsize=8,
                )

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.output_dir}/performance_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def create_noise_analysis_plot(self, result: ComparisonResult) -> plt.Figure:
        """Create noise robustness analysis plot."""
        if not result.noise_analysis:
            logger.warning("No noise analysis data available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Noise Robustness Analysis", fontsize=16, fontweight="bold")

        noise_levels = sorted(result.noise_analysis.keys())

        # Extract quantum method performance under noise
        vqe_sharpe = []
        qaoa_sharpe = []
        classical_sharpe = []

        for noise_level in noise_levels:
            data = result.noise_analysis[noise_level]
            if "vqe" in data:
                vqe_sharpe.append(data["vqe"].sharpe_ratio)
            if "qaoa" in data:
                qaoa_sharpe.append(data["qaoa"].sharpe_ratio)

            # Classical performance (unaffected by quantum noise)
            classical_methods = [k for k in data.keys() if k not in ["vqe", "qaoa"]]
            if classical_methods:
                best_classical = max(
                    classical_methods, key=lambda x: data[x].sharpe_ratio
                )
                classical_sharpe.append(data[best_classical].sharpe_ratio)

        # Plot 1: Sharpe Ratio vs Noise
        axes[0, 0].plot(noise_levels, vqe_sharpe, "ro-", label="VQE", linewidth=2)
        axes[0, 0].plot(noise_levels, qaoa_sharpe, "bo-", label="QAOA", linewidth=2)
        if classical_sharpe:
            axes[0, 0].axhline(
                y=classical_sharpe[0],
                color="green",
                linestyle="--",
                label="Best Classical",
                linewidth=2,
            )
        axes[0, 0].set_title("Sharpe Ratio vs Noise Level", fontweight="bold")
        axes[0, 0].set_xlabel("Noise Level (Error Probability)")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale("log")

        # Plot 2: Fidelity vs Performance
        fidelities = []
        quantum_performance = []
        for noise_level in noise_levels:
            if noise_level > 0:
                # Estimate fidelity based on noise level (simplified)
                estimated_fidelity = (1 - noise_level) ** 10  # Rough approximation
                fidelities.append(estimated_fidelity)

                data = result.noise_analysis[noise_level]
                if "vqe" in data:
                    quantum_performance.append(data["vqe"].sharpe_ratio)

        if fidelities and quantum_performance:
            axes[0, 1].scatter(
                fidelities, quantum_performance, c="red", s=50, alpha=0.7
            )
            axes[0, 1].set_title("Performance vs Circuit Fidelity", fontweight="bold")
            axes[0, 1].set_xlabel("Estimated Circuit Fidelity")
            axes[0, 1].set_ylabel("Sharpe Ratio")
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Execution Time vs Noise
        vqe_times = []
        qaoa_times = []
        for noise_level in noise_levels:
            data = result.noise_analysis[noise_level]
            if "vqe" in data:
                vqe_times.append(data["vqe"].execution_time)
            if "qaoa" in data:
                qaoa_times.append(data["qaoa"].execution_time)

        axes[1, 0].plot(noise_levels, vqe_times, "ro-", label="VQE", linewidth=2)
        axes[1, 0].plot(noise_levels, qaoa_times, "bo-", label="QAOA", linewidth=2)
        axes[1, 0].set_title("Execution Time vs Noise Level", fontweight="bold")
        axes[1, 0].set_xlabel("Noise Level (Error Probability)")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale("log")

        # Plot 4: Success Rate vs Noise
        vqe_success = []
        qaoa_success = []
        for noise_level in noise_levels:
            data = result.noise_analysis[noise_level]
            if "vqe" in data:
                vqe_success.append(data["vqe"].success_rate)
            if "qaoa" in data:
                qaoa_success.append(data["qaoa"].success_rate)

        axes[1, 1].plot(noise_levels, vqe_success, "ro-", label="VQE", linewidth=2)
        axes[1, 1].plot(noise_levels, qaoa_success, "bo-", label="QAOA", linewidth=2)
        axes[1, 1].set_title("Success Rate vs Noise Level", fontweight="bold")
        axes[1, 1].set_xlabel("Noise Level (Error Probability)")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_ylim(0, 1.1)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.output_dir}/noise_analysis.png", dpi=300, bbox_inches="tight"
            )

        return fig

    def create_scalability_plot(self, result: ComparisonResult) -> plt.Figure:
        """Create scalability analysis plot."""
        if not result.scalability_analysis:
            logger.warning("No scalability analysis data available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Scalability Analysis", fontsize=16, fontweight="bold")

        asset_counts = sorted(result.scalability_analysis.keys())

        # Extract performance metrics for different asset counts
        method_performance = {}
        method_times = {}

        for num_assets in asset_counts:
            data = result.scalability_analysis[num_assets]
            for method, metrics in data.items():
                if method not in method_performance:
                    method_performance[method] = []
                    method_times[method] = []
                method_performance[method].append(metrics.sharpe_ratio)
                method_times[method].append(metrics.execution_time)

        # Plot 1: Performance vs Problem Size
        for method, performance in method_performance.items():
            color = (
                "red"
                if "vqe" in method.lower()
                else ("blue" if "qaoa" in method.lower() else "green")
            )
            axes[0, 0].plot(
                asset_counts, performance, "o-", label=method, color=color, linewidth=2
            )

        axes[0, 0].set_title("Performance vs Problem Size", fontweight="bold")
        axes[0, 0].set_xlabel("Number of Assets")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Execution Time vs Problem Size
        for method, times in method_times.items():
            color = (
                "red"
                if "vqe" in method.lower()
                else ("blue" if "qaoa" in method.lower() else "green")
            )
            axes[0, 1].plot(
                asset_counts, times, "o-", label=method, color=color, linewidth=2
            )

        axes[0, 1].set_title("Execution Time vs Problem Size", fontweight="bold")
        axes[0, 1].set_xlabel("Number of Assets")
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        # Plot 3: Quantum vs Classical Performance Ratio
        if "vqe" in method_performance and "mean_variance" in method_performance:
            vqe_perf = method_performance["vqe"]
            classical_perf = method_performance["mean_variance"]
            performance_ratio = [
                v / c if c > 0 else 0 for v, c in zip(vqe_perf, classical_perf)
            ]

            axes[1, 0].plot(
                asset_counts,
                performance_ratio,
                "ro-",
                linewidth=2,
                label="VQE vs Classical",
            )
            axes[1, 0].axhline(
                y=1.0, color="black", linestyle="--", alpha=0.5, label="Parity"
            )
            axes[1, 0].set_title("Quantum Advantage vs Problem Size", fontweight="bold")
            axes[1, 0].set_xlabel("Number of Assets")
            axes[1, 0].set_ylabel("Performance Ratio (Quantum/Classical)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Circuit Complexity
        circuit_depths = []
        for num_assets in asset_counts:
            data = result.scalability_analysis[num_assets]
            if (
                "vqe" in data
                and hasattr(data["vqe"], "circuit_depth")
                and data["vqe"].circuit_depth
            ):
                circuit_depths.append(data["vqe"].circuit_depth)
            else:
                circuit_depths.append(None)

        # Filter out None values
        valid_assets = [
            a for a, d in zip(asset_counts, circuit_depths) if d is not None
        ]
        valid_depths = [d for d in circuit_depths if d is not None]

        if valid_assets and valid_depths:
            axes[1, 1].plot(
                valid_assets,
                valid_depths,
                "ro-",
                linewidth=2,
                label="VQE Circuit Depth",
            )
            axes[1, 1].set_title(
                "Circuit Complexity vs Problem Size", fontweight="bold"
            )
            axes[1, 1].set_xlabel("Number of Assets")
            axes[1, 1].set_ylabel("Circuit Depth")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.output_dir}/scalability_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def create_quantum_volume_plot(
        self, qv_results: Dict[int, QuantumVolumeResult]
    ) -> plt.Figure:
        """Create quantum volume analysis plot."""
        if not qv_results:
            logger.warning("No quantum volume results available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Quantum Volume Analysis", fontsize=16, fontweight="bold")

        qubit_counts = sorted(qv_results.keys())
        quantum_volumes = [qv_results[q].quantum_volume for q in qubit_counts]
        success_rates = [qv_results[q].success_probability for q in qubit_counts]
        heavy_output_probs = [
            qv_results[q].heavy_output_probability for q in qubit_counts
        ]
        fidelity_estimates = [
            qv_results[q].circuit_fidelity_estimate for q in qubit_counts
        ]

        # Plot 1: Quantum Volume vs Qubits
        axes[0, 0].semilogy(
            qubit_counts, quantum_volumes, "bo-", linewidth=2, markersize=8
        )
        axes[0, 0].set_title("Achieved Quantum Volume", fontweight="bold")
        axes[0, 0].set_xlabel("Number of Qubits")
        axes[0, 0].set_ylabel("Quantum Volume")
        axes[0, 0].grid(True, alpha=0.3)

        # Add theoretical maximum line
        theoretical_max = [2**q for q in qubit_counts]
        axes[0, 0].semilogy(
            qubit_counts, theoretical_max, "r--", alpha=0.5, label="Theoretical Maximum"
        )
        axes[0, 0].legend()

        # Plot 2: Success Rate vs Qubits
        axes[0, 1].plot(qubit_counts, success_rates, "go-", linewidth=2, markersize=8)
        axes[0, 1].axhline(
            y=2 / 3, color="red", linestyle="--", alpha=0.7, label="QV Threshold"
        )
        axes[0, 1].set_title("QV Success Rate", fontweight="bold")
        axes[0, 1].set_xlabel("Number of Qubits")
        axes[0, 1].set_ylabel("Success Probability")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Heavy Output Probability
        axes[1, 0].plot(
            qubit_counts, heavy_output_probs, "mo-", linewidth=2, markersize=8
        )
        axes[1, 0].axhline(
            y=0.5, color="black", linestyle="--", alpha=0.5, label="Random Threshold"
        )
        axes[1, 0].set_title("Heavy Output Probability", fontweight="bold")
        axes[1, 0].set_xlabel("Number of Qubits")
        axes[1, 0].set_ylabel("Heavy Output Probability")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Circuit Fidelity
        axes[1, 1].plot(
            qubit_counts, fidelity_estimates, "co-", linewidth=2, markersize=8
        )
        axes[1, 1].set_title("Circuit Fidelity Estimate", fontweight="bold")
        axes[1, 1].set_xlabel("Number of Qubits")
        axes[1, 1].set_ylabel("Fidelity")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.output_dir}/quantum_volume_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def create_summary_report(self, result: ComparisonResult) -> plt.Figure:
        """Create comprehensive summary report."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

        # Title
        fig.suptitle(
            "Quantum vs Classical Portfolio Optimization Benchmark Report",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        # Summary statistics
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis("off")

        summary_text = f"""
        Benchmark Configuration:
        • Assets: {result.config.num_assets}  • Trials: {result.config.num_trials}  • Runtime: {result.total_runtime:.1f}s
        
        Results Summary:
        • Best Quantum Method: {result.best_quantum_method.upper()}
        • Best Classical Method: {result.best_classical_method}
        • Quantum Advantage: {'✓' if result.quantum_advantage_achieved else '✗'}
        """

        ax_summary.text(
            0.05,
            0.5,
            summary_text,
            transform=ax_summary.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
        )

        # Main comparison plots
        self._add_subplot_to_grid(fig, gs[1, 0], result, "sharpe_comparison")
        self._add_subplot_to_grid(fig, gs[1, 1], result, "time_comparison")
        self._add_subplot_to_grid(fig, gs[1, 2], result, "risk_return")

        # Bottom row - additional analyses
        if result.noise_analysis:
            self._add_subplot_to_grid(fig, gs[2, 0], result, "noise_summary")
        if result.scalability_analysis:
            self._add_subplot_to_grid(fig, gs[2, 1], result, "scalability_summary")
        if result.quantum_volume_results:
            self._add_subplot_to_grid(fig, gs[2, 2], result, "qv_summary")

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.output_dir}/benchmark_summary.png", dpi=300, bbox_inches="tight"
            )

        return fig

    def _add_subplot_to_grid(
        self, fig, grid_spec, result: ComparisonResult, plot_type: str
    ):
        """Helper method to add specific plots to grid layout."""
        ax = fig.add_subplot(grid_spec)

        # Collect all methods
        all_methods = {}
        all_methods.update(result.quantum_vqe_results)
        all_methods.update(result.quantum_qaoa_results)
        all_methods.update(result.classical_results)

        if plot_type == "sharpe_comparison":
            method_names = list(all_methods.keys())
            sharpe_ratios = [all_methods[name].sharpe_ratio for name in method_names]
            colors = [
                (
                    "red"
                    if "vqe" in name.lower()
                    else "blue" if "qaoa" in name.lower() else "green"
                )
                for name in method_names
            ]

            ax.bar(method_names, sharpe_ratios, color=colors, alpha=0.7)
            ax.set_title("Sharpe Ratio", fontweight="bold")
            ax.set_ylabel("Sharpe Ratio")
            ax.tick_params(axis="x", rotation=45)

        elif plot_type == "time_comparison":
            method_names = list(all_methods.keys())
            times = [all_methods[name].execution_time for name in method_names]
            colors = [
                (
                    "red"
                    if "vqe" in name.lower()
                    else "blue" if "qaoa" in name.lower() else "green"
                )
                for name in method_names
            ]

            ax.bar(method_names, times, color=colors, alpha=0.7)
            ax.set_title("Execution Time", fontweight="bold")
            ax.set_ylabel("Time (s)")
            ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=45)

        # Add more plot types as needed...

        ax.grid(True, alpha=0.3)


# Convenience functions
def create_performance_comparison_plot(
    result: ComparisonResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a performance comparison plot."""
    visualizer = BenchmarkVisualizer(figsize=figsize, save_plots=save_path is not None)
    fig = visualizer.create_performance_comparison(result)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_convergence_plot(
    convergence_data: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create convergence analysis plot."""
    fig, ax = plt.subplots(figsize=figsize)

    for method, values in convergence_data.items():
        iterations = range(len(values))
        color = (
            "red"
            if "vqe" in method.lower()
            else "blue" if "qaoa" in method.lower() else "green"
        )
        ax.plot(iterations, values, label=method, color=color, linewidth=2)

    ax.set_title("Convergence Analysis", fontweight="bold")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_quantum_volume_plot(
    qv_results: Dict[int, QuantumVolumeResult],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create quantum volume analysis plot."""
    visualizer = BenchmarkVisualizer(figsize=figsize, save_plots=save_path is not None)
    fig = visualizer.create_quantum_volume_plot(qv_results)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
