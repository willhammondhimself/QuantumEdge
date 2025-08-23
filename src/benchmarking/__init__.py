"""
Benchmarking framework for quantum vs classical optimization comparison.

This module provides tools for comprehensive performance analysis and
comparison between quantum-inspired and classical portfolio optimization methods.
"""

from .quantum_classical_comparison import (
    ComparisonResult,
    BenchmarkConfiguration,
    QuantumClassicalBenchmark,
    PerformanceMetrics,
    run_comprehensive_benchmark,
)

from .visualization import (
    BenchmarkVisualizer,
    create_performance_comparison_plot,
    create_convergence_plot,
    create_quantum_volume_plot,
)

__all__ = [
    "ComparisonResult",
    "BenchmarkConfiguration",
    "QuantumClassicalBenchmark",
    "PerformanceMetrics",
    "run_comprehensive_benchmark",
    "BenchmarkVisualizer",
    "create_performance_comparison_plot",
    "create_convergence_plot",
    "create_quantum_volume_plot",
]
