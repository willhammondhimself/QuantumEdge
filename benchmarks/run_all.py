#!/usr/bin/env python3
"""
Comprehensive benchmark suite for QuantumEdge performance validation.
Measures optimization speed, memory usage, and throughput across scenarios.
"""

import json
import time
import tracemalloc
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time_ms: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    success: bool
    metadata: Dict[str, Any] = None


class QuantumEdgeBenchmarks:
    """Professional benchmark suite for QuantumEdge performance validation."""
    
    def __init__(self):
        self.results = []
    
    def run_optimization_benchmarks(self) -> None:
        """Run portfolio optimization benchmarks."""
        print("🔬 Running optimization benchmarks...")
        
        # Test scenarios with increasing complexity
        scenarios = [
            {"name": "small_portfolio", "assets": 10, "periods": 252},
            {"name": "medium_portfolio", "assets": 50, "periods": 252},
            {"name": "large_portfolio", "assets": 200, "periods": 252},
            {"name": "xlarge_portfolio", "assets": 500, "periods": 252},
        ]
        
        for scenario in scenarios:
            result = self._benchmark_optimization(**scenario)
            self.results.append(result)
            print(f"  ✅ {scenario['name']}: {result.execution_time_ms:.2f}ms")
    
    def _benchmark_optimization(self, name: str, assets: int, periods: int) -> BenchmarkResult:
        """Benchmark single optimization scenario."""
        # Generate synthetic market data
        np.random.seed(42)  # Reproducible benchmarks
        returns = np.random.multivariate_normal(
            mean=np.full(assets, 0.001),
            cov=self._generate_covariance_matrix(assets),
            size=periods
        )
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            # Simulate optimization (placeholder - replace with actual QuantumEdge calls)
            weights = self._mock_optimization(returns)
            success = True
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            success = False
            weights = np.zeros(assets)
        
        # Measure performance
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        # Calculate throughput (optimizations per second)
        throughput = 1000 / execution_time_ms if execution_time_ms > 0 else 0
        
        return BenchmarkResult(
            name=name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            throughput_ops_per_sec=throughput,
            success=success,
            metadata={
                "assets": assets,
                "periods": periods,
                "weights_sum": float(np.sum(weights)),
                "max_weight": float(np.max(weights))
            }
        )
    
    def _generate_covariance_matrix(self, n_assets: int) -> np.ndarray:
        """Generate realistic covariance matrix."""
        # Create factor-based covariance structure
        n_factors = min(5, n_assets // 3)
        factor_loadings = np.random.normal(0, 0.3, (n_assets, n_factors))
        factor_cov = np.eye(n_factors) * 0.04
        idiosyncratic_var = np.random.uniform(0.01, 0.09, n_assets)
        
        covariance = (
            factor_loadings @ factor_cov @ factor_loadings.T + 
            np.diag(idiosyncratic_var)
        )
        return covariance
    
    def _mock_optimization(self, returns: np.ndarray) -> np.ndarray:
        """Mock optimization for benchmarking (replace with actual implementation)."""
        n_assets = returns.shape[1]
        
        # Simulate computational work
        cov_matrix = np.cov(returns.T)
        mean_returns = np.mean(returns, axis=0)
        
        # Simple mean-variance optimization (placeholder)
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones(n_assets)
        
        # Equal-weight portfolio (fast computation)
        weights = ones / n_assets
        
        # Add some computation to simulate quantum algorithms
        for _ in range(10):
            gradient = 2 * cov_matrix @ weights - mean_returns
            weights -= 0.01 * gradient
            weights = np.maximum(weights, 0)  # Long-only constraint
            weights /= np.sum(weights)  # Normalize
        
        return weights
    
    def run_memory_benchmarks(self) -> None:
        """Run memory usage benchmarks."""
        print("💾 Running memory benchmarks...")
        
        test_cases = [
            {"name": "covariance_100x100", "size": 100},
            {"name": "covariance_500x500", "size": 500},
            {"name": "covariance_1000x1000", "size": 1000},
        ]
        
        for case in test_cases:
            result = self._benchmark_memory_usage(**case)
            self.results.append(result)
            print(f"  ✅ {case['name']}: {result.memory_usage_mb:.1f}MB")
    
    def _benchmark_memory_usage(self, name: str, size: int) -> BenchmarkResult:
        """Benchmark memory usage for matrix operations."""
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            # Allocate and process large matrices
            matrix = np.random.normal(0, 1, (size, size))
            cov_matrix = matrix @ matrix.T / size
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Simulate processing
            processed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            success = True
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            success = False
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        return BenchmarkResult(
            name=name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            throughput_ops_per_sec=1000 / execution_time_ms if execution_time_ms > 0 else 0,
            success=success,
            metadata={"matrix_size": size}
        )
    
    def run_throughput_benchmarks(self) -> None:
        """Run throughput benchmarks."""
        print("⚡ Running throughput benchmarks...")
        
        scenarios = [
            {"name": "batch_optimization_10", "batch_size": 10, "duration_sec": 5},
            {"name": "batch_optimization_50", "batch_size": 50, "duration_sec": 10},
        ]
        
        for scenario in scenarios:
            result = self._benchmark_throughput(**scenario)
            self.results.append(result)
            print(f"  ✅ {scenario['name']}: {result.throughput_ops_per_sec:.0f} ops/sec")
    
    def _benchmark_throughput(self, name: str, batch_size: int, duration_sec: int) -> BenchmarkResult:
        """Benchmark optimization throughput."""
        start_time = time.perf_counter()
        end_time = start_time + duration_sec
        operations_completed = 0
        tracemalloc.start()
        
        try:
            while time.perf_counter() < end_time:
                # Simulate batch optimization
                for _ in range(batch_size):
                    returns = np.random.normal(0.001, 0.02, (252, 20))
                    weights = self._mock_optimization(returns)
                    operations_completed += 1
            success = True
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            success = False
        
        actual_duration = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        throughput = operations_completed / actual_duration
        
        return BenchmarkResult(
            name=name,
            execution_time_ms=actual_duration * 1000,
            memory_usage_mb=memory_usage_mb,
            throughput_ops_per_sec=throughput,
            success=success,
            metadata={
                "operations_completed": operations_completed,
                "target_duration": duration_sec
            }
        )
    
    def generate_report(self, output_format: str = "json", output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        # Organize results by category
        optimization_results = {}
        memory_results = {}
        throughput_results = {}
        
        for result in self.results:
            if "portfolio" in result.name:
                optimization_results[result.name] = result.execution_time_ms
            elif "covariance" in result.name:
                memory_results[result.name] = result.memory_usage_mb
            elif "batch" in result.name:
                throughput_results[result.name] = result.throughput_ops_per_sec
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "successful_benchmarks": sum(1 for r in self.results if r.success),
            "optimization": optimization_results,
            "memory": memory_results,
            "throughput": throughput_results,
            "summary": {
                "fastest_optimization_ms": min(optimization_results.values()) if optimization_results else 0,
                "peak_memory_mb": max(memory_results.values()) if memory_results else 0,
                "max_throughput_ops_sec": max(throughput_results.values()) if throughput_results else 0
            },
            "detailed_results": [asdict(result) for result in self.results]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                if output_format == "json":
                    json.dump(report, f, indent=2)
                else:
                    # CSV format for spreadsheet analysis
                    df = pd.DataFrame([asdict(r) for r in self.results])
                    df.to_csv(f, index=False)
        
        return report


def main():
    """Run comprehensive benchmark suite."""
    parser = argparse.ArgumentParser(description="QuantumEdge Performance Benchmarks")
    parser.add_argument("--output-format", choices=["json", "csv"], default="json",
                       help="Output format for results")
    parser.add_argument("--output-file", type=str, default="benchmark-results.json",
                       help="Output file path")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling")
    
    args = parser.parse_args()
    
    print("🚀 Starting QuantumEdge Performance Benchmarks")
    print("=" * 50)
    
    benchmarks = QuantumEdgeBenchmarks()
    
    # Run all benchmark categories
    benchmarks.run_optimization_benchmarks()
    benchmarks.run_memory_benchmarks()
    benchmarks.run_throughput_benchmarks()
    
    # Generate and save report
    report = benchmarks.generate_report(args.output_format, args.output_file)
    
    print("\n📊 Benchmark Summary:")
    print("=" * 30)
    print(f"Total Benchmarks: {report['total_benchmarks']}")
    print(f"Successful: {report['successful_benchmarks']}")
    print(f"Fastest Optimization: {report['summary']['fastest_optimization_ms']:.2f}ms")
    print(f"Peak Memory Usage: {report['summary']['peak_memory_mb']:.1f}MB")
    print(f"Max Throughput: {report['summary']['max_throughput_ops_sec']:.0f} ops/sec")
    print(f"\n✅ Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()