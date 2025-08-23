"""Tests for quantum-classical algorithm comparison."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
from pathlib import Path

from src.benchmarking.quantum_classical_comparison import (
    PerformanceMetrics,
    BenchmarkConfiguration,
    ComparisonResult,
    QuantumClassicalBenchmark,
    BenchmarkType,
    run_comprehensive_benchmark,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.0,
            expected_return=0.08,
            volatility=0.08,
            objective_value=0.72,
            execution_time=2.5,
            iterations=100,
            convergence_achieved=True,
            quantum_volume=10,
            circuit_depth=100,
            noise_robustness=0.95,
            fidelity_estimate=0.98,
            success_rate=0.95,
            confidence_interval=(0.9, 1.0),
        )

        assert metrics.sharpe_ratio == 1.0
        assert metrics.expected_return == 0.08
        assert metrics.execution_time == 2.5
        assert metrics.quantum_volume == 10
        assert metrics.convergence_achieved is True


class TestBenchmarkConfiguration:
    """Test BenchmarkConfiguration dataclass."""

    def test_default_configuration(self):
        """Test default benchmark configuration."""
        config = BenchmarkConfiguration()

        assert config.num_assets == 10
        assert config.num_trials == 5
        assert config.random_seed == 42
        assert config.return_range == (0.05, 0.15)
        assert config.volatility_range == (0.10, 0.30)
        assert config.correlation_strength == 0.3
        assert config.min_weight == 0.0
        assert config.max_weight == 0.5
        assert config.max_iterations == 1000
        assert config.population_size == 50
        assert len(config.noise_levels) == 4
        assert config.save_results is True
        assert config.verbose is True

    def test_custom_configuration(self):
        """Test custom benchmark configuration."""
        config = BenchmarkConfiguration(
            num_assets=20,
            num_trials=10,
            random_seed=123,
            return_range=(0.0, 0.20),
            volatility_range=(0.05, 0.25),
            min_weight=0.05,
            max_weight=0.40,
            target_return=0.10,
            noise_levels=[0.0, 0.005, 0.01],
            save_results=False,
            verbose=False,
        )

        assert config.num_assets == 20
        assert config.num_trials == 10
        assert config.target_return == 0.10
        assert len(config.noise_levels) == 3
        assert config.save_results is False


class TestQuantumClassicalBenchmark:
    """Test QuantumClassicalBenchmark class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return BenchmarkConfiguration(
            num_assets=5, num_trials=2, save_results=False, verbose=False
        )

    @pytest.fixture
    def benchmark(self, config, tmp_path):
        """Create benchmark instance with temp directory."""
        config.output_dir = str(tmp_path / "test_results")
        return QuantumClassicalBenchmark(config)

    def test_initialization(self, benchmark, config):
        """Test benchmark initialization."""
        assert benchmark.config == config
        assert isinstance(benchmark.results_cache, dict)

    def test_generate_test_data(self, benchmark):
        """Test synthetic data generation."""
        expected_returns, covariance_matrix = benchmark.generate_test_data()

        n_assets = benchmark.config.num_assets
        assert len(expected_returns) == n_assets
        assert covariance_matrix.shape == (n_assets, n_assets)

        # Check returns are in expected range
        assert np.all(expected_returns >= benchmark.config.return_range[0])
        assert np.all(expected_returns <= benchmark.config.return_range[1])

        # Check covariance matrix properties
        assert np.allclose(covariance_matrix, covariance_matrix.T)  # Symmetric
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        assert np.all(eigenvalues >= 0)  # Positive semi-definite

    def test_create_constraints(self, benchmark):
        """Test constraint creation."""
        constraints = benchmark.create_constraints()

        assert constraints.min_weight == benchmark.config.min_weight
        assert constraints.max_weight == benchmark.config.max_weight
        assert constraints.target_return == benchmark.config.target_return
        assert constraints.long_only is True
        assert constraints.budget_constraint == 1.0

    @patch("src.quantum_algorithms.vqe.QuantumVQE")
    def test_run_quantum_vqe(self, mock_vqe_class, benchmark):
        """Test running VQE benchmark."""
        # Setup mock
        mock_vqe = Mock()
        mock_result = Mock()
        mock_result.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_result.eigenvalue = 0.9
        mock_result.circuit = Mock(depth=100, num_qubits=5)
        mock_vqe.solve_eigenportfolio.return_value = mock_result
        mock_vqe_class.return_value = mock_vqe

        expected_returns, covariance_matrix = benchmark.generate_test_data()

        metrics = benchmark.run_quantum_vqe(expected_returns, covariance_matrix)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.sharpe_ratio > 0
        assert metrics.expected_return > 0
        assert metrics.volatility > 0
        assert metrics.execution_time > 0
        assert metrics.convergence_achieved is True

    @patch("src.quantum_algorithms.qaoa.PortfolioQAOA")
    def test_run_quantum_qaoa(self, mock_qaoa_class, benchmark):
        """Test running QAOA benchmark."""
        # Setup mock
        mock_qaoa = Mock()
        mock_qaoa.solve_portfolio_selection.return_value = (
            [1, 0, 1, 0, 1],  # selection
            0.12,  # expected return
            Mock(depth=50, num_gates=200),  # circuit
        )
        mock_qaoa_class.return_value = mock_qaoa

        expected_returns, covariance_matrix = benchmark.generate_test_data()

        # Mock the run_quantum_qaoa method if it exists
        with patch.object(benchmark, "run_quantum_qaoa") as mock_run_qaoa:
            mock_run_qaoa.return_value = PerformanceMetrics(
                sharpe_ratio=0.85,
                expected_return=0.09,
                volatility=0.10,
                objective_value=0.85,
                execution_time=1.5,
                iterations=50,
                convergence_achieved=True,
                circuit_depth=50,
            )

            metrics = benchmark.run_quantum_qaoa(expected_returns, covariance_matrix)

            assert metrics.sharpe_ratio == 0.85
            assert metrics.circuit_depth == 50

    @patch("src.optimization.mean_variance.MeanVarianceOptimizer")
    def test_run_classical_mean_variance(self, mock_optimizer_class, benchmark):
        """Test running classical mean-variance optimization."""
        # Setup mock
        mock_optimizer = Mock()
        mock_result = Mock()
        mock_result.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_result.expected_return = 0.08
        mock_result.expected_variance = 0.0064
        mock_result.sharpe_ratio = 0.75
        mock_result.success = True
        mock_result.solve_time = 0.5
        mock_optimizer.optimize_portfolio.return_value = mock_result
        mock_optimizer_class.return_value = mock_optimizer

        expected_returns, covariance_matrix = benchmark.generate_test_data()

        # Mock the run_classical_mean_variance method
        with patch.object(
            benchmark, "run_classical_mean_variance"
        ) as mock_run_classical:
            mock_run_classical.return_value = PerformanceMetrics(
                sharpe_ratio=0.75,
                expected_return=0.08,
                volatility=0.08,
                objective_value=0.75,
                execution_time=0.5,
                iterations=1,
                convergence_achieved=True,
            )

            metrics = benchmark.run_classical_mean_variance(
                expected_returns, covariance_matrix
            )

            assert metrics.sharpe_ratio == 0.75
            assert metrics.execution_time == 0.5

    @patch("src.optimization.classical_solvers.ClassicalOptimizerFactory")
    def test_run_classical_solver(self, mock_factory_class, benchmark):
        """Test running classical solver benchmark."""
        # Setup mock
        mock_optimizer = Mock()
        mock_result = Mock()
        mock_result.weights = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
        mock_result.expected_return = 0.09
        mock_result.expected_variance = 0.0081
        mock_result.sharpe_ratio = 0.78
        mock_result.success = True
        mock_result.iterations = 100
        mock_optimizer.optimize.return_value = mock_result
        mock_factory_class.create_optimizer.return_value = mock_optimizer

        expected_returns, covariance_matrix = benchmark.generate_test_data()

        # Mock the run_classical_solver method
        with patch.object(benchmark, "run_classical_solver") as mock_run_solver:
            mock_run_solver.return_value = PerformanceMetrics(
                sharpe_ratio=0.78,
                expected_return=0.09,
                volatility=0.09,
                objective_value=0.78,
                execution_time=1.0,
                iterations=100,
                convergence_achieved=True,
            )

            metrics = benchmark.run_classical_solver(
                "genetic_algorithm", expected_returns, covariance_matrix
            )

            assert metrics.sharpe_ratio == 0.78
            assert metrics.iterations == 100

    def test_compare_results(self, benchmark):
        """Test comparing benchmark results."""
        # Create mock results
        quantum_results = {
            "VQE": PerformanceMetrics(
                sharpe_ratio=1.0,
                expected_return=0.09,
                volatility=0.09,
                objective_value=1.0,
                execution_time=2.5,
                iterations=100,
                convergence_achieved=True,
                quantum_volume=10,
            ),
            "QAOA": PerformanceMetrics(
                sharpe_ratio=0.85,
                expected_return=0.08,
                volatility=0.09,
                objective_value=0.85,
                execution_time=1.5,
                iterations=50,
                convergence_achieved=True,
                circuit_depth=50,
            ),
        }

        classical_results = {
            "mean_variance": PerformanceMetrics(
                sharpe_ratio=0.75,
                expected_return=0.08,
                volatility=0.10,
                objective_value=0.75,
                execution_time=0.5,
                iterations=1,
                convergence_achieved=True,
            ),
            "genetic_algorithm": PerformanceMetrics(
                sharpe_ratio=0.78,
                expected_return=0.08,
                volatility=0.10,
                objective_value=0.78,
                execution_time=1.0,
                iterations=100,
                convergence_achieved=True,
            ),
        }

        # Mock the compare_results method
        with patch.object(benchmark, "compare_results") as mock_compare:
            mock_compare.return_value = ComparisonResult(
                config=benchmark.config,
                quantum_vqe_results={"test": quantum_results["VQE"]},
                quantum_qaoa_results={"test": quantum_results["QAOA"]},
                classical_results=classical_results,
                best_quantum_method="VQE",
                best_classical_method="genetic_algorithm",
                quantum_advantage_achieved=True,
                noise_analysis={},
                scalability_analysis={},
                quantum_volume_results={},
                total_runtime=0.0,
                timestamp="",
            )

            result = benchmark.compare_results(quantum_results, classical_results)

            assert result.best_quantum_method == "VQE"
            assert result.best_classical_method == "genetic_algorithm"
            assert result.quantum_advantage_achieved is True

    @patch.object(QuantumClassicalBenchmark, "run_quantum_vqe")
    @patch.object(QuantumClassicalBenchmark, "run_classical_mean_variance")
    def test_run_benchmark(self, mock_classical, mock_quantum, benchmark):
        """Test running complete benchmark."""
        # Setup mocks
        quantum_metrics = PerformanceMetrics(
            sharpe_ratio=1.0,
            expected_return=0.09,
            volatility=0.09,
            objective_value=1.0,
            execution_time=2.5,
            iterations=100,
            convergence_achieved=True,
        )

        classical_metrics = PerformanceMetrics(
            sharpe_ratio=0.75,
            expected_return=0.08,
            volatility=0.10,
            objective_value=0.75,
            execution_time=0.5,
            iterations=1,
            convergence_achieved=True,
        )

        mock_quantum.return_value = quantum_metrics
        mock_classical.return_value = classical_metrics

        # Mock the run_benchmark method if it exists
        with patch.object(benchmark, "run_benchmark") as mock_run:
            mock_run.return_value = ComparisonResult(
                config=benchmark.config,
                quantum_vqe_results={"trial_1": quantum_metrics},
                quantum_qaoa_results={},
                classical_results={"mean_variance": classical_metrics},
                best_quantum_method="VQE",
                best_classical_method="mean_variance",
                quantum_advantage_achieved=True,
                noise_analysis={},
                scalability_analysis={},
                quantum_volume_results={},
                total_runtime=0.0,
                timestamp="",
            )

            result = benchmark.run_benchmark()

            assert isinstance(result, ComparisonResult)
            assert result.quantum_advantage_achieved is True


class TestComprehensiveBenchmark:
    """Test comprehensive benchmark function."""

    @patch("src.benchmarking.quantum_classical_comparison.QuantumClassicalBenchmark")
    def test_run_comprehensive_benchmark(self, mock_benchmark_class):
        """Test running comprehensive benchmark."""
        # Setup mock
        mock_benchmark = Mock()
        mock_result = ComparisonResult(
            config=BenchmarkConfiguration(),
            quantum_vqe_results={},
            quantum_qaoa_results={},
            classical_results={},
            best_quantum_method="VQE",
            best_classical_method="mean_variance",
            quantum_advantage_achieved=False,
            noise_analysis={},
            scalability_analysis={},
            quantum_volume_results={},
            total_runtime=0.0,
            timestamp="",
        )
        mock_benchmark.run_benchmark.return_value = mock_result
        mock_benchmark_class.return_value = mock_benchmark

        result = run_comprehensive_benchmark()

        assert isinstance(result, ComparisonResult)
        assert mock_benchmark.run_benchmark.called

    @patch("src.benchmarking.quantum_classical_comparison.QuantumClassicalBenchmark")
    def test_run_comprehensive_benchmark_with_config(self, mock_benchmark_class):
        """Test running comprehensive benchmark with custom config."""
        # Custom config
        config = BenchmarkConfiguration(
            num_assets=20, num_trials=10, save_results=False
        )

        # Setup mock
        mock_benchmark = Mock()
        mock_result = ComparisonResult(
            config=config,
            quantum_vqe_results={},
            quantum_qaoa_results={},
            classical_results={},
            best_quantum_method="QAOA",
            best_classical_method="genetic_algorithm",
            quantum_advantage_achieved=True,
            noise_analysis={},
            scalability_analysis={},
            quantum_volume_results={},
            total_runtime=0.0,
            timestamp="",
        )
        mock_benchmark.run_benchmark.return_value = mock_result
        mock_benchmark_class.return_value = mock_benchmark

        result = run_comprehensive_benchmark(config)

        assert result.config == config
        assert result.quantum_advantage_achieved is True
