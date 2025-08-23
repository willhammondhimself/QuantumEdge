"""Tests for quantum volume protocol."""

import unittest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quantum_algorithms.quantum_volume import (
    QuantumVolumeProtocol,
    QuantumVolumeResult,
    QVCircuitSpecification,
)
from quantum_algorithms.quantum_circuit import QuantumCircuit
from quantum_algorithms.noise_models import NoiseModel


class TestQVCircuitSpecification(unittest.TestCase):
    """Test quantum volume circuit specification."""

    def test_default_specification(self):
        """Test default circuit specification."""
        spec = QVCircuitSpecification(num_qubits=4, depth=4)

        self.assertEqual(spec.num_qubits, 4)
        self.assertEqual(spec.depth, 4)
        self.assertIsNotNone(spec.gate_set)
        self.assertIn("H", spec.gate_set)
        self.assertIn("CNOT", spec.gate_set)

    def test_custom_specification(self):
        """Test custom circuit specification."""
        custom_gates = ["H", "CNOT", "Ry"]
        spec = QVCircuitSpecification(
            num_qubits=3, depth=5, gate_set=custom_gates, seed=42
        )

        self.assertEqual(spec.num_qubits, 3)
        self.assertEqual(spec.depth, 5)
        self.assertEqual(spec.gate_set, custom_gates)
        self.assertEqual(spec.seed, 42)


class TestQuantumVolumeProtocol(unittest.TestCase):
    """Test quantum volume protocol implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.protocol = QuantumVolumeProtocol()
        self.spec = QVCircuitSpecification(num_qubits=3, depth=3, seed=42)

    def test_protocol_initialization(self):
        """Test protocol initialization."""
        self.assertEqual(self.protocol.confidence_level, 0.95)
        self.assertEqual(self.protocol.min_success_probability, 2.0 / 3.0)
        self.assertGreater(self.protocol.min_circuits, 0)
        self.assertGreater(self.protocol.min_shots_per_circuit, 0)

    def test_random_circuit_generation(self):
        """Test random circuit generation."""
        circuit = self.protocol.generate_random_circuit(self.spec)

        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.spec.num_qubits)
        self.assertGreater(len(circuit.operations), 0)

    def test_reproducible_circuit_generation(self):
        """Test that circuits are reproducible with same seed."""
        spec1 = QVCircuitSpecification(num_qubits=3, depth=3, seed=42)
        spec2 = QVCircuitSpecification(num_qubits=3, depth=3, seed=42)

        circuit1 = self.protocol.generate_random_circuit(spec1)
        circuit2 = self.protocol.generate_random_circuit(spec2)

        # Should have same number of operations
        self.assertEqual(len(circuit1.operations), len(circuit2.operations))

        # Operations should be the same
        for op1, op2 in zip(circuit1.operations, circuit2.operations):
            self.assertEqual(op1.gate, op2.gate)
            self.assertEqual(op1.qubits, op2.qubits)

    def test_heavy_output_probability_calculation(self):
        """Test heavy output probability calculation."""
        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)  # Create superposition

        # Calculate heavy output probability
        heavy_prob = self.protocol.compute_heavy_output_probability(
            circuit, num_shots=100
        )

        # Should be a valid probability
        self.assertGreaterEqual(heavy_prob, 0.0)
        self.assertLessEqual(heavy_prob, 1.0)

    def test_quantum_volume_protocol_small(self):
        """Test quantum volume protocol on small instance."""
        # Run on very small instance for fast testing
        result = self.protocol.run_quantum_volume_protocol(
            num_qubits=2,
            num_circuits=10,  # Reduced for testing
            num_shots=100,  # Reduced for testing
        )

        self.assertIsInstance(result, QuantumVolumeResult)
        self.assertEqual(result.width, 2)
        self.assertEqual(result.depth, 2)
        self.assertEqual(result.num_circuits, 10)
        self.assertEqual(result.num_shots_per_circuit, 100)
        self.assertGreater(result.execution_time, 0)

        # Success probability should be valid
        self.assertGreaterEqual(result.success_probability, 0.0)
        self.assertLessEqual(result.success_probability, 1.0)

        # Quantum volume should be reasonable
        self.assertGreaterEqual(result.quantum_volume, 0)

    def test_quantum_volume_with_noise(self):
        """Test quantum volume with noise model."""
        # Create noise model
        noise_model = NoiseModel()
        noise_model.add_depolarizing_noise(error_probability=0.1)

        # Run protocol with noise
        result = self.protocol.run_quantum_volume_protocol(
            num_qubits=2,
            num_circuits=5,  # Very small for testing
            num_shots=50,
            noise_model=noise_model,
        )

        self.assertIsInstance(result, QuantumVolumeResult)
        # Noise should generally reduce performance
        self.assertLessEqual(result.circuit_fidelity_estimate, 1.0)

    def test_circuit_complexity_analysis(self):
        """Test circuit complexity analysis."""
        # Create a test circuit
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.ry(1, np.pi / 4)
        circuit.cnot(1, 2)

        analysis = self.protocol.analyze_circuit_complexity(circuit)

        self.assertIsInstance(analysis, dict)
        self.assertIn("total_gates", analysis)
        self.assertIn("single_qubit_gates", analysis)
        self.assertIn("two_qubit_gates", analysis)
        self.assertIn("circuit_depth", analysis)
        self.assertIn("circuit_width", analysis)
        self.assertIn("complexity_score", analysis)

        # Check values make sense
        self.assertEqual(analysis["total_gates"], 4)
        self.assertEqual(analysis["two_qubit_gates"], 2)
        self.assertEqual(analysis["circuit_width"], 3)
        self.assertGreater(analysis["complexity_score"], 0)

    def test_max_quantum_volume_estimation(self):
        """Test maximum quantum volume estimation."""
        # Run quick test for small systems
        results = self.protocol.estimate_max_quantum_volume(
            max_qubits=3, quick_test=True
        )

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

        # Each result should be valid
        for num_qubits, result in results.items():
            self.assertIsInstance(result, QuantumVolumeResult)
            self.assertEqual(result.width, num_qubits)
            self.assertEqual(result.depth, num_qubits)


class TestQuantumVolumeResult(unittest.TestCase):
    """Test quantum volume result validation."""

    def test_valid_result(self):
        """Test valid quantum volume result."""
        result = QuantumVolumeResult(
            quantum_volume=8,
            depth=3,
            width=3,
            success_probability=0.75,
            confidence_interval=(0.65, 0.85),
            num_circuits=100,
            num_shots_per_circuit=1024,
            heavy_output_probability=0.6,
            threshold_probability=2 / 3,
            success=True,
            execution_time=120.5,
            circuit_fidelity_estimate=0.95,
        )

        self.assertEqual(result.quantum_volume, 8)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.success_probability, 0.75)

    def test_invalid_probability(self):
        """Test invalid probability values."""
        with self.assertRaises(ValueError):
            QuantumVolumeResult(
                quantum_volume=0,
                depth=2,
                width=2,
                success_probability=1.5,  # Invalid
                confidence_interval=(0, 1),
                num_circuits=100,
                num_shots_per_circuit=1024,
                heavy_output_probability=0.5,
                threshold_probability=2 / 3,
                success=False,
                execution_time=10,
                circuit_fidelity_estimate=1.0,
            )

        with self.assertRaises(ValueError):
            QuantumVolumeResult(
                quantum_volume=0,
                depth=2,
                width=2,
                success_probability=0.5,
                confidence_interval=(0, 1),
                num_circuits=100,
                num_shots_per_circuit=1024,
                heavy_output_probability=1.5,  # Invalid
                threshold_probability=2 / 3,
                success=False,
                execution_time=10,
                circuit_fidelity_estimate=1.0,
            )


if __name__ == "__main__":
    unittest.main()
