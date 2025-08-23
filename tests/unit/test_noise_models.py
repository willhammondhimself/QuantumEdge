"""Tests for quantum noise models."""

import unittest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quantum_algorithms.noise_models import (
    NoiseModel,
    DepolarizingChannel,
    NoiseParameters,
    NoiseType,
)
from quantum_algorithms.quantum_circuit import QuantumCircuit


class TestNoiseParameters(unittest.TestCase):
    """Test noise parameters validation."""

    def test_valid_parameters(self):
        """Test valid noise parameters."""
        params = NoiseParameters(error_probability=0.01)
        self.assertEqual(params.error_probability, 0.01)

    def test_invalid_probability(self):
        """Test invalid error probability."""
        with self.assertRaises(ValueError):
            NoiseParameters(error_probability=-0.1)

        with self.assertRaises(ValueError):
            NoiseParameters(error_probability=1.5)


class TestDepolarizingChannel(unittest.TestCase):
    """Test depolarizing noise channel."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = NoiseParameters(error_probability=0.1)
        self.single_qubit_channel = DepolarizingChannel(self.params, num_qubits=1)
        self.two_qubit_channel = DepolarizingChannel(self.params, num_qubits=2)

    def test_single_qubit_kraus_operators(self):
        """Test single-qubit Kraus operators."""
        kraus_ops = self.single_qubit_channel.kraus_operators

        # Should have 4 operators (I, X, Y, Z)
        self.assertEqual(len(kraus_ops), 4)

        # Each should be 2x2
        for op in kraus_ops:
            self.assertEqual(op.shape, (2, 2))

        # Completeness relation: sum(K†K) = I
        total = sum(op.conj().T @ op for op in kraus_ops)
        np.testing.assert_allclose(total, np.eye(2), atol=1e-10)

    def test_two_qubit_kraus_operators(self):
        """Test two-qubit Kraus operators."""
        kraus_ops = self.two_qubit_channel.kraus_operators

        # Should have 16 operators (all Pauli combinations)
        self.assertEqual(len(kraus_ops), 16)

        # Each should be 4x4
        for op in kraus_ops:
            self.assertEqual(op.shape, (4, 4))

        # Completeness relation
        total = sum(op.conj().T @ op for op in kraus_ops)
        np.testing.assert_allclose(total, np.eye(4), atol=1e-10)

    def test_fidelity_calculation(self):
        """Test fidelity calculation."""
        # Single qubit: F = 1 - 3p/4
        expected_fidelity = 1 - 3 * 0.1 / 4
        actual_fidelity = self.single_qubit_channel.get_channel_fidelity()
        self.assertAlmostEqual(actual_fidelity, expected_fidelity, places=10)

        # Two qubit: F = 1 - 15p/16
        expected_fidelity = 1 - 15 * 0.1 / 16
        actual_fidelity = self.two_qubit_channel.get_channel_fidelity()
        self.assertAlmostEqual(actual_fidelity, expected_fidelity, places=10)

    def test_apply_noise(self):
        """Test noise application to state vector."""
        # Create simple 2-qubit state |00⟩
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        # Apply two-qubit noise should return a valid state vector
        noisy_state = self.two_qubit_channel.apply_noise(state, [0, 1])

        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(noisy_state), 1.0, places=10)

        # State should be different (with high probability)
        # Note: This is probabilistic, so we just check it's a valid state
        self.assertEqual(len(noisy_state), 4)
        self.assertTrue(np.all(np.isfinite(noisy_state)))


class TestNoiseModel(unittest.TestCase):
    """Test noise model container."""

    def setUp(self):
        """Set up test fixtures."""
        self.noise_model = NoiseModel()

    def test_empty_noise_model(self):
        """Test empty noise model."""
        # No noise channels should be present
        self.assertEqual(len(self.noise_model.global_noise_channels), 0)
        self.assertEqual(len(self.noise_model.gate_noise_map), 0)
        self.assertEqual(len(self.noise_model.qubit_noise_map), 0)

    def test_add_global_depolarizing_noise(self):
        """Test adding global depolarizing noise."""
        self.noise_model.add_depolarizing_noise(error_probability=0.01)

        # Should have one global noise channel
        self.assertEqual(len(self.noise_model.global_noise_channels), 1)

        # Should be applicable to any gate
        applicable_noise = self.noise_model.get_applicable_noise("X", [0])
        self.assertEqual(len(applicable_noise), 1)

    def test_add_gate_specific_noise(self):
        """Test adding gate-specific noise."""
        self.noise_model.add_depolarizing_noise(
            error_probability=0.01, gate_types=["CNOT"], num_qubits=2
        )

        # Should apply to CNOT gates
        cnot_noise = self.noise_model.get_applicable_noise("CNOT", [0, 1])
        self.assertEqual(len(cnot_noise), 1)

        # Should not apply to single-qubit gates
        x_noise = self.noise_model.get_applicable_noise("X", [0])
        self.assertEqual(len(x_noise), 0)

    def test_add_qubit_specific_noise(self):
        """Test adding qubit-specific noise."""
        self.noise_model.add_depolarizing_noise(
            error_probability=0.01, qubits=[0, 1], num_qubits=1
        )

        # Should apply to operations on specified qubits
        qubit0_noise = self.noise_model.get_applicable_noise("X", [0])
        self.assertEqual(len(qubit0_noise), 1)

        qubit2_noise = self.noise_model.get_applicable_noise("X", [2])
        self.assertEqual(len(qubit2_noise), 0)

    def test_fidelity_estimation(self):
        """Test circuit fidelity estimation."""
        # Empty model should have perfect fidelity
        fidelity = self.noise_model.get_total_fidelity_estimate(circuit_depth=10)
        self.assertEqual(fidelity, 1.0)

        # With noise, fidelity should be less than 1
        self.noise_model.add_depolarizing_noise(error_probability=0.1)
        fidelity = self.noise_model.get_total_fidelity_estimate(circuit_depth=10)
        self.assertLess(fidelity, 1.0)
        self.assertGreater(fidelity, 0.0)


class TestQuantumCircuitNoiseIntegration(unittest.TestCase):
    """Test noise integration with quantum circuits."""

    def test_circuit_with_noise_model(self):
        """Test quantum circuit with noise model."""
        # Create noise model
        noise_model = NoiseModel()
        noise_model.add_depolarizing_noise(error_probability=0.01)

        # Create circuit with noise
        circuit = QuantumCircuit(2, noise_model=noise_model)

        # Apply some gates
        circuit.h(0)
        circuit.cnot(0, 1)

        # Check that noise tracking is working
        self.assertIsNotNone(circuit.noise_model)
        self.assertGreaterEqual(circuit.noise_applications, 0)

        # Get noise statistics
        stats = circuit.get_noise_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("noise_model_active", stats)
        self.assertTrue(stats["noise_model_active"])

    def test_circuit_without_noise(self):
        """Test quantum circuit without noise model."""
        circuit = QuantumCircuit(2)

        # Apply gates
        circuit.h(0)
        circuit.cnot(0, 1)

        # No noise should be applied
        self.assertEqual(circuit.noise_applications, 0)

        # Fidelity should be perfect
        self.assertEqual(circuit.get_estimated_fidelity(), 1.0)

    def test_set_noise_model(self):
        """Test setting noise model after circuit creation."""
        circuit = QuantumCircuit(2)

        # Initially no noise
        self.assertIsNone(circuit.noise_model)

        # Add noise model
        noise_model = NoiseModel()
        noise_model.add_depolarizing_noise(error_probability=0.05)
        circuit.set_noise_model(noise_model)

        # Now should have noise
        self.assertIsNotNone(circuit.noise_model)

        # Future operations should apply noise
        circuit.h(0)
        stats = circuit.get_noise_statistics()
        self.assertTrue(stats["noise_model_active"])


if __name__ == "__main__":
    unittest.main()
