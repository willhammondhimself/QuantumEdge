"""
Unit tests for quantum circuit implementation.
"""

import numpy as np
import pytest
from src.quantum_algorithms.quantum_circuit import QuantumCircuit, Operation


class TestQuantumCircuit:
    """Test suite for QuantumCircuit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.circuit = QuantumCircuit(2)
        self.tolerance = 1e-10

    def test_initialization(self):
        """Test circuit initialization."""
        # Should start in |00⟩ state
        expected_state = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

        # Probabilities should be [1, 0, 0, 0]
        expected_probs = np.array([1, 0, 0, 0])
        assert np.allclose(self.circuit.get_probabilities(), expected_probs)

    def test_single_qubit_gates(self):
        """Test single qubit gate operations."""
        # Apply X gate to first qubit: |00⟩ → |10⟩
        self.circuit.x(0)
        expected_state = np.array([0, 0, 1, 0], dtype=complex)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

        # Apply X gate again: |10⟩ → |00⟩
        self.circuit.x(0)
        expected_state = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

    def test_hadamard_superposition(self):
        """Test Hadamard gate creates superposition."""
        # Apply H to first qubit: |00⟩ → (|00⟩ + |10⟩)/√2
        self.circuit.h(0)
        expected_state = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

        # Probabilities should be [0.5, 0, 0.5, 0]
        expected_probs = np.array([0.5, 0, 0.5, 0])
        assert np.allclose(self.circuit.get_probabilities(), expected_probs)

    def test_cnot_gate(self):
        """Test CNOT gate operation."""
        # Start with |10⟩ state
        self.circuit.x(0)

        # Apply CNOT: |10⟩ → |11⟩
        self.circuit.cnot(0, 1)
        expected_state = np.array([0, 0, 0, 1], dtype=complex)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

    def test_bell_state_creation(self):
        """Test Bell state creation."""
        # Create Bell state: H(0), CNOT(0,1)
        self.circuit.h(0)
        self.circuit.cnot(0, 1)

        # Should be (|00⟩ + |11⟩)/√2
        expected_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

    def test_rotation_gates(self):
        """Test rotation gate operations."""
        # Test Ry rotation
        theta = np.pi / 2
        self.circuit.ry(0, theta)

        # After Ry(π/2): |0⟩ → (|0⟩ + |1⟩)/√2
        state = self.circuit.get_state_vector()
        expected_state = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        assert np.allclose(state, expected_state)

    def test_reset_functionality(self):
        """Test circuit reset."""
        # Modify circuit state
        self.circuit.h(0)
        self.circuit.cnot(0, 1)

        # Reset circuit
        self.circuit.reset()

        # Should be back to |00⟩
        expected_state = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(self.circuit.get_state_vector(), expected_state)

        # Operations should be cleared
        assert len(self.circuit.operations) == 0

    def test_expectation_value(self):
        """Test expectation value calculation."""
        # Prepare |+⟩ state on first qubit
        self.circuit.h(0)

        # Z expectation value should be 0
        z_observable = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        )
        expectation = self.circuit.expectation_value(z_observable)
        assert abs(expectation) < self.tolerance

        # X expectation value should be 1
        x_observable = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
        )
        expectation = self.circuit.expectation_value(x_observable)
        assert abs(expectation - 1.0) < self.tolerance

    def test_operation_tracking(self):
        """Test operation tracking functionality."""
        # Apply some gates
        self.circuit.h(0)
        self.circuit.cnot(0, 1)
        self.circuit.rz(1, np.pi / 4)

        # Check operations are recorded
        assert len(self.circuit.operations) == 3

        # Check operation details
        ops = self.circuit.operations
        assert ops[0].gate == "H"
        assert ops[0].qubits == [0]

        assert ops[1].gate == "CNOT"
        assert ops[1].qubits == [0, 1]

        assert ops[2].gate == "Rz"
        assert ops[2].qubits == [1]
        assert ops[2].params == [np.pi / 4]

    def test_gate_count(self):
        """Test gate counting functionality."""
        # Apply various gates
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.cnot(0, 1)
        self.circuit.x(0)

        gate_count = self.circuit.get_gate_count()
        assert gate_count["H"] == 2
        assert gate_count["CNOT"] == 1
        assert gate_count["X"] == 1

    def test_circuit_depth(self):
        """Test circuit depth calculation."""
        initial_depth = self.circuit.get_circuit_depth()
        assert initial_depth == 0

        # Add some gates
        self.circuit.h(0)
        self.circuit.cnot(0, 1)

        depth = self.circuit.get_circuit_depth()
        assert depth == 2

    def test_multi_qubit_circuit(self):
        """Test larger quantum circuits."""
        # Create 3-qubit circuit
        circuit = QuantumCircuit(3)

        # Create GHZ state: H(0), CNOT(0,1), CNOT(1,2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)

        # Should be (|000⟩ + |111⟩)/√2
        state = circuit.get_state_vector()
        expected_state = np.zeros(8, dtype=complex)
        expected_state[0] = 1 / np.sqrt(2)  # |000⟩
        expected_state[7] = 1 / np.sqrt(2)  # |111⟩

        assert np.allclose(state, expected_state)

    def test_measurement_simulation(self):
        """Test measurement simulation."""
        # Prepare superposition state
        self.circuit.h(0)

        # Measure multiple times
        results = []
        for _ in range(100):
            # Reset to superposition state
            self.circuit.reset()
            self.circuit.h(0)
            result = self.circuit.measure_all()
            results.append(result[0])  # First qubit result

        # Should get roughly equal distribution of 0s and 1s
        zeros = results.count(0)
        ones = results.count(1)

        # Allow for statistical variation (at least 30% of each)
        assert zeros >= 30
        assert ones >= 30

    def test_parameterized_gates(self):
        """Test parameterized gate operations."""
        # Test U3 gate
        theta, phi, lam = np.pi / 3, np.pi / 4, np.pi / 6
        self.circuit.apply_gate("U3", [0], [theta, phi, lam])

        # State should be modified
        state = self.circuit.get_state_vector()
        assert not np.allclose(state, [1, 0, 0, 0])

        # Check operation was recorded with parameters
        op = self.circuit.operations[0]
        assert op.gate == "U3"
        assert op.params == [theta, phi, lam]

    def test_invalid_gate_error(self):
        """Test error handling for invalid gates."""
        with pytest.raises(ValueError, match="Unknown gate"):
            self.circuit.apply_gate("INVALID_GATE", [0])

    def test_string_representation(self):
        """Test string representation of circuit."""
        self.circuit.h(0)
        self.circuit.cnot(0, 1)

        circuit_str = str(self.circuit)
        assert "QuantumCircuit(2 qubits)" in circuit_str
        assert "H on qubits [0]" in circuit_str
        assert "CNOT on qubits [0, 1]" in circuit_str


class TestOperation:
    """Test suite for Operation dataclass."""

    def test_operation_creation(self):
        """Test Operation creation."""
        op = Operation("H", [0])
        assert op.gate == "H"
        assert op.qubits == [0]
        assert op.params is None

        # With parameters
        op_param = Operation("Rx", [0], [np.pi / 2])
        assert op_param.gate == "Rx"
        assert op_param.qubits == [0]
        assert op_param.params == [np.pi / 2]


if __name__ == "__main__":
    pytest.main([__file__])
