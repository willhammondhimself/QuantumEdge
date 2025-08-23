"""
Unit tests for quantum gates implementation.
"""

import numpy as np
import pytest
from src.quantum_algorithms.quantum_gates import QuantumGates, create_controlled_gate


class TestQuantumGates:
    """Test suite for QuantumGates class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gates = QuantumGates()
        self.tolerance = 1e-10

    def test_pauli_gates(self):
        """Test Pauli gates properties."""
        # Test X gate
        assert np.allclose(self.gates.X @ self.gates.X, self.gates.I)

        # Test Y gate
        assert np.allclose(self.gates.Y @ self.gates.Y, self.gates.I)

        # Test Z gate
        assert np.allclose(self.gates.Z @ self.gates.Z, self.gates.I)

    def test_hadamard_gate(self):
        """Test Hadamard gate properties."""
        # H^2 = I
        assert np.allclose(self.gates.H @ self.gates.H, self.gates.I)

        # Test normalization
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(self.gates.H, expected)

    def test_rotation_gates(self):
        """Test rotation gates."""
        theta = np.pi / 4

        # Test Rx rotation
        rx = self.gates.Rx(theta)
        assert self.gates.is_unitary(rx)

        # Test Ry rotation
        ry = self.gates.Ry(theta)
        assert self.gates.is_unitary(ry)

        # Test Rz rotation
        rz = self.gates.Rz(theta)
        assert self.gates.is_unitary(rz)

        # Test 4π rotation returns to identity (2π gives -I due to spinor representation)
        assert np.allclose(self.gates.Rx(4 * np.pi), self.gates.I, atol=self.tolerance)
        assert np.allclose(self.gates.Ry(4 * np.pi), self.gates.I, atol=self.tolerance)
        assert np.allclose(self.gates.Rz(4 * np.pi), self.gates.I, atol=self.tolerance)

        # Test 2π rotation gives -I
        assert np.allclose(self.gates.Rx(2 * np.pi), -self.gates.I, atol=self.tolerance)
        assert np.allclose(self.gates.Ry(2 * np.pi), -self.gates.I, atol=self.tolerance)
        assert np.allclose(self.gates.Rz(2 * np.pi), -self.gates.I, atol=self.tolerance)

    def test_controlled_gates(self):
        """Test controlled gates."""
        # CNOT gate should be unitary
        assert self.gates.is_unitary(self.gates.CNOT)

        # Test CNOT truth table
        # |00⟩ → |00⟩
        state_00 = np.array([1, 0, 0, 0])
        result = self.gates.CNOT @ state_00
        assert np.allclose(result, [1, 0, 0, 0])

        # |01⟩ → |01⟩
        state_01 = np.array([0, 1, 0, 0])
        result = self.gates.CNOT @ state_01
        assert np.allclose(result, [0, 1, 0, 0])

        # |10⟩ → |11⟩
        state_10 = np.array([0, 0, 1, 0])
        result = self.gates.CNOT @ state_10
        assert np.allclose(result, [0, 0, 0, 1])

        # |11⟩ → |10⟩
        state_11 = np.array([0, 0, 0, 1])
        result = self.gates.CNOT @ state_11
        assert np.allclose(result, [0, 0, 1, 0])

    def test_u3_gate(self):
        """Test universal U3 gate."""
        theta, phi, lam = np.pi / 3, np.pi / 4, np.pi / 6
        u3 = self.gates.U3(theta, phi, lam)

        # Should be unitary
        assert self.gates.is_unitary(u3)

        # Test special cases
        # U3(π, 0, π) = X
        u3_x = self.gates.U3(np.pi, 0, np.pi)
        assert np.allclose(u3_x, self.gates.X, atol=self.tolerance)

        # U3(π, π/2, π/2) = Y
        u3_y = self.gates.U3(np.pi, np.pi / 2, np.pi / 2)
        assert np.allclose(u3_y, self.gates.Y, atol=self.tolerance)

    def test_tensor_product(self):
        """Test tensor product functionality."""
        # I ⊗ I should be 4x4 identity
        ii = self.gates.tensor_product(self.gates.I, self.gates.I)
        expected = np.eye(4)
        assert np.allclose(ii, expected)

        # X ⊗ I
        xi = self.gates.tensor_product(self.gates.X, self.gates.I)
        expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.allclose(xi, expected)

    def test_is_unitary(self):
        """Test unitary check function."""
        # Standard gates should be unitary
        assert self.gates.is_unitary(self.gates.I)
        assert self.gates.is_unitary(self.gates.X)
        assert self.gates.is_unitary(self.gates.Y)
        assert self.gates.is_unitary(self.gates.Z)
        assert self.gates.is_unitary(self.gates.H)
        assert self.gates.is_unitary(self.gates.CNOT)

        # Non-unitary matrix should fail
        non_unitary = np.array([[1, 1], [0, 1]])
        assert not self.gates.is_unitary(non_unitary)

    def test_phase_gates(self):
        """Test phase gates."""
        # S gate should be unitary
        assert self.gates.is_unitary(self.gates.S)

        # S^2 = Z
        assert np.allclose(self.gates.S @ self.gates.S, self.gates.Z)

        # T gate should be unitary
        assert self.gates.is_unitary(self.gates.T)

        # T^2 = S
        assert np.allclose(self.gates.T @ self.gates.T, self.gates.S)

    def test_swap_gate(self):
        """Test SWAP gate."""
        # SWAP should be unitary
        assert self.gates.is_unitary(self.gates.SWAP)

        # SWAP^2 = I
        swap_squared = self.gates.SWAP @ self.gates.SWAP
        assert np.allclose(swap_squared, np.eye(4))

        # Test SWAP truth table
        # |01⟩ → |10⟩
        state_01 = np.array([0, 1, 0, 0])
        result = self.gates.SWAP @ state_01
        assert np.allclose(result, [0, 0, 1, 0])

        # |10⟩ → |01⟩
        state_10 = np.array([0, 0, 1, 0])
        result = self.gates.SWAP @ state_10
        assert np.allclose(result, [0, 1, 0, 0])

    def test_controlled_rotation(self):
        """Test controlled rotation gates."""
        theta = np.pi / 3
        crz = self.gates.CRz(theta)

        # Should be unitary
        assert self.gates.is_unitary(crz)

        # Should be 4x4 matrix
        assert crz.shape == (4, 4)


class TestControlledGateConstruction:
    """Test controlled gate construction utility."""

    def test_controlled_x_gate(self):
        """Test controlled X gate construction."""
        gates = QuantumGates()
        controlled_x = create_controlled_gate(gates.X)

        # Should be equivalent to CNOT
        assert np.allclose(controlled_x, gates.CNOT)

    def test_controlled_z_gate(self):
        """Test controlled Z gate construction."""
        gates = QuantumGates()
        controlled_z = create_controlled_gate(gates.Z)

        # Should be equivalent to CZ
        assert np.allclose(controlled_z, gates.CZ)


if __name__ == "__main__":
    pytest.main([__file__])
