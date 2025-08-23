"""
Basic quantum gates implementation for classical simulation.

This module provides quantum gate operations using NumPy arrays to simulate
quantum circuits on classical hardware.
"""

import numpy as np
from typing import Tuple, Union, Optional
from functools import lru_cache


class QuantumGates:
    """Implementation of basic quantum gates."""

    def __init__(self, precision: float = 1e-10):
        """
        Initialize quantum gates.

        Args:
            precision: Numerical precision for calculations
        """
        self.precision = precision
        self._sqrt2 = np.sqrt(2)

    @property
    @lru_cache(maxsize=1)
    def I(self) -> np.ndarray:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=complex)

    @property
    @lru_cache(maxsize=1)
    def X(self) -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @property
    @lru_cache(maxsize=1)
    def Y(self) -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @property
    @lru_cache(maxsize=1)
    def Z(self) -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @property
    @lru_cache(maxsize=1)
    def H(self) -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / self._sqrt2

    @property
    @lru_cache(maxsize=1)
    def S(self) -> np.ndarray:
        """Phase gate (S gate)."""
        return np.array([[1, 0], [0, 1j]], dtype=complex)

    @property
    @lru_cache(maxsize=1)
    def T(self) -> np.ndarray:
        """T gate (π/8 gate)."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    def Rx(self, theta: float) -> np.ndarray:
        """
        Rotation around X-axis.

        Args:
            theta: Rotation angle in radians

        Returns:
            2x2 rotation matrix
        """
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex)

    def Ry(self, theta: float) -> np.ndarray:
        """
        Rotation around Y-axis.

        Args:
            theta: Rotation angle in radians

        Returns:
            2x2 rotation matrix
        """
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)

    def Rz(self, theta: float) -> np.ndarray:
        """
        Rotation around Z-axis.

        Args:
            theta: Rotation angle in radians

        Returns:
            2x2 rotation matrix
        """
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )

    def U3(self, theta: float, phi: float, lam: float) -> np.ndarray:
        """
        General single-qubit unitary gate.

        Args:
            theta: Rotation angle around Y-axis
            phi: First phase angle
            lam: Second phase angle

        Returns:
            2x2 unitary matrix
        """
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array(
            [
                [cos, -np.exp(1j * lam) * sin],
                [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos],
            ],
            dtype=complex,
        )

    @property
    @lru_cache(maxsize=1)
    def CNOT(self) -> np.ndarray:
        """Controlled-NOT (CNOT) gate."""
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )

    @property
    @lru_cache(maxsize=1)
    def CZ(self) -> np.ndarray:
        """Controlled-Z gate."""
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )

    def CRz(self, theta: float) -> np.ndarray:
        """
        Controlled rotation around Z-axis.

        Args:
            theta: Rotation angle in radians

        Returns:
            4x4 controlled rotation matrix
        """
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * theta)]],
            dtype=complex,
        )

    @property
    @lru_cache(maxsize=1)
    def SWAP(self) -> np.ndarray:
        """SWAP gate."""
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @property
    @lru_cache(maxsize=1)
    def Toffoli(self) -> np.ndarray:
        """Toffoli (CCNOT) gate."""
        toffoli = np.eye(8, dtype=complex)
        toffoli[6:8, 6:8] = self.X
        return toffoli

    def is_unitary(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is unitary.

        Args:
            matrix: Matrix to check

        Returns:
            True if matrix is unitary
        """
        n = matrix.shape[0]
        product = matrix @ matrix.conj().T
        identity = np.eye(n, dtype=complex)
        return np.allclose(product, identity, atol=self.precision)

    def tensor_product(self, *matrices: np.ndarray) -> np.ndarray:
        """
        Compute tensor product of multiple matrices.

        Args:
            *matrices: Matrices to compute tensor product

        Returns:
            Tensor product of all matrices
        """
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result


def create_controlled_gate(gate: np.ndarray, control_qubits: int = 1) -> np.ndarray:
    """
    Create a controlled version of a gate.

    Args:
        gate: The gate to control
        control_qubits: Number of control qubits

    Returns:
        Controlled gate matrix
    """
    n = gate.shape[0]
    total_dim = 2 ** (control_qubits + int(np.log2(n)))
    controlled = np.eye(total_dim, dtype=complex)

    # Apply gate only when all control qubits are |1⟩
    start_idx = total_dim - n
    controlled[start_idx:, start_idx:] = gate

    return controlled
