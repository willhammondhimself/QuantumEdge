"""
Quantum noise models for realistic circuit simulation.

This module implements various quantum noise channels with a focus on
depolarizing noise for portfolio optimization algorithm evaluation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of quantum noise channels."""

    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    BIT_PHASE_FLIP = "bit_phase_flip"


@dataclass
class NoiseParameters:
    """Parameters for noise channels."""

    error_probability: float = 0.001
    gate_types: List[str] = field(default_factory=lambda: ["all"])
    qubits: Optional[List[int]] = None

    def __post_init__(self):
        """Validate noise parameters."""
        if not 0 <= self.error_probability <= 1:
            raise ValueError(
                f"Error probability must be in [0,1], got {self.error_probability}"
            )


class NoiseChannel(ABC):
    """Abstract base class for quantum noise channels."""

    def __init__(self, parameters: NoiseParameters):
        """Initialize noise channel with parameters."""
        self.parameters = parameters
        self.kraus_operators = self._compute_kraus_operators()

    @abstractmethod
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for the noise channel."""
        pass

    @abstractmethod
    def apply_noise(self, state_vector: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Apply noise to state vector on specified qubits."""
        pass

    def get_channel_fidelity(self) -> float:
        """Calculate average fidelity of the noise channel."""
        p = self.parameters.error_probability
        if hasattr(self, "_fidelity_formula"):
            return self._fidelity_formula(p)
        return 1.0 - p  # Default approximation


class DepolarizingChannel(NoiseChannel):
    """
    Depolarizing noise channel implementation.

    Models the quantum depolarizing channel where a qubit state is
    replaced with maximally mixed state with probability p.
    """

    def __init__(self, parameters: NoiseParameters, num_qubits: int = 1):
        """
        Initialize depolarizing channel.

        Args:
            parameters: Noise parameters
            num_qubits: Number of qubits the channel acts on (1 or 2)
        """
        self.num_qubits = num_qubits
        if num_qubits not in [1, 2]:
            raise ValueError("Depolarizing channel supports only 1 or 2 qubits")
        super().__init__(parameters)

    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for depolarizing channel."""
        p = self.parameters.error_probability

        if self.num_qubits == 1:
            return self._single_qubit_kraus_operators(p)
        else:
            return self._two_qubit_kraus_operators(p)

    def _single_qubit_kraus_operators(self, p: float) -> List[np.ndarray]:
        """Generate single-qubit depolarizing Kraus operators."""
        # Pauli matrices
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Kraus operators for single-qubit depolarizing channel
        sqrt_factor = np.sqrt(1 - 3 * p / 4)
        error_factor = np.sqrt(p / 4)

        kraus_ops = [
            sqrt_factor * I,  # No error
            error_factor * X,  # Bit flip
            error_factor * Y,  # Bit-phase flip
            error_factor * Z,  # Phase flip
        ]

        return kraus_ops

    def _two_qubit_kraus_operators(self, p: float) -> List[np.ndarray]:
        """Generate two-qubit depolarizing Kraus operators."""
        # Single qubit Paulis
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Two-qubit Pauli basis
        paulis = [I, X, Y, Z]

        # Generate all 16 two-qubit Pauli operators
        two_qubit_paulis = []
        for p1 in paulis:
            for p2 in paulis:
                two_qubit_paulis.append(np.kron(p1, p2))

        # Kraus operators for two-qubit depolarizing channel
        sqrt_no_error = np.sqrt(1 - 15 * p / 16)
        sqrt_error = np.sqrt(p / 16)

        kraus_ops = [sqrt_no_error * two_qubit_paulis[0]]  # II (no error)

        # Add error terms (exclude II)
        for i in range(1, 16):
            kraus_ops.append(sqrt_error * two_qubit_paulis[i])

        return kraus_ops

    def apply_noise(self, state_vector: np.ndarray, qubits: List[int]) -> np.ndarray:
        """
        Apply depolarizing noise to state vector.

        Args:
            state_vector: Current quantum state vector
            qubits: Qubits to apply noise to

        Returns:
            State vector after applying noise
        """
        if len(qubits) != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} qubits, got {len(qubits)}")

        # For state vector simulation, we sample from the Kraus operators
        # In practice, this introduces stochastic behavior
        probabilities = [
            np.real(np.trace(K.conj().T @ K)) for K in self.kraus_operators
        ]

        # Normalize probabilities to ensure they sum to 1
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)

        # Randomly select which Kraus operator to apply
        selected_op = np.random.choice(len(self.kraus_operators), p=probabilities)
        kraus_op = self.kraus_operators[selected_op]

        # Expand Kraus operator to full circuit size
        full_kraus_op = self._expand_to_full_circuit(
            kraus_op, qubits, len(state_vector)
        )

        # Apply the selected Kraus operator
        new_state = full_kraus_op @ state_vector

        # Renormalize (should be approximately normalized already)
        norm = np.linalg.norm(new_state)
        if norm > 1e-10:
            new_state /= norm

        return new_state

    def _expand_to_full_circuit(
        self, kraus_op: np.ndarray, target_qubits: List[int], state_size: int
    ) -> np.ndarray:
        """Expand Kraus operator to full circuit size."""
        num_total_qubits = int(np.log2(state_size))

        if self.num_qubits == 1:
            return self._expand_single_qubit_kraus(
                kraus_op, target_qubits[0], num_total_qubits
            )
        else:
            return self._expand_two_qubit_kraus(
                kraus_op, target_qubits, num_total_qubits
            )

    def _expand_single_qubit_kraus(
        self, kraus_op: np.ndarray, target: int, num_total_qubits: int
    ) -> np.ndarray:
        """Expand single-qubit Kraus operator to full circuit."""
        # Build tensor product with identities
        ops = []
        I = np.eye(2, dtype=complex)

        for i in range(num_total_qubits):
            if i == target:
                ops.append(kraus_op)
            else:
                ops.append(I)

        # Compute tensor product
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)

        return result

    def _expand_two_qubit_kraus(
        self, kraus_op: np.ndarray, targets: List[int], num_total_qubits: int
    ) -> np.ndarray:
        """Expand two-qubit Kraus operator to full circuit."""
        if len(targets) != 2:
            raise ValueError("Two-qubit Kraus expansion requires exactly 2 targets")

        # For simplicity, assume targets are adjacent or handle general case
        # This is a simplified implementation - production version would be more sophisticated
        ops = []
        I = np.eye(2, dtype=complex)
        targets_set = set(targets)
        kraus_applied = False

        for i in range(num_total_qubits):
            if i in targets_set and not kraus_applied:
                ops.append(kraus_op)
                kraus_applied = True
                # Skip next qubit if it's the other target
                if i + 1 in targets_set:
                    continue
            elif i not in targets_set:
                ops.append(I)

        # Compute tensor product
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)

        return result

    def _fidelity_formula(self, p: float) -> float:
        """Calculate exact fidelity for depolarizing channel."""
        if self.num_qubits == 1:
            return 1 - 3 * p / 4
        else:
            return 1 - 15 * p / 16


class NoiseModel:
    """
    Container for managing multiple noise channels in a quantum circuit.

    Provides a centralized way to specify and apply different types of noise
    to different gates and qubits in a quantum circuit.
    """

    def __init__(self):
        """Initialize empty noise model."""
        self.noise_channels: Dict[str, List[NoiseChannel]] = {}
        self.gate_noise_map: Dict[str, List[NoiseChannel]] = {}
        self.qubit_noise_map: Dict[int, List[NoiseChannel]] = {}
        self.global_noise_channels: List[NoiseChannel] = []

    def add_depolarizing_noise(
        self,
        error_probability: float,
        gate_types: Optional[List[str]] = None,
        qubits: Optional[List[int]] = None,
        num_qubits: int = 1,
    ) -> "NoiseModel":
        """
        Add depolarizing noise to the model.

        Args:
            error_probability: Probability of depolarizing error
            gate_types: Gate types to apply noise to (None for all gates)
            qubits: Specific qubits to apply noise to (None for all qubits)
            num_qubits: Number of qubits the noise acts on (1 or 2)

        Returns:
            Self for method chaining
        """
        parameters = NoiseParameters(
            error_probability=error_probability,
            gate_types=gate_types or ["all"],
            qubits=qubits,
        )

        channel = DepolarizingChannel(parameters, num_qubits)

        # Register channel based on targeting
        if gate_types:
            for gate_type in gate_types:
                if gate_type not in self.gate_noise_map:
                    self.gate_noise_map[gate_type] = []
                self.gate_noise_map[gate_type].append(channel)

        if qubits:
            for qubit in qubits:
                if qubit not in self.qubit_noise_map:
                    self.qubit_noise_map[qubit] = []
                self.qubit_noise_map[qubit].append(channel)

        if not gate_types and not qubits:
            self.global_noise_channels.append(channel)

        return self

    def get_applicable_noise(
        self, gate_name: str, qubits: List[int]
    ) -> List[NoiseChannel]:
        """
        Get noise channels applicable to a specific gate operation.

        Args:
            gate_name: Name of the gate being applied
            qubits: Qubits the gate operates on

        Returns:
            List of applicable noise channels
        """
        applicable_channels = []

        # Global noise (applies to all gates)
        applicable_channels.extend(self.global_noise_channels)

        # Gate-specific noise
        if gate_name in self.gate_noise_map:
            applicable_channels.extend(self.gate_noise_map[gate_name])

        if "all" in self.gate_noise_map:
            applicable_channels.extend(self.gate_noise_map["all"])

        # Qubit-specific noise
        for qubit in qubits:
            if qubit in self.qubit_noise_map:
                applicable_channels.extend(self.qubit_noise_map[qubit])

        # Filter channels that match the number of qubits
        filtered_channels = []
        for channel in applicable_channels:
            if hasattr(channel, "num_qubits") and channel.num_qubits == len(qubits):
                # Check if this channel should apply to these specific qubits
                if channel.parameters.qubits is None or any(
                    q in channel.parameters.qubits for q in qubits
                ):
                    filtered_channels.append(channel)

        return filtered_channels

    def apply_noise_to_state(
        self, state_vector: np.ndarray, gate_name: str, qubits: List[int]
    ) -> np.ndarray:
        """
        Apply all applicable noise channels to a state vector.

        Args:
            state_vector: Current quantum state vector
            gate_name: Name of the gate that was just applied
            qubits: Qubits the gate was applied to

        Returns:
            State vector after applying noise
        """
        current_state = state_vector.copy()
        applicable_channels = self.get_applicable_noise(gate_name, qubits)

        for channel in applicable_channels:
            try:
                current_state = channel.apply_noise(current_state, qubits)
            except Exception as e:
                logger.warning(
                    f"Failed to apply noise channel {type(channel).__name__}: {e}"
                )

        return current_state

    def get_total_fidelity_estimate(
        self, circuit_depth: int, avg_qubits_per_gate: float = 1.5
    ) -> float:
        """
        Estimate total circuit fidelity under this noise model.

        Args:
            circuit_depth: Number of gate layers in the circuit
            avg_qubits_per_gate: Average number of qubits per gate

        Returns:
            Estimated circuit fidelity
        """
        if (
            not self.global_noise_channels
            and not self.gate_noise_map
            and not self.qubit_noise_map
        ):
            return 1.0  # No noise

        # Simple approximation: multiply single-gate fidelities
        # This assumes uncorrelated errors (optimistic)
        avg_gate_fidelity = 1.0

        # Consider global noise channels
        for channel in self.global_noise_channels:
            avg_gate_fidelity *= channel.get_channel_fidelity()

        # Approximate effect of gate-specific and qubit-specific noise
        # This is a simplified model for estimation purposes
        if self.gate_noise_map or self.qubit_noise_map:
            additional_noise_factor = 0.95  # Conservative estimate
            avg_gate_fidelity *= additional_noise_factor

        # Circuit fidelity scales with depth
        circuit_fidelity = avg_gate_fidelity**circuit_depth

        return max(0.0, circuit_fidelity)

    def __str__(self) -> str:
        """String representation of noise model."""
        lines = ["NoiseModel:"]
        lines.append(f"  Global channels: {len(self.global_noise_channels)}")
        lines.append(f"  Gate-specific channels: {len(self.gate_noise_map)}")
        lines.append(f"  Qubit-specific channels: {len(self.qubit_noise_map)}")
        return "\n".join(lines)
