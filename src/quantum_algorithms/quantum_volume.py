"""
Quantum Volume implementation following IBM's protocol.

This module implements the standard quantum volume benchmark to measure
the effective size and quality of quantum circuits that can be reliably
executed on a quantum device.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from .quantum_circuit import QuantumCircuit
from .quantum_gates import QuantumGates
import itertools
import time

logger = logging.getLogger(__name__)


class QVProtocolType(Enum):
    """Types of quantum volume protocols."""

    STANDARD_IBM = "standard_ibm"
    HEAVY_OUTPUT = "heavy_output"
    CUSTOM = "custom"


@dataclass
class QuantumVolumeResult:
    """Result from quantum volume measurement."""

    quantum_volume: int
    depth: int
    width: int  # number of qubits
    success_probability: float
    confidence_interval: Tuple[float, float]
    num_circuits: int
    num_shots_per_circuit: int
    heavy_output_probability: float
    threshold_probability: float
    success: bool
    execution_time: float
    circuit_fidelity_estimate: float

    def __post_init__(self):
        """Validate quantum volume result."""
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Success probability must be in [0,1]")
        if not 0 <= self.heavy_output_probability <= 1:
            raise ValueError("Heavy output probability must be in [0,1]")


@dataclass
class QVCircuitSpecification:
    """Specification for quantum volume circuit generation."""

    num_qubits: int
    depth: int
    seed: Optional[int] = None
    gate_set: List[str] = None
    coupling_map: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        """Set default gate set if not provided."""
        if self.gate_set is None:
            self.gate_set = ["X", "Y", "Z", "H", "CNOT", "Ry", "Rz"]


class QuantumVolumeProtocol:
    """
    Implementation of IBM's Quantum Volume protocol.

    Quantum Volume is a single-number metric that captures the largest
    square (m×m) circuit that can be reliably executed on a quantum device.
    """

    def __init__(self, precision: float = 1e-10):
        """
        Initialize quantum volume protocol.

        Args:
            precision: Numerical precision for calculations
        """
        self.precision = precision
        self.gates = QuantumGates(precision)

        # Protocol parameters (IBM standard)
        self.confidence_level = 0.95
        self.min_success_probability = 2.0 / 3.0  # 2σ threshold
        self.min_circuits = 100
        self.min_shots_per_circuit = 1024

    def generate_random_circuit(self, spec: QVCircuitSpecification) -> QuantumCircuit:
        """
        Generate a random quantum volume circuit.

        Args:
            spec: Circuit specification

        Returns:
            Random quantum circuit for quantum volume testing
        """
        if spec.seed is not None:
            np.random.seed(spec.seed)

        circuit = QuantumCircuit(spec.num_qubits, self.precision)

        # Generate circuit with alternating layers
        for layer in range(spec.depth):
            # Generate random permutation of qubits for this layer
            qubit_pairs = self._generate_qubit_pairs(spec.num_qubits, spec.coupling_map)

            # Apply random two-qubit gates to each pair
            for qubit1, qubit2 in qubit_pairs:
                # For quantum volume, we typically use random SU(4) gates
                # For simulation, we'll use a combination of single-qubit rotations + CNOT
                self._apply_random_su4_gate(circuit, qubit1, qubit2)

            # Apply random single-qubit gates to remaining qubits
            used_qubits = set()
            for q1, q2 in qubit_pairs:
                used_qubits.add(q1)
                used_qubits.add(q2)

            for qubit in range(spec.num_qubits):
                if qubit not in used_qubits:
                    self._apply_random_single_qubit_gate(circuit, qubit, spec.gate_set)

        return circuit

    def _generate_qubit_pairs(
        self, num_qubits: int, coupling_map: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """Generate random qubit pairs for a layer."""
        if coupling_map is not None:
            # Use provided coupling map
            available_pairs = coupling_map.copy()
            np.random.shuffle(available_pairs)

            # Greedy selection to avoid conflicts
            selected_pairs = []
            used_qubits = set()

            for q1, q2 in available_pairs:
                if q1 not in used_qubits and q2 not in used_qubits:
                    selected_pairs.append((q1, q2))
                    used_qubits.add(q1)
                    used_qubits.add(q2)

            return selected_pairs
        else:
            # All-to-all connectivity
            qubits = list(range(num_qubits))
            np.random.shuffle(qubits)

            pairs = []
            for i in range(0, len(qubits) - 1, 2):
                pairs.append((qubits[i], qubits[i + 1]))

            return pairs

    def _apply_random_su4_gate(
        self, circuit: QuantumCircuit, qubit1: int, qubit2: int
    ) -> None:
        """Apply a random SU(4) gate decomposition."""
        # Decompose random SU(4) into single-qubit + CNOT gates
        # This is a simplified decomposition for simulation purposes

        # Random single-qubit rotations
        for qubit in [qubit1, qubit2]:
            theta_y = np.random.uniform(0, 2 * np.pi)
            theta_z = np.random.uniform(0, 2 * np.pi)
            circuit.ry(qubit, theta_y)
            circuit.rz(qubit, theta_z)

        # CNOT
        circuit.cnot(qubit1, qubit2)

        # More random single-qubit rotations
        for qubit in [qubit1, qubit2]:
            theta_y = np.random.uniform(0, 2 * np.pi)
            theta_z = np.random.uniform(0, 2 * np.pi)
            circuit.ry(qubit, theta_y)
            circuit.rz(qubit, theta_z)

    def _apply_random_single_qubit_gate(
        self, circuit: QuantumCircuit, qubit: int, gate_set: List[str]
    ) -> None:
        """Apply a random single-qubit gate."""
        single_qubit_gates = [
            g for g in gate_set if g in ["X", "Y", "Z", "H", "Ry", "Rz"]
        ]

        if not single_qubit_gates:
            return

        gate = np.random.choice(single_qubit_gates)

        if gate == "X":
            circuit.x(qubit)
        elif gate == "Y":
            circuit.y(qubit)
        elif gate == "Z":
            circuit.z(qubit)
        elif gate == "H":
            circuit.h(qubit)
        elif gate == "Ry":
            theta = np.random.uniform(0, 2 * np.pi)
            circuit.ry(qubit, theta)
        elif gate == "Rz":
            theta = np.random.uniform(0, 2 * np.pi)
            circuit.rz(qubit, theta)

    def compute_heavy_output_probability(
        self, circuit: QuantumCircuit, num_shots: int = 1024
    ) -> float:
        """
        Compute heavy output probability for a quantum volume circuit.

        Heavy outputs are those measurement outcomes with probability >= median.

        Args:
            circuit: Quantum circuit to evaluate
            num_shots: Number of measurement shots

        Returns:
            Probability of measuring heavy outputs
        """
        # Get ideal probability distribution
        ideal_probabilities = circuit.get_probabilities()

        # Find median probability
        median_prob = np.median(ideal_probabilities)

        # Identify heavy outputs (those with probability >= median)
        heavy_outputs = np.where(ideal_probabilities >= median_prob)[0]

        # Simulate measurements
        heavy_count = 0
        for _ in range(num_shots):
            measurement = circuit.measure_all()
            # Convert bit string to integer
            outcome = sum(bit * (2**i) for i, bit in enumerate(reversed(measurement)))
            if outcome in heavy_outputs:
                heavy_count += 1

        return heavy_count / num_shots

    def run_quantum_volume_protocol(
        self,
        num_qubits: int,
        num_circuits: Optional[int] = None,
        num_shots: Optional[int] = None,
        noise_model=None,
    ) -> QuantumVolumeResult:
        """
        Run the complete quantum volume protocol.

        Args:
            num_qubits: Number of qubits (circuit width and depth)
            num_circuits: Number of random circuits to test
            num_shots: Number of shots per circuit
            noise_model: Optional noise model for realistic simulation

        Returns:
            Quantum volume measurement result
        """
        start_time = time.time()

        # Use default parameters if not specified
        if num_circuits is None:
            num_circuits = self.min_circuits
        if num_shots is None:
            num_shots = self.min_shots_per_circuit

        logger.info(
            f"Running quantum volume protocol: {num_qubits} qubits, "
            f"{num_circuits} circuits, {num_shots} shots per circuit"
        )

        # Generate circuit specification
        spec = QVCircuitSpecification(
            num_qubits=num_qubits,
            depth=num_qubits,  # Square circuits: depth = width
        )

        heavy_output_probabilities = []
        circuit_fidelities = []

        # Run protocol on multiple random circuits
        for circuit_idx in range(num_circuits):
            # Generate random circuit
            spec.seed = circuit_idx  # For reproducibility
            circuit = self.generate_random_circuit(spec)

            # Apply noise model if provided
            if noise_model is not None:
                circuit.set_noise_model(noise_model)

            # Compute heavy output probability
            heavy_prob = self.compute_heavy_output_probability(circuit, num_shots)
            heavy_output_probabilities.append(heavy_prob)

            # Track circuit fidelity if noise model is present
            if noise_model is not None:
                circuit_fidelities.append(circuit.get_estimated_fidelity())
            else:
                circuit_fidelities.append(1.0)

            if (circuit_idx + 1) % 20 == 0:
                logger.debug(f"Completed {circuit_idx + 1}/{num_circuits} circuits")

        # Analyze results
        heavy_probs = np.array(heavy_output_probabilities)
        mean_heavy_prob = np.mean(heavy_probs)

        # Statistical analysis
        success_count = np.sum(heavy_probs >= self.min_success_probability)
        success_probability = success_count / num_circuits

        # Confidence interval (binomial proportion)
        from scipy import stats

        try:
            ci_low, ci_high = stats.binom.interval(
                self.confidence_level, num_circuits, success_probability
            )
            confidence_interval = (ci_low / num_circuits, ci_high / num_circuits)
        except ImportError:
            # Fallback if scipy not available
            std_err = np.sqrt(
                success_probability * (1 - success_probability) / num_circuits
            )
            confidence_interval = (
                max(0, success_probability - 1.96 * std_err),
                min(1, success_probability + 1.96 * std_err),
            )

        # Determine if quantum volume is achieved
        qv_success = success_probability >= self.min_success_probability
        quantum_volume = (2**num_qubits) if qv_success else 0

        execution_time = time.time() - start_time
        avg_fidelity = np.mean(circuit_fidelities)

        result = QuantumVolumeResult(
            quantum_volume=quantum_volume,
            depth=num_qubits,
            width=num_qubits,
            success_probability=success_probability,
            confidence_interval=confidence_interval,
            num_circuits=num_circuits,
            num_shots_per_circuit=num_shots,
            heavy_output_probability=mean_heavy_prob,
            threshold_probability=self.min_success_probability,
            success=qv_success,
            execution_time=execution_time,
            circuit_fidelity_estimate=avg_fidelity,
        )

        logger.info(
            f"Quantum volume protocol completed: QV={quantum_volume}, "
            f"success_rate={success_probability:.3f}, "
            f"fidelity={avg_fidelity:.3f}"
        )

        return result

    def estimate_max_quantum_volume(
        self, max_qubits: int, noise_model=None, quick_test: bool = True
    ) -> Dict[int, QuantumVolumeResult]:
        """
        Estimate maximum achievable quantum volume by testing multiple qubit counts.

        Args:
            max_qubits: Maximum number of qubits to test
            noise_model: Optional noise model
            quick_test: Use reduced parameters for faster testing

        Returns:
            Dictionary mapping qubit count to quantum volume results
        """
        results = {}

        # Parameters for quick vs thorough testing
        if quick_test:
            test_circuits = max(50, self.min_circuits // 2)
            test_shots = max(256, self.min_shots_per_circuit // 4)
        else:
            test_circuits = self.min_circuits
            test_shots = self.min_shots_per_circuit

        logger.info(
            f"Estimating max quantum volume up to {max_qubits} qubits "
            f"({'quick' if quick_test else 'thorough'} test)"
        )

        for num_qubits in range(2, max_qubits + 1):
            logger.info(f"Testing {num_qubits} qubits...")

            try:
                result = self.run_quantum_volume_protocol(
                    num_qubits=num_qubits,
                    num_circuits=test_circuits,
                    num_shots=test_shots,
                    noise_model=noise_model,
                )
                results[num_qubits] = result

                # If we fail quantum volume, likely won't succeed at higher qubit counts
                if not result.success and num_qubits >= 4:
                    logger.info(f"Failed QV at {num_qubits} qubits, stopping search")
                    break

            except Exception as e:
                logger.error(f"Error testing {num_qubits} qubits: {e}")
                break

        return results

    def analyze_circuit_complexity(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Analyze the complexity characteristics of a quantum circuit.

        Args:
            circuit: Quantum circuit to analyze

        Returns:
            Dictionary containing complexity metrics
        """
        gate_counts = circuit.get_gate_count()
        total_gates = sum(gate_counts.values())

        # Calculate circuit metrics
        two_qubit_gates = sum(
            gate_counts.get(gate, 0) for gate in ["CNOT", "CZ", "SWAP", "CRz"]
        )
        single_qubit_gates = total_gates - two_qubit_gates

        # Estimate quantum volume requirements
        # This is a heuristic based on circuit structure
        effective_depth = circuit.get_circuit_depth()
        effective_width = circuit.num_qubits

        # Circuit complexity score (heuristic)
        complexity_score = (
            effective_depth * 0.4
            + effective_width * 0.3
            + two_qubit_gates * 0.2
            + single_qubit_gates * 0.1
        ) / (
            effective_depth + effective_width + 1
        )  # Normalize

        analysis = {
            "total_gates": total_gates,
            "single_qubit_gates": single_qubit_gates,
            "two_qubit_gates": two_qubit_gates,
            "circuit_depth": effective_depth,
            "circuit_width": effective_width,
            "gate_distribution": gate_counts,
            "complexity_score": complexity_score,
            "estimated_qv_requirement": max(
                2, 2 ** int(np.ceil(np.log2(max(effective_depth, effective_width))))
            ),
            "cnot_depth": two_qubit_gates,  # Approximation
        }

        return analysis
