"""
Quantum circuit simulation for classical hardware.

This module provides a quantum circuit simulator that can be used to implement
quantum-inspired algorithms for portfolio optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from .quantum_gates import QuantumGates


@dataclass
class Operation:
    """Represents a quantum operation in a circuit."""
    gate: str
    qubits: List[int]
    params: Optional[List[float]] = None
    

class QuantumCircuit:
    """
    Quantum circuit simulator using state vector representation.
    
    This implementation uses classical simulation to approximate quantum
    behavior for portfolio optimization algorithms.
    """
    
    def __init__(self, num_qubits: int, precision: float = 1e-10):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            precision: Numerical precision for calculations
        """
        self.num_qubits = num_qubits
        self.precision = precision
        self.gates = QuantumGates(precision)
        
        # Initialize state vector |00...0⟩
        self.state_vector = np.zeros(2 ** num_qubits, dtype=complex)
        self.state_vector[0] = 1.0
        
        # Track operations for circuit analysis
        self.operations: List[Operation] = []
        
        # Cache for frequently used gate expansions
        self._gate_cache: Dict[str, np.ndarray] = {}
    
    def reset(self) -> None:
        """Reset circuit to initial state |00...0⟩."""
        self.state_vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        self.state_vector[0] = 1.0
        self.operations.clear()
        self._gate_cache.clear()
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector."""
        return self.state_vector.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states."""
        return np.abs(self.state_vector) ** 2
    
    def _expand_gate(self, gate: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """
        Expand a gate to act on specified qubits in the full circuit.
        
        Args:
            gate: Gate matrix to expand
            target_qubits: Qubits the gate acts on
            
        Returns:
            Expanded gate matrix for full circuit
        """
        # Create cache key
        cache_key = f"{id(gate)}_{tuple(sorted(target_qubits))}"
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]
        
        if len(target_qubits) == 1:
            expanded = self._expand_single_qubit_gate(gate, target_qubits[0])
        elif len(target_qubits) == 2:
            expanded = self._expand_two_qubit_gate(gate, target_qubits)
        else:
            raise NotImplementedError("Gates on >2 qubits not implemented")
        
        self._gate_cache[cache_key] = expanded
        return expanded
    
    def _expand_single_qubit_gate(self, gate: np.ndarray, target: int) -> np.ndarray:
        """Expand single-qubit gate to full circuit."""
        gates_list = []
        for i in range(self.num_qubits):
            if i == target:
                gates_list.append(gate)
            else:
                gates_list.append(self.gates.I)
        
        return self.gates.tensor_product(*gates_list)
    
    def _expand_two_qubit_gate(self, gate: np.ndarray, targets: List[int]) -> np.ndarray:
        """Expand two-qubit gate to full circuit."""
        if len(targets) != 2:
            raise ValueError("Two-qubit gate requires exactly 2 targets")
        
        control, target = sorted(targets)
        
        # For simplicity, we'll use a direct expansion approach
        # This can be optimized for larger circuits
        expanded = np.eye(2 ** self.num_qubits, dtype=complex)
        
        for i in range(2 ** self.num_qubits):
            # Extract control and target bits
            control_bit = (i >> (self.num_qubits - 1 - control)) & 1
            target_bit = (i >> (self.num_qubits - 1 - target)) & 1
            
            # Apply gate logic based on the specific gate
            if np.array_equal(gate, self.gates.CNOT):
                if control_bit == 1:
                    # Flip target bit
                    j = i ^ (1 << (self.num_qubits - 1 - target))
                    expanded[j, i] = 1
                    expanded[i, i] = 0
                else:
                    expanded[i, i] = 1
            else:
                # For general two-qubit gates, use tensor product approach
                return self._expand_general_two_qubit_gate(gate, targets)
        
        return expanded
    
    def _expand_general_two_qubit_gate(self, gate: np.ndarray, targets: List[int]) -> np.ndarray:
        """General expansion for two-qubit gates."""
        # This is a simplified implementation
        # For production use, consider more efficient tensor network methods
        gates_list = []
        target_set = set(targets)
        
        gate_index = 0
        for i in range(self.num_qubits):
            if i in target_set:
                if gate_index == 0:
                    gates_list.append(gate)
                    gate_index += 1
                # Skip second target qubit as it's included in the gate
            else:
                gates_list.append(self.gates.I)
        
        # This is a simplified approach - real implementation would be more complex
        return self.gates.tensor_product(*gates_list)
    
    def apply_gate(self, gate_name: str, qubits: List[int], params: Optional[List[float]] = None) -> None:
        """
        Apply a quantum gate to specified qubits.
        
        Args:
            gate_name: Name of the gate to apply
            qubits: List of qubit indices
            params: Parameters for parameterized gates
        """
        # Get gate matrix
        gate = self._get_gate_matrix(gate_name, params)
        
        # Expand gate to full circuit
        expanded_gate = self._expand_gate(gate, qubits)
        
        # Apply gate to state vector
        self.state_vector = expanded_gate @ self.state_vector
        
        # Record operation
        self.operations.append(Operation(gate_name, qubits.copy(), params))
    
    def _get_gate_matrix(self, gate_name: str, params: Optional[List[float]] = None) -> np.ndarray:
        """Get gate matrix by name."""
        if gate_name == 'I':
            return self.gates.I
        elif gate_name == 'X':
            return self.gates.X
        elif gate_name == 'Y':
            return self.gates.Y
        elif gate_name == 'Z':
            return self.gates.Z
        elif gate_name == 'H':
            return self.gates.H
        elif gate_name == 'S':
            return self.gates.S
        elif gate_name == 'T':
            return self.gates.T
        elif gate_name == 'CNOT':
            return self.gates.CNOT
        elif gate_name == 'CZ':
            return self.gates.CZ
        elif gate_name == 'SWAP':
            return self.gates.SWAP
        elif gate_name == 'Rx' and params:
            return self.gates.Rx(params[0])
        elif gate_name == 'Ry' and params:
            return self.gates.Ry(params[0])
        elif gate_name == 'Rz' and params:
            return self.gates.Rz(params[0])
        elif gate_name == 'U3' and params and len(params) >= 3:
            return self.gates.U3(params[0], params[1], params[2])
        elif gate_name == 'CRz' and params:
            return self.gates.CRz(params[0])
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    
    # Convenience methods for common gates
    def x(self, qubit: int) -> None:
        """Apply X gate."""
        self.apply_gate('X', [qubit])
    
    def y(self, qubit: int) -> None:
        """Apply Y gate."""
        self.apply_gate('Y', [qubit])
    
    def z(self, qubit: int) -> None:
        """Apply Z gate."""
        self.apply_gate('Z', [qubit])
    
    def h(self, qubit: int) -> None:
        """Apply Hadamard gate."""
        self.apply_gate('H', [qubit])
    
    def rx(self, qubit: int, theta: float) -> None:
        """Apply Rx rotation."""
        self.apply_gate('Rx', [qubit], [theta])
    
    def ry(self, qubit: int, theta: float) -> None:
        """Apply Ry rotation."""
        self.apply_gate('Ry', [qubit], [theta])
    
    def rz(self, qubit: int, theta: float) -> None:
        """Apply Rz rotation."""
        self.apply_gate('Rz', [qubit], [theta])
    
    def cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        self.apply_gate('CNOT', [control, target])
    
    def cz(self, control: int, target: int) -> None:
        """Apply CZ gate."""
        self.apply_gate('CZ', [control, target])
    
    def measure_all(self) -> List[int]:
        """
        Simulate measurement of all qubits.
        
        Returns:
            Classical bit string as list of integers
        """
        probabilities = self.get_probabilities()
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        # Convert to bit string
        bit_string = []
        for i in range(self.num_qubits):
            bit = (outcome >> (self.num_qubits - 1 - i)) & 1
            bit_string.append(bit)
        
        return bit_string
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """
        Calculate expectation value of an observable.
        
        Args:
            observable: Observable matrix (Hermitian)
            
        Returns:
            Expectation value
        """
        if observable.shape != (2 ** self.num_qubits, 2 ** self.num_qubits):
            raise ValueError("Observable dimension mismatch")
        
        expectation = np.real(
            self.state_vector.conj() @ observable @ self.state_vector
        )
        return expectation
    
    def get_circuit_depth(self) -> int:
        """Get circuit depth (number of gate layers)."""
        return len(self.operations)
    
    def get_gate_count(self) -> Dict[str, int]:
        """Get count of each gate type used."""
        gate_count = {}
        for op in self.operations:
            gate_count[op.gate] = gate_count.get(op.gate, 0) + 1
        return gate_count
    
    def __str__(self) -> str:
        """String representation of circuit."""
        lines = [f"QuantumCircuit({self.num_qubits} qubits)"]
        for i, op in enumerate(self.operations):
            params_str = f"({', '.join(map(str, op.params))})" if op.params else ""
            qubits_str = ', '.join(map(str, op.qubits))
            lines.append(f"  {i}: {op.gate}{params_str} on qubits [{qubits_str}]")
        return '\n'.join(lines)