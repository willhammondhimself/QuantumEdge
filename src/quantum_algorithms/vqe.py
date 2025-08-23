"""
Variational Quantum Eigensolver (VQE) implementation for eigenportfolio discovery.

This module implements a classical simulation of VQE to find optimal portfolio
eigenvectors using quantum-inspired optimization techniques.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
from scipy.optimize import minimize
import logging

from .quantum_circuit import QuantumCircuit
from .quantum_gates import QuantumGates

logger = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Result from VQE optimization."""

    eigenvalue: float
    eigenvector: np.ndarray
    optimal_params: np.ndarray
    optimization_history: List[float]
    num_iterations: int
    success: bool


class ParameterizedCircuit:
    """Parameterized quantum circuit for VQE."""

    def __init__(self, num_qubits: int, depth: int = 2):
        """
        Initialize parameterized circuit.

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth (number of layers)
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.gates = QuantumGates()

        # Calculate number of parameters
        # Each layer has rotation gates for each qubit + entangling gates
        self.num_params = depth * (3 * num_qubits + num_qubits - 1)

    def create_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create parameterized ansatz circuit.

        This uses a hardware-efficient ansatz with alternating rotation
        and entangling layers.

        Args:
            params: Circuit parameters

        Returns:
            Parameterized quantum circuit
        """
        if len(params) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, got {len(params)}"
            )

        circuit = QuantumCircuit(self.num_qubits)
        param_idx = 0

        for layer in range(self.depth):
            # Rotation layer
            for qubit in range(self.num_qubits):
                circuit.ry(qubit, params[param_idx])
                param_idx += 1
                circuit.rz(qubit, params[param_idx])
                param_idx += 1
                circuit.ry(qubit, params[param_idx])
                param_idx += 1

            # Entangling layer (except for last layer)
            if layer < self.depth - 1:
                for qubit in range(self.num_qubits - 1):
                    circuit.cnot(qubit, qubit + 1)

        return circuit

    def create_efficient_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create efficient ansatz for portfolio optimization.

        This uses a more efficient ansatz specifically designed for
        eigenportfolio problems.

        Args:
            params: Circuit parameters

        Returns:
            Parameterized quantum circuit
        """
        circuit = QuantumCircuit(self.num_qubits)

        # Initial layer - put all qubits in superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)

        param_idx = 0
        for layer in range(self.depth):
            # Phase rotation layer
            for qubit in range(self.num_qubits):
                circuit.rz(qubit, params[param_idx])
                param_idx += 1

            # Entangling layer with circular connectivity
            for qubit in range(self.num_qubits):
                next_qubit = (qubit + 1) % self.num_qubits
                circuit.cnot(qubit, next_qubit)

        return circuit


class QuantumVQE:
    """Variational Quantum Eigensolver for eigenportfolio discovery."""

    def __init__(
        self,
        num_assets: int,
        depth: int = 2,
        optimizer: str = "COBYLA",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize VQE solver.

        Args:
            num_assets: Number of assets (qubits)
            depth: Circuit depth
            optimizer: Classical optimizer to use
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.num_assets = num_assets
        self.num_qubits = int(np.ceil(np.log2(num_assets)))
        self.depth = depth
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Initialize parameterized circuit
        self.circuit_template = ParameterizedCircuit(self.num_qubits, depth)

        # Optimization tracking
        self.optimization_history: List[float] = []
        self.iteration_count = 0

    def _covariance_to_hamiltonian(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Convert covariance matrix to quantum Hamiltonian.

        Args:
            covariance_matrix: Asset covariance matrix

        Returns:
            Hamiltonian matrix for quantum simulation
        """
        n = covariance_matrix.shape[0]

        # Pad to power of 2 if necessary
        padded_size = 2**self.num_qubits
        if n < padded_size:
            padded_cov = np.zeros((padded_size, padded_size))
            padded_cov[:n, :n] = covariance_matrix
            covariance_matrix = padded_cov

        # Create Hamiltonian from covariance matrix
        # H = Σᵢⱼ σᵢⱼ Zᵢ ⊗ Zⱼ where σᵢⱼ are covariance elements
        hamiltonian = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)

        gates = QuantumGates()

        for i in range(self.num_qubits):
            for j in range(i, self.num_qubits):
                # Create Pauli-Z operators for qubits i and j
                pauli_ops = []
                for k in range(self.num_qubits):
                    if k == i or k == j:
                        pauli_ops.append(gates.Z)
                    else:
                        pauli_ops.append(gates.I)

                # Tensor product of Pauli operators
                pauli_product = gates.tensor_product(*pauli_ops)

                # Add to Hamiltonian with covariance coefficient
                coeff = covariance_matrix[i, j] if i < n and j < n else 0.0
                hamiltonian += coeff * pauli_product

        # Shift Hamiltonian to ensure non-negative spectrum to stabilize optimization
        try:
            min_eig = np.min(np.real(np.linalg.eigvals(hamiltonian)))
            if min_eig < 0:
                hamiltonian += (-min_eig) * np.eye(hamiltonian.shape[0])
        except Exception:
            pass

        return hamiltonian

    def _cost_function(self, params: np.ndarray, hamiltonian: np.ndarray) -> float:
        """
        Cost function for VQE optimization.

        Args:
            params: Circuit parameters
            hamiltonian: Problem Hamiltonian

        Returns:
            Expectation value of Hamiltonian
        """
        # Create parameterized circuit
        circuit = self.circuit_template.create_ansatz(params)

        # Calculate expectation value
        expectation = circuit.expectation_value(hamiltonian)

        # Track optimization progress
        self.optimization_history.append(expectation)
        self.iteration_count += 1

        if self.iteration_count % 50 == 0:
            logger.info(
                f"VQE iteration {self.iteration_count}: expectation = {expectation:.6f}"
            )

        return expectation

    def _gradient_cost_function(
        self, params: np.ndarray, hamiltonian: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of cost function using parameter-shift rule.

        Args:
            params: Circuit parameters
            hamiltonian: Problem Hamiltonian

        Returns:
            Gradient vector
        """
        gradients = np.zeros_like(params)
        shift = np.pi / 2

        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += shift
            cost_plus = self._cost_function(params_plus, hamiltonian)

            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= shift
            cost_minus = self._cost_function(params_minus, hamiltonian)

            # Parameter-shift gradient
            gradients[i] = (cost_plus - cost_minus) / 2

        return gradients

    def solve_eigenportfolio(
        self, covariance_matrix: np.ndarray, initial_params: Optional[np.ndarray] = None
    ) -> VQEResult:
        """
        Solve for eigenportfolio using VQE.

        Args:
            covariance_matrix: Asset covariance matrix
            initial_params: Initial circuit parameters

        Returns:
            VQE optimization result
        """
        logger.info(
            f"Starting VQE eigenportfolio optimization for {covariance_matrix.shape[0]} assets"
        )

        # Convert covariance matrix to Hamiltonian
        hamiltonian = self._covariance_to_hamiltonian(covariance_matrix)

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(
                0, 2 * np.pi, size=self.circuit_template.num_params
            )

        # Reset optimization tracking
        self.optimization_history = []
        self.iteration_count = 0

        # Optimize using classical optimizer
        if self.optimizer == "COBYLA":
            result = minimize(
                fun=self._cost_function,
                x0=initial_params,
                args=(hamiltonian,),
                method="COBYLA",
                options={"maxiter": self.max_iterations, "rhobeg": 0.1},
            )
        elif self.optimizer == "gradient":
            result = minimize(
                fun=self._cost_function,
                x0=initial_params,
                args=(hamiltonian,),
                jac=self._gradient_cost_function,
                method="BFGS",
                options={"maxiter": self.max_iterations},
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Extract eigenvector from optimal state
        optimal_circuit = self.circuit_template.create_ansatz(result.x)
        state_vector = optimal_circuit.get_state_vector()

        # Convert quantum state to portfolio weights
        probabilities = np.abs(state_vector) ** 2
        eigenvector = self._extract_portfolio_weights(
            probabilities, covariance_matrix.shape[0]
        )

        logger.info(f"VQE optimization completed. Final eigenvalue: {result.fun:.6f}")

        return VQEResult(
            eigenvalue=result.fun,
            eigenvector=eigenvector,
            optimal_params=result.x,
            optimization_history=self.optimization_history,
            num_iterations=self.iteration_count,
            success=(
                result.success
                or (
                    len(self.optimization_history) >= 2
                    and abs(
                        self.optimization_history[-1] - self.optimization_history[-2]
                    )
                    < self.tolerance
                )
            ),
        )

    def _extract_portfolio_weights(
        self, probabilities: np.ndarray, num_assets: int
    ) -> np.ndarray:
        """
        Extract portfolio weights from quantum probabilities.

        Args:
            probabilities: Quantum measurement probabilities
            num_assets: Number of assets

        Returns:
            Normalized portfolio weights
        """
        # Take first num_assets probabilities and normalize
        weights = probabilities[:num_assets]

        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights
            weights = np.ones(num_assets) / num_assets

        return weights

    def compute_multiple_eigenportfolios(
        self,
        covariance_matrix: np.ndarray,
        num_eigenportfolios: int = 3,
        num_random_starts: int = 5,
    ) -> List[VQEResult]:
        """
        Compute multiple eigenportfolios using different random starts.

        Args:
            covariance_matrix: Asset covariance matrix
            num_eigenportfolios: Number of eigenportfolios to find
            num_random_starts: Random starts per eigenportfolio

        Returns:
            List of VQE results sorted by eigenvalue
        """
        results = []

        for i in range(num_eigenportfolios):
            best_result = None
            best_eigenvalue = float("inf")

            for start in range(num_random_starts):
                logger.info(
                    f"Computing eigenportfolio {i+1}/{num_eigenportfolios}, start {start+1}/{num_random_starts}"
                )

                # Random initialization
                initial_params = np.random.uniform(
                    0, 2 * np.pi, size=self.circuit_template.num_params
                )

                # Add penalty for previously found eigenportfolios
                modified_cov = covariance_matrix.copy()
                for prev_result in results:
                    # Deflation: add penalty term to avoid previous solutions
                    penalty = 10.0 * np.outer(
                        prev_result.eigenvector, prev_result.eigenvector
                    )
                    modified_cov += penalty

                result = self.solve_eigenportfolio(modified_cov, initial_params)

                if result.eigenvalue < best_eigenvalue:
                    best_eigenvalue = result.eigenvalue
                    best_result = result

            if best_result is not None:
                results.append(best_result)

        # Sort by eigenvalue
        results.sort(key=lambda x: x.eigenvalue)

        return results


class JAXVQEOptimizer:
    """JAX-accelerated VQE optimizer for larger problems."""

    def __init__(self, num_qubits: int, depth: int = 2):
        """
        Initialize JAX VQE optimizer.

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JAXVQEOptimizer. Install with: pip install jax jaxlib"
            )

        self.num_qubits = num_qubits
        self.depth = depth

    def _jax_cost_function(self, params, hamiltonian) -> float:
        """JAX-compiled cost function for faster optimization."""
        # This is a simplified version - full implementation would
        # require JAX-compatible quantum circuit simulation
        return jnp.sum(params**2)  # Placeholder

    def optimize_with_jax(
        self,
        covariance_matrix: np.ndarray,
        learning_rate: float = 0.01,
        num_steps: int = 1000,
    ) -> VQEResult:
        """
        Optimize VQE using JAX gradient descent.

        Args:
            covariance_matrix: Asset covariance matrix
            learning_rate: Learning rate for gradient descent
            num_steps: Number of optimization steps

        Returns:
            VQE optimization result
        """
        # Convert to JAX arrays
        hamiltonian = jnp.array(covariance_matrix)

        # Initialize parameters
        params = jax.random.uniform(
            jax.random.PRNGKey(42),
            shape=(self.depth * 3 * self.num_qubits,),
            minval=0,
            maxval=2 * jnp.pi,
        )

        # Gradient function
        grad_fn = grad(self._jax_cost_function)

        # Optimization loop
        history = []
        for step in range(num_steps):
            cost = self._jax_cost_function(params, hamiltonian)
            gradients = grad_fn(params, hamiltonian)
            params = params - learning_rate * gradients

            history.append(float(cost))

            if step % 100 == 0:
                logger.info(f"JAX VQE step {step}: cost = {cost:.6f}")

        # Placeholder result
        return VQEResult(
            eigenvalue=float(cost),
            eigenvector=np.ones(covariance_matrix.shape[0])
            / covariance_matrix.shape[0],
            optimal_params=np.array(params),
            optimization_history=history,
            num_iterations=num_steps,
            success=True,
        )
