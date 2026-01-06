"""
Quantum Circuit Architecture for Lorenz System Solver

This module implements a parameterized quantum circuit that learns to approximate
the solution to the Lorenz system using a physics-informed approach.

Architecture:
1. Time Encoding: Encodes time t into the circuit
2. Variational Ansatz: Parameterized layers that learn the dynamics  
3. Measurement: Extract x(t), y(t), z(t) from expectation values

The circuit is differentiable and can be trained using gradient descent.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, List


class LorenzQuantumCircuit:
    """
    Differentiable Quantum Circuit for solving the Lorenz system.
    
    The circuit architecture:
    1. Encodes time t using angle encoding
    2. Applies variational layers with trainable parameters
    3. Measures expectation values to extract x(t), y(t), z(t)
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 3,
        x_range: Tuple[float, float] = (-20.0, 20.0),
        y_range: Tuple[float, float] = (-30.0, 30.0),
        z_range: Tuple[float, float] = (0.0, 50.0),
        t_max: float = 50.0
    ):
        """
        Initialize the Lorenz quantum circuit.
        
        Args:
            n_qubits: Number of qubits (default: 3, one per Lorenz variable)
            n_layers: Number of variational layers (default: 3)
            x_range: Range for x variable mapping (min, max)
            y_range: Range for y variable mapping (min, max)
            z_range: Range for z variable mapping (min, max)
            t_max: Maximum time for normalization (default: 50.0)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.t_max = t_max  # For time normalization
        
        # Time parameter (input)
        self.t_param = Parameter('t')
        
        # Variational parameters (trainable)
        # Each layer has: rotation angles (3 per qubit) + entangling angles (n_qubits-1)
        n_params_per_layer = 3 * n_qubits + (n_qubits - 1)
        self.n_params = n_params_per_layer * n_layers
        self.theta = ParameterVector('θ', self.n_params)
        
        # Build the circuit
        self.circuit = self._build_circuit()
        
        # Observables for measuring x, y, z
        self.observables = self._build_observables()
    
    def _build_circuit(self) -> QuantumCircuit:
        """
        Build the parameterized quantum circuit.
        
        Uses Fourier-like time encoding to maintain expressivity while avoiding saturation.
        
        Returns:
            Parameterized quantum circuit
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Time encoding layer with Fourier-style features
        # Key idea: Normalize time to [0, 1], then encode with sin/cos patterns
        # This keeps all angles bounded in [-π, π] regardless of t
        #
        # For each qubit, we use: RY(ω_i * φ(t)) where φ(t) = 2π * t / t_max
        # and ω_i are frequency factors that stay small enough to prevent wrapping
        #
        # With 3 qubits, we use frequencies [1, 1, 1] and different gate types
        # to create diverse time-dependent features
        
        t_normalized = 2 * np.pi * self.t_param / self.t_max  # Maps [0, t_max] → [0, 2π]
        
        # Encode with different rotation gates for diversity
        # All angles stay in [0, 2π] range
        qc.ry(t_normalized, qr[0])  # RY(2πt/t_max)
        if self.n_qubits > 1:
            qc.rz(t_normalized, qr[1])  # RZ(2πt/t_max)
        if self.n_qubits > 2:
            qc.rx(t_normalized, qr[2])  # RX(2πt/t_max)
        
        # For additional qubits (if any), use combinations
        for i in range(3, self.n_qubits):
            qc.ry(t_normalized * 0.5, qr[i])
        
        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations for each qubit
            for i in range(self.n_qubits):
                qc.rx(self.theta[param_idx], qr[i])
                param_idx += 1
                qc.ry(self.theta[param_idx], qr[i])
                param_idx += 1
                qc.rz(self.theta[param_idx], qr[i])
                param_idx += 1
            
            # Entangling layer (CNOT ladder)
            for i in range(self.n_qubits - 1):
                qc.cx(qr[i], qr[i + 1])
                # Add parameterized rotation after CNOT
                qc.rz(self.theta[param_idx], qr[i + 1])
                param_idx += 1
        
        return qc
    
    def _build_observables(self) -> List[SparsePauliOp]:
        """
        Build Pauli observables for measuring x, y, z.
        
        Each qubit is measured with Pauli-Z operator.
        We always measure only the first 3 qubits for x, y, z,
        even if n_qubits > 3 (extra qubits provide entanglement).
        
        Returns:
            List of observables [obs_x, obs_y, obs_z]
        """
        observables = []
        
        # Always use first 3 qubits for x, y, z
        for i in range(min(3, self.n_qubits)):
            # Create Pauli string with Z on qubit i, I on others
            pauli_str = ['I'] * self.n_qubits
            pauli_str[i] = 'Z'
            pauli_str_formatted = ''.join(reversed(pauli_str))  # Qiskit ordering
            
            obs = SparsePauliOp(pauli_str_formatted)
            observables.append(obs)
        
        return observables
    
    def get_circuit(self) -> QuantumCircuit:
        """
        Get the parameterized quantum circuit.
        
        Returns:
            Quantum circuit with parameters
        """
        return self.circuit
    
    def get_observables(self) -> List[SparsePauliOp]:
        """
        Get the observables for measuring x, y, z.
        
        Returns:
            List of observables
        """
        return self.observables
    
    def map_expectation_to_value(
        self,
        expectation: float,
        variable: str
    ) -> float:
        """
        Map expectation value (in [-1, 1]) to physical variable range.
        
        Args:
            expectation: Expectation value from measurement (in [-1, 1])
            variable: Variable name ('x', 'y', or 'z')
        
        Returns:
            Mapped value in physical range
        """
        if variable == 'x':
            range_min, range_max = self.x_range
        elif variable == 'y':
            range_min, range_max = self.y_range
        elif variable == 'z':
            range_min, range_max = self.z_range
        else:
            raise ValueError(f"Unknown variable: {variable}")
        
        # Linear mapping from [-1, 1] to [range_min, range_max]
        value = (expectation + 1) / 2  # Map to [0, 1]
        value = range_min + value * (range_max - range_min)  # Map to range
        
        return value
    
    def assign_parameters(
        self,
        t_value: float,
        theta_values: np.ndarray
    ) -> QuantumCircuit:
        """
        Assign specific values to the circuit parameters.
        
        Args:
            t_value: Time value
            theta_values: Array of variational parameter values
        
        Returns:
            Circuit with parameters bound
        """
        if len(theta_values) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(theta_values)}"
            )
        
        # Create parameter binding dictionary
        param_dict = {self.t_param: t_value}
        for i, val in enumerate(theta_values):
            param_dict[self.theta[i]] = val
        
        # Bind parameters to circuit
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        return bound_circuit
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return self.n_params
    
    def initialize_parameters(self, seed: int = 42, scale: float = None) -> np.ndarray:
        """
        Initialize variational parameters randomly.
        
        Args:
            seed: Random seed for reproducibility
            scale: Scale for initialization (default: π, or smaller if specified)
        
        Returns:
            Array of initial parameter values
        """
        rng = np.random.default_rng(seed)
        # Use smaller initialization if specified (helps avoid barren plateaus)
        if scale is None:
            scale = np.pi
        return rng.uniform(-scale, scale, self.n_params)


def create_lorenz_circuit(
    n_qubits: int = 3,
    n_layers: int = 3,
    t_max: float = 50.0
) -> LorenzQuantumCircuit:
    """
    Factory function to create a Lorenz quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        t_max: Maximum time for normalization
    
    Returns:
        LorenzQuantumCircuit instance
    """
    return LorenzQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers, t_max=t_max)

