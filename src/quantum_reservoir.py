"""
Quantum Reservoir Computing for Lorenz System

Based on:
- arXiv:2311.14105: Hybrid quantum-classical reservoir computing
- arXiv:2204.13951: Quantum reservoir dynamics

Key Concept:
- Fixed random quantum circuit acts as reservoir (high-dimensional feature extractor)
- Only classical readout layer is trained (simple, fast)
- Natural for temporal/chaotic dynamics
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer.primitives import SamplerV2 as AerSampler
from typing import Tuple, List, Optional
import warnings


class QuantumReservoir:
    """
    Quantum Reservoir for temporal dynamics.
    
    The reservoir is a fixed (random) quantum circuit that transforms
    input states into high-dimensional quantum features. These features
    are then read out classically.
    
    Architecture:
    1. Input encoding: Classical state → Quantum state
    2. Reservoir dynamics: Fixed random quantum gates
    3. Measurement: Sample all qubits → Classical features
    """
    
    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 2,
        coupling_strength: float = 0.5,
        random_seed: int = 42
    ):
        """
        Initialize quantum reservoir.
        
        Args:
            n_qubits: Number of qubits in reservoir (more = higher dimensional features)
            n_layers: Number of random gate layers
            coupling_strength: Strength of inter-qubit coupling (0-1)
            random_seed: Seed for reproducible random gates
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.coupling_strength = coupling_strength
        self.random_seed = random_seed
        
        # Generate fixed random parameters
        self.rng = np.random.default_rng(random_seed)
        
        # Create sampler for measurements
        self.sampler = AerSampler()
        
        # Build reservoir circuit template
        self.reservoir_circuit_template = self._build_reservoir()
        
    def _build_reservoir(self) -> QuantumCircuit:
        """
        Build fixed random reservoir circuit.
        
        The reservoir consists of random rotation gates and entangling gates.
        These are fixed after initialization (NOT trained).
        
        Returns:
            Quantum circuit template (without input encoding)
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Random reservoir layers
        for layer in range(self.n_layers):
            # Random single-qubit rotations
            for i in range(self.n_qubits):
                # Random angles for RX, RY, RZ
                theta_x = self.rng.uniform(0, 2 * np.pi)
                theta_y = self.rng.uniform(0, 2 * np.pi)
                theta_z = self.rng.uniform(0, 2 * np.pi)
                
                qc.rx(theta_x, qr[i])
                qc.ry(theta_y, qr[i])
                qc.rz(theta_z, qr[i])
            
            # Entangling layer (coupling between qubits)
            for i in range(self.n_qubits - 1):
                # CNOT with random rotation after
                qc.cx(qr[i], qr[i + 1])
                
                # Controlled rotation (strength determined by coupling)
                rotation_angle = self.coupling_strength * self.rng.uniform(0, 2 * np.pi)
                qc.rz(rotation_angle, qr[i + 1])
            
            # Ring topology: connect last to first
            if self.n_qubits > 2:
                qc.cx(qr[-1], qr[0])
                rotation_angle = self.coupling_strength * self.rng.uniform(0, 2 * np.pi)
                qc.rz(rotation_angle, qr[0])
        
        return qc
    
    def encode_input(self, state: np.ndarray) -> QuantumCircuit:
        """
        Encode classical state into quantum circuit.
        
        Uses angle encoding: each component of the state controls
        rotation angles on qubits.
        
        Args:
            state: Classical state vector [x, y, z]
        
        Returns:
            Quantum circuit with input encoding
        """
        if len(state) != 3:
            raise ValueError(f"Expected 3-dimensional state, got {len(state)}")
        
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Normalize state to [0, 2π] for encoding
        # Lorenz ranges: x∈[-20,20], y∈[-30,30], z∈[0,50]
        x_norm = (state[0] + 20) / 40 * 2 * np.pi  # [-20,20] → [0,2π]
        y_norm = (state[1] + 30) / 60 * 2 * np.pi  # [-30,30] → [0,2π]
        z_norm = state[2] / 50 * 2 * np.pi          # [0,50] → [0,2π]
        
        # Encode on first 3 qubits
        qc.ry(x_norm, qr[0])
        qc.ry(y_norm, qr[1])
        qc.ry(z_norm, qr[2])
        
        # If more qubits, encode combinations
        if self.n_qubits > 3:
            qc.ry((x_norm + y_norm) / 2, qr[3])
        if self.n_qubits > 4:
            qc.ry((y_norm + z_norm) / 2, qr[4])
        if self.n_qubits > 5:
            qc.ry((x_norm + z_norm) / 2, qr[5])
        
        return qc
    
    def get_reservoir_states(
        self,
        input_states: np.ndarray,
        n_shots: int = 1024
    ) -> np.ndarray:
        """
        Get reservoir feature vectors for a sequence of input states.
        
        This is the key operation: transform classical states into
        high-dimensional quantum features via the reservoir.
        
        Args:
            input_states: Array of shape (n_timesteps, 3) - sequence of [x,y,z]
            n_shots: Number of measurement shots per state
        
        Returns:
            Array of shape (n_timesteps, 2^n_qubits) - quantum features
            Each row is a probability distribution over measurement outcomes
        """
        n_timesteps = len(input_states)
        n_features = 2 ** self.n_qubits  # Number of possible measurement outcomes
        
        reservoir_states = np.zeros((n_timesteps, n_features))
        
        # Process each timestep
        circuits = []
        for t in range(n_timesteps):
            # Encode input
            encoding_circuit = self.encode_input(input_states[t])
            
            # Apply reservoir dynamics
            full_circuit = encoding_circuit.compose(self.reservoir_circuit_template)
            
            # Add measurements
            full_circuit.measure_all()
            
            circuits.append(full_circuit)
        
        # Run all circuits at once (batch processing)
        job = self.sampler.run(circuits, shots=n_shots)
        result = job.result()
        
        # Extract measurement probabilities
        for t in range(n_timesteps):
            counts = result[t].data.meas.get_counts()
            
            # Convert counts to probability distribution
            for bitstring, count in counts.items():
                index = int(bitstring, 2)  # Convert binary string to integer
                reservoir_states[t, index] = count / n_shots
        
        return reservoir_states
    
    def get_feature_dimension(self) -> int:
        """Get dimension of feature space (2^n_qubits)."""
        return 2 ** self.n_qubits


class ReservoirReadout:
    """
    Classical readout layer for quantum reservoir.
    
    Simple linear regression from reservoir features to outputs.
    Trained using Ridge regression (regularized least squares).
    """
    
    def __init__(self, input_dim: int, output_dim: int = 3):
        """
        Initialize readout layer.
        
        Args:
            input_dim: Dimension of reservoir features (2^n_qubits)
            output_dim: Dimension of output (3 for x,y,z)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Weights will be learned
        self.W = None
        self.b = None
        
    def fit(
        self,
        reservoir_features: np.ndarray,
        targets: np.ndarray,
        alpha: float = 1.0
    ):
        """
        Train readout using Ridge regression.
        
        Closed-form solution: W = (X^T X + αI)^{-1} X^T y
        
        Args:
            reservoir_features: Array of shape (n_samples, input_dim)
            targets: Array of shape (n_samples, output_dim)
            alpha: Regularization strength (higher = more regularization)
        """
        n_samples = len(reservoir_features)
        
        # Add bias term
        X = np.hstack([reservoir_features, np.ones((n_samples, 1))])
        
        # Ridge regression solution
        # (X^T X + αI)^{-1} X^T y
        XtX = X.T @ X
        reg_term = alpha * np.eye(X.shape[1])
        
        # Solve for weights
        W_full = np.linalg.solve(XtX + reg_term, X.T @ targets)
        
        # Split into weights and bias
        self.W = W_full[:-1, :]  # Shape: (input_dim, output_dim)
        self.b = W_full[-1, :]   # Shape: (output_dim,)
        
    def predict(self, reservoir_features: np.ndarray) -> np.ndarray:
        """
        Predict outputs from reservoir features.
        
        Args:
            reservoir_features: Array of shape (n_samples, input_dim)
        
        Returns:
            Predictions of shape (n_samples, output_dim)
        """
        if self.W is None:
            raise ValueError("Readout not trained yet. Call fit() first.")
        
        return reservoir_features @ self.W + self.b


def create_quantum_reservoir(
    n_qubits: int = 5,
    n_layers: int = 2,
    coupling_strength: float = 0.5,
    random_seed: int = 42
) -> QuantumReservoir:
    """Factory function to create quantum reservoir."""
    return QuantumReservoir(
        n_qubits=n_qubits,
        n_layers=n_layers,
        coupling_strength=coupling_strength,
        random_seed=random_seed
    )

