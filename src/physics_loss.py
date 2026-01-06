"""
Physics-Informed Loss Function for Lorenz System

This module implements a physics-informed loss function that enforces:
1. The Lorenz differential equations at all time points
2. Initial boundary conditions

The loss guides the quantum circuit to learn the correct dynamics.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


class LorenzPhysicsLoss:
    """
    Physics-informed loss function for training the Lorenz quantum circuit.
    
    Loss = L_diff + λ * L_boundary
    
    Where:
    - L_diff: Enforces Lorenz differential equations
    - L_boundary: Enforces initial conditions
    - λ: Weighting parameter for boundary loss
    """
    
    def __init__(
        self,
        circuit_obj,
        t_points: np.ndarray,
        initial_condition: np.ndarray,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        lambda_boundary: float = 10.0,
        estimator: Optional[AerEstimator] = None
    ):
        """
        Initialize the physics-informed loss.
        
        Args:
            circuit_obj: LorenzQuantumCircuit instance
            t_points: Array of time points for training
            initial_condition: Initial condition [x0, y0, z0]
            sigma: Lorenz parameter σ
            rho: Lorenz parameter ρ
            beta: Lorenz parameter β
            lambda_boundary: Weight for boundary loss
            estimator: Qiskit Estimator (creates default if None)
        """
        self.circuit_obj = circuit_obj
        self.t_points = t_points
        self.initial_condition = initial_condition
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.lambda_boundary = lambda_boundary
        
        # Create estimator if not provided
        self.estimator = estimator if estimator is not None else AerEstimator()
        
        # Cache for storing computed values
        self._cache = {}
    
    def evaluate_circuit_at_times(
        self,
        theta: np.ndarray,
        t_points: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate quantum circuit at multiple time points.
        
        Args:
            theta: Variational parameters
            t_points: Array of time points
        
        Returns:
            Array of shape (n_points, 3) with [x, y, z] at each time
        """
        n_points = len(t_points)
        results = np.zeros((n_points, 3))
        
        observables = self.circuit_obj.get_observables()
        
        # Build list of PUBs (Parameterized Unit Blocks) for batch execution
        pubs = []
        for t in t_points:
            bound_circuit = self.circuit_obj.assign_parameters(t, theta)
            for obs in observables:
                pubs.append((bound_circuit, obs))
        
        # Run all circuits at once
        job = self.estimator.run(pubs)
        result = job.result()
        
        # Extract results
        for i in range(n_points):
            for j, var in enumerate(['x', 'y', 'z']):
                idx = i * 3 + j
                exp_val = result[idx].data.evs
                results[i, j] = self.circuit_obj.map_expectation_to_value(exp_val, var)
        
        return results
    
    def compute_time_derivatives(
        self,
        theta: np.ndarray,
        t_points: np.ndarray,
        states: Optional[np.ndarray] = None,
        epsilon: float = 1e-3
    ) -> np.ndarray:
        """
        Compute time derivatives using finite differences on the CIRCUIT itself.
        
        This properly implements ∂f/∂t where f(t,θ) is the quantum circuit.
        We perturb the TIME INPUT and measure how the circuit OUTPUT changes.
        
        This is the correct approach for Physics-Informed Neural Networks:
        - NOT: finite differences on sampled outputs (what we had before)
        - YES: finite differences on the function itself w.r.t. its input
        
        Args:
            theta: Variational parameters
            t_points: Array of time points
            states: Pre-computed states (optional, for efficiency)
            epsilon: Finite difference step size
        
        Returns:
            Array of shape (n_points, 3) with [∂x/∂t, ∂y/∂t, ∂z/∂t]
        """
        n_points = len(t_points)
        derivatives = np.zeros((n_points, 3))
        
        # For each time point, compute derivative using finite differences
        # on the quantum circuit function
        for i, t in enumerate(t_points):
            # Central finite difference: [f(t+ε) - f(t-ε)] / (2ε)
            # This gives us ∂QC/∂t at time t
            
            # Evaluate circuit at t + epsilon
            t_plus = t + epsilon
            state_plus = self.evaluate_circuit_at_times(theta, np.array([t_plus]))
            
            # Evaluate circuit at t - epsilon
            t_minus = max(0.0, t - epsilon)  # Don't go negative
            state_minus = self.evaluate_circuit_at_times(theta, np.array([t_minus]))
            
            # Central difference (or forward if at t=0)
            if t < epsilon:
                # Forward difference at t=0
                state_current = self.evaluate_circuit_at_times(theta, np.array([t]))
                derivatives[i] = (state_plus[0] - state_current[0]) / epsilon
            else:
                # Central difference
                derivatives[i] = (state_plus[0] - state_minus[0]) / (2 * epsilon)
        
        return derivatives
    
    def lorenz_rhs(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Compute right-hand side of Lorenz equations.
        
        Args:
            x, y, z: Current state
        
        Returns:
            Tuple (dx/dt, dy/dt, dz/dt) from Lorenz equations
        """
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt
    
    def compute_differential_loss(
        self,
        theta: np.ndarray,
        epsilon: float = 1e-3
    ) -> float:
        """
        Compute L_diff: loss from violating differential equations.
        
        This is the core of the Physics-Informed approach:
        L_diff = mean_over_time( ||∂f/∂t - F(f)||² )
        
        where:
        - f(t,θ) is the quantum circuit output
        - ∂f/∂t is computed via finite differences on the circuit
        - F is the Lorenz RHS: [σ(y-x), x(ρ-z)-y, xy-βz]
        
        Args:
            theta: Variational parameters
            epsilon: Finite difference step size for time derivatives
        
        Returns:
            Differential loss value
        """
        # Get states at all time points
        states = self.evaluate_circuit_at_times(theta, self.t_points)
        
        # Compute time derivatives properly (using circuit derivatives)
        derivatives = self.compute_time_derivatives(theta, self.t_points, epsilon=epsilon)
        
        # Compute expected derivatives from Lorenz equations
        expected_derivatives = np.zeros_like(derivatives)
        for i in range(len(self.t_points)):
            x, y, z = states[i]
            expected_derivatives[i] = self.lorenz_rhs(x, y, z)
        
        # Mean squared error
        diff = derivatives - expected_derivatives
        loss = np.mean(diff ** 2)
        
        return loss
    
    def compute_boundary_loss(
        self,
        theta: np.ndarray
    ) -> float:
        """
        Compute L_boundary: loss from violating initial conditions.
        
        Args:
            theta: Variational parameters
        
        Returns:
            Boundary loss value
        """
        # Evaluate at t=0 (first time point)
        t0 = self.t_points[0]
        state_t0 = self.evaluate_circuit_at_times(theta, np.array([t0]))[0]
        
        # Compare with initial condition
        diff = state_t0 - self.initial_condition
        loss = np.mean(diff ** 2)
        
        return loss
    
    def compute_total_loss(
        self,
        theta: np.ndarray,
        return_components: bool = False,
        epsilon: float = 1e-3
    ) -> float:
        """
        Compute total physics-informed loss.
        
        Args:
            theta: Variational parameters
            return_components: If True, return (total, L_diff, L_boundary)
            epsilon: Finite difference step size for time derivatives
        
        Returns:
            Total loss value (or tuple if return_components=True)
        """
        L_diff = self.compute_differential_loss(theta, epsilon=epsilon)
        L_boundary = self.compute_boundary_loss(theta)
        
        total_loss = L_diff + self.lambda_boundary * L_boundary
        
        if return_components:
            return total_loss, L_diff, L_boundary
        else:
            return total_loss
    
    def compute_gradient(
        self,
        theta: np.ndarray,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """
        Compute gradient of loss with respect to parameters using finite differences.
        
        Args:
            theta: Variational parameters
            epsilon: Step size for finite differences (for parameter gradient)
        
        Returns:
            Gradient array of same shape as theta
        """
        grad = np.zeros_like(theta)
        
        # Compute loss at current point (use smaller epsilon for time derivatives)
        loss_current = self.compute_total_loss(theta, epsilon=1e-3)
        
        # Finite difference for each parameter
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            
            loss_plus = self.compute_total_loss(theta_plus, epsilon=1e-3)
            
            grad[i] = (loss_plus - loss_current) / epsilon
        
        return grad


def create_physics_loss(
    circuit_obj,
    t_points: np.ndarray,
    initial_condition: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    lambda_boundary: float = 10.0
) -> LorenzPhysicsLoss:
    """
    Factory function to create physics-informed loss.
    
    Args:
        circuit_obj: LorenzQuantumCircuit instance
        t_points: Time points for training
        initial_condition: Initial condition [x0, y0, z0]
        sigma, rho, beta: Lorenz parameters
        lambda_boundary: Weight for boundary loss
    
    Returns:
        LorenzPhysicsLoss instance
    """
    return LorenzPhysicsLoss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition,
        sigma=sigma,
        rho=rho,
        beta=beta,
        lambda_boundary=lambda_boundary
    )

