"""
Hybrid physics-informed + data-driven loss function.

Based on research showing that pure physics-informed loss can suffer from
barren plateaus. Adding a data-driven component improves trainability.

References:
- arXiv:2311.14105 (Hybrid quantum-classical reservoir computing)
- arXiv:1906.01170 (Adaptive activation functions in PINNs)
"""

import numpy as np
from typing import Tuple, Optional
from physics_loss import LorenzPhysicsLoss


class HybridLorenzLoss(LorenzPhysicsLoss):
    """
    Hybrid loss combining physics-informed and data-driven components.
    
    L_total = α * L_physics + β * L_data
    
    where:
    - L_physics = L_diff + λ_boundary * L_boundary (from parent class)
    - L_data = MSE between circuit output and classical solution samples
    """
    
    def __init__(
        self,
        circuit_obj,
        t_points: np.ndarray,
        initial_condition: np.ndarray,
        reference_states: Optional[np.ndarray] = None,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        lambda_boundary: float = 10.0,
        alpha_physics: float = 1.0,
        beta_data: float = 10.0
    ):
        """
        Initialize hybrid loss.
        
        Args:
            circuit_obj: Lorenz quantum circuit
            t_points: Time points for evaluation
            initial_condition: Initial conditions [x0, y0, z0]
            reference_states: Classical solution states at t_points (n_points, 3)
            sigma, rho, beta: Lorenz parameters
            lambda_boundary: Weight for boundary condition loss
            alpha_physics: Weight for physics-informed loss
            beta_data: Weight for data-driven loss
        """
        super().__init__(
            circuit_obj=circuit_obj,
            t_points=t_points,
            initial_condition=initial_condition,
            sigma=sigma,
            rho=rho,
            beta=beta,
            lambda_boundary=lambda_boundary
        )
        
        self.reference_states = reference_states
        self.alpha_physics = alpha_physics
        self.beta_data = beta_data
    
    def compute_data_loss(self, theta: np.ndarray) -> float:
        """
        Compute data-driven loss: MSE with reference solution.
        
        Args:
            theta: Variational parameters
        
        Returns:
            Data loss value
        """
        if self.reference_states is None:
            return 0.0
        
        # Evaluate circuit at time points
        states = self.evaluate_circuit_at_times(theta, self.t_points)
        
        # MSE with reference
        diff = states - self.reference_states
        loss = np.mean(diff ** 2)
        
        return loss
    
    def compute_total_loss(
        self,
        theta: np.ndarray,
        return_components: bool = False,
        epsilon: float = 1e-3
    ) -> float:
        """
        Compute hybrid total loss.
        
        Args:
            theta: Variational parameters
            return_components: If True, return all components
            epsilon: Finite difference step size
        
        Returns:
            Total loss (or tuple if return_components=True)
        """
        # Physics-informed components
        L_diff = self.compute_differential_loss(theta, epsilon=epsilon)
        L_boundary = self.compute_boundary_loss(theta)
        L_physics = L_diff + self.lambda_boundary * L_boundary
        
        # Data-driven component
        L_data = self.compute_data_loss(theta)
        
        # Hybrid loss
        total_loss = self.alpha_physics * L_physics + self.beta_data * L_data
        
        if return_components:
            return total_loss, L_physics, L_data, L_diff, L_boundary
        else:
            return total_loss
    
    def compute_gradient(
        self,
        theta: np.ndarray,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """
        Compute gradient of hybrid loss.
        
        Args:
            theta: Variational parameters
            epsilon: Step size for finite differences
        
        Returns:
            Gradient array
        """
        grad = np.zeros_like(theta)
        
        # Compute loss at current point
        loss_current = self.compute_total_loss(theta, epsilon=1e-3)
        
        # Finite difference for each parameter
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            
            loss_plus = self.compute_total_loss(theta_plus, epsilon=1e-3)
            
            grad[i] = (loss_plus - loss_current) / epsilon
        
        return grad


def create_hybrid_loss(
    circuit_obj,
    t_points: np.ndarray,
    initial_condition: np.ndarray,
    reference_states: Optional[np.ndarray] = None,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    lambda_boundary: float = 10.0,
    alpha_physics: float = 1.0,
    beta_data: float = 10.0
) -> HybridLorenzLoss:
    """
    Factory function to create hybrid loss.
    
    Args:
        circuit_obj: Lorenz quantum circuit
        t_points: Time points
        initial_condition: Initial conditions
        reference_states: Classical solution (if available)
        sigma, rho, beta: Lorenz parameters
        lambda_boundary: Boundary loss weight
        alpha_physics: Physics loss weight
        beta_data: Data loss weight
    
    Returns:
        HybridLorenzLoss instance
    """
    return HybridLorenzLoss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition,
        reference_states=reference_states,
        sigma=sigma,
        rho=rho,
        beta=beta,
        lambda_boundary=lambda_boundary,
        alpha_physics=alpha_physics,
        beta_data=beta_data
    )

