"""
Lorenz System of Differential Equations

This module implements the classical Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Standard parameters (chaotic regime):
    σ (sigma) = 10
    ρ (rho) = 28
    β (beta) = 8/3
"""

import numpy as np
from typing import Tuple, Callable


def lorenz_equations(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0
) -> np.ndarray:
    """
    Compute the derivatives of the Lorenz system at a given state.
    
    Args:
        t: Time (not used in Lorenz equations, but required for ODE solver interface)
        state: Current state vector [x, y, z]
        sigma: Lorenz parameter σ (default: 10.0)
        rho: Lorenz parameter ρ (default: 28.0)
        beta: Lorenz parameter β (default: 8/3)
    
    Returns:
        Derivative vector [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return np.array([dx_dt, dy_dt, dz_dt])


def runge_kutta4(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta 4th order (RK4) method for solving ODEs.
    
    Args:
        f: Function f(t, y) that returns dy/dt
        y0: Initial state vector
        t_span: Tuple (t_start, t_end)
        n_steps: Number of time steps
        **kwargs: Additional arguments to pass to f
    
    Returns:
        Tuple (t_array, y_array) where:
            t_array: Array of time points
            y_array: Array of state vectors, shape (n_steps+1, state_dim)
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    
    # Initialize arrays
    t_array = np.linspace(t_start, t_end, n_steps + 1)
    y_array = np.zeros((n_steps + 1, len(y0)))
    y_array[0] = y0
    
    # RK4 integration
    for i in range(n_steps):
        t = t_array[i]
        y = y_array[i]
        
        # Compute four slopes
        k1 = dt * f(t, y, **kwargs)
        k2 = dt * f(t + dt/2, y + k1/2, **kwargs)
        k3 = dt * f(t + dt/2, y + k2/2, **kwargs)
        k4 = dt * f(t + dt, y + k3, **kwargs)
        
        # Update state
        y_array[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_array, y_array


def solve_lorenz(
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int = 1000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Lorenz system using RK4 method.
    
    Args:
        initial_state: Initial state [x0, y0, z0]
        t_span: Tuple (t_start, t_end) for time integration
        n_steps: Number of time steps (default: 1000)
        sigma: Lorenz parameter σ (default: 10.0)
        rho: Lorenz parameter ρ (default: 28.0)
        beta: Lorenz parameter β (default: 8/3)
    
    Returns:
        Tuple (t_array, state_array) where:
            t_array: Array of time points
            state_array: Array of state vectors, shape (n_steps+1, 3)
                        Columns are [x, y, z]
    """
    return runge_kutta4(
        lorenz_equations,
        initial_state,
        t_span,
        n_steps,
        sigma=sigma,
        rho=rho,
        beta=beta
    )

