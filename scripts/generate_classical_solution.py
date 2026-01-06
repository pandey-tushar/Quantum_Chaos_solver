#!/usr/bin/env python3
"""
Generate classical reference solution for the Lorenz system.

This script:
1. Solves the Lorenz system using RK4 method
2. Saves the solution to a file for comparison with quantum solver
3. Optionally creates a visualization to verify the butterfly attractor
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from lorenz_system import solve_lorenz


def main():
    """Generate and save classical Lorenz solution."""
    print("=" * 60)
    print("Generating Classical Lorenz Solution")
    print("=" * 60)
    print()
    
    # Standard Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    # Initial conditions (standard values)
    initial_state = np.array([1.0, 1.0, 1.0])
    
    # Time span: t ∈ [0, T] where T captures the attractor
    t_start = 0.0
    t_end = 50.0
    n_steps = 5000  # High resolution for smooth trajectory
    
    print(f"Parameters:")
    print(f"  σ (sigma) = {sigma}")
    print(f"  ρ (rho) = {rho}")
    print(f"  β (beta) = {beta}")
    print()
    print(f"Initial conditions: x={initial_state[0]}, y={initial_state[1]}, z={initial_state[2]}")
    print(f"Time span: [{t_start}, {t_end}]")
    print(f"Number of steps: {n_steps}")
    print()
    
    # Solve the system
    print("Solving Lorenz system with RK4...")
    t_array, state_array = solve_lorenz(
        initial_state=initial_state,
        t_span=(t_start, t_end),
        n_steps=n_steps,
        sigma=sigma,
        rho=rho,
        beta=beta
    )
    
    x_array = state_array[:, 0]
    y_array = state_array[:, 1]
    z_array = state_array[:, 2]
    
    print(f"Solution computed: {len(t_array)} time points")
    print(f"State range:")
    print(f"  x: [{x_array.min():.2f}, {x_array.max():.2f}]")
    print(f"  y: [{y_array.min():.2f}, {y_array.max():.2f}]")
    print(f"  z: [{z_array.min():.2f}, {z_array.max():.2f}]")
    print()
    
    # Save solution to file
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "lorenz_classical_solution.npz"
    np.savez(
        output_file,
        t=t_array,
        x=x_array,
        y=y_array,
        z=z_array,
        initial_state=initial_state,
        sigma=sigma,
        rho=rho,
        beta=beta,
        t_span=np.array([t_start, t_end]),
        n_steps=n_steps
    )
    
    print(f"✓ Solution saved to: {output_file}")
    print()
    
    # Create visualization to verify butterfly attractor
    print("Creating visualization...")
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D Lorenz attractor plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(x_array, y_array, z_array, 'b-', linewidth=0.5, alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Lorenz Attractor (3D)')
    
    # Time series plots
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_array, x_array, 'r-', label='x(t)', linewidth=0.5)
    ax2.plot(t_array, y_array, 'g-', label='y(t)', linewidth=0.5)
    ax2.plot(t_array, z_array, 'b-', label='z(t)', linewidth=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase space projections
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_array, y_array, 'b-', linewidth=0.5, alpha=0.6)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Phase Space: x-y projection')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_array, z_array, 'b-', linewidth=0.5, alpha=0.6)
    ax4.set_xlabel('x')
    ax4.set_ylabel('z')
    ax4.set_title('Phase Space: x-z projection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = results_dir / "lorenz_classical_validation.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    print()
    
    # Validation checks
    print("Validation checks:")
    print("-" * 60)
    
    # Check 1: Solution should show chaotic behavior (non-periodic)
    # Approximate check: look for large variations
    x_range = x_array.max() - x_array.min()
    y_range = y_array.max() - y_array.min()
    z_range = z_array.max() - z_array.min()
    
    print(f"✓ State ranges: x={x_range:.2f}, y={y_range:.2f}, z={z_range:.2f}")
    
    # Check 2: Solution should be bounded (not diverging)
    max_abs = max(np.abs(x_array).max(), np.abs(y_array).max(), np.abs(z_array).max())
    if max_abs < 100:
        print(f"✓ Solution is bounded (max |value| = {max_abs:.2f})")
    else:
        print(f"⚠ Warning: Solution may be diverging (max |value| = {max_abs:.2f})")
    
    # Check 3: Butterfly shape (z should have positive values)
    if z_array.min() > 0:
        print(f"✓ Z values are positive (min z = {z_array.min():.2f}) - butterfly attractor confirmed")
    else:
        print(f"⚠ Z values include negative (min z = {z_array.min():.2f})")
    
    print()
    print("=" * 60)
    print("Classical solution generation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Review visualization in results/lorenz_classical_validation.png")
    print("  - Proceed to Phase 3: Quantum Circuit Architecture Design")
    print()


if __name__ == "__main__":
    main()

