"""
Visualization tools for the Lorenz quantum circuit solver.

Creates publication-quality plots comparing quantum and classical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Dict, Tuple
import json


def plot_3d_attractor(
    t_classical: np.ndarray,
    xyz_classical: np.ndarray,
    t_quantum: np.ndarray,
    xyz_quantum: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Lorenz Attractor: Quantum vs Classical"
):
    """
    Plot 3D Lorenz attractor comparing quantum and classical solutions.
    
    Args:
        t_classical: Time array for classical solution
        xyz_classical: Classical solution array (n_points, 3)
        t_quantum: Time array for quantum solution
        xyz_quantum: Quantum solution array (n_points, 3)
        save_path: Path to save figure
        title: Plot title
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Classical solution
    ax1 = fig.add_subplot(131, projection='3d')
    x_c, y_c, z_c = xyz_classical[:, 0], xyz_classical[:, 1], xyz_classical[:, 2]
    ax1.plot(x_c, y_c, z_c, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Classical Solution (RK4)')
    
    # Quantum solution
    ax2 = fig.add_subplot(132, projection='3d')
    x_q, y_q, z_q = xyz_quantum[:, 0], xyz_quantum[:, 1], xyz_quantum[:, 2]
    ax2.plot(x_q, y_q, z_q, 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Quantum Solution (DQC)')
    
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(x_c, y_c, z_c, 'b-', linewidth=0.5, alpha=0.5, label='Classical')
    ax3.plot(x_q, y_q, z_q, 'r-', linewidth=0.5, alpha=0.5, label='Quantum')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Overlay')
    ax3.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D attractor plot to {save_path}")
    
    return fig


def plot_time_series(
    t_classical: np.ndarray,
    xyz_classical: np.ndarray,
    t_quantum: np.ndarray,
    xyz_quantum: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot time series comparison of x(t), y(t), z(t).
    
    Args:
        t_classical: Time array for classical solution
        xyz_classical: Classical solution array (n_points, 3)
        t_quantum: Time array for quantum solution
        xyz_quantum: Quantum solution array (n_points, 3)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    variables = ['x', 'y', 'z']
    colors_classical = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_quantum = ['#d62728', '#9467bd', '#8c564b']
    
    for i, (var, ax) in enumerate(zip(variables, axes)):
        ax.plot(t_classical, xyz_classical[:, i], 
                color=colors_classical[i], linewidth=1.5, 
                label=f'{var}(t) Classical', alpha=0.7)
        ax.plot(t_quantum, xyz_quantum[:, i], 
                color=colors_quantum[i], linewidth=1.5, 
                label=f'{var}(t) Quantum', alpha=0.7, linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        ax.set_title(f'{var}(t) vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series plot to {save_path}")
    
    return fig


def plot_phase_space(
    xyz_classical: np.ndarray,
    xyz_quantum: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot phase space projections (x-y, x-z, y-z).
    
    Args:
        xyz_classical: Classical solution array (n_points, 3)
        xyz_quantum: Quantum solution array (n_points, 3)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    projections = [
        ('x', 'y', 0, 1),
        ('x', 'z', 0, 2),
        ('y', 'z', 1, 2)
    ]
    
    for col, (var1, var2, idx1, idx2) in enumerate(projections):
        # Classical
        ax_c = axes[0, col]
        ax_c.plot(xyz_classical[:, idx1], xyz_classical[:, idx2], 
                 'b-', linewidth=0.5, alpha=0.6)
        ax_c.set_xlabel(var1)
        ax_c.set_ylabel(var2)
        ax_c.set_title(f'Classical: {var1}-{var2} Projection')
        ax_c.grid(True, alpha=0.3)
        
        # Quantum
        ax_q = axes[1, col]
        ax_q.plot(xyz_quantum[:, idx1], xyz_quantum[:, idx2], 
                 'r-', linewidth=0.5, alpha=0.6)
        ax_q.set_xlabel(var1)
        ax_q.set_ylabel(var2)
        ax_q.set_title(f'Quantum: {var1}-{var2} Projection')
        ax_q.grid(True, alpha=0.3)
    
    plt.suptitle('Phase Space Projections', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase space plot to {save_path}")
    
    return fig


def plot_training_metrics(
    history: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot training metrics: loss curves and gradient norm.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = history['iteration']
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(iterations, history['loss_total'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss vs Iteration')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Loss components
    ax = axes[0, 1]
    ax.plot(iterations, history['loss_diff'], 'r-', label='L_diff', linewidth=2)
    ax.plot(iterations, history['loss_boundary'], 'g-', label='L_boundary', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Gradient norm
    ax = axes[1, 0]
    ax.plot(iterations, history['grad_norm'], 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm vs Iteration')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Training time
    ax = axes[1, 1]
    ax.plot(iterations, np.array(history['time']) / 60, 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Training Time')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training metrics plot to {save_path}")
    
    return fig


def plot_error_analysis(
    t_quantum: np.ndarray,
    xyz_classical: np.ndarray,
    xyz_quantum: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot error analysis: pointwise and cumulative errors.
    
    Args:
        t_quantum: Time array
        xyz_classical: Classical solution (interpolated to quantum times)
        xyz_quantum: Quantum solution
        save_path: Path to save figure
    """
    # Compute errors
    errors = np.abs(xyz_quantum - xyz_classical)
    error_x, error_y, error_z = errors[:, 0], errors[:, 1], errors[:, 2]
    error_total = np.linalg.norm(errors, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pointwise errors
    ax = axes[0, 0]
    ax.plot(t_quantum, error_x, 'r-', label='|x_q - x_c|', linewidth=1.5)
    ax.plot(t_quantum, error_y, 'g-', label='|y_q - y_c|', linewidth=1.5)
    ax.plot(t_quantum, error_z, 'b-', label='|z_q - z_c|', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Pointwise Absolute Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total error
    ax = axes[0, 1]
    ax.plot(t_quantum, error_total, 'purple', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('L2 Error')
    ax.set_title('Total L2 Error ||x_q - x_c||')
    ax.grid(True, alpha=0.3)
    
    # Error statistics
    ax = axes[1, 0]
    ax.bar(['x', 'y', 'z'], [error_x.mean(), error_y.mean(), error_z.mean()],
           color=['r', 'g', 'b'], alpha=0.7)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Mean Errors by Component')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative error
    ax = axes[1, 1]
    cumulative_error = np.cumsum(error_total) * (t_quantum[1] - t_quantum[0])
    ax.plot(t_quantum, cumulative_error, 'orange', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Error')
    ax.set_title('Cumulative Error over Time')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error analysis plot to {save_path}")
    
    # Print statistics
    print("\nError Statistics:")
    print(f"  Mean Absolute Error: x={error_x.mean():.4f}, y={error_y.mean():.4f}, z={error_z.mean():.4f}")
    print(f"  Max Absolute Error: x={error_x.max():.4f}, y={error_y.max():.4f}, z={error_z.max():.4f}")
    print(f"  Mean L2 Error: {error_total.mean():.4f}")
    print(f"  Max L2 Error: {error_total.max():.4f}")
    
    return fig


def create_all_visualizations(
    results_dir: Path,
    classical_data_path: Path,
    output_dir: Optional[Path] = None
):
    """
    Create all visualizations for a trained model.
    
    Args:
        results_dir: Directory with training results
        classical_data_path: Path to classical solution data
        output_dir: Directory to save plots (defaults to results_dir/plots)
    """
    from data_utils import load_classical_solution
    from training import load_training_results
    from quantum_circuit import create_lorenz_circuit
    from physics_loss import create_physics_loss
    
    # Load training results
    best_theta, history = load_training_results(results_dir)
    
    # Load classical solution
    classical_data = load_classical_solution(classical_data_path.parent)
    t_classical = classical_data['t']
    xyz_classical = np.column_stack([classical_data['x'], classical_data['y'], classical_data['z']])
    
    # Generate quantum solution
    print("Generating quantum solution...")
    # Get time points from training (reconstruct from history)
    # For now, use same time range as classical
    t_quantum = np.linspace(t_classical[0], t_classical[-1], 50)
    
    # Create circuit and evaluate
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=3)
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_quantum,
        initial_condition=classical_data['initial_state']
    )
    
    xyz_quantum = loss_fn.evaluate_circuit_at_times(best_theta, t_quantum)
    
    # Setup output directory
    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nCreating visualizations in {output_dir}...")
    
    # Create all plots
    plot_3d_attractor(t_classical, xyz_classical, t_quantum, xyz_quantum,
                     save_path=output_dir / "3d_attractor.png")
    
    plot_time_series(t_classical, xyz_classical, t_quantum, xyz_quantum,
                    save_path=output_dir / "time_series.png")
    
    plot_phase_space(xyz_classical, xyz_quantum,
                    save_path=output_dir / "phase_space.png")
    
    plot_training_metrics(history,
                         save_path=output_dir / "training_metrics.png")
    
    # Interpolate classical to quantum times for error analysis
    from scipy.interpolate import interp1d
    xyz_classical_interp = np.zeros_like(xyz_quantum)
    for i in range(3):
        interp_func = interp1d(t_classical, xyz_classical[:, i], kind='cubic')
        xyz_classical_interp[:, i] = interp_func(t_quantum)
    
    plot_error_analysis(t_quantum, xyz_classical_interp, xyz_quantum,
                       save_path=output_dir / "error_analysis.png")
    
    print(f"\n✓ All visualizations created in {output_dir}")
    plt.close('all')

