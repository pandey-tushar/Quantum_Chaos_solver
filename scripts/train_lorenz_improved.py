#!/usr/bin/env python3
"""
Improved training script based on QCPINN (2024) and barren plateau research.

Key improvements:
1. Increased to 4 qubits (more expressivity without excessive barren plateaus)
2. 3 variational layers (proven optimal depth)
3. Small parameter initialization (scale=0.1)
4. Hybrid loss (physics + data-driven)
5. Adaptive momentum + LR decay + gradient clipping

References:
- arXiv:2503.16678 (QCPINN 2024)
- arXiv:2311.14105 (Hybrid quantum-classical)
- arXiv:2411.09226 (Mitigating barren plateaus)
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_circuit import create_lorenz_circuit
from hybrid_loss import create_hybrid_loss
from training import AdamOptimizer, TrainingMonitor
from data_utils import load_classical_solution


def train_improved(
    n_qubits: int = 4,  # Increased from 3
    n_layers: int = 3,  # Back to 3 layers
    n_time_points: int = 15,
    t_max: float = 3.0,
    n_iterations: int = 1000,
    learning_rate: float = 0.05,  # Lower initial LR
    lambda_boundary: float = 50.0,
    alpha_physics: float = 0.5,
    beta_data: float = 10.0,
    seed: int = 42,
    results_dir: str = None
):
    """Train with improved settings based on QCPINN paper (2024)."""
    
    print("=" * 80)
    print("IMPROVED LORENZ QUANTUM CIRCUIT SOLVER")
    print("=" * 80)
    print()
    print("Research-backed improvements (QCPINN 2024):")
    print("  ✓ Increased to 4 qubits (more expressivity)")
    print("  ✓ 3 layers (proven optimal depth)")
    print("  ✓ Smaller parameter initialization (scale=0.1)")
    print("  ✓ Hybrid loss (physics + data-driven)")
    print("  ✓ Gradient clipping + LR decay")
    print("  ✓ Shorter time span (t ∈ [0, 3])")
    print()
    
    # Create quantum circuit
    print("Creating quantum circuit...")
    circuit_obj = create_lorenz_circuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        t_max=t_max
    )
    n_params = circuit_obj.get_num_parameters()
    print(f"  Circuit: {n_qubits} qubits, {n_layers} layers")
    print(f"  Parameters: {n_params}")
    print()
    
    # Setup training data
    print("Loading classical reference solution...")
    classical = load_classical_solution(project_root / "data")
    
    # Sample time points from classical solution
    t_points = np.linspace(0, t_max, n_time_points)
    
    # Get reference states at these time points
    reference_states = np.zeros((n_time_points, 3))
    for i, t in enumerate(t_points):
        idx = np.argmin(np.abs(classical['t'] - t))
        reference_states[i, 0] = classical['x'][idx]
        reference_states[i, 1] = classical['y'][idx]
        reference_states[i, 2] = classical['z'][idx]
    
    initial_condition = reference_states[0]  # Use actual initial state from classical
    
    print(f"  Time points: {len(t_points)} from 0 to {t_max}")
    print(f"  Initial condition: [{initial_condition[0]:.2f}, {initial_condition[1]:.2f}, {initial_condition[2]:.2f}]")
    print()
    
    # Create hybrid loss
    print("Creating hybrid loss function...")
    loss_fn = create_hybrid_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition,
        reference_states=reference_states,
        sigma=10.0,
        rho=28.0,
        beta=8.0/3.0,
        lambda_boundary=lambda_boundary,
        alpha_physics=alpha_physics,
        beta_data=beta_data
    )
    print(f"  α_physics = {alpha_physics} (physics loss weight)")
    print(f"  β_data = {beta_data} (data loss weight)")
    print(f"  λ_boundary = {lambda_boundary} (boundary loss weight)")
    print()
    
    # Initialize parameters with small values
    print("Initializing parameters...")
    theta = circuit_obj.initialize_parameters(seed=seed, scale=0.1)
    print(f"  Initialization scale: 0.1 (small to avoid barren plateaus)")
    print(f"  Parameter range: [{theta.min():.4f}, {theta.max():.4f}]")
    print()
    
    # Setup optimizer based on QCPINN recommendations
    optimizer = AdamOptimizer(
        learning_rate=learning_rate,  # Use function parameter
        adaptive_momentum=True,
        loss_threshold=100.0,
        grad_threshold=0.5,
        lr_decay=0.99  # Faster decay: 1% per iteration (was 0.995)
    )
    
    # Early stopping configuration
    patience = 20  # Stop if no improvement for 20 iterations
    best_loss = float('inf')
    best_theta = None
    no_improve_count = 0
    
    # Setup monitoring
    if results_dir is None:
        results_dir = project_root / "results" / f"training_improved_{seed}"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor()
    monitor.start()  # Start timing
    
    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()
    
    # Determine print frequency
    print_freq = max(1, n_iterations // 20)  # Print ~20 times
    
    for iteration in range(n_iterations):
        # Compute loss and components
        total_loss, l_physics, l_data, l_diff, l_boundary = loss_fn.compute_total_loss(
            theta, return_components=True, epsilon=1e-3
        )
        
        # Compute gradient
        grad = loss_fn.compute_gradient(theta, epsilon=1e-4)
        grad_norm = np.linalg.norm(grad)
        
        # Update parameters with adaptive momentum
        theta = optimizer.step(theta, grad, loss=total_loss)
        
        # Early stopping: track best parameters
        if total_loss < best_loss:
            best_loss = total_loss
            best_theta = theta.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Stop if no improvement for patience iterations
        if no_improve_count >= patience:
            print(f"\n⚠ Early stopping at iteration {iteration} (no improvement for {patience} iters)")
            print(f"   Best loss: {best_loss:.2f} at iteration {iteration - patience}")
            theta = best_theta  # Restore best parameters
            break
        
        # Log
        monitor.log(
            iteration=iteration,
            loss_total=float(total_loss),
            loss_diff=float(l_diff),
            loss_boundary=float(l_boundary),
            grad_norm=float(grad_norm)
        )
        
        # Add extra fields for hybrid loss
        monitor.history.setdefault('loss_physics', []).append(float(l_physics))
        monitor.history.setdefault('loss_data', []).append(float(l_data))
        
        # Print progress
        if iteration % print_freq == 0 or iteration == n_iterations - 1:
            # Show ACTUAL gradient norm (not clipped for display)
            current_lr = optimizer.learning_rate
            print(f"Iter {iteration:4d} | L_total: {total_loss:8.2f} | "
                  f"L_data: {l_data:8.2f} | |∇|: {grad_norm:8.2f} | LR: {current_lr:.4f}")
        
        # Save checkpoints
        if (iteration + 1) % 250 == 0:
            checkpoint_path = results_dir / f"checkpoint_iter_{iteration+1}.npz"
            np.savez(checkpoint_path, theta=theta, iteration=iteration, loss=total_loss)
    
    # Save final results (use best parameters if early stopping triggered)
    print()
    print("Saving final results...")
    results_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    final_path = results_dir / "final_parameters.npz"
    
    # Use best parameters if found, otherwise use final
    if best_theta is not None:
        save_theta = best_theta
        save_loss = best_loss
    else:
        save_theta = theta
        save_loss = total_loss
    
    np.savez(
        final_path,
        theta=save_theta,
        final_theta=theta,
        final_loss=total_loss,
        best_loss=save_loss,
        best_theta=save_theta
    )
    
    # Save history manually
    import json
    with open(results_dir / "training_history.json", 'w') as f:
        json.dump(monitor.history, f, indent=2)
    
    print(f"  Final loss: {total_loss:.2f}")
    print(f"  Final L_data: {l_data:.2f} (should be < 10 for good fit)")
    print(f"  Results saved to: {results_dir}")
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return theta, total_loss


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train improved Lorenz quantum circuit")
    parser.add_argument("--n-qubits", type=int, default=4, help="Increased to 4")
    parser.add_argument("--n-layers", type=int, default=3, help="Back to 3 layers")
    parser.add_argument("--n-time-points", type=int, default=15)
    parser.add_argument("--t-max", type=float, default=3.0, help="Shorter time span")
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Increased")
    parser.add_argument("--lambda-boundary", type=float, default=50.0)
    parser.add_argument("--alpha-physics", type=float, default=0.5)
    parser.add_argument("--beta-data", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    train_improved(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_time_points=args.n_time_points,
        t_max=args.t_max,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        lambda_boundary=args.lambda_boundary,
        alpha_physics=args.alpha_physics,
        beta_data=args.beta_data,
        seed=args.seed,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()

