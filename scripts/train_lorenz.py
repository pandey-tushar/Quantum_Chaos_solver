#!/usr/bin/env python3
"""
Train the Lorenz quantum circuit solver.

This script trains the quantum circuit to solve the Lorenz system
using physics-informed loss.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_circuit import create_lorenz_circuit
from physics_loss import create_physics_loss
from training import train_lorenz_circuit


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Lorenz quantum circuit")
    parser.add_argument("--n-qubits", type=int, default=3, help="Number of qubits")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of circuit layers")
    parser.add_argument("--n-time-points", type=int, default=10, help="Number of time points")
    parser.add_argument("--t-max", type=float, default=2.0, help="Maximum time")
    parser.add_argument("--n-iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lambda-boundary", type=float, default=10.0, help="Boundary loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LORENZ QUANTUM CIRCUIT SOLVER - TRAINING")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Circuit: {args.n_qubits} qubits, {args.n_layers} layers")
    print(f"  Time points: {args.n_time_points} from 0 to {args.t_max}")
    print(f"  Training: {args.n_iterations} iterations, lr={args.learning_rate}")
    print(f"  Loss weighting: λ_boundary={args.lambda_boundary}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Create quantum circuit
    print("Creating quantum circuit...")
    circuit_obj = create_lorenz_circuit(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        t_max=args.t_max
    )
    n_params = circuit_obj.get_num_parameters()
    print(f"  Number of parameters: {n_params}")
    print()
    
    # Setup training data
    print("Setting up physics-informed loss...")
    t_points = np.linspace(0, args.t_max, args.n_time_points)
    initial_condition = np.array([1.0, 1.0, 1.0])  # Standard Lorenz IC
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition,
        sigma=10.0,
        rho=28.0,
        beta=8.0/3.0,
        lambda_boundary=args.lambda_boundary
    )
    print(f"  Time points: {len(t_points)}")
    print(f"  Initial condition: {initial_condition}")
    print()
    
    # Setup results directory
    if args.results_dir is None:
        results_dir = project_root / "results" / f"training_{args.seed}"
    else:
        results_dir = Path(args.results_dir)
    
    # Train
    print("Starting training...")
    print()
    
    best_theta, history = train_lorenz_circuit(
        circuit_obj=circuit_obj,
        loss_fn=loss_fn,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        print_every=max(1, args.n_iterations // 10),
        save_every=max(10, args.n_iterations // 2),
        results_dir=results_dir,
        random_seed=args.seed
    )
    
    # Summary
    print()
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Initial loss: {history['loss_total'][0]:.6f}")
    print(f"Final loss:   {history['loss_total'][-1]:.6f}")
    print(f"Best loss:    {min(history['loss_total']):.6f}")
    print(f"Reduction:    {history['loss_total'][0] - min(history['loss_total']):.6f}")
    print(f"Relative:     {(1 - min(history['loss_total'])/history['loss_total'][0])*100:.2f}%")
    print()
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

