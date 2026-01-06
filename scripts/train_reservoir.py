#!/usr/bin/env python3
"""
Train Quantum Reservoir Computing model on Lorenz system.

Much faster than PINN because:
1. Reservoir is fixed (no gradient descent through quantum circuit)
2. Only linear readout is trained (closed-form Ridge regression)
3. Training takes seconds, not hours
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_reservoir import create_quantum_reservoir, ReservoirReadout
from data_utils import load_classical_solution
from temporal_features import create_temporal_features, align_targets_with_temporal_features


def train_reservoir(
    n_qubits: int = 5,
    n_layers: int = 2,
    coupling_strength: float = 0.5,
    n_shots: int = 1024,
    alpha_ridge: float = 1.0,
    t_max: float = 3.0,
    n_train_points: int = 50,
    n_test_points: int = 20,
    window_size: int = 3,  # NEW: temporal window
    seed: int = 42,
    results_dir: str = None
):
    """
    Train quantum reservoir computing model.
    
    Args:
        n_qubits: Number of qubits in reservoir
        n_layers: Number of reservoir layers
        coupling_strength: Inter-qubit coupling (0-1)
        n_shots: Measurement shots per state
        alpha_ridge: Ridge regression regularization
        t_max: Maximum time for training
        n_train_points: Number of training time points
        n_test_points: Number of test time points
        window_size: Temporal window size (3 = use t-2, t-1, t to predict t)
        seed: Random seed
        results_dir: Where to save results
    """
    print("=" * 80)
    print("QUANTUM RESERVOIR COMPUTING FOR LORENZ SYSTEM")
    print("=" * 80)
    print()
    print("Key Advantages over PINN:")
    print("  ✓ Reservoir is fixed (no gradient descent)")
    print("  ✓ Only linear readout trained (closed-form solution)")
    print("  ✓ Fast training (seconds vs hours)")
    print("  ✓ Natural for temporal dynamics")
    print(f"  ✓ Uses temporal context (window={window_size})")
    print()
    
    # Create reservoir
    print("Creating quantum reservoir...")
    reservoir = create_quantum_reservoir(
        n_qubits=n_qubits,
        n_layers=n_layers,
        coupling_strength=coupling_strength,
        random_seed=seed
    )
    feature_dim = reservoir.get_feature_dimension()
    print(f"  Reservoir: {n_qubits} qubits, {n_layers} layers")
    print(f"  Feature dimension: {feature_dim} (2^{n_qubits})")
    print(f"  Coupling strength: {coupling_strength}")
    print()
    
    # Load classical data
    print("Loading classical reference solution...")
    classical = load_classical_solution(project_root / "data")
    
    # Split into train and test
    t_train = np.linspace(0, t_max, n_train_points)
    t_test = np.linspace(t_max, t_max + 1.0, n_test_points)  # Test on future
    
    # Get training data
    train_states = np.zeros((n_train_points, 3))
    for i, t in enumerate(t_train):
        idx = np.argmin(np.abs(classical['t'] - t))
        train_states[i, 0] = classical['x'][idx]
        train_states[i, 1] = classical['y'][idx]
        train_states[i, 2] = classical['z'][idx]
    
    # Get test data
    test_states = np.zeros((n_test_points, 3))
    for i, t in enumerate(t_test):
        idx = np.argmin(np.abs(classical['t'] - t))
        test_states[i, 0] = classical['x'][idx]
        test_states[i, 1] = classical['y'][idx]
        test_states[i, 2] = classical['z'][idx]
    
    print(f"  Training: {len(t_train)} points from t∈[0, {t_max}]")
    print(f"  Testing: {len(t_test)} points from t∈[{t_max}, {t_max+1}]")
    print()
    
    # Generate reservoir features
    print("Generating reservoir features (this takes time)...")
    print(f"  Processing {n_train_points} training states...")
    start_feature = time.time()
    train_features = reservoir.get_reservoir_states(train_states, n_shots=n_shots)
    feature_time = time.time() - start_feature
    print(f"  ✓ Features computed in {feature_time:.1f}s")
    print(f"  Feature shape: {train_features.shape}")
    print(f"  Feature sparsity: {np.sum(train_features == 0) / train_features.size * 100:.1f}% zeros")
    print()
    
    # Create temporal features
    print(f"Creating temporal features (window={window_size})...")
    train_features_temporal = create_temporal_features(train_features, window_size=window_size)
    train_states_aligned = align_targets_with_temporal_features(train_states, window_size=window_size)
    print(f"  Temporal features: {train_features_temporal.shape}")
    print(f"  Aligned targets: {train_states_aligned.shape}")
    print(f"  Effective training samples: {len(train_states_aligned)}")
    print()
    
    # Train readout
    print("Training readout layer...")
    start_train = time.time()
    readout = ReservoirReadout(input_dim=train_features_temporal.shape[1], output_dim=3)
    readout.fit(train_features_temporal, train_states_aligned, alpha=alpha_ridge)
    train_time = time.time() - start_train
    print(f"  ✓ Readout trained in {train_time:.3f}s (Ridge regression)")
    print(f"  Weights shape: {readout.W.shape}")
    print()
    
    # Evaluate on training set
    print("Evaluating on training set...")
    train_predictions = readout.predict(train_features_temporal)
    train_error = np.mean((train_predictions - train_states_aligned) ** 2)
    train_mae = np.mean(np.abs(train_predictions - train_states_aligned))
    print(f"  Train MSE: {train_error:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    
    # Component-wise errors
    train_errors_per_var = np.mean((train_predictions - train_states_aligned) ** 2, axis=0)
    print(f"  Train MSE per variable:")
    print(f"    x: {train_errors_per_var[0]:.4f}")
    print(f"    y: {train_errors_per_var[1]:.4f}")
    print(f"    z: {train_errors_per_var[2]:.4f}")
    print()
    
    # Evaluate on test set (future prediction)
    print("Evaluating on test set (future prediction)...")
    print(f"  Generating test features...")
    test_features = reservoir.get_reservoir_states(test_states, n_shots=n_shots)
    test_features_temporal = create_temporal_features(test_features, window_size=window_size)
    test_states_aligned = align_targets_with_temporal_features(test_states, window_size=window_size)
    test_predictions = readout.predict(test_features_temporal)
    test_error = np.mean((test_predictions - test_states_aligned) ** 2)
    test_mae = np.mean(np.abs(test_predictions - test_states_aligned))
    print(f"  Test MSE: {test_error:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    
    # Component-wise errors
    test_errors_per_var = np.mean((test_predictions - test_states_aligned) ** 2, axis=0)
    print(f"  Test MSE per variable:")
    print(f"    x: {test_errors_per_var[0]:.4f}")
    print(f"    y: {test_errors_per_var[1]:.4f}")
    print(f"    z: {test_errors_per_var[2]:.4f}")
    print()
    
    # Save results
    if results_dir is None:
        results_dir = project_root / "results" / f"reservoir_{seed}"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving results...")
    np.savez(
        results_dir / "reservoir_model.npz",
        W=readout.W,
        b=readout.b,
        train_error=train_error,
        test_error=test_error,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        train_states=train_states,
        test_states=test_states,
        t_train=t_train,
        t_test=t_test
    )
    
    # Save config
    import json
    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "coupling_strength": coupling_strength,
        "n_shots": n_shots,
        "alpha_ridge": alpha_ridge,
        "feature_dim": feature_dim,
        "train_error": float(train_error),
        "test_error": float(test_error),
        "feature_time": float(feature_time),
        "train_time": float(train_time)
    }
    
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Results saved to: {results_dir}")
    print()
    
    # Summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {feature_time + train_time:.1f}s")
    print(f"  Feature generation: {feature_time:.1f}s")
    print(f"  Readout training: {train_time:.3f}s")
    print()
    print("Performance:")
    print(f"  Train MSE: {train_error:.4f}")
    print(f"  Test MSE: {test_error:.4f}")
    print()
    print("Comparison with PINN:")
    pinn_mse = 37.77
    print(f"  PINN MSE: {pinn_mse:.2f}")
    print(f"  Reservoir MSE: {train_error:.2f}")
    if train_error < pinn_mse:
        improvement = (pinn_mse - train_error) / pinn_mse * 100
        print(f"  ✓ Reservoir is {improvement:.1f}% better!")
    else:
        decline = (train_error - pinn_mse) / pinn_mse * 100
        print(f"  ✗ Reservoir is {decline:.1f}% worse")
    print("=" * 80)
    
    return train_error, test_error


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Quantum Reservoir for Lorenz")
    parser.add_argument("--n-qubits", type=int, default=5, help="Number of qubits")
    parser.add_argument("--n-layers", type=int, default=2, help="Reservoir layers")
    parser.add_argument("--coupling", type=float, default=0.5, help="Coupling strength")
    parser.add_argument("--n-shots", type=int, default=1024, help="Measurement shots")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    parser.add_argument("--t-max", type=float, default=3.0, help="Max training time")
    parser.add_argument("--n-train", type=int, default=50, help="Training points")
    parser.add_argument("--n-test", type=int, default=20, help="Test points")
    parser.add_argument("--window", type=int, default=3, help="Temporal window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    
    args = parser.parse_args()
    
    train_reservoir(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        coupling_strength=args.coupling,
        n_shots=args.n_shots,
        alpha_ridge=args.alpha,
        t_max=args.t_max,
        n_train_points=args.n_train,
        n_test_points=args.n_test,
        window_size=args.window,
        seed=args.seed,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()

