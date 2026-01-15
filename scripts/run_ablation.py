#!/usr/bin/env python3
"""
Run ablation experiments for window sizes.
Verifies the numbers for the paper.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_reservoir import create_quantum_reservoir, ReservoirReadout
from data_utils import load_classical_solution
from temporal_features import create_temporal_features, align_targets_with_temporal_features


def run_single_experiment(n_qubits=5, n_layers=2, window_size=5, n_train=50, seed=42):
    """Run a single reservoir experiment and return MSE."""
    
    # Create reservoir
    reservoir = create_quantum_reservoir(
        n_qubits=n_qubits,
        n_layers=n_layers,
        coupling_strength=0.5,
        random_seed=seed
    )
    
    # Load classical data
    classical = load_classical_solution(project_root / "data")
    
    # Get training data
    t_max = 3.0
    t_train = np.linspace(0, t_max, n_train)
    
    train_states = np.zeros((n_train, 3))
    for i, t in enumerate(t_train):
        idx = np.argmin(np.abs(classical['t'] - t))
        train_states[i, 0] = classical['x'][idx]
        train_states[i, 1] = classical['y'][idx]
        train_states[i, 2] = classical['z'][idx]
    
    # Generate reservoir features
    train_features = reservoir.get_reservoir_states(train_states, n_shots=1024)
    
    # Create temporal features
    if window_size > 1:
        train_features_temporal = create_temporal_features(train_features, window_size=window_size)
        train_states_aligned = align_targets_with_temporal_features(train_states, window_size=window_size)
    else:
        train_features_temporal = train_features
        train_states_aligned = train_states
    
    # Train readout
    readout = ReservoirReadout(input_dim=train_features_temporal.shape[1], output_dim=3)
    readout.fit(train_features_temporal, train_states_aligned, alpha=1.0)
    
    # Evaluate
    train_predictions = readout.predict(train_features_temporal)
    train_mse = np.mean((train_predictions - train_states_aligned) ** 2)
    
    return train_mse, train_features_temporal.shape[1]


def main():
    print("=" * 60)
    print("ABLATION STUDY: Window Size Effect")
    print("=" * 60)
    print()
    
    window_sizes = [1, 3, 5]
    results = []
    
    for w in window_sizes:
        print(f"Running window_size={w}...")
        start = time.time()
        mse, feat_dim = run_single_experiment(window_size=w)
        elapsed = time.time() - start
        results.append({
            'window': w,
            'mse': mse,
            'feat_dim': feat_dim,
            'time': elapsed
        })
        print(f"  MSE: {mse:.2f}, Feature dim: {feat_dim}, Time: {elapsed:.1f}s")
    
    print()
    print("=" * 60)
    print("ABLATION RESULTS (for paper)")
    print("=" * 60)
    print()
    print("| Window Size | Train MSE | Feature Dim |")
    print("|-------------|-----------|-------------|")
    for r in results:
        print(f"| w={r['window']}         | {r['mse']:.2f}     | {r['feat_dim']}          |")
    
    print()
    print("LaTeX table format:")
    print("\\begin{tabular}{lcc}")
    print("    \\toprule")
    print("    \\textbf{Window Size} & \\textbf{Train MSE} & \\textbf{Feature Dim} \\\\")
    print("    \\midrule")
    for r in results:
        bold = "\\textbf{" if r['window'] == 5 else ""
        bold_end = "}" if r['window'] == 5 else ""
        no_window = " (no window)" if r['window'] == 1 else ""
        print(f"    $w={r['window']}${no_window} & {bold}{r['mse']:.1f}{bold_end} & {r['feat_dim']} \\\\")
    print("    \\bottomrule")
    print("\\end{tabular}")
    
    # Save results
    import json
    with open(project_root / "results" / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/ablation_results.json")


if __name__ == "__main__":
    main()

