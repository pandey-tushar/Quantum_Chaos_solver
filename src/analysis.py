"""
Analysis and comparison tools for quantum vs classical Lorenz solver.

Computes quantitative metrics and generates comparison reports.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime


def compute_error_metrics(
    xyz_classical: np.ndarray,
    xyz_quantum: np.ndarray
) -> Dict:
    """
    Compute comprehensive error metrics between quantum and classical solutions.
    
    Args:
        xyz_classical: Classical solution array (n_points, 3)
        xyz_quantum: Quantum solution array (n_points, 3)
    
    Returns:
        Dictionary of error metrics
    """
    # Absolute errors
    errors = np.abs(xyz_quantum - xyz_classical)
    
    # Component-wise metrics
    metrics = {}
    for i, var in enumerate(['x', 'y', 'z']):
        metrics[f'mae_{var}'] = float(np.mean(errors[:, i]))
        metrics[f'mse_{var}'] = float(np.mean(errors[:, i]**2))
        metrics[f'rmse_{var}'] = float(np.sqrt(metrics[f'mse_{var}']))
        metrics[f'max_error_{var}'] = float(np.max(errors[:, i]))
        
        # Relative errors (avoid division by zero)
        classical_vals = xyz_classical[:, i]
        nonzero_mask = np.abs(classical_vals) > 1e-10
        if np.any(nonzero_mask):
            rel_errors = errors[nonzero_mask, i] / np.abs(classical_vals[nonzero_mask])
            metrics[f'mean_rel_error_{var}'] = float(np.mean(rel_errors))
            metrics[f'max_rel_error_{var}'] = float(np.max(rel_errors))
        else:
            metrics[f'mean_rel_error_{var}'] = float('inf')
            metrics[f'max_rel_error_{var}'] = float('inf')
    
    # Overall metrics
    l2_errors = np.linalg.norm(errors, axis=1)
    metrics['mae_total'] = float(np.mean(np.sum(errors, axis=1)))
    metrics['mse_total'] = float(np.mean(np.sum(errors**2, axis=1)))
    metrics['rmse_total'] = float(np.sqrt(metrics['mse_total']))
    metrics['mean_l2_error'] = float(np.mean(l2_errors))
    metrics['max_l2_error'] = float(np.max(l2_errors))
    
    # Coefficient of determination (R²) for each component
    for i, var in enumerate(['x', 'y', 'z']):
        ss_res = np.sum((xyz_quantum[:, i] - xyz_classical[:, i])**2)
        ss_tot = np.sum((xyz_classical[:, i] - np.mean(xyz_classical[:, i]))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics[f'r2_{var}'] = float(r2)
    
    return metrics


def analyze_convergence(history: Dict) -> Dict:
    """
    Analyze training convergence behavior.
    
    Args:
        history: Training history dictionary
    
    Returns:
        Dictionary of convergence metrics
    """
    iterations = np.array(history['iteration'])
    loss_total = np.array(history['loss_total'])
    loss_diff = np.array(history['loss_diff'])
    loss_boundary = np.array(history['loss_boundary'])
    grad_norm = np.array(history['grad_norm'])
    
    metrics = {
        'initial_loss': float(loss_total[0]),
        'final_loss': float(loss_total[-1]),
        'best_loss': float(np.min(loss_total)),
        'loss_reduction': float(loss_total[0] - loss_total[-1]),
        'loss_reduction_percent': float((loss_total[0] - loss_total[-1]) / loss_total[0] * 100),
        'total_iterations': int(iterations[-1] + 1),
        'convergence_rate': float((loss_total[0] - loss_total[-1]) / iterations[-1]),
    }
    
    # Find iteration where loss dropped below certain thresholds
    thresholds = [0.9, 0.5, 0.1]
    for thresh in thresholds:
        target = loss_total[0] * thresh
        idx = np.where(loss_total <= target)[0]
        if len(idx) > 0:
            metrics[f'iter_to_{int(thresh*100)}pct'] = int(iterations[idx[0]])
        else:
            metrics[f'iter_to_{int(thresh*100)}pct'] = None
    
    # Gradient convergence
    metrics['initial_grad_norm'] = float(grad_norm[0])
    metrics['final_grad_norm'] = float(grad_norm[-1])
    metrics['grad_reduction'] = float(grad_norm[0] - grad_norm[-1])
    
    # Loss component analysis
    metrics['initial_loss_diff'] = float(loss_diff[0])
    metrics['final_loss_diff'] = float(loss_diff[-1])
    metrics['initial_loss_boundary'] = float(loss_boundary[0])
    metrics['final_loss_boundary'] = float(loss_boundary[-1])
    
    # Training time
    if 'time' in history:
        metrics['total_time_seconds'] = float(history['time'][-1])
        metrics['total_time_minutes'] = float(history['time'][-1] / 60)
        metrics['time_per_iteration'] = float(history['time'][-1] / len(iterations))
    
    return metrics


def compare_computational_resources(
    n_quantum_params: int,
    n_time_points: int,
    n_iterations: int,
    training_time: float
) -> Dict:
    """
    Compare computational resources between quantum and classical approaches.
    
    Args:
        n_quantum_params: Number of quantum circuit parameters
        n_time_points: Number of time points used
        n_iterations: Number of training iterations
        training_time: Total training time in seconds
    
    Returns:
        Dictionary of resource comparisons
    """
    # Classical RK4 computation
    # For RK4: 4 function evaluations per step
    classical_evals_per_step = 4
    classical_total_ops = n_time_points * classical_evals_per_step * 3  # 3 ODEs
    
    # Quantum circuit evaluations during training
    # Rough estimate: loss eval + gradient eval per iteration
    quantum_evals_per_iter = n_time_points * 3 + n_quantum_params  # 3 observables
    quantum_total_evals = quantum_evals_per_iter * n_iterations
    
    metrics = {
        'quantum_parameters': n_quantum_params,
        'training_iterations': n_iterations,
        'training_time_minutes': training_time / 60,
        'circuit_evaluations_total': quantum_total_evals,
        'circuit_evaluations_per_iter': quantum_evals_per_iter,
        'classical_function_evals': classical_total_ops,
        'classical_parameters': 0,  # RK4 has no trainable parameters
    }
    
    return metrics


def generate_analysis_report(
    results_dir: Path,
    classical_data_path: Path,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate comprehensive analysis report.
    
    Args:
        results_dir: Directory with training results
        classical_data_path: Path to classical solution
        output_path: Path to save report JSON
    
    Returns:
        Complete analysis dictionary
    """
    from data_utils import load_classical_solution
    from training import load_training_results
    from quantum_circuit import create_lorenz_circuit
    from physics_loss import create_physics_loss
    from scipy.interpolate import interp1d
    
    print("=" * 80)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 80)
    
    # Load training results
    print("Loading training results...")
    best_theta, history = load_training_results(results_dir)
    
    # Load classical solution
    print("Loading classical solution...")
    classical_data = load_classical_solution(classical_data_path.parent)
    t_classical = classical_data['t']
    xyz_classical = np.column_stack([
        classical_data['x'],
        classical_data['y'],
        classical_data['z']
    ])
    
    # Generate quantum solution
    print("Generating quantum solution...")
    t_quantum = np.linspace(t_classical[0], t_classical[-1], 50)
    
    # Infer circuit configuration from parameters
    n_params = len(best_theta)
    # For 3 qubits: n_layers * (3*3 + 2) = n_layers * 11
    # Try different layer counts
    n_layers = None
    for layers in range(1, 10):
        if layers * 11 == n_params:
            n_layers = layers
            break
    if n_layers is None:
        # Default to 3 layers
        n_layers = 3
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=n_layers)
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_quantum,
        initial_condition=classical_data['initial_state']
    )
    
    xyz_quantum = loss_fn.evaluate_circuit_at_times(best_theta, t_quantum)
    
    # Interpolate classical solution to quantum time points
    print("Computing error metrics...")
    xyz_classical_interp = np.zeros_like(xyz_quantum)
    for i in range(3):
        interp_func = interp1d(t_classical, xyz_classical[:, i], kind='cubic')
        xyz_classical_interp[:, i] = interp_func(t_quantum)
    
    # Compute metrics
    error_metrics = compute_error_metrics(xyz_classical_interp, xyz_quantum)
    convergence_metrics = analyze_convergence(history)
    
    training_time = history['time'][-1] if 'time' in history else 0
    resource_metrics = compare_computational_resources(
        n_quantum_params=len(best_theta),
        n_time_points=len(t_quantum),
        n_iterations=len(history['iteration']),
        training_time=training_time
    )
    
    # Compile full report
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_qubits': 3,
            'n_layers': n_layers,
            'n_parameters': len(best_theta),
            'n_time_points': len(t_quantum),
            'time_range': [float(t_quantum[0]), float(t_quantum[-1])],
        },
        'error_metrics': error_metrics,
        'convergence_metrics': convergence_metrics,
        'resource_metrics': resource_metrics,
    }
    
    # Save report
    if output_path is None:
        output_path = results_dir / "analysis_report.json"
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Quantum circuit: {3} qubits, {n_layers} layers, {len(best_theta)} parameters")
    print(f"  Time points: {len(t_quantum)}")
    
    print("\nError Metrics:")
    print(f"  Mean L2 Error: {error_metrics['mean_l2_error']:.4f}")
    print(f"  Max L2 Error: {error_metrics['max_l2_error']:.4f}")
    print(f"  RMSE (x, y, z): ({error_metrics['rmse_x']:.4f}, "
          f"{error_metrics['rmse_y']:.4f}, {error_metrics['rmse_z']:.4f})")
    print(f"  R² (x, y, z): ({error_metrics['r2_x']:.4f}, "
          f"{error_metrics['r2_y']:.4f}, {error_metrics['r2_z']:.4f})")
    
    print("\nConvergence:")
    print(f"  Initial loss: {convergence_metrics['initial_loss']:.2f}")
    print(f"  Final loss: {convergence_metrics['final_loss']:.2f}")
    print(f"  Reduction: {convergence_metrics['loss_reduction_percent']:.2f}%")
    print(f"  Iterations: {convergence_metrics['total_iterations']}")
    print(f"  Training time: {convergence_metrics.get('total_time_minutes', 0):.1f} minutes")
    
    print("\nComputational Resources:")
    print(f"  Quantum parameters: {resource_metrics['quantum_parameters']}")
    print(f"  Training iterations: {resource_metrics['training_iterations']}")
    print(f"  Circuit evaluations: {resource_metrics['circuit_evaluations_total']}")
    print(f"  Training time: {resource_metrics['training_time_minutes']:.1f} minutes")
    
    print("\n" + "=" * 80)
    
    return report

