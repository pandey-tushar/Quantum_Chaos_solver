#!/usr/bin/env python3
"""
Test the physics-informed loss function.
Verifies that loss components can be computed and gradients are accessible.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_circuit import create_lorenz_circuit
from physics_loss import create_physics_loss


def test_loss_creation():
    """Test that we can create the loss function."""
    print("TEST 1: Loss function creation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    t_points = np.linspace(0, 2, 5)  # Small number for testing
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition
    )
    
    print(f"  Number of time points: {len(t_points)}")
    print(f"  Initial condition: {initial_condition}")
    print(f"  Lorenz parameters: σ={loss_fn.sigma}, ρ={loss_fn.rho}, β={loss_fn.beta:.3f}")
    print(f"  Boundary weight λ: {loss_fn.lambda_boundary}")
    
    if loss_fn is not None:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_circuit_evaluation():
    """Test evaluating circuit at multiple time points."""
    print("\nTEST 2: Circuit evaluation at time points")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    t_points = np.array([0.0, 0.5, 1.0])
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition
    )
    
    theta = circuit_obj.initialize_parameters(seed=42)
    
    print(f"  Evaluating at {len(t_points)} time points...")
    states = loss_fn.evaluate_circuit_at_times(theta, t_points)
    
    print(f"  States shape: {states.shape}")
    print(f"  Expected shape: ({len(t_points)}, 3)")
    
    for i, t in enumerate(t_points):
        x, y, z = states[i]
        print(f"    t={t:.1f}: x={x:7.3f}, y={y:7.3f}, z={z:7.3f}")
    
    if states.shape == (len(t_points), 3):
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_boundary_loss():
    """Test boundary loss computation."""
    print("\nTEST 3: Boundary loss computation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    t_points = np.array([0.0, 0.5, 1.0])
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition
    )
    
    theta = circuit_obj.initialize_parameters(seed=42)
    
    print(f"  Computing boundary loss...")
    L_boundary = loss_fn.compute_boundary_loss(theta)
    
    print(f"  L_boundary = {L_boundary:.6f}")
    print(f"  Initial condition: {initial_condition}")
    
    # Get state at t=0
    state_t0 = loss_fn.evaluate_circuit_at_times(theta, np.array([t_points[0]]))[0]
    print(f"  Circuit at t=0: [{state_t0[0]:.3f}, {state_t0[1]:.3f}, {state_t0[2]:.3f}]")
    
    # Boundary loss should be non-negative
    if L_boundary >= 0 and np.isfinite(L_boundary):
        print("  ✓ PASS (boundary loss is finite and non-negative)")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_differential_loss():
    """Test differential loss computation."""
    print("\nTEST 4: Differential loss computation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    t_points = np.linspace(0, 1, 4)  # Few points for speed
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition
    )
    
    theta = circuit_obj.initialize_parameters(seed=42)
    
    print(f"  Computing differential loss at {len(t_points)} points...")
    print(f"  (This computes derivatives via finite differences)")
    
    try:
        L_diff = loss_fn.compute_differential_loss(theta)
        
        print(f"  L_diff = {L_diff:.6f}")
        
        # Differential loss should be non-negative
        if L_diff >= 0 and np.isfinite(L_diff):
            print("  ✓ PASS (differential loss is finite and non-negative)")
            return True
        else:
            print("  ✗ FAIL (loss is negative or not finite)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL (error: {e})")
        return False


def test_total_loss():
    """Test total loss computation."""
    print("\nTEST 5: Total loss computation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    t_points = np.linspace(0, 1, 4)
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition,
        lambda_boundary=10.0
    )
    
    theta = circuit_obj.initialize_parameters(seed=42)
    
    print(f"  Computing total loss...")
    
    try:
        total, L_diff, L_boundary = loss_fn.compute_total_loss(theta, return_components=True)
        
        print(f"  L_diff      = {L_diff:.6f}")
        print(f"  L_boundary  = {L_boundary:.6f}")
        print(f"  λ           = {loss_fn.lambda_boundary}")
        print(f"  Total loss  = {total:.6f}")
        print(f"  Expected    = {L_diff + loss_fn.lambda_boundary * L_boundary:.6f}")
        
        # Check formula
        expected_total = L_diff + loss_fn.lambda_boundary * L_boundary
        matches = np.isclose(total, expected_total)
        
        if matches and np.isfinite(total):
            print("  ✓ PASS (total loss computed correctly)")
            return True
        else:
            print("  ✗ FAIL (total loss doesn't match expected formula)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL (error: {e})")
        return False


def test_gradient_computation():
    """Test gradient computation."""
    print("\nTEST 6: Gradient computation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=1)  # Small for speed
    t_points = np.array([0.0, 0.5, 1.0])
    initial_condition = np.array([1.0, 1.0, 1.0])
    
    loss_fn = create_physics_loss(
        circuit_obj=circuit_obj,
        t_points=t_points,
        initial_condition=initial_condition
    )
    
    theta = circuit_obj.initialize_parameters(seed=42)
    n_params = len(theta)
    
    print(f"  Number of parameters: {n_params}")
    print(f"  Computing gradient (this may take a moment)...")
    
    try:
        grad = loss_fn.compute_gradient(theta, epsilon=1e-3)
        
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient norm: {np.linalg.norm(grad):.6f}")
        print(f"  Gradient range: [{grad.min():.6f}, {grad.max():.6f}]")
        
        # Check a few gradient values
        print(f"  Sample gradients: {grad[:3]}")
        
        if grad.shape == theta.shape and np.all(np.isfinite(grad)):
            print("  ✓ PASS (gradient computed, all values finite)")
            return True
        else:
            print("  ✗ FAIL (gradient has wrong shape or non-finite values)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL (error: {e})")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("PHYSICS-INFORMED LOSS FUNCTION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Loss Creation", test_loss_creation),
        ("Circuit Evaluation", test_circuit_evaluation),
        ("Boundary Loss", test_boundary_loss),
        ("Differential Loss", test_differential_loss),
        ("Total Loss", test_total_loss),
        ("Gradient Computation", test_gradient_computation),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"  {status:8s} {name}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print()
    
    if failed == 0 and passed > 0:
        print("=" * 70)
        print("✓ ALL TESTS PASSED - Physics-informed loss is ready")
        print("=" * 70)
        print()
        print("Next: Phase 5 - Training & Optimization")
        return 0
    elif failed > 0:
        print("=" * 70)
        print("✗ SOME TESTS FAILED - Loss function needs fixes")
        print("=" * 70)
        return 1
    else:
        print("=" * 70)
        print("⚠ NO TESTS RUN")
        print("=" * 70)
        return 2


if __name__ == "__main__":
    sys.exit(main())

