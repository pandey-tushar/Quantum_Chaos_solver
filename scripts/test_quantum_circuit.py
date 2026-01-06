#!/usr/bin/env python3
"""
Test the quantum circuit architecture for the Lorenz system solver.
This script verifies that the circuit can be created, parameterized, and evaluated.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_circuit import create_lorenz_circuit


def test_circuit_creation():
    """Test that we can create the circuit."""
    print("TEST 1: Circuit creation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    qc = circuit_obj.get_circuit()
    
    print(f"  Number of qubits: {qc.num_qubits}")
    print(f"  Number of parameters: {len(qc.parameters)}")
    print(f"  Expected parameters: {circuit_obj.get_num_parameters() + 1}")  # +1 for time
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Number of gates: {len(qc.data)}")
    
    # Check parameter names
    param_names = [p.name for p in qc.parameters]
    has_t = any('t' in name for name in param_names)
    has_theta = any('θ' in name for name in param_names)
    
    print(f"  Has time parameter 't': {has_t}")
    print(f"  Has variational parameters 'θ': {has_theta}")
    
    if qc.num_qubits == 3 and has_t and has_theta:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_parameter_binding():
    """Test that we can bind parameters to the circuit."""
    print("\nTEST 2: Parameter binding")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    
    # Initialize parameters
    theta_values = circuit_obj.initialize_parameters(seed=42)
    t_value = 1.5
    
    print(f"  Time value: {t_value}")
    print(f"  Number of variational parameters: {len(theta_values)}")
    print(f"  Parameter range: [{theta_values.min():.2f}, {theta_values.max():.2f}]")
    
    # Bind parameters
    try:
        bound_circuit = circuit_obj.assign_parameters(t_value, theta_values)
        print(f"  Bound circuit parameters: {len(bound_circuit.parameters)}")
        
        if len(bound_circuit.parameters) == 0:
            print("  ✓ PASS (all parameters bound)")
            return True
        else:
            print(f"  ✗ FAIL (still has {len(bound_circuit.parameters)} unbound parameters)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL (error: {e})")
        return False


def test_observables():
    """Test that observables are created correctly."""
    print("\nTEST 3: Observable creation")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    observables = circuit_obj.get_observables()
    
    print(f"  Number of observables: {len(observables)}")
    print(f"  Expected: 3 (one per variable x, y, z)")
    
    # Check each observable
    for i, obs in enumerate(observables):
        var_name = ['x', 'y', 'z'][i]
        print(f"  Observable for {var_name}: {obs}")
        print(f"    Number of qubits: {obs.num_qubits}")
    
    if len(observables) == 3:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_expectation_mapping():
    """Test mapping expectation values to physical ranges."""
    print("\nTEST 4: Expectation value mapping")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    
    # Test mapping for each variable
    test_cases = [
        ('x', -1.0, -20.0),  # Min expectation -> min range
        ('x', 1.0, 20.0),    # Max expectation -> max range
        ('x', 0.0, 0.0),     # Middle expectation -> middle range
        ('y', -1.0, -30.0),
        ('y', 1.0, 30.0),
        ('z', -1.0, 0.0),
        ('z', 1.0, 50.0),
    ]
    
    all_pass = True
    for var, exp_val, expected in test_cases:
        mapped = circuit_obj.map_expectation_to_value(exp_val, var)
        print(f"  {var}: exp={exp_val:5.1f} -> value={mapped:6.1f} (expected {expected:6.1f})")
        if not np.isclose(mapped, expected, atol=0.01):
            print(f"    ✗ Mismatch!")
            all_pass = False
    
    if all_pass:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


def test_forward_pass_with_estimator():
    """Test forward pass using Qiskit Aer Estimator."""
    print("\nTEST 5: Forward pass with Estimator (simulation)")
    print("-" * 60)
    
    try:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    except ImportError:
        print("  SKIP (qiskit_aer not available)")
        return None
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    
    # Initialize parameters
    theta_values = circuit_obj.initialize_parameters(seed=42)
    t_value = 1.0
    
    # Bind parameters
    bound_circuit = circuit_obj.assign_parameters(t_value, theta_values)
    
    # Get observables
    observables = circuit_obj.get_observables()
    
    # Create estimator
    estimator = AerEstimator()
    
    # Run estimator for each observable
    print(f"  Running circuit at t={t_value}")
    expectation_values = []
    
    for i, obs in enumerate(observables):
        var_name = ['x', 'y', 'z'][i]
        
        # Run estimator
        job = estimator.run([(bound_circuit, obs)])
        result = job.result()
        exp_val = result[0].data.evs
        
        # Map to physical range
        physical_val = circuit_obj.map_expectation_to_value(exp_val, var_name)
        
        print(f"    {var_name}(t={t_value}): expectation={exp_val:.4f}, value={physical_val:.4f}")
        expectation_values.append(physical_val)
    
    # Check that we got reasonable values
    x, y, z = expectation_values
    x_in_range = circuit_obj.x_range[0] <= x <= circuit_obj.x_range[1]
    y_in_range = circuit_obj.y_range[0] <= y <= circuit_obj.y_range[1]
    z_in_range = circuit_obj.z_range[0] <= z <= circuit_obj.z_range[1]
    
    print(f"  x in range {circuit_obj.x_range}: {x_in_range}")
    print(f"  y in range {circuit_obj.y_range}: {y_in_range}")
    print(f"  z in range {circuit_obj.z_range}: {z_in_range}")
    
    if x_in_range and y_in_range and z_in_range:
        print("  ✓ PASS (forward pass works, values in expected ranges)")
        return True
    else:
        print("  ✗ FAIL (values outside expected ranges)")
        return False


def test_circuit_visualization():
    """Test circuit visualization (optional)."""
    print("\nTEST 6: Circuit visualization")
    print("-" * 60)
    
    circuit_obj = create_lorenz_circuit(n_qubits=3, n_layers=2)
    qc = circuit_obj.get_circuit()
    
    try:
        # Try to draw the circuit
        circuit_text = qc.draw(output='text', fold=-1)
        print("  Circuit diagram:")
        print()
        # Print first few lines
        lines = str(circuit_text).split('\n')
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print(f"    ... ({len(lines) - 10} more lines)")
        print()
        print("  ✓ PASS (circuit can be visualized)")
        return True
    except Exception as e:
        print(f"  ⚠ WARNING (visualization failed: {e})")
        return None


def main():
    """Run all tests."""
    print("=" * 70)
    print("QUANTUM CIRCUIT ARCHITECTURE TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Circuit Creation", test_circuit_creation),
        ("Parameter Binding", test_parameter_binding),
        ("Observable Creation", test_observables),
        ("Expectation Mapping", test_expectation_mapping),
        ("Forward Pass", test_forward_pass_with_estimator),
        ("Circuit Visualization", test_circuit_visualization),
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
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
            passed += 1
        elif result is False:
            status = "✗ FAIL"
            failed += 1
        else:
            status = "⊘ SKIP"
            skipped += 1
        print(f"  {status:8s} {name}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print()
    
    if failed == 0 and passed > 0:
        print("=" * 70)
        print("✓ ALL TESTS PASSED - Quantum circuit is ready")
        print("=" * 70)
        print()
        print("Next: Phase 4 - Physics-Informed Loss Function")
        return 0
    elif failed > 0:
        print("=" * 70)
        print("✗ SOME TESTS FAILED - Circuit needs fixes")
        print("=" * 70)
        return 1
    else:
        print("=" * 70)
        print("⚠ NO TESTS RUN - Cannot verify")
        print("=" * 70)
        return 2


if __name__ == "__main__":
    sys.exit(main())

