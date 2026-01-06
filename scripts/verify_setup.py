#!/usr/bin/env python3
"""
Verify that the environment is set up correctly for the quantum chaos solver.
Checks that all required dependencies are installed and accessible.
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name} is NOT installed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Quantum Chaos Solver - Environment Verification")
    print("=" * 60)
    print()
    
    print("Python version:", sys.version)
    print()
    
    print("Checking dependencies...")
    print("-" * 60)
    
    checks = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("qiskit", "Qiskit"),
        ("qiskit_aer", "Qiskit Aer"),
    ]
    
    results = []
    for module, name in checks:
        results.append(check_import(module, name))
    
    print()
    print("-" * 60)
    
    if all(results):
        print("✓ All dependencies are installed correctly!")
        print()
        print("Environment is ready for Phase 2: Classical Baseline Implementation")
        return 0
    else:
        print("✗ Some dependencies are missing.")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

