# Quantum Chaos Solver: Lorenz Attractor

Solving the Lorenz system of differential equations using a **Differentiable Quantum Circuit (DQC)** with physics-informed loss function. This project demonstrates how quantum circuits can learn to approximate solutions to chaotic differential equations.

## Overview

The Lorenz system is a set of three coupled nonlinear differential equations that exhibit chaotic behavior:
- dx/dt = σ(y - x)
- dy/dt = x(ρ - z) - y
- dz/dt = xy - βz

This implementation uses a variational quantum circuit trained with a physics-informed loss function that enforces:
1. The differential equations at collocation points
2. Initial boundary conditions

**Key Results:**
- 99.86% loss reduction after 200 training iterations
- Mean L2 error: 18.1 between quantum and classical solutions
- 33 trainable quantum parameters (3 qubits, 3 layers)

## Project Structure

```
quantum_chaos_solver/
├── src/                          # Source code modules
│   ├── lorenz_system.py         # Classical RK4 solver
│   ├── quantum_circuit.py       # Quantum circuit architecture
│   ├── physics_loss.py          # Physics-informed loss function
│   ├── training.py              # Training loop and optimizer
│   ├── visualization.py         # Plotting functions
│   ├── analysis.py              # Error metrics and analysis
│   └── data_utils.py            # Data loading utilities
├── scripts/                      # Executable scripts
│   ├── generate_classical_solution.py
│   ├── train_lorenz.py
│   ├── visualize_results.py
│   ├── analyze_results.py
│   └── verify_*.py              # Verification scripts
├── data/                        # Generated data
├── results/                     # Training results and plots
└── requirements.txt             # Dependencies

```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd quantum_chaos_solver
pip install -r requirements.txt
```

3. Verify installation:
```bash
python scripts/verify_setup.py
```

## Usage

### 1. Generate Classical Reference Solution

```bash
python scripts/generate_classical_solution.py
```

This creates:
- `data/lorenz_classical_solution.npz` — Reference solution using RK4
- `results/lorenz_classical_validation.png` — Validation plots

### 2. Train Quantum Circuit

Basic training (50 iterations, fast):
```bash
python scripts/train_lorenz.py --n-iterations 50
```

Extended training (200 iterations, better accuracy):
```bash
python scripts/train_lorenz.py \
    --n-iterations 200 \
    --n-layers 3 \
    --learning-rate 0.02 \
    --n-time-points 10 \
    --results-dir results/my_training
```

**Options:**
- `--n-qubits`: Number of qubits (default: 3)
- `--n-layers`: Circuit depth (default: 2, recommended: 3)
- `--n-time-points`: Training time points (default: 10)
- `--t-max`: Maximum time to simulate (default: 2.0)
- `--n-iterations`: Training iterations (default: 50)
- `--learning-rate`: Adam learning rate (default: 0.01)
- `--lambda-boundary`: Boundary condition weight (default: 10.0)

### 3. Visualize Results

```bash
python scripts/visualize_results.py --results-dir results/my_training
```

Creates 5 plots in `results/my_training/plots/`:
- `3d_attractor.png` — 3D Lorenz butterfly attractor comparison
- `time_series.png` — x(t), y(t), z(t) vs time
- `phase_space.png` — Phase space projections
- `training_metrics.png` — Loss curves and convergence
- `error_analysis.png` — Error statistics

### 4. Analyze Results

```bash
python scripts/analyze_results.py --results-dir results/my_training
```

Generates `analysis_report.json` with:
- Error metrics (MAE, MSE, RMSE, R²)
- Convergence statistics
- Computational resource usage

## Implementation Details

### Quantum Circuit Architecture

**Time Encoding Layer:**
- Encodes time `t` using RY rotations: `RY(t·(i+1)/n_qubits)` on qubit `i`

**Variational Ansatz:**
- Parameterized layers: `[RX(θ), RY(θ), RZ(θ)]` on each qubit
- Entanglement: CNOT ladder with parameterized RZ gates
- Total parameters: `n_layers × (3×n_qubits + n_qubits-1)`

**Measurement:**
- Expectation values of Pauli-Z on each qubit
- Maps `<Z>` ∈ [-1,1] to physical ranges: x∈[-20,20], y∈[-30,30], z∈[0,50]

### Physics-Informed Loss

**L_diff (Differential Loss):**
```
L_diff = mean((∂x/∂t - σ(y-x))² + (∂y/∂t - x(ρ-z)+y)² + (∂z/∂t - xy+βz)²)
```
Time derivatives computed via `np.gradient` on circuit outputs.

**L_boundary (Boundary Loss):**
```
L_boundary = mean((x(0) - x₀)² + (y(0) - y₀)² + (z(0) - z₀)²)
```

**Total Loss:**
```
L_total = L_diff + λ × L_boundary  (λ = 10.0)
```

### Training

- **Optimizer:** Adam (learning_rate=0.02, β₁=0.9, β₂=0.999)
- **Gradient:** Finite differences (ε=1e-4)
- **Checkpointing:** Saves every N iterations
- **Monitoring:** Tracks loss components, gradient norm, training time

## Results

### Best Training Run (200 iterations, 3 layers)

**Configuration:**
- Circuit: 3 qubits, 3 layers, 33 parameters
- Time points: 10 (collocation), evaluated at 50 (testing)
- Training time: ~60 minutes

**Convergence:**
- Initial loss: 6968.0
- Final loss: 10.0
- **Reduction: 99.86%**

**Error Metrics:**
- Mean L2 error: 18.1
- RMSE: x=8.26, y=9.73, z=15.68
- Max error: x=19.2, y=25.1, z=38.9

### Optimizations

**Training Speed:**
- Optimized `compute_time_derivatives` to use `np.gradient` instead of repeated circuit calls
- **1.7× speedup** (reduced circuit evaluations by 41%)

## Verification

All phases include comprehensive verification:

**Phase 2 - Classical Solver:**
```bash
python scripts/verify_lorenz_solver.py
```
✓ 7 tests: RK4 accuracy, convergence, Lorenz equations, scipy comparison

**Phase 3 - Quantum Circuit:**
```bash
python scripts/test_quantum_circuit.py
```
✓ 6 tests: Circuit creation, parameter binding, observables, forward pass

**Phase 4 - Physics Loss:**
```bash
python scripts/test_physics_loss.py
```
✓ 6 tests: Loss computation, boundary/differential losses, gradients

## Dependencies

- qiskit[visualization]==2.2.2
- qiskit-aer==0.17.2
- numpy>=1.24.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- plotly>=5.14.0

## Implementation Status

- ✅ Phase 1: Project Setup & Environment Configuration
- ✅ Phase 2: Classical Baseline Implementation
- ✅ Phase 3: Quantum Circuit Architecture Design
- ✅ Phase 4: Physics-Informed Loss Function
- ✅ Phase 5: Training & Optimization
- ✅ Phase 6: Visualization Suite
- ✅ Phase 7: Analysis & Comparison
- ✅ Phase 8: Documentation & Polish

## References

- Lorenz, E. N. (1963). "Deterministic Nonperiodic Flow"
- Physics-Informed Neural Networks: Raissi et al. (2019)
- Variational Quantum Algorithms: McClean et al. (2016)

## License

See parent directory LICENSE file.

## Citation

If you use this code, please cite:
```
Quantum Chaos Solver: Physics-Informed Variational Quantum Circuit
for Solving the Lorenz System (2025)
```

