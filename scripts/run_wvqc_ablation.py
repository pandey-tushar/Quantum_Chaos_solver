#!/usr/bin/env python3
"""
run_wvqc_ablation.py - Windowed Variational Quantum Circuit ablation.

Closes the information-asymmetry concern in the QRC vs QPINN comparison
by giving the variational circuit the same w=5 windowed input that QRC
uses. Same Adam training, same iteration budget, same matched seeds.

The variational circuit is identical to QPINN except for the input
encoding stage: instead of encoding scalar time t, the circuit encodes
the last w state vectors as repeated rotation layers on the first three
qubits (data re-uploading style).

Two scientifically informative outcomes:
  - WVQC closes the gap to QRC  -> windowed input is the load-bearing
    factor and the paper's framing shifts.
  - WVQC stays in QPINN's regime -> the fixed-vs-trained reservoir
    choice is load-bearing even with information-symmetric input,
    strengthening the architectural argument.

Estimated runtime: ~2.4 h per seed x 5 seeds = ~12 h total. Run
overnight. Output is a JSON summary with per-seed train/test MSE
plus the per-iteration loss curves.

Outputs:
    results/wvqc_ablation/wvqc_summary.json
    results/wvqc_ablation/wvqc_loss_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Lorenz reference dynamics (RK4)
# ---------------------------------------------------------------------------

def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = state
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])


def integrate(rhs, x0, t_end, dt=0.01):
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# WVQC circuit
# ---------------------------------------------------------------------------
#
# Architecture: 4 qubits, 3 variational layers, 45 trainable parameters
# (matches QPINN). Difference from QPINN is the encoding stage.
#
# Encoding stage:
#   For each of w=5 windowed past states x(t-w+1), ..., x(t):
#     apply RY(phi_x), RZ(phi_y), RX(phi_z) on q0, q1, q2
#     and RY(0.5*(phi_x + phi_y)) on q3
#   Angles are normalised to [0, 2*pi] using fixed training-set bounds
#   (per-component global, matches the QRC encoding contract).
#
# Variational stage (same as QPINN):
#   3 layers x [RX(theta), RY(theta), RZ(theta) per qubit
#                + CNOT chain + RZ(theta) phases]
#   = 15 params per layer x 3 layers = 45 trainable parameters
#
# Output: <Z_0>, <Z_1>, <Z_2> linearly mapped to physical ranges
# (x in [-20, 20], y in [-30, 30], z in [0, 50]).
#
# Target: x(t + Delta_t), the next state.

WINDOW = 5
N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = 45


def angle_encode(state, lb, ub):
    """Per-component global normalisation onto [0, 2*pi]."""
    return (state - lb) / (ub - lb + 1e-8) * 2.0 * np.pi


def build_wvqc_circuit(window_states, theta, lb, ub, n_qubits=N_QUBITS):
    """Construct the WVQC circuit for one input window.

    window_states : ndarray, shape (w, 3)
    theta         : ndarray, shape (45,) trainable parameters
    lb, ub        : ndarray, shape (3,) per-component bounds
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    # ---- Encoding stage: data re-uploading over the window ----
    for w_step in range(window_states.shape[0]):
        phi = angle_encode(window_states[w_step], lb, ub)
        qc.ry(float(phi[0]), 0)
        qc.rz(float(phi[1]), 1)
        qc.rx(float(phi[2]), 2)
        if n_qubits >= 4:
            qc.ry(float(0.5 * (phi[0] + phi[1])), 3)

    # ---- Variational stage: 3 layers of RxRyRz + CNOT chain + RZ phases ----
    p = 0
    for _ in range(N_LAYERS):
        for q in range(n_qubits):
            qc.rx(float(theta[p + 0]), q)
            qc.ry(float(theta[p + 1]), q)
            qc.rz(float(theta[p + 2]), q)
            p += 3
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(float(theta[p]), q + 1)
            p += 1
    assert p == N_PARAMS, f"parameter count mismatch: {p} != {N_PARAMS}"
    return qc


def expectation_z(qc, q):
    """<Z_q> from a circuit's statevector."""
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    n = qc.num_qubits
    probs = np.abs(sv.data) ** 2
    expectation = 0.0
    for s in range(2 ** n):
        bit = (s >> q) & 1
        sign = 1.0 - 2.0 * bit
        expectation += sign * probs[s]
    return float(expectation)


def predict_state(window_states, theta, lb, ub,
                   x_range=(-20.0, 20.0),
                   y_range=(-30.0, 30.0),
                   z_range=(0.0, 50.0)):
    """Map the circuit output to a predicted state."""
    qc = build_wvqc_circuit(window_states, theta, lb, ub)
    z0 = expectation_z(qc, 0)
    z1 = expectation_z(qc, 1)
    z2 = expectation_z(qc, 2)
    x = (z0 + 1.0) / 2.0 * (x_range[1] - x_range[0]) + x_range[0]
    y = (z1 + 1.0) / 2.0 * (y_range[1] - y_range[0]) + y_range[0]
    z = (z2 + 1.0) / 2.0 * (z_range[1] - z_range[0]) + z_range[0]
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Training (Adam with finite-difference gradient)
# ---------------------------------------------------------------------------

def loss_on_window_pairs(theta, window_inputs, targets, lb, ub):
    """Mean squared error of predictions against targets."""
    preds = np.array([predict_state(w, theta, lb, ub) for w in window_inputs])
    return float(np.mean((preds - targets) ** 2))


def finite_diff_gradient(theta, window_inputs, targets, lb, ub, eps=1e-4):
    """Forward finite differences over all 45 parameters."""
    base = loss_on_window_pairs(theta, window_inputs, targets, lb, ub)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_p = theta.copy()
        theta_p[i] += eps
        grad[i] = (loss_on_window_pairs(theta_p, window_inputs, targets, lb, ub)
                    - base) / eps
    return base, grad


def adam_step(theta, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


def train_wvqc(window_inputs, targets, lb, ub, n_iters=200, lr=1e-3,
                lr_decay=0.995, seed=0, log_every=10):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, N_PARAMS)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    losses = []
    for it in range(1, n_iters + 1):
        loss, grad = finite_diff_gradient(theta, window_inputs, targets, lb, ub)
        # Adaptive gradient clipping at the 99th percentile (matches src/training.py)
        cap = np.percentile(np.abs(grad), 99) * 5.0
        grad = np.clip(grad, -cap, cap)
        cur_lr = lr * (lr_decay ** it)
        theta, m, v = adam_step(theta, grad, m, v, it, lr=cur_lr)
        losses.append(loss)
        if it % log_every == 0 or it == 1:
            print(f"  [seed {seed}] iter {it:>3d}  loss={loss:.3e}  "
                   f"|grad|={np.linalg.norm(grad):.3e}  lr={cur_lr:.2e}")
    return theta, losses


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, n_iters: int = 200, dt: float = 0.01,
              t_train_end: float = 3.0, t_test_end: float = 4.0,
              n_train: int = 50, n_test: int = 20):
    x0 = np.array([1.0, 1.0, 1.0])
    full_traj = integrate(lorenz_rhs, x0, t_test_end, dt=dt)
    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    test_idx  = np.linspace(int(t_train_end / dt),
                              int(t_test_end / dt) - 1, n_test, dtype=int)
    train_states = full_traj[train_idx]
    test_states  = full_traj[test_idx]

    # Per-component bounds from training (matches QRC global encoding)
    lb = train_states.min(axis=0)
    ub = train_states.max(axis=0)

    # Build (window, target) pairs from training trajectory
    win_in_train = []
    target_train = []
    for i in range(WINDOW, len(train_states)):
        win_in_train.append(train_states[i - WINDOW:i])
        target_train.append(train_states[i])
    win_in_train = np.array(win_in_train)
    target_train = np.array(target_train)

    print(f"\n=== seed={seed}  n_train_pairs={len(win_in_train)} ===")
    t0 = time.time()
    theta, losses = train_wvqc(win_in_train, target_train, lb, ub,
                                  n_iters=n_iters, seed=seed)
    train_time = time.time() - t0

    # Evaluate train MSE at the final theta
    train_mse = loss_on_window_pairs(theta, win_in_train, target_train, lb, ub)

    # Evaluate test MSE: teacher-forced (drive encoder with ground truth)
    win_in_test = []
    target_test = []
    full = np.concatenate([train_states, test_states])
    for i in range(len(train_states), len(full)):
        win_in_test.append(full[i - WINDOW:i])
        target_test.append(full[i])
    win_in_test = np.array(win_in_test)
    target_test = np.array(target_test)
    test_mse = loss_on_window_pairs(theta, win_in_test, target_test, lb, ub)

    return {
        "seed": seed,
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_time_s": train_time,
        "loss_curve": losses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "wvqc_ablation"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "wvqc_summary.json"

    results = []
    for seed in range(args.n_seeds):
        res = run_seed(seed, n_iters=args.n_iters)
        results.append(res)
        # Incremental save
        with open(out_json, "w") as f:
            json.dump({
                "n_seeds": args.n_seeds,
                "n_iters": args.n_iters,
                "results": results,
            }, f, indent=2)
        print(f"  -> seed {seed}: train={res['train_mse']:.3e}  "
               f"test={res['test_mse']:.3e}  ({res['train_time_s']:.1f}s)")

    train_v = np.array([r["train_mse"] for r in results])
    test_v  = np.array([r["test_mse"]  for r in results])
    print(f"\n=== summary ({args.n_seeds} seeds) ===")
    print(f"  WVQC train MSE: {train_v.mean():.3e} +/- {train_v.std():.3e}")
    print(f"  WVQC test  MSE: {test_v.mean():.3e} +/- {test_v.std():.3e}")
    print(f"\n  For reference (from arXiv 2604.23743):")
    print(f"    QPINN train MSE: 91.3 +/- 21.9")
    print(f"    QRC   train MSE: 17.1 +/- 3.7")
    print(f"\n  Saved {out_json}")

    # ----- Plot loss curves -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            ax.plot(r["loss_curve"], lw=1.2, alpha=0.7,
                     label=f"seed {r['seed']}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Train MSE")
        ax.set_yscale("log")
        ax.set_title(f"WVQC training loss across {args.n_seeds} seeds")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "wvqc_loss_curves.png", dpi=150)
        print(f"  Saved {out_dir / 'wvqc_loss_curves.png'}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
