#!/usr/bin/env python3
"""
run_qpinn_gradient_logging.py - Re-run QPINN training with per-layer
gradient-norm logging for the capacity-vs-barren-plateau diagnosis
figure (paper/qst_submission/main.tex, fig5).

The original QPINN training only logs scalar ||grad|| via the adaptive
clipping logic. For the diagnosis figure we need per-layer ||grad_l||
at each iteration to plot against the McClean barren-plateau threshold.

Per-layer grouping (45 params total, 4 qubits, 3 variational layers):
  Layer 1 : params  0..14  (R_X, R_Y, R_Z on 4 qubits + 3 R_Z phases)
  Layer 2 : params 15..29
  Layer 3 : params 30..44

McClean threshold for our ansatz class (n=4 qubits, L=3 layers):
  Var[d_theta_i C] ~ 2^(-2 max(d, L)) ~ 2^-8 ~ 4e-3
  typical |d_theta_i C| ~ sqrt(4e-3) ~ 0.06
  typical ||grad|| over 45 params ~ sqrt(45) * 0.06 ~ 0.4

Measured ||grad|| from prior QPINN runs: ~10^3 to 10^4. Plot will
show measured per-layer norms (filled markers, log scale) with the
McClean threshold (dashed horizontal line at ||grad|| ~ 0.4) far below.

Estimated runtime: ~2.4 h per seed x 3 seeds = ~7.2 h total. The
finite-difference gradient computation already evaluates each
parameter individually, so per-layer logging adds no compute overhead
beyond bookkeeping.

Outputs:
    results/qpinn_gradients/per_layer_norms.json
    results/qpinn_gradients/per_layer_gradient_norms.png  (fig5)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Lorenz reference dynamics
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
        k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# QPINN circuit (matches src/quantum_circuit.py LorenzQuantumCircuit pattern)
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_LAYERS = 3
N_PARAMS_PER_LAYER = 15        # 12 single-qubit + 3 RZ phases
N_PARAMS = N_LAYERS * N_PARAMS_PER_LAYER

# Parameter index ranges per layer for grouping ||grad||
LAYER_RANGES = [
    (0,  N_PARAMS_PER_LAYER),
    (N_PARAMS_PER_LAYER,     2 * N_PARAMS_PER_LAYER),
    (2 * N_PARAMS_PER_LAYER, 3 * N_PARAMS_PER_LAYER),
]


def build_qpinn_circuit(t, theta, t_max, n_qubits=N_QUBITS):
    """Build the QPINN circuit at time t with parameters theta."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    # Time encoding (fixed)
    theta_t = 2.0 * np.pi * t / t_max
    qc.ry(float(theta_t),     0)
    qc.rz(float(theta_t),     1)
    qc.rx(float(theta_t),     2)
    qc.ry(float(np.pi * t / t_max), 3)

    # Variational layers
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
    assert p == N_PARAMS
    return qc


def expectation_z(qc, q):
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


def predict_state(t, theta, t_max,
                   x_range=(-20.0, 20.0),
                   y_range=(-30.0, 30.0),
                   z_range=(0.0, 50.0)):
    qc = build_qpinn_circuit(t, theta, t_max)
    z0 = expectation_z(qc, 0); z1 = expectation_z(qc, 1); z2 = expectation_z(qc, 2)
    x = (z0 + 1.0) / 2.0 * (x_range[1] - x_range[0]) + x_range[0]
    y = (z1 + 1.0) / 2.0 * (y_range[1] - y_range[0]) + y_range[0]
    z = (z2 + 1.0) / 2.0 * (z_range[1] - z_range[0]) + z_range[0]
    return np.array([x, y, z])


def time_derivative(t, theta, t_max, eps=1e-3):
    return (predict_state(t + eps, theta, t_max)
             - predict_state(t - eps, theta, t_max)) / (2.0 * eps)


def physics_loss(theta, t_points, x0, t_max, lam=10.0):
    """L_phys + lam * L_bc."""
    L_phys = 0.0
    for t in t_points:
        u = predict_state(t, theta, t_max)
        u_dot = time_derivative(t, theta, t_max)
        F_u = lorenz_rhs(u)
        L_phys += float(np.sum((u_dot - F_u) ** 2))
    L_phys /= len(t_points)
    u0 = predict_state(0.0, theta, t_max)
    L_bc = float(np.sum((u0 - x0) ** 2))
    return L_phys + lam * L_bc


def finite_diff_per_parameter(theta, t_points, x0, t_max, eps=1e-4):
    """Forward finite diff over all 45 params; returns base loss + per-param grad."""
    base = physics_loss(theta, t_points, x0, t_max)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_p = theta.copy()
        theta_p[i] += eps
        grad[i] = (physics_loss(theta_p, t_points, x0, t_max) - base) / eps
    return base, grad


def adam_step(theta, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


def per_layer_norms(grad):
    """||grad_l||_2 for l in {1, 2, 3}."""
    return [float(np.linalg.norm(grad[lo:hi])) for lo, hi in LAYER_RANGES]


# ---------------------------------------------------------------------------
# Single-seed run with per-layer gradient logging
# ---------------------------------------------------------------------------

def run_seed(seed: int, n_iters: int = 200, t_train_end: float = 3.0,
              n_train: int = 50, dt: float = 0.01):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, N_PARAMS)
    m = np.zeros_like(theta); v = np.zeros_like(theta)

    x0 = np.array([1.0, 1.0, 1.0])
    t_max = t_train_end
    t_points = np.linspace(0, t_train_end, n_train)

    log = {"loss": [], "grad_norm_total": [],
            "grad_norm_layer_1": [], "grad_norm_layer_2": [],
            "grad_norm_layer_3": []}

    print(f"\n=== seed={seed}  n_iters={n_iters} ===")
    t0 = time.time()
    for it in range(1, n_iters + 1):
        loss, grad = finite_diff_per_parameter(theta, t_points, x0, t_max)
        # adaptive clipping at the 99th percentile (matches src/training.py)
        cap = np.percentile(np.abs(grad), 99) * 5.0
        grad_clipped = np.clip(grad, -cap, cap)
        theta, m, v = adam_step(theta, grad_clipped, m, v, it,
                                  lr=1e-3 * (0.995 ** it))
        # log per-layer norms (BEFORE clipping, to expose true gradient regime)
        norms = per_layer_norms(grad)
        log["loss"].append(loss)
        log["grad_norm_total"].append(float(np.linalg.norm(grad)))
        log["grad_norm_layer_1"].append(norms[0])
        log["grad_norm_layer_2"].append(norms[1])
        log["grad_norm_layer_3"].append(norms[2])
        if it % 10 == 0 or it == 1:
            print(f"  iter {it:>3d}  loss={loss:.3e}  "
                   f"|grad|={np.linalg.norm(grad):.3e}  "
                   f"L1={norms[0]:.3e}  L2={norms[1]:.3e}  L3={norms[2]:.3e}")

    elapsed = time.time() - t0
    print(f"  done in {elapsed/60.0:.1f} min, final loss={loss:.3e}")
    return {"seed": seed, "wall_time_s": elapsed, "log": log}


# ---------------------------------------------------------------------------
# McClean barren-plateau threshold for our ansatz class
# ---------------------------------------------------------------------------
#
# For an L-layer hardware-efficient ansatz on n qubits, McClean et al.
# 2018 give Var[d_theta_i C] ~ 2^(-2 max(d, L)) where d is the
# entangling depth needed to reach a 2-design. For our setting
# (n=4, L=3) with shallow entanglement, max(d, L) ~ 3, so
#   Var ~ 2^-6 ... 2^-8  =>  typical |d_theta_i C| ~ 0.04 ... 0.12
#   typical ||grad|| over 45 params ~ sqrt(45) * 0.06 ~ 0.4

MCCLEAN_PER_PARAM = 0.06     # typical individual-parameter gradient
MCCLEAN_TOTAL     = float(np.sqrt(N_PARAMS) * MCCLEAN_PER_PARAM)
MCCLEAN_LAYER     = float(np.sqrt(N_PARAMS_PER_LAYER) * MCCLEAN_PER_PARAM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "qpinn_gradients"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "per_layer_norms.json"

    runs = []
    for seed in range(args.n_seeds):
        r = run_seed(seed, n_iters=args.n_iters)
        runs.append(r)
        with open(out_json, "w") as f:
            json.dump({
                "n_seeds": args.n_seeds, "n_iters": args.n_iters,
                "n_params": N_PARAMS, "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "mcclean_per_layer_threshold": MCCLEAN_LAYER,
                "mcclean_total_threshold": MCCLEAN_TOTAL,
                "runs": runs,
            }, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Figure 5: per-layer gradient norms vs iteration -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        layer_keys = ["grad_norm_layer_1", "grad_norm_layer_2",
                       "grad_norm_layer_3"]
        layer_labels = ["Layer 1", "Layer 2", "Layer 3"]
        colors = ["#0F4C5C", "#9A4836", "#5E7C6B"]
        for ax, key, label in zip(axes, layer_keys, layer_labels):
            for r in runs:
                ax.plot(r["log"][key], lw=1.2, alpha=0.7,
                         label=f"seed {r['seed']}")
            ax.axhline(MCCLEAN_LAYER, color="#222", lw=1.5, ls="--",
                        label=fr"McClean BP threshold $\sim$ {MCCLEAN_LAYER:.2f}")
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Iteration")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
        axes[0].set_ylabel(r"$\|\nabla_{\theta_\ell} \mathcal{L}\|$  (log scale)")
        fig.suptitle("Per-layer QPINN gradient norms across training "
                      "(measured vs McClean barren-plateau threshold)",
                      fontsize=13)
        fig.tight_layout()
        out_fig = out_dir / "per_layer_gradient_norms.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
